import sys
import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(Path(__file__).parent.parent))
from snowflake_conn import get_snowflake_connection

# Mapping classes anglais 
CLASS_MAP = {
    'asthma':    'Asthme',
    'bronchial': 'Bronchite',
    'copd':      'BPCO',
    'healthy':   'Sain',
    'pneumonia': 'Pneumonie',
}

DISEASE_COLORS = {
    'Asthme':    '#F59E0B',
    'BPCO':      '#EF4444',
    'Bronchite': '#8B5CF6',
    'Pneumonie': '#F97316',
    'Sain':      '#10B981',
}

DISEASE_COLORS_HEX = {k: v for k, v in DISEASE_COLORS.items()}

CLASSES_FR = ['Asthme', 'BPCO', 'Bronchite', 'Pneumonie', 'Sain']

FOLIUM_ICON_COLOR = {
    'Asthme':    'orange',
    'BPCO':      'red',
    'Bronchite': 'purple',
    'Pneumonie': 'darkred',
    'Sain':      'green',
}

@st.cache_data
def load_pharmacies():
    df = pd.read_csv(ROOT / 'tessan_pharmacies.csv', encoding='utf-8-sig')
    df['DEP_CODE'] = df['DEPARTEMENT'].astype(str).str.strip().str.zfill(2)
    return df

def build_folium_map(df, disease_filter):

    # Agrégation par département
    dep_counts = df.groupby(['DEP_CODE', 'classe_predite']).size().reset_index(name='cas')
    dep_total  = df.groupby('DEP_CODE').size().reset_index(name='total')
    dominant = (
        dep_counts.sort_values('cas', ascending=False)
        .drop_duplicates('DEP_CODE')[['DEP_CODE', 'classe_predite', 'cas']]
        .rename(columns={'classe_predite': 'dominante', 'cas': 'cas_dominante'})
    )
    dep_agg = dep_total.merge(dominant, on='DEP_CODE', how='left')
    dep_agg['dominante'] = dep_agg['dominante'].fillna('Sain')

    for disease in CLASSES_FR:
        col = f'cas_{disease}'
        sub = dep_counts[dep_counts['classe_predite'] == disease][['DEP_CODE', 'cas']].rename(columns={'cas': col})
        dep_agg = dep_agg.merge(sub, on='DEP_CODE', how='left')
        dep_agg[col] = dep_agg[col].fillna(0).astype(int)

    # GeoJSON départements
    geojson = load_geojson()

    # Carte Folium
    m = folium.Map(
        location=[46.5, 2.5],
        zoom_start=6,
        tiles=None,
        prefer_canvas=True,
    )
    folium.TileLayer(
        tiles='https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
        attr=' ',
        name='CartoDB Positron',
    ).add_to(m)

    # Choroplèthe
    if disease_filter == 'Tous':
        # Convertit dominante 
        dep_color_map = dep_agg.set_index('DEP_CODE')['dominante'].to_dict()

        def style_dominant(feature):
            code     = feature['properties']['code']
            disease  = dep_color_map.get(code, 'Sain')
            color    = DISEASE_COLORS.get(disease, '#94A3B8')
            return {
                'fillColor':   color,
                'color':       'white',
                'weight':      0.8,
                'fillOpacity': 0.55,
            }

        folium.GeoJson(
            geojson,
            style_function=style_dominant,
            tooltip=folium.GeoJsonTooltip(
                fields=['code', 'nom'],
                aliases=['Département :', 'Nom :'],
                localize=True,
            ),
            name='Pathologie dominante',
        ).add_to(m)

    else:
        col = f'cas_{disease_filter}'
        max_val = dep_agg[col].max() or 1
        dep_val_map = dep_agg.set_index('DEP_CODE')[col].to_dict()
        base_color  = DISEASE_COLORS.get(disease_filter, '#94A3B8')

        def style_disease(feature):
            code  = feature['properties']['code']
            val   = dep_val_map.get(code, 0)
            alpha = 0.1 + 0.7 * (val / max_val)
            return {
                'fillColor':   base_color,
                'color':       'white',
                'weight':      0.8,
                'fillOpacity': alpha,
            }

        folium.GeoJson(
            geojson,
            style_function=style_disease,
            tooltip=folium.GeoJsonTooltip(
                fields=['code', 'nom'],
                aliases=['Département :', 'Nom :'],
            ),
            name=f'Cas {disease_filter}',
        ).add_to(m)

    # Légende colorée 
    if disease_filter == 'Tous':
        legend_html = """
        <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                    background:white;padding:12px 16px;border-radius:10px;
                    box-shadow:0 2px 8px rgba(0,0,0,.2);font-size:13px;">
          <b style="display:block;margin-bottom:8px;">Pathologie dominante</b>
        """
        for disease, color in DISEASE_COLORS.items():
            legend_html += (
                f'<span style="display:inline-block;width:12px;height:12px;'
                f'border-radius:50%;background:{color};margin-right:6px;vertical-align:middle;"></span>'
                f'{disease}<br>'
            )
        legend_html += '</div>'
        m.get_root().html.add_child(folium.Element(legend_html))

    # Clusters pharmacies
    pharm_stats = (
        df.groupby(['PHARMACIE_ID', 'NOM', 'VILLE'])
        .agg(
            lat=('LATITUDE',  'first'),
            lon=('LONGITUDE', 'first'),
            total=('classe_predite', 'count'),
            dominante=('classe_predite', lambda x: x.value_counts().idxmax()),
        )
        .reset_index()
    )
    for disease in CLASSES_FR:
        pharm_stats[disease] = (
            df[df['classe_predite'] == disease]
            .groupby('PHARMACIE_ID').size()
            .reindex(pharm_stats['PHARMACIE_ID']).fillna(0).astype(int).values
        )

    cluster = MarkerCluster(
        name='Cabines Tessan',
        options={
            'maxClusterRadius': 40,
            'disableClusteringAtZoom': 10,
        },
    ).add_to(m)

    for _, row in pharm_stats.iterrows():
        if pd.isna(row['lat']) or pd.isna(row['lon']):
            continue

        dominant   = row['dominante']
        icon_color = FOLIUM_ICON_COLOR.get(dominant, 'gray')
        dom_color  = DISEASE_COLORS.get(dominant, '#94A3B8')
        total      = row['total']

        # Barres de répartition
        bars_html = ''
        for d in CLASSES_FR:
            cnt   = row[d]
            pct   = (cnt / total * 100) if total > 0 else 0
            color = DISEASE_COLORS.get(d, '#94A3B8')
            bars_html += f"""
            <div style="margin-bottom:5px">
              <div style="display:flex;justify-content:space-between;
                          font-size:11px;margin-bottom:2px">
                <span style="color:{color};font-weight:600">{d}</span>
                <span style="color:#374151">{cnt} cas &nbsp;({pct:.0f}%)</span>
              </div>
              <div style="background:#F3F4F6;border-radius:4px;height:7px">
                <div style="width:{pct}%;background:{color};
                            border-radius:4px;height:7px;
                            transition:width .3s"></div>
              </div>
            </div>"""

        popup_html = f"""
        <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
                    width:280px">

          <!-- En-tête -->
          <div style="background:{dom_color};padding:12px 14px;
                      border-radius:6px 6px 0 0;margin:-1px -1px 0">
            <div style="font-size:14px;font-weight:700;color:white;
                        white-space:nowrap;overflow:hidden;
                        text-overflow:ellipsis">{row['NOM']}</div>
            <div style="font-size:12px;color:rgba(255,255,255,.8);
                        margin-top:2px">{row['VILLE']}</div>
          </div>

          <!-- Stats rapides -->
          <div style="display:flex;gap:8px;padding:10px 14px;
                      background:#F9FAFB;border-bottom:1px solid #E5E7EB">
            <div style="text-align:center;flex:1">
              <div style="font-size:20px;font-weight:700;color:#111827">{total}</div>
              <div style="font-size:10px;color:#6B7280;text-transform:uppercase;
                          letter-spacing:.5px">Analyses</div>
            </div>
            <div style="text-align:center;flex:1">
              <div style="font-size:14px;font-weight:700;color:{dom_color}">{dominant}</div>
              <div style="font-size:10px;color:#6B7280;text-transform:uppercase;
                          letter-spacing:.5px">Dominante</div>
            </div>
          </div>

          <!-- Répartition -->
          <div style="padding:12px 14px">
            <div style="font-size:11px;font-weight:700;color:#6B7280;
                        text-transform:uppercase;letter-spacing:.5px;
                        margin-bottom:8px">Répartition</div>
            {bars_html}
          </div>
        </div>
        """

        folium.Marker(
            location=[row['lat'], row['lon']],
            popup=folium.Popup(popup_html, max_width=320),
            tooltip=folium.Tooltip(
                f"<b>{row['NOM']}</b><br>"
                f"<span style='color:{dom_color}'>{dominant}</span> · {total} analyses",
                style="font-family:sans-serif;font-size:12px;padding:6px 10px;"
                      "background:white;border:none;box-shadow:0 2px 6px rgba(0,0,0,.15);"
                      "border-radius:6px",
            ),
            icon=folium.Icon(color=icon_color, icon='plus-sign', prefix='glyphicon'),
        ).add_to(cluster)

    folium.LayerControl().add_to(m)
    return m

@st.cache_data
def load_geojson():
    url = 'https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/departements-version-simplifiee.geojson'
    return requests.get(url, timeout=10).json()

@st.cache_data(ttl=300, show_spinner="Chargement depuis Snowflake…")
def load_predictions_sf(totp: str, days: int) -> pd.DataFrame | None:
    """Charge les prédictions depuis Snowflake et les enrichit avec les coordonnées GPS."""
    try:
        conn   = get_snowflake_connection(totp)
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT TIMESTAMP, PHARMACIE_ID, CLASSE_PREDITE, CONFIANCE
            FROM PREDICTIONS
            WHERE TIMESTAMP >= DATEADD(day, -{days}, CURRENT_TIMESTAMP())
            ORDER BY TIMESTAMP DESC
        """)
        rows = cursor.fetchall()
        cols = [d[0].lower() for d in cursor.description]
        cursor.close()

        df = pd.DataFrame(rows, columns=cols)
        if df.empty:
            return df

        df['timestamp']      = pd.to_datetime(df['timestamp'])
        df['confiance']      = df['confiance'].astype(float)
        df['classe_predite'] = df['classe_predite'].map(CLASS_MAP).fillna(df['classe_predite'])
        df = df.rename(columns={'pharmacie_id': 'PHARMACIE_ID'})
        df['PHARMACIE_ID'] = df['PHARMACIE_ID'].astype(str)

        pharmacies_df = load_pharmacies()
        pharmacies_df = pharmacies_df.copy()
        pharmacies_df['PHARMACIE_ID'] = pharmacies_df['PHARMACIE_ID'].astype(str)
        df = df.merge(
            pharmacies_df[['PHARMACIE_ID', 'NOM', 'VILLE', 'DEPARTEMENT', 'DEP_CODE', 'REGION', 'LATITUDE', 'LONGITUDE']],
            on='PHARMACIE_ID', how='left',
        )
        return df
    except Exception as e:
        if 'TOTP' in str(e) or '250001' in str(e):
            if '_sf' in st.query_params:
                del st.query_params['_sf']
            get_snowflake_connection.clear()
            st.rerun()
        st.warning(f"Snowflake indisponible ({e})")
        return None


PERIOD_OPTIONS = {
    "7 derniers jours":   7,
    "1 mois":             30,
    "3 mois":             90,
    "1 an":               365,
}

def dashboard_page():
    pharmacies_df = load_pharmacies()

    # Sidebar : connexion Snowflake 
    with st.sidebar:
        st.subheader("🔌 Snowflake")
        if "_sf" not in st.query_params:
            totp = st.text_input("Code MFA (6 chiffres)", max_chars=6, type="password")
            if st.button("Connecter") and totp:
                try:
                    get_snowflake_connection(totp)
                    st.query_params["_sf"] = totp
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ {e}")
        else:
            st.success("✅ Connecté")
            if st.button("Déconnecter"):
                del st.query_params["_sf"]
                st.rerun()

    # Chargement des données Snowflake
    totp = st.query_params.get("_sf")
    if not totp:
        st.title("Tableau de Bord Épidémiologique")
        st.info("Connectez-vous à Snowflake via la sidebar pour afficher les données.")
        st.stop()

    st.title("Tableau de Bord Épidémiologique")
    st.caption(f"Réseau Tessan · {len(pharmacies_df):,} cabines actives · Données Snowflake en direct")
    st.markdown("---")

    # Sélecteur de période 
    period_label = st.pills(
        "Période d'analyse",
        options=list(PERIOD_OPTIONS.keys()),
        default="1 mois",
        key="period_filter",
    )
    date_range = PERIOD_OPTIONS[period_label]

    df = load_predictions_sf(totp, date_range)
    if df is None or df.empty:
        st.warning("Aucune donnée disponible pour cette période.")
        st.stop()

    st.markdown("---")

    # KPIs 
    _, k1, k2, k3, k4, _ = st.columns([1, 2, 2, 2, 2, 1])
    k1.metric("Analyses",                      f"{len(df):,}")
    k2.metric("Cas graves (BPCO + Pneumonie)", f"{len(df[df['classe_predite'].isin(['Pneumonie','BPCO'])]):,}")
    k3.metric("Confiance IA",                  f"{df['confiance'].mean():.1%}")
    k4.metric("Cabines actives",               f"{df['PHARMACIE_ID'].nunique():,}")

    st.markdown("---")

    # Carte 
    st.subheader("Carte épidémiologique par département")

    disease_filter = st.pills(
        "Filtrer par pathologie",
        options=['Tous'] + CLASSES_FR,
        default='Tous',
        key='map_disease_filter',
    )

    fmap = build_folium_map(df, disease_filter)
    st_folium(fmap, use_container_width=True, height=620, returned_objects=[])

    st.markdown("---")

    # Évolution + Répartition
    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("Évolution temporelle")
        df_trend = (
            df.groupby([df['timestamp'].dt.date, 'classe_predite'])
            .size().reset_index(name='cas')
        )
        fig_trend = px.line(
            df_trend, x='timestamp', y='cas', color='classe_predite',
            color_discrete_map=DISEASE_COLORS, markers=True,
            labels={'timestamp': 'Date', 'cas': 'Cas', 'classe_predite': 'Pathologie'},
        )
        fig_trend.update_layout(
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
            margin=dict(l=0, r=0, t=30, b=0),
            hovermode='x unified',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig_trend, use_container_width=True, config={'displayModeBar': False})

    with col2:
        st.subheader("Répartition")
        dist = df['classe_predite'].value_counts().reset_index()
        dist.columns = ['Pathologie', 'Cas']
        fig_pie = px.pie(
            dist, names='Pathologie', values='Cas',
            color='Pathologie', color_discrete_map=DISEASE_COLORS, hole=0.45,
            height=320,
        )
        fig_pie.update_traces(textposition='outside', textinfo='percent+label')
        fig_pie.update_layout(
            showlegend=False,
            margin=dict(l=40, r=40, t=40, b=40),
            paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar': False})

    # Top départements barres empilées 
    st.subheader("Départements les plus actifs")

    geojson = load_geojson()
    code_to_name = {f['properties']['code']: f['properties']['nom'] for f in geojson['features']}

    dep_disease = (
        df.groupby(['DEP_CODE', 'classe_predite'])
        .size().reset_index(name='cas')
    )
    dep_disease['Département'] = dep_disease['DEP_CODE'].map(code_to_name).fillna(dep_disease['DEP_CODE'])

    dep_totals  = dep_disease.groupby('Département')['cas'].sum().nlargest(20).index
    dep_top     = dep_disease[dep_disease['Département'].isin(dep_totals)]

    fig_bar = px.bar(
        dep_top,
        x='Département', y='cas', color='classe_predite',
        color_discrete_map=DISEASE_COLORS,
        labels={'cas': 'Analyses', 'classe_predite': 'Pathologie'},
        category_orders={'Département': list(dep_totals)},
    )
    fig_bar.update_layout(
        barmode='stack',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(type='category', tickangle=-35),
    )
    st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})
