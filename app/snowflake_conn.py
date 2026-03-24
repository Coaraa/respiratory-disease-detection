import os
import streamlit as st
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent.parent / '.env')

@st.cache_resource(show_spinner=False)
def get_snowflake_connection(totp: str):
    """Connexion Snowflake partagée entre toutes les pages (cache unique)."""
    import snowflake.connector
    return snowflake.connector.connect(
        account       = os.environ['SNOWFLAKE_ACCOUNT'],
        user          = os.environ['SNOWFLAKE_USER'],
        password      = os.environ['SNOWFLAKE_PASSWORD'],
        authenticator = 'username_password_mfa',
        passcode      = totp,
        warehouse     = os.environ['SNOWFLAKE_WAREHOUSE'],
        database      = os.environ['SNOWFLAKE_DATABASE'],
        schema        = os.environ['SNOWFLAKE_SCHEMA'],
        role          = os.environ['SNOWFLAKE_ROLE'],
    )
