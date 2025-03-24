# db_connection.py
import streamlit as st
from supabase import create_client

@st.cache_resource
def get_supabase_client():
    """
    Creates and returns a Supabase client object (cached as a resource).
    """
    supabase_url = st.secrets["supabase"]["url"]
    supabase_key = st.secrets["supabase"]["key"]
    return create_client(supabase_url, supabase_key)
