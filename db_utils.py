# db_utils.py

import pandas as pd
import streamlit as st
import requests
from db_connection import get_supabase_client
from datetime import date, datetime

##################################################
#            Supabase Client & Helpers
##################################################

def get_supabase():
    """Return the Supabase client from a global connection."""
    return get_supabase_client()

def client_table():
    """Shortcut to the 'clients' table."""
    return get_supabase().table("clients")

def portfolio_table():
    """Shortcut to the 'portfolios' table."""
    return get_supabase().table("portfolios")

def performance_table():
    """Shortcut to the 'performance_periods' table."""
    return get_supabase().table("performance_periods")

##################################################
#               MASI Fetch
##################################################

def fetch_masi_from_cb():
    """
    Fetches JSON from Casablanca Bourse for 'Principaux indices'
    and returns the current MASI index value as float.
    If not found, returns 0.0 (or logs an error).
    """
    url = "https://www.casablanca-bourse.com/api/proxy/fr/api/bourse/dashboard/grouped_index_watch?"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        # data = { "data": [ { "title": "...", "items": [...] }, ... ] }
        for block in data.get("data", []):
            if block.get("title") == "Principaux indices":
                for item in block.get("items", []):
                    if item.get("index") == "MASI":
                        val_str = item.get("field_index_value", "0")
                        return float(val_str)
        return 0.0
    except Exception as e:
        print("Error fetching MASI index from Casablanca Bourse:", e)
        return 0.0

##################################################
#       Fetching Stocks & Instruments
##################################################

@st.cache_data(ttl=60)
def _cached_fetch_stocks():
    """
    Actually fetch from IDBourse API, returning a DataFrame with columns: [valeur, cours].
    Adds a 'Cash' row with cours=1.
    """
    try:
        response = requests.get("https://backend.idbourse.com/api_2/get_all_data", timeout=10)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(
            [(s.get("name", "N/A"), s.get("dernier_cours", 0)) for s in data],
            columns=["valeur", "cours"]
        )
        # Add CASH row
        cash_row = pd.DataFrame([{"valeur": "Cash", "cours": 1}])
        return pd.concat([df, cash_row], ignore_index=True)
    except Exception as e:
        st.error(f"Failed to fetch stock data from IDBourse: {e}")
        return pd.DataFrame(columns=["valeur", "cours"])

def fetch_stocks():
    """Return the 'stocks' DataFrame from the IDBourse API, cached for 60s."""
    return _cached_fetch_stocks()

def fetch_instruments():
    """
    Return a DataFrame [instrument_name, nombre_de_titres, facteur_flottant]
    from the 'instruments' Supabase table.
    """
    client = get_supabase()
    res = client.table("instruments").select("*").execute()
    if not res.data:
        return pd.DataFrame(columns=["instrument_name","nombre_de_titres","facteur_flottant"])
    df = pd.DataFrame(res.data)
    needed_cols = ["instrument_name","nombre_de_titres","facteur_flottant"]
    for col in needed_cols:
        if col not in df.columns:
            df[col] = None
    return df[needed_cols].copy()

##################################################
#           Client / Portfolio / Performance
##################################################

def get_all_clients():
    """Return a list of all client names from 'clients' table."""
    res = client_table().select("*").execute()
    if not res.data:
        return []
    return [r["name"] for r in res.data]

def get_client_info(client_name: str):
    """
    Return the client row as a dict:
    { id, name, exchange_commission_rate, tax_on_gains_rate, 
      is_pea, management_fee_rate, bill_surperformance, ...}
    or None if not found.
    """
    res = client_table().select("*").eq("name", client_name).execute()
    if res.data:
        return res.data[0]
    return None

def get_client_id(client_name: str):
    """Return integer ID for this client or None if not found."""
    cinfo = get_client_info(client_name)
    if not cinfo:
        return None
    return int(cinfo["id"])

def client_has_portfolio(client_name: str) -> bool:
    """Check if 'client_name' already has at least one row in 'portfolios'."""
    cid = get_client_id(client_name)
    if cid is None:
        return False
    port = portfolio_table().select("*").eq("client_id", cid).execute()
    return len(port.data) > 0

def get_portfolio(client_name: str) -> pd.DataFrame:
    """Return a DataFrame with portfolio rows for 'client_name'."""
    cid = get_client_id(client_name)
    if cid is None:
        return pd.DataFrame()
    res = portfolio_table().select("*").eq("client_id", cid).execute()
    return pd.DataFrame(res.data)

##################################################
#        CRUD for Clients & Rates
##################################################

def create_client(name: str):
    """
    Insert a new client with minimal fields. 
    """
    if not name:
        st.error("Nom du client invalide.")
        return
    try:
        client_table().insert({"name": name}).execute()
        st.success(f"Client '{name}' créé avec succès!")
        st.rerun()
    except Exception as e:
        st.error(f"Erreur lors de la création du client: {e}")

def rename_client(old_name: str, new_name: str):
    """
    Update the 'name' field for an existing client.
    """
    cid = get_client_id(old_name)
    if cid is None:
        st.error("Client introuvable.")
        return
    try:
        client_table().update({"name": new_name}).eq("id", cid).execute()
        st.success(f"Client '{old_name}' renommé en '{new_name}'!")
        st.rerun()
    except Exception as e:
        st.error(f"Erreur lors du renommage: {e}")

def delete_client(cname: str):
    """Delete a client by name."""
    cid = get_client_id(cname)
    if cid is None:
        st.error("Client introuvable.")
        return
    try:
        client_table().delete().eq("id", cid).execute()
        st.success(f"Client '{cname}' supprimé.")
        st.rerun()
    except Exception as e:
        st.error(f"Erreur lors de la suppression du client: {e}")

def update_client_rates(client_name: str, 
                        exchange_comm: float, 
                        is_pea: bool, 
                        custom_tax: float, 
                        mgmt_fee: float, 
                        bill_surperf: bool):
    """
    Update the client's financial parameters:
      - exchange_commission_rate
      - tax_on_gains_rate (0 if is_pea=True)
      - is_pea
      - management_fee_rate
      - bill_surperformance => bool
    """
    cid = get_client_id(client_name)
    if cid is None:
        st.error("Client introuvable.")
        return
    try:
        final_tax = 0.0 if is_pea else float(custom_tax)
        client_table().update({
            "exchange_commission_rate": float(exchange_comm),
            "tax_on_gains_rate": final_tax,
            "is_pea": bool(is_pea),
            "management_fee_rate": float(mgmt_fee),
            "bill_surperformance": bool(bill_surperf)
        }).eq("id", cid).execute()
        st.success(f"Paramètres mis à jour pour « {client_name} ».")
        st.rerun()
    except Exception as e:
        st.error(f"Erreur lors de la mise à jour des taux: {e}")

##################################################
#       Performance Periods (start_value, etc.)
##################################################

def create_performance_period(client_id: int, 
                              start_date_str: str, 
                              start_val: float, 
                              masi_start_value: float):
    """
    Insert a new row in 'performance_periods'.
    Fields:
      - client_id
      - start_date => string 'YYYY-MM-DD'
      - start_value => float
      - masi_start_value => float
    """
    if not client_id:
        st.error("ID client invalide.")
        return
    try:
        row_data = {
            "client_id": client_id,
            "start_date": start_date_str,
            "start_value": start_val,
            "masi_start_value": masi_start_value
        }
        performance_table().insert(row_data).execute()
    except Exception as e:
        st.error(f"Erreur lors de la création d'une période de performance: {e}")

def get_performance_periods_for_client(client_id: int) -> pd.DataFrame:
    """Return all rows from 'performance_periods' for a given client, sorted by date ascending."""
    res = performance_table().select("*").eq("client_id", client_id).order("start_date", desc=False).execute()
    if not res.data:
        return pd.DataFrame()
    return pd.DataFrame(res.data)

def get_latest_performance_period_for_all_clients() -> pd.DataFrame:
    """
    For each client, pick the row with the max start_date from performance_periods.
    Return DataFrame with columns: [id, client_id, start_date, start_value, masi_start_value, ...].
    """
    res = performance_table().select("*").execute()
    if not res.data:
        return pd.DataFrame()
    df = pd.DataFrame(res.data)
    if df.empty or "start_date" not in df.columns:
        return pd.DataFrame()

    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df_sorted = df.sort_values(["client_id", "start_date"], ascending=[True, False])
    # groupby client_id => top row (most recent date)
    df_latest = df_sorted.groupby("client_id", as_index=False).head(1)
    return df_latest

def update_performance_period_rows(old_df: pd.DataFrame, new_df: pd.DataFrame):
    """
    Compare old_df vs new_df from st.data_editor and apply changes to 'performance_periods'.
    We'll assume each row has an integer 'id' primary key.
    Only updates existing rows (no new row insertion or old row deletion here).
    """
    # Convert date => string if needed
    for idx, row in new_df.iterrows():
        rec_id = row.get("id", None)
        if rec_id is None:
            continue  # if there's no ID, skip
        # We do a simple approach: re-insert start_date as isoformat if it's a date
        start_dt = row.get("start_date")
        if isinstance(start_dt, date):
            start_dt_str = start_dt.isoformat()
        elif isinstance(start_dt, datetime):
            start_dt_str = start_dt.date().isoformat()
        else:
            start_dt_str = str(start_dt)

        new_start_val = float(row.get("start_value", 0))
        new_masi_val = float(row.get("masi_start_value", 0))

        # Attempt an update
        try:
            performance_table().update({
                "start_date": start_dt_str,
                "start_value": new_start_val,
                "masi_start_value": new_masi_val
            }).eq("id", rec_id).execute()
        except Exception as e:
            st.error(f"Erreur lors de la mise à jour de la ligne id={rec_id}: {e}")
