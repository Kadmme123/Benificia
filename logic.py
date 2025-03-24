# logic.py

import streamlit as st
import pandas as pd

import db_utils
from db_utils import (
    get_portfolio,
    get_client_info,
    get_client_id,
    portfolio_table,
    fetch_instruments,
    fetch_stocks,
    client_has_portfolio,
    create_performance_period,
    get_performance_periods_for_client,
    get_latest_performance_period_for_all_clients,
)

######################################################
#     Real-time MASI Fetch
######################################################

def get_current_masi():
    """Return the real-time MASI index from Casablanca Bourse."""
    return db_utils.fetch_masi_from_cb()

######################################################
#  Compute Poids Masi for each "valeur"
######################################################

def compute_poids_masi():
    """
    Creates a dictionary { valeur: {"capitalisation": X, "poids_masi": Y}, ... }
    by merging instruments + stocks => capitalisation => floated_cap => sum => percentage.
    """
    instruments_df = fetch_instruments()
    if instruments_df.empty:
        return {}

    stocks_df = fetch_stocks()
    instr_renamed = instruments_df.rename(columns={"instrument_name": "valeur"})
    merged = pd.merge(instr_renamed, stocks_df, on="valeur", how="left")

    merged["cours"] = merged["cours"].fillna(0.0).astype(float)
    merged["nombre_de_titres"] = merged["nombre_de_titres"].fillna(0.0).astype(float)
    merged["facteur_flottant"] = merged["facteur_flottant"].fillna(0.0).astype(float)

    # exclude zero
    merged = merged[(merged["cours"] != 0.0) & (merged["nombre_de_titres"] != 0.0)].copy()

    merged["capitalisation"] = merged["cours"] * merged["nombre_de_titres"]
    merged["floated_cap"]    = merged["capitalisation"] * merged["facteur_flottant"]
    tot_floated = merged["floated_cap"].sum()
    if tot_floated <= 0:
        merged["poids_masi"] = 0.0
    else:
        merged["poids_masi"] = (merged["floated_cap"] / tot_floated) * 100.0

    outdict = {}
    for _, row in merged.iterrows():
        val = row["valeur"]
        outdict[val] = {
            "capitalisation": row["capitalisation"],
            "poids_masi": row["poids_masi"]
        }
    return outdict

# Global dictionary for Poids Masi
poids_masi_map = compute_poids_masi()

######################################################
#   Create a brand-new portfolio
######################################################

def create_portfolio_rows(client_name: str, holdings: dict):
    """
    Upserts rows (valeur -> quantity) into 'portfolios' if the client has no portfolio.
    If they do, we do a warning.
    """
    cid = get_client_id(client_name)
    if cid is None:
        st.error("Client not found.")
        return

    if client_has_portfolio(client_name):
        st.warning(f"Le client '{client_name}' possède déjà un portefeuille.")
        return

    rows = []
    for stock, qty in holdings.items():
        if qty > 0:
            rows.append({
                "client_id": cid,
                "valeur": str(stock),
                "quantité": float(qty),
                "vwap": 0.0,
                "cours": 0.0,
                "valorisation": 0.0
            })

    if not rows:
        st.warning("Aucun actif fourni pour la création du portefeuille.")
        return

    try:
        portfolio_table().upsert(rows, on_conflict="client_id,valeur").execute()
        st.success(f"Portefeuille créé pour '{client_name}'!")
        st.rerun()
    except Exception as e:
        st.error(f"Erreur création du portefeuille: {e}")

def new_portfolio_creation_ui(client_name: str):
    """
    Lets the user pick stocks/cash to add to a brand-new portfolio via st.session_state.
    """
    st.subheader(f"➕ Définir les actifs initiaux pour {client_name}")

    if "temp_holdings" not in st.session_state:
        st.session_state.temp_holdings = {}

    all_stocks = fetch_stocks()
    chosen_val = st.selectbox(f"Choisir une valeur ou 'Cash'", all_stocks["valeur"].tolist(), key=f"new_stock_{client_name}")
    qty = st.number_input(
        f"Quantité pour {client_name}",
        min_value=1.0,
        value=1.0,
        step=1.0,
        key=f"new_qty_{client_name}"
    )

    if st.button(f"➕ Ajouter {chosen_val}", key=f"add_btn_{client_name}"):
        st.session_state.temp_holdings[chosen_val] = float(qty)
        st.success(f"Ajouté {qty} de {chosen_val}")

    if st.session_state.temp_holdings:
        st.write("### Actifs Sélectionnés :")
        df_hold = pd.DataFrame([
            {"valeur": k, "quantité": v} for k, v in st.session_state.temp_holdings.items()
        ])
        st.dataframe(df_hold, use_container_width=True)

        if st.button(f"💾 Créer le Portefeuille pour {client_name}", key=f"create_pf_btn_{client_name}"):
            create_portfolio_rows(client_name, st.session_state.temp_holdings)
            del st.session_state.temp_holdings

######################################################
#        Buy / Sell
######################################################

def buy_shares(client_name: str, stock_name: str, transaction_price: float, quantity: float):
    cinfo = get_client_info(client_name)
    if not cinfo:
        st.error("Informations du client introuvables.")
        return

    cid = get_client_id(client_name)
    if cid is None:
        st.error("Client introuvable.")
        return

    dfp = get_portfolio(client_name)
    exchange_rate = float(cinfo.get("exchange_commission_rate") or 0.0)

    raw_cost = transaction_price * quantity
    commission = raw_cost * (exchange_rate / 100.0)
    cost_with_comm = raw_cost + commission

    # Check Cash
    cash_match = dfp[dfp["valeur"] == "Cash"]
    current_cash = float(cash_match["quantité"].values[0]) if not cash_match.empty else 0.0
    if cost_with_comm > current_cash:
        st.error(f"Montant insuffisant en Cash: {current_cash:,.2f} < {cost_with_comm:,.2f}")
        return

    # Check if stock exists
    match = dfp[dfp["valeur"] == stock_name]
    if match.empty:
        # Insert new
        new_vwap = cost_with_comm / quantity
        try:
            portfolio_table().upsert([{
                "client_id": cid,
                "valeur": stock_name,
                "quantité": quantity,
                "vwap": new_vwap,
                "cours": 0.0,
                "valorisation": 0.0
            }], on_conflict="client_id,valeur").execute()
        except Exception as e:
            st.error(f"Erreur lors de l'ajout de {stock_name}: {e}")
            return
    else:
        # update
        old_qty = float(match["quantité"].values[0])
        old_vwap = float(match["vwap"].values[0])
        old_cost = old_qty * old_vwap
        new_cost = old_cost + cost_with_comm
        new_qty = old_qty + quantity
        new_vwap = new_cost / new_qty if new_qty > 0 else 0.0
        try:
            portfolio_table().update({
                "quantité": new_qty,
                "vwap": new_vwap
            }).eq("client_id", cid).eq("valeur", stock_name).execute()
        except Exception as e:
            st.error(f"Erreur mise à jour stock {stock_name}: {e}")
            return

    # Update Cash
    new_cash = current_cash - cost_with_comm
    if cash_match.empty:
        try:
            portfolio_table().upsert([{
                "client_id": cid,
                "valeur": "Cash",
                "quantité": new_cash,
                "vwap": 1.0,
                "cours": 0.0,
                "valorisation": 0.0
            }], on_conflict="client_id,valeur").execute()
        except Exception as e:
            st.error(f"Erreur insertion Cash: {e}")
            return
    else:
        try:
            portfolio_table().update({
                "quantité": new_cash,
                "vwap": 1.0
            }).eq("client_id", cid).eq("valeur", "Cash").execute()
        except Exception as e:
            st.error(f"Erreur mise à jour Cash: {e}")
            return

    st.success(
        f"Achat de {quantity:.0f} '{stock_name}' @ {transaction_price:,.2f}, "
        f"coût total {cost_with_comm:,.2f} (commission incluse)."
    )
    st.rerun()

def sell_shares(client_name: str, stock_name: str, transaction_price: float, quantity: float):
    cinfo = get_client_info(client_name)
    if not cinfo:
        st.error("Client introuvable.")
        return

    cid = get_client_id(client_name)
    if cid is None:
        st.error("Client introuvable.")
        return

    exchange_rate = float(cinfo.get("exchange_commission_rate") or 0.0)
    tax_rate      = float(cinfo.get("tax_on_gains_rate") or 15.0)

    dfp = get_portfolio(client_name)
    match = dfp[dfp["valeur"] == stock_name]
    if match.empty:
        st.error(f"Le client ne possède pas {stock_name}.")
        return

    old_qty = float(match["quantité"].values[0])
    if quantity > old_qty:
        st.error(f"Quantité insuffisante: vend {quantity}, possède {old_qty}.")
        return

    old_vwap      = float(match["vwap"].values[0])
    raw_proceeds  = transaction_price * quantity
    commission    = raw_proceeds * (exchange_rate / 100.0)
    net_proceeds  = raw_proceeds - commission

    cost_of_shares = old_vwap * quantity
    profit = net_proceeds - cost_of_shares
    if profit > 0:
        tax = profit * (tax_rate / 100.0)
        net_proceeds -= tax

    new_qty = old_qty - quantity
    try:
        if new_qty <= 0:
            portfolio_table().delete().eq("client_id", cid).eq("valeur", stock_name).execute()
        else:
            portfolio_table().update({"quantité": new_qty}).eq("client_id", cid).eq("valeur", stock_name).execute()
    except Exception as e:
        st.error(f"Erreur mise à jour après vente: {e}")
        return

    # Update Cash
    cash_match = dfp[dfp["valeur"] == "Cash"]
    old_cash = float(cash_match["quantité"].values[0]) if not cash_match.empty else 0.0
    new_cash = old_cash + net_proceeds

    try:
        if cash_match.empty:
            portfolio_table().upsert([{
                "client_id": cid,
                "valeur": "Cash",
                "quantité": new_cash,
                "vwap": 1.0,
                "cours": 0.0,
                "valorisation": 0.0
            }], on_conflict="client_id,valeur").execute()
        else:
            portfolio_table().update({
                "quantité": new_cash,
                "vwap": 1.0
            }).eq("client_id", cid).eq("valeur", "Cash").execute()
    except Exception as e:
        st.error(f"Erreur mise à jour Cash: {e}")
        return

    st.success(
        f"Vendu {quantity:.0f} '{stock_name}' @ {transaction_price:,.2f}, "
        f"net {net_proceeds:,.2f} (commission + taxe gains)."
    )
    st.rerun()
