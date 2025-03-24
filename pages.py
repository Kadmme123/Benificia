import streamlit as st
import pandas as pd
import json  
from collections import defaultdict
from datetime import date

import db_utils 
from db_utils import (
    get_all_clients,
    get_client_id,
    get_client_info,
    create_client,
    rename_client,
    delete_client,
    update_client_rates,
    client_has_portfolio,
    get_portfolio,
    get_supabase, 
    get_performance_periods_for_client,
    create_performance_period,
    get_latest_performance_period_for_all_clients,
    fetch_stocks
)
from logic import (
    buy_shares,
    sell_shares,
    new_portfolio_creation_ui,
    poids_masi_map,
    get_current_masi
)


########################################
# 1) Manage Clients Page
########################################
def page_manage_clients():
    st.title("Gestion des Clients")
    existing = get_all_clients()

    # --- Form: Create New Client ---
    with st.form("add_client_form", clear_on_submit=True):
        new_client_name = st.text_input("Nom du nouveau client", key="new_client_input")
        if st.form_submit_button("➕ Créer le client"):
            create_client(new_client_name)

    # --- If clients exist, allow rename & delete ---
    if existing:
        with st.form("rename_client_form", clear_on_submit=True):
            rename_choice = st.selectbox("Sélectionner le client à renommer", options=existing, key="rename_choice")
            rename_new = st.text_input("Nouveau nom du client", key="rename_text")
            if st.form_submit_button("✏️ Renommer ce client"):
                rename_client(rename_choice, rename_new)

        with st.form("delete_client_form", clear_on_submit=True):
            delete_choice = st.selectbox("Sélectionner le client à supprimer", options=existing, key="delete_choice")
            if st.form_submit_button("🗑️ Supprimer ce client"):
                delete_client(delete_choice)


########################################
# 2) Create Portfolio Page
########################################
def page_create_portfolio():
    st.title("Création d'un Portefeuille Client")
    clist = get_all_clients()
    if not clist:
        st.warning("Aucun client trouvé. Veuillez d'abord créer un client.")
    else:
        cselect = st.selectbox("Sélectionner un client", clist, key="create_pf_select")
        if cselect:
            if client_has_portfolio(cselect):
                st.warning(f"Le client '{cselect}' dispose déjà d'un portefeuille.")
            else:
                new_portfolio_creation_ui(cselect)


########################################
# 3) Afficher / Gérer un portefeuille
########################################
def show_portfolio(client_name, read_only=False):
    cid = get_client_id(client_name)
    if cid is None:
        st.warning("Client introuvable.")
        return

    df = get_portfolio(client_name)
    if df.empty:
        st.warning(f"Aucun portefeuille trouvé pour « {client_name} ».")
        return

    stocks = db_utils.fetch_stocks()
    df = df.copy()

    # Convert "quantité" to integer if it exists
    if "quantité" in df.columns:
        # We attempt an integer cast: if there's any fractional you want to floor or round
        df["quantité"] = df["quantité"].astype(int, errors="ignore")

    # Recalculate columns
    for i, row in df.iterrows():
        val = str(row["valeur"])
        match = stocks[stocks["valeur"] == val]
        live_price = float(match["cours"].values[0]) if not match.empty else 0.0
        df.at[i, "cours"] = live_price

        qty_ = float(row.get("quantité", 0))
        vw_  = float(row.get("vwap", 0.0))
        val_ = round(qty_ * live_price, 2)
        df.at[i, "valorisation"] = val_

        cost_ = round(qty_ * vw_, 2)
        df.at[i, "cost_total"] = cost_
        df.at[i, "performance_latente"] = round(val_ - cost_, 2)

        # Poids Masi => 0 if "Cash"
        if val == "Cash":
            df.at[i, "poids_masi"] = 0.0
        else:
            info = poids_masi_map.get(val, {"poids_masi": 0.0})
            df.at[i, "poids_masi"] = info["poids_masi"]

    # Compute total
    total_val = df["valorisation"].sum()
    if total_val > 0:
        df["poids"] = ((df["valorisation"] / total_val) * 100).round(2)
    else:
        df["poids"] = 0.0

    # Put "Cash" at bottom
    df["__cash_marker"] = df["valeur"].apply(lambda x: 1 if x == "Cash" else 0)
    df.sort_values("__cash_marker", inplace=True, ignore_index=True)

    st.subheader(f"Portefeuille de {client_name}")
    st.write(f"**Valorisation totale du portefeuille :** {total_val:,.2f}")

    # If read_only => style only
    if read_only:
        drop_cols = ["id", "client_id", "is_cash", "__cash_marker"]
        for c in drop_cols:
            if c in df.columns:
                df.drop(columns=c, inplace=True)

        columns_display = [
            "valeur", "quantité", "vwap", "cours",
            "cost_total", "valorisation", "performance_latente",
            "poids", "poids_masi"
        ]
        df_disp = df[columns_display].copy()

        def color_perf(x):
            if isinstance(x, (float, int)) and x > 0:
                return "color:green;"
            elif isinstance(x, (float, int)) and x < 0:
                return "color:red;"
            return ""

        def bold_cash(row):
            if row["valeur"] == "Cash":
                return ["font-weight:bold;"] * len(row)
            return ["" for _ in row]

        df_styled = df_disp.style.format(
            "{:,.2f}",
            subset=["quantité", "vwap", "cours", "cost_total", "valorisation", "performance_latente", "poids", "poids_masi"]
        ).applymap(color_perf, subset=["performance_latente"]) \
         .apply(bold_cash, axis=1)

        st.dataframe(df_styled, use_container_width=True)
        return

    # Not read_only => let user edit commissions + buy/sell
    cinfo = get_client_info(client_name)
    if cinfo:
        with st.expander(f"Modifier Commissions / Taxes / Frais pour {client_name}", expanded=False):
            exch = float(cinfo.get("exchange_commission_rate") or 0.0)
            mgf  = float(cinfo.get("management_fee_rate") or 0.0)
            pea  = bool(cinfo.get("is_pea") or False)
            tax  = float(cinfo.get("tax_on_gains_rate") or 15.0)
            bill_surf = bool(cinfo.get("bill_surperformance", False))

            new_exch = st.number_input(
                "Commission d'intermédiation (%)", min_value=0.0, value=exch, step=0.01
            )
            new_mgmt = st.number_input(
                "Frais de gestion (%)", min_value=0.0, value=mgf, step=0.01
            )
            new_pea  = st.checkbox("Compte PEA ?", value=pea)
            new_tax  = st.number_input("Taux d'imposition sur les gains (%)", min_value=0.0, value=tax, step=0.01)
            new_bill = st.checkbox("Facturer Surperformance ?", value=bill_surf)

            if st.button(f"Mettre à jour les paramètres pour {client_name}"):
                update_client_rates(client_name, new_exch, new_pea, new_tax, new_mgmt, new_bill)

    # Display the portfolio again
    columns_display = [
        "valeur", "quantité", "vwap", "cours",
        "cost_total", "valorisation", "performance_latente",
        "poids_masi", "poids", "__cash_marker"
    ]
    df2 = df[columns_display].copy()

    def color_perf(x):
        if isinstance(x, (float,int)) and x>0:
            return "color:green;"
        elif isinstance(x,(float,int)) and x<0:
            return "color:red;"
        return ""

    def bold_cash(row):
        if row["valeur"]=="Cash":
            return ["font-weight:bold;"]*len(row)
        return ["" for _ in row]

    df_styled = df2.drop(columns="__cash_marker").style.format(
        "{:,.2f}",
        subset=["quantité","vwap","cours","cost_total","valorisation","performance_latente","poids_masi","poids"]
    ).applymap(color_perf, subset=["performance_latente"]) \
     .apply(bold_cash, axis=1)

    st.write("#### Actifs actuels du portefeuille")
    st.dataframe(df_styled, use_container_width=True)

    with st.expander("Édition manuelle (Quantité / VWAP)", expanded=False):
        edit_cols = ["valeur", "quantité", "vwap"]
        edf = df2[edit_cols].drop(columns="__cash_marker", errors="ignore").copy()

        # force 'quantité' to int
        edf["quantité"] = edf["quantité"].astype(int, errors="ignore")

        updated_df = st.data_editor(edf, use_container_width=True)
        if st.button("💾 Enregistrer modifications"):
            from db_utils import portfolio_table
            cid2 = get_client_id(client_name)
            for idx, row2 in updated_df.iterrows():
                valn = str(row2["valeur"])
                qn   = int(row2["quantité"])
                vw   = float(row2["vwap"])
                try:
                    portfolio_table().update({
                        "quantité": qn,
                        "vwap": vw
                    }).eq("client_id", cid2).eq("valeur", valn).execute()
                except Exception as e:
                    st.error(f"Erreur lors de la sauvegarde pour {valn}: {e}")
            st.success(f"Portefeuille de « {client_name} » mis à jour avec succès!")
            st.rerun()

    # BUY
    st.write("### Opération d'Achat")
    _stocks = db_utils.fetch_stocks()
    buy_stock = st.selectbox("Choisir la valeur à acheter", _stocks["valeur"].tolist())
    buy_price = st.number_input("Prix d'achat", min_value=0.0, value=0.0, step=0.01)
    buy_qty   = st.number_input("Quantité à acheter", min_value=1, value=1, step=1)
    if st.button("Acheter"):
        buy_shares(client_name, buy_stock, buy_price, float(buy_qty))

    # SELL
    st.write("### Opération de Vente")
    existing_stocks = df2[df2["valeur"] != "Cash"]["valeur"].unique().tolist()
    sell_stock = st.selectbox("Choisir la valeur à vendre", existing_stocks)
    sell_price = st.number_input("Prix de vente", min_value=0.0, value=0.0, step=0.01)
    sell_qty   = st.number_input("Quantité à vendre", min_value=1, value=1, step=1)
    if st.button("Vendre"):
        sell_shares(client_name, sell_stock, sell_price, float(sell_qty))


########################################
# 4) View Single Portfolio
########################################
def page_view_client_portfolio():
    st.title("Portefeuille d'un Client")
    c2 = get_all_clients()
    if not c2:
        st.warning("Aucun client trouvé.")
        return

    client_selected = st.selectbox("Sélectionner un client", c2)
    if client_selected:
        show_portfolio(client_selected, read_only=False)


########################################
# 5) View All Portfolios
########################################
def page_view_all_portfolios():
    st.title("Vue Globale de Tous les Portefeuilles")
    clients = get_all_clients()
    if not clients:
        st.warning("Aucun client n'est disponible.")
        return
    for cname in clients:
        st.write(f"### Client: {cname}")
        show_portfolio(cname, read_only=True)
        st.write("---")


########################################
# 6) Inventory
########################################
def page_inventory():
    st.title("Inventaire des Actifs")

    from db_utils import fetch_stocks
    stocks = fetch_stocks()

    clients = get_all_clients()
    if not clients:
        st.warning("Aucun client n'est disponible.")
        return

    master_data = defaultdict(lambda: {"quantity": 0.0, "clients": set()})
    overall_val = 0.0

    for c in clients:
        dfp = get_portfolio(c)
        if not dfp.empty:
            portf_val = 0.0
            for _, row in dfp.iterrows():
                val = str(row["valeur"])
                qty = float(row["quantité"])
                match = stocks[stocks["valeur"] == val]
                price = float(match["cours"].values[0]) if not match.empty else 0.0
                total_ = qty * price
                portf_val += total_
                master_data[val]["quantity"] += qty
                master_data[val]["clients"].add(c)
            overall_val += portf_val

    if not master_data:
        st.write("Aucun actif trouvé dans les portefeuilles.")
        return

    rows = []
    sum_stocks_val = 0.0

    for val, info in master_data.items():
        match = stocks[stocks["valeur"] == val]
        price = float(match["cours"].values[0]) if not match.empty else 0.0
        agg_val = info["quantity"] * price
        sum_stocks_val += agg_val
        rows.append({
            "valeur": val,
            "quantité total": info["quantity"],
            "valorisation": agg_val,
            "portefeuille": ", ".join(sorted(info["clients"]))
        })

    for row in rows:
        if sum_stocks_val > 0:
            row["poids"] = round((row["valorisation"] / sum_stocks_val) * 100, 2)
        else:
            row["poids"] = 0.0

    df_inv = pd.DataFrame(rows)
    fmt_dict = {
        "quantité total": "{:,.0f}",
        "valorisation": "{:,.2f}",
        "poids": "{:,.2f}"
    }
    styled_inv = df_inv.style.format(fmt_dict)
    st.dataframe(styled_inv, use_container_width=True)
    st.write(f"### Actif sous gestion: {overall_val:,.2f}")


########################################
# 7) Market Page
########################################
def page_market():
    st.title("Marché Boursier")
    st.write("Les cours affichés peuvent avoir un décalage (~15 min).")

    from logic import compute_poids_masi
    from db_utils import fetch_stocks

    mm = compute_poids_masi()
    if not mm:
        st.warning("Aucun instrument trouvé / BD vide.")
        return

    stx = fetch_stocks()

    rows = []
    for val, info in mm.items():
        rows.append({
            "valeur": val,
            "Capitalisation": info["capitalisation"],
            "Poids Masi": info["poids_masi"]
        })
    df_mkt = pd.DataFrame(rows)
    df_mkt = pd.merge(df_mkt, stx, on="valeur", how="left")
    df_mkt.rename(columns={"cours":"Cours"}, inplace=True)
    df_mkt = df_mkt[["valeur","Cours","Capitalisation","Poids Masi"]]

    styled_mkt = df_mkt.style.format({
        "Cours":"{:,.2f}",
        "Capitalisation":"{:,.2f}",
        "Poids Masi":"{:,.2f}"
    })
    st.dataframe(styled_mkt, use_container_width=True)


########################################
# 8) Performance & Fees
########################################
def page_performance_fees():
    st.title("Performance et Frais")

    clients = get_all_clients()
    if not clients:
        st.warning("Aucun client trouvé. Veuillez créer un client.")
        return

    client_name = st.selectbox("Sélectionner un client", clients)
    if not client_name:
        st.info("Veuillez choisir un client pour continuer.")
        return

    cid = get_client_id(client_name)
    if cid is None:
        st.error("Client non valide.")
        return

    # --------------------------------------------------------------
    # Show / Edit existing periods in an expander
    # --------------------------------------------------------------
    with st.expander("Périodes de Performance Existantes", expanded=False):
        df_periods = get_performance_periods_for_client(cid)
        if df_periods.empty:
            st.info("Aucune période n'existe pour ce client.")
        else:
            # Convert start_date to date
            df_periods = df_periods.copy()
            if "start_date" in df_periods.columns:
                df_periods["start_date"] = pd.to_datetime(df_periods["start_date"], errors="coerce").dt.date

            # We'll do data_editor to allow editing
            # If the table has 'id' or 'created_at' we'll keep them read-only or hidden
            col_cfg = {
                "start_date": st.column_config.DateColumn("Date de Début", required=True),
                "start_value": st.column_config.NumberColumn("Portefeuille Départ", format="%.2f"),
                "masi_start_value": st.column_config.NumberColumn("MASI Départ", format="%.2f"),
            }
            if "id" in df_periods.columns:
                col_cfg["id"] = st.column_config.Column("id", disabled=True)

            updated = st.data_editor(
                df_periods,
                use_container_width=True,
                column_config=col_cfg
            )

            if st.button("Enregistrer modifications sur ces périodes"):
                # We do a naive approach => we will check row by row
                for idx in range(len(updated)):
                    row_new = updated.iloc[idx]
                    if "id" in updated.columns and "id" in df_periods.columns:
                        # locate by id
                        old_row = df_periods[df_periods["id"] == row_new["id"]]
                    else:
                        # fallback => locate by start_date or index
                        old_row = df_periods.iloc[idx]

                    # prepare the data to update
                    row_data = {
                        "start_date": str(row_new["start_date"]),
                        "start_value": float(row_new["start_value"] or 0),
                        "masi_start_value": float(row_new["masi_start_value"] or 0)
                    }
                    # do the update
                    try:
                        # use the primary key => "id" or a composite of client_id + start_date
                        # assume we have a column "id"
                        if "id" in updated.columns and "id" in row_new:
                            p_id = row_new["id"]
                            db_utils.performance_table().update(row_data).eq("id", p_id).execute()
                        else:
                            # fallback => do eq("client_id",cid).eq("start_date", old_row["start_date"])
                            odt = str(old_row["start_date"])
                            db_utils.performance_table().update(row_data)\
                                .eq("client_id", cid).eq("start_date", odt).execute()
                    except Exception as e:
                        st.error(f"Erreur lors de la mise à jour: {e}")
                st.success("Périodes mises à jour avec succès.")
                st.rerun()

    # --------------------------------------------------------------
    # Add new period in an expander
    # --------------------------------------------------------------
    with st.expander("Ajouter une nouvelle période de performance", expanded=False):
        with st.form("add_perf_period_form", clear_on_submit=True):
            start_date_input = st.date_input("Date de Début")
            start_val_port   = st.number_input("Portefeuille Départ", min_value=0.0, step=0.01, value=0.0)
            start_val_masi   = st.number_input("MASI Départ", min_value=0.0, step=0.01, value=0.0)
            s_sub = st.form_submit_button("Enregistrer")
            if s_sub:
                sd_str = str(start_date_input)
                create_performance_period(cid, sd_str, start_val_port, start_val_masi)
                st.rerun()

    # --------------------------------------------------------------
    # Calculate performance for a chosen period
    # --------------------------------------------------------------
    with st.expander("Calculer la Performance sur une Période", expanded=False):
        df_periods2 = get_performance_periods_for_client(cid)
        if df_periods2.empty:
            st.info("Aucune période n'existe.")
        else:
            df_periods2 = df_periods2.copy()
            df_periods2["start_date"] = pd.to_datetime(df_periods2["start_date"], errors="coerce").dt.date
            df_periods2 = df_periods2.sort_values("start_date", ascending=False)
            start_choices = df_periods2["start_date"].unique().tolist()

            pick = st.selectbox("Choisir la date de début", start_choices)
            row_chosen = df_periods2[df_periods2["start_date"]==pick].iloc[0]
            portfolio_start = float(row_chosen.get("start_value",0))
            masi_start      = float(row_chosen.get("masi_start_value",0))

            # Current portfolio value
            pdf = get_portfolio(client_name)
            if pdf.empty:
                st.warning("Pas de portefeuille pour ce client.")
            else:
                stx = db_utils.fetch_stocks()
                cur_val = 0.0
                for _, prow in pdf.iterrows():
                    val = str(prow["valeur"])
                    qty_ = float(prow["quantité"])
                    matchp = stx[stx["valeur"] == val]
                    px_ = float(matchp["cours"].values[0]) if not matchp.empty else 0.0
                    cur_val += (qty_ * px_)

                gains_port = cur_val - portfolio_start
                perf_port = 0.0
                if portfolio_start > 0:
                    perf_port = (gains_port / portfolio_start)*100.0

                masi_now = get_current_masi()
                gains_masi = masi_now - masi_start
                perf_masi  = 0.0
                if masi_start>0:
                    perf_masi = (gains_masi / masi_start)*100.0

                # surperf% = perf_port - perf_masi
                surp_pct = perf_port - perf_masi
                # surperf_abs => (surp_pct / 100) * portfolio_start
                surp_abs = (surp_pct / 100.0)* portfolio_start

                cinfo_ = get_client_info(client_name)
                mgmt_rate = float(cinfo_.get("management_fee_rate",0))/100.0
                # if surperformance is billed
                if cinfo_.get("bill_surperformance", False):
                    # we charge on surperf
                    base_ = max(0, surp_abs)
                    fees_ = base_* mgmt_rate
                else:
                    # we charge on actual gains
                    base_ = max(0, gains_port)
                    fees_ = base_* mgmt_rate

                # Display in small table
                results_df = pd.DataFrame([{
                    "Portf Départ": portfolio_start,
                    "Portf Actuel": cur_val,
                    "Gains Portf": gains_port,
                    "Perf Portf %": perf_port,
                    "MASI Départ": masi_start,
                    "MASI Actuel": masi_now,
                    "Gains MASI": gains_masi,
                    "Perf MASI %": perf_masi,
                    "Surperf %": surp_pct,
                    "Surperf Abs.": surp_abs,
                    "Frais": fees_,
                }])
                numcols = results_df.select_dtypes(include=["int","float"]).columns
                rstyled = results_df.style.format("{:,.2f}", subset=numcols)
                st.dataframe(rstyled, use_container_width=True)

    # --------------------------------------------------------------
    # Summary for all clients
    # --------------------------------------------------------------
    with st.expander("Résumé de Performance (tous les clients)", expanded=False):
        all_latest = get_latest_performance_period_for_all_clients()
        if all_latest.empty:
            st.info("Aucune donnée globale de performance.")
        else:
            stx2 = db_utils.fetch_stocks()
            masi_now2 = get_current_masi()
            all_list = []
            all_cs = get_all_clients()

            for _, rowL in all_latest.iterrows():
                c_id = rowL["client_id"]
                st_val = float(rowL.get("start_value",0))
                ms_val = float(rowL.get("masi_start_value",0))
                ddate  = str(rowL.get("start_date",""))

                # find name
                name_ = None
                for cc_ in all_cs:
                    if get_client_id(cc_)== c_id:
                        name_ = cc_
                        break
                if not name_:
                    continue

                # compute portf current
                pdf2 = get_portfolio(name_)
                cur_val2=0.0
                if not pdf2.empty:
                    for _, prow2 in pdf2.iterrows():
                        v2= str(prow2["valeur"])
                        q2= float(prow2["quantité"])
                        mt2= stx2[stx2["valeur"]== v2]
                        px2= float(mt2["cours"].values[0]) if not mt2.empty else 0.0
                        cur_val2 += (q2*px2)

                # perf client
                gains_port2 = cur_val2 - st_val
                perf_port2  = 0.0
                if st_val>0:
                    perf_port2= (gains_port2/st_val)*100.0

                # perf masi
                gains_masi2= masi_now2- ms_val
                perf_masi2= 0.0
                if ms_val>0:
                    perf_masi2= (gains_masi2/ms_val)*100.0

                # surperf% = perf_port2 - perf_masi2
                surp_pct2= perf_port2- perf_masi2
                # surperf_abs => (surp_pct2/100)* st_val
                surp_abs2= (surp_pct2/100.0)* st_val

                cinfo2 = get_client_info(name_)
                mgmtr2 = float(cinfo2.get("management_fee_rate",0))/100.0
                if cinfo2.get("bill_surperformance",False):
                    base2= max(0, surp_abs2)
                    fee2 = base2* mgmtr2
                else:
                    base2= max(0, gains_port2)
                    fee2 = base2* mgmtr2

                all_list.append({
                    "Client": name_,
                    "Date Début": ddate,
                    "Portf Départ": st_val,
                    "Portf Actuel": cur_val2,
                    "Perf Portf %": perf_port2,
                    "MASI Départ": ms_val,
                    "MASI Actuel": masi_now2,
                    "Perf MASI %": perf_masi2,
                    "Surperf %": surp_pct2,
                    "Surperf Abs.": surp_abs2,
                    "Frais": fee2
                })

            if not all_list:
                st.info("Aucune info dispo.")
            else:
                df_sum = pd.DataFrame(all_list)
                numeric_cols = df_sum.select_dtypes(include=["int","float"]).columns
                styd = df_sum.style.format("{:,.2f}", subset=numeric_cols)
                st.dataframe(styd, use_container_width=True)

                tot_start= df_sum["Portf Départ"].sum()
                tot_cur  = df_sum["Portf Actuel"].sum()
                tot_fee  = df_sum["Frais"].sum()
                df_tots  = pd.DataFrame([{
                    "Total Portf Départ": tot_start,
                    "Total Portf Actuel": tot_cur,
                    "Total Frais": tot_fee
                }])
                st.write("#### Totaux Globaux")
                st.dataframe(df_tots.style.format("{:,.2f}"), use_container_width=True)



########################################
# DATABASE FUNCTIONS FOR STRATÉGIES
########################################

def strategy_table():
    """Return a Supabase table object for 'strategies'."""
    return get_supabase().table("strategies")

def get_strategies():
    """Retrieve all strategies as a DataFrame."""
    res = strategy_table().select("*").execute()
    if not res.data:
        return pd.DataFrame()
    return pd.DataFrame(res.data)

def create_strategy(name, targets):
    """
    Create a new strategy.
    targets: a dictionary mapping asset names to target percentages.
             (Cash is not entered – it is auto‑calculated as 100 minus the sum of percentages.)
    """
    try:
        row = {"name": name, "targets": json.dumps(targets)}
        strategy_table().insert(row).execute()
        st.success(f"Stratégie « {name} » créée avec succès.")
    except Exception as e:
        st.error(f"Erreur lors de la création de la stratégie : {e}")

def update_strategy(strategy_id, name, targets):
    try:
        row = {"name": name, "targets": json.dumps(targets)}
        strategy_table().update(row).eq("id", strategy_id).execute()
        st.success("Stratégie mise à jour avec succès.")
    except Exception as e:
        st.error(f"Erreur lors de la mise à jour de la stratégie : {e}")

def delete_strategy(strategy_id):
    try:
        strategy_table().delete().eq("id", strategy_id).execute()
        st.success("Stratégie supprimée avec succès.")
    except Exception as e:
        st.error(f"Erreur lors de la suppression de la stratégie : {e}")

def assign_strategy_to_client(client_name, strategy_id):
    """
    Assign a strategy to a client by updating the client's record.
    (This assumes you have added a column "strategy_id" in your clients table.)
    """
    from db_utils import client_table
    cid = get_client_id(client_name)
    if cid is None:
        st.error("Client introuvable.")
        return
    try:
        client_table().update({"strategy_id": strategy_id}).eq("id", cid).execute()
        st.success(f"Stratégie assignée à {client_name}.")
    except Exception as e:
        st.error(f"Erreur lors de l'assignation de la stratégie : {e}")


########################################
# SIMULATION FUNCTIONS AND HELPERS
########################################

def simulation_for_client_updated(client_name):
    """
    Updated simulation for a single portfolio.
    Displays a table with columns:
      Valeur | Cours (Prix) | Quantité actuelle | Poids Actuel (%) | Quantité Cible | Poids Cible (%) | Écart
    Even if an asset from the strategy is not present in the portfolio, its target is computed.
    The Cash row is always placed at the bottom.
    """
    client = get_client_info(client_name)
    if not client:
        st.error("Client non trouvé.")
        return
    strategies_df = get_strategies()
    if "strategy_id" in client and client["strategy_id"]:
        strat = strategies_df[strategies_df["id"] == client["strategy_id"]]
        targets = json.loads(strat.iloc[0]["targets"]) if not strat.empty else {}
    else:
        targets = {}
    # Ensure all assets in the strategy appear even if not in portfolio.
    pf = get_portfolio(client_name)
    if pf.empty:
        st.error("Portefeuille vide pour ce client.")
        return
    stocks_df = fetch_stocks()
    total_val = 0.0
    portfolio_assets = {}
    for _, row in pf.iterrows():
        asset = row["valeur"]
        qty = float(row["quantité"])
        # If asset not in portfolio, try fetching its price from stocks_df
        match = stocks_df[stocks_df["valeur"] == asset]
        if not match.empty:
            price = float(match["cours"].iloc[0])
        else:
            price = 0.0
        total_val += qty * price
        portfolio_assets[asset] = {"qty": qty, "price": price}
    # Include any asset from targets not in portfolio_assets.
    for asset in targets.keys():
        if asset not in portfolio_assets:
            match = stocks_df[stocks_df["valeur"] == asset]
            if not match.empty:
                price = float(match["cours"].iloc[0])
            else:
                price = 0.0
            portfolio_assets[asset] = {"qty": 0, "price": price}
    # Build simulation rows
    sim_rows = []
    # Ensure Cash row is processed last.
    assets_ordered = [a for a in portfolio_assets if a.lower() != "cash"] + (["Cash"] if "Cash" in portfolio_assets else [])
    for asset in assets_ordered:
        current_qty = portfolio_assets[asset]["qty"]
        price = portfolio_assets[asset]["price"]
        current_value = current_qty * price
        current_weight = (current_value / total_val * 100) if total_val > 0 else 0
        target_pct = targets.get(asset, 0)
        if asset.lower() == "cash":
            target_pct = 100 - sum(targets.values())
        target_value = total_val * (target_pct / 100)
        target_qty = round(target_value / price) if price > 0 else 0
        ecart = current_qty - target_qty
        sim_rows.append({
            "Valeur": asset,
            "Cours (Prix)": price,
            "Quantité actuelle": current_qty,
            "Poids Actuel (%)": round(current_weight, 2),
            "Quantité Cible": target_qty,
            "Poids Cible (%)": target_pct,
            "Écart": ecart
        })
    sim_df = pd.DataFrame(sim_rows, columns=["Valeur", "Cours (Prix)", "Quantité actuelle", "Poids Actuel (%)", "Quantité Cible", "Poids Cible (%)", "Écart"])
    st.dataframe(sim_df, use_container_width=True)


def aggregate_portfolios(client_list):
    """
    Aggregate portfolios for a list of clients.
    Returns a DataFrame with aggregated quantities per asset.
    """
    agg = {}
    for client in client_list:
        pf = get_portfolio(client)
        if not pf.empty:
            for _, row in pf.iterrows():
                asset = row["valeur"]
                qty = float(row["quantité"])
                agg[asset] = agg.get(asset, 0) + qty
    return pd.DataFrame(list(agg.items()), columns=["valeur", "quantité"])


def simulation_for_aggregated(agg_pf, strategy):
    """
    Run simulation on an aggregated portfolio.
    Uses the same columns as the single portfolio simulation.
    """
    targets = json.loads(strategy["targets"])
    targets["Cash"] = 100 - sum(targets.values())
    stocks_df = fetch_stocks()
    total_val = 0.0
    portfolio_assets = {}
    for _, row in agg_pf.iterrows():
        asset = row["valeur"]
        qty = float(row["quantité"])
        match = stocks_df[stocks_df["valeur"] == asset]
        price = 1.0 if asset.lower() == "cash" else (float(match["cours"].iloc[0]) if not match.empty else 0.0)
        total_val += qty * price
        portfolio_assets[asset] = {"qty": qty, "price": price}
    # Ensure Cash row is at the bottom.
    assets_ordered = [a for a in portfolio_assets if a.lower() != "cash"] + (["Cash"] if "Cash" in portfolio_assets else [])
    sim_rows = []
    for asset in assets_ordered:
        current_qty = portfolio_assets[asset]["qty"]
        price = portfolio_assets[asset]["price"]
        current_value = current_qty * price
        current_weight = (current_value / total_val * 100) if total_val > 0 else 0
        target_pct = targets.get(asset, 0)
        if asset.lower() == "cash":
            target_pct = 100 - sum(targets[k] for k in targets if k.lower() != "cash")
        target_value = total_val * (target_pct / 100)
        target_qty = round(target_value / price) if price > 0 else 0
        ecart = current_qty - target_qty
        sim_rows.append({
            "Valeur": asset,
            "Cours (Prix)": price,
            "Quantité actuelle": current_qty,
            "Poids Actuel (%)": round(current_weight, 2),
            "Quantité Cible": target_qty,
            "Poids Cible (%)": target_pct,
            "Écart": ecart
        })
    sim_df = pd.DataFrame(sim_rows, columns=["Valeur", "Cours (Prix)", "Quantité actuelle", "Poids Actuel (%)", "Quantité Cible", "Poids Cible (%)", "Écart"])
    st.dataframe(sim_df, use_container_width=True)


def simulation_stock_details(selected_stock, strategy, client_list):
    """
    For multiple portfolios, returns detailed breakdown for a selected stock.
    It returns two items:
      1. An aggregated details dictionary with:
         - "Action": the selected stock.
         - "Prix": the stock price (rounded to 2 decimals).
         - "Quantité actuelle agrégée": aggregated current quantity (integer).
         - "Poids cible (%)": target percentage from the strategy.
         - "Quantité cible agrégée": aggregated target quantity (integer).
         - "Ajustement (à acheter si positif, à vendre si négatif)": aggregated adjustment in quantity (integer).
         - "Valeur de l'ajustement (MAD)": adjustment value in Moroccan Dirhams (rounded to 2 decimals).
         - "Cash disponible": total cash available across all portfolios.
      2. A "Pré‑répartition" DataFrame with per‑client details including:
         - "Client"
         - "Quantité actuelle"
         - "Quantité Cible"
         - "Valeur de l'ajustement (MAD)"
         - "Cash disponible"
    """
    stocks_df = fetch_stocks()
    # Get the price for the selected stock using its raw value (ensuring 2 decimals)
    match = stocks_df[stocks_df["valeur"].str.lower() == selected_stock.lower()]
    if not match.empty:
        price = float(match["cours"].iloc[0])
    else:
        price = 0.0

    # Get the target percentage for the selected stock from the strategy
    strategy_targets = json.loads(strategy["targets"])
    target_pct = strategy_targets.get(selected_stock, 0)
    if selected_stock.lower() == "cash":
        target_pct = 100 - sum(strategy_targets.values())

    aggregated_qty = 0
    total_cash_available = 0
    total_value_all = 0.0
    per_client_details = []

    # Process each client to get per-client details
    for client in client_list:
        pf = get_portfolio(client)
        client_value = 0.0
        current_qty = 0
        cash_available = 0
        if not pf.empty:
            for _, row in pf.iterrows():
                asset = row["valeur"]
                qty = float(row["quantité"])
                # Use 1.0 as price for Cash, otherwise look it up
                m = stocks_df[stocks_df["valeur"].str.lower() == asset.lower()]
                p = 1.0 if asset.lower() == "cash" else (float(m["cours"].iloc[0]) if not m.empty else 0.0)
                client_value += qty * p
                if asset.lower() == selected_stock.lower():
                    current_qty = qty
                if asset.lower() == "cash":
                    cash_available = qty
        # Compute the client's target quantity for the selected stock
        target_qty_client = round(client_value * (target_pct / 100) / price) if price > 0 else 0
        adjustment_client = target_qty_client - current_qty
        per_client_details.append({
            "Client": client,
            "Quantité actuelle": int(current_qty),
            "Quantité Cible": int(target_qty_client),
            "Valeur de l'ajustement (MAD)": round(adjustment_client * price, 2),
            "Cash disponible": cash_available
        })
        aggregated_qty += current_qty
        total_cash_available += cash_available
        total_value_all += client_value

    # Aggregated target quantity for the selected stock
    target_qty_agg = round(total_value_all * (target_pct / 100) / price) if price > 0 else 0
    adjustment_agg = target_qty_agg - aggregated_qty

    agg_details = {
        "Action": selected_stock,
        "Prix": round(price, 2),
        "Quantité actuelle agrégée": int(aggregated_qty),
        "Poids cible (%)": target_pct,
        "Quantité cible agrégée": int(target_qty_agg),
        "Ajustement (à acheter si positif, à vendre si négatif)": int(adjustment_agg),
        "Valeur de l'ajustement (MAD)": round(adjustment_agg * price, 2),
        "Cash disponible": total_cash_available
    }

    repartition_df = pd.DataFrame(per_client_details)
    return agg_details, repartition_df



########################################
# PAGE : STRATÉGIES ET SIMULATION
########################################

def page_strategies_and_simulation():
    st.title("Stratégies et Simulation")
    tabs = st.tabs(["Gestion des Stratégies", "Assignation aux Clients", "Simulation de Stratégie"])

    # Tab 0: Gestion des Stratégies
    with tabs[0]:
        with st.expander("Stratégies existantes", expanded=False):
            strategies_df = get_strategies()
            if not strategies_df.empty:
                display_rows = []
                for _, row in strategies_df.iterrows():
                    targets = json.loads(row["targets"])
                    cash = 100 - sum(targets.values())
                    targets["Cash"] = cash
                    details = ", ".join([f"{k} : {v}%" for k, v in targets.items()])
                    display_rows.append({"Nom": row["name"], "Détails": details})
                st.table(pd.DataFrame(display_rows))
            else:
                st.info("Aucune stratégie existante.")
        with st.expander("Créer une nouvelle stratégie", expanded=False):
            if "new_strategy_targets" not in st.session_state:
                st.session_state.new_strategy_targets = {}
            col1, col2, col3 = st.columns([3,1,1])
            stocks_df = fetch_stocks()
            stock_options = [s for s in stocks_df["valeur"].tolist() if s.lower() != "cash"]
            with col1:
                new_stock = st.selectbox("Action à ajouter", stock_options, key="new_strat_stock_create")
            with col2:
                new_weight = st.number_input("Pourcentage", min_value=0.0, max_value=100.0, value=0.0, step=0.5, key="new_strat_weight_create")
            with col3:
                if st.button("Ajouter"):
                    st.session_state.new_strategy_targets[new_stock] = new_weight
                    st.success(f"{new_stock} ajouté avec {new_weight}%")
            if st.session_state.new_strategy_targets:
                df_new = pd.DataFrame(list(st.session_state.new_strategy_targets.items()), columns=["Action", "Pourcentage"])
                total_weight = df_new["Pourcentage"].sum()
                cash_pct = 100 - total_weight
                df_display = pd.concat([df_new, pd.DataFrame([{"Action": "Cash", "Pourcentage": cash_pct}])], ignore_index=True)
                st.table(df_display)
                if total_weight > 100:
                    st.error(f"Le total dépasse 100% de {total_weight - 100}%.")
            strat_name_new = st.text_input("Nom de la stratégie", key="new_strat_name")
            if st.button("Créer la stratégie"):
                if not strat_name_new:
                    st.error("Veuillez entrer un nom pour la stratégie.")
                elif not st.session_state.new_strategy_targets:
                    st.error("Veuillez ajouter au moins une action.")
                elif sum(st.session_state.new_strategy_targets.values()) > 100:
                    st.error("Le total des pourcentages dépasse 100%.")
                else:
                    create_strategy(strat_name_new, st.session_state.new_strategy_targets)
                    st.session_state.new_strategy_targets = {}
                    st.success("Stratégie créée.")
        with st.expander("Modifier/Supprimer une stratégie", expanded=False):
            strategies_df = get_strategies()
            if not strategies_df.empty:
                strat_options = strategies_df["name"].tolist()
                selected_strat_name = st.selectbox("Sélectionnez une stratégie à modifier", strat_options, key="edit_strat_select")
                selected_strategy = strategies_df[strategies_df["name"] == selected_strat_name].iloc[0]
                current_targets = json.loads(selected_strategy["targets"])
                st.write("Actions actuelles dans la stratégie :")
                updated_targets = {}
                for action, pct in current_targets.items():
                    colA, colB, colC = st.columns([3,1,1])
                    with colA:
                        new_pct = st.number_input(f"{action} (%)", min_value=0.0, max_value=100.0, value=float(pct), step=0.5, key=f"edit_{action}")
                    with colB:
                        remove = st.checkbox(f"Supprimer", key=f"remove_{action}")
                    if not remove:
                        updated_targets[action] = new_pct
                st.write("Ajouter une nouvelle action :")
                colD, colE = st.columns(2)
                with colD:
                    add_action = st.selectbox("Nouvelle action", stock_options, key="add_strat_stock")
                with colE:
                    add_pct = st.number_input("Pourcentage", min_value=0.0, max_value=100.0, value=0.0, step=0.5, key="add_strat_pct")
                if st.button("Ajouter l'action"):
                    if add_action in updated_targets:
                        st.error("Action déjà présente.")
                    else:
                        updated_targets[add_action] = add_pct
                        st.success(f"{add_action} ajouté avec {add_pct}%")
                total_updated = sum(updated_targets.values())
                cash_updated = 100 - total_updated
                display_df = pd.DataFrame(list(updated_targets.items()), columns=["Action", "Pourcentage"])
                display_df = pd.concat([display_df, pd.DataFrame([{"Action": "Cash", "Pourcentage": cash_updated}])], ignore_index=True)
                st.table(display_df)
                if st.button("Mettre à jour la stratégie"):
                    if total_updated > 100:
                        st.error(f"Le total dépasse 100% de {total_updated - 100}%.")
                    else:
                        update_strategy(selected_strategy["id"], selected_strat_name, updated_targets)
                # Added delete button
                if st.button("Supprimer la stratégie"):
                    delete_strategy(selected_strategy["id"])
            else:
                st.info("Aucune stratégie à modifier.")

    # Tab 1: Assignation aux Clients
    with tabs[1]:
        st.header("Assignation de Stratégies aux Clients")
        clients = get_all_clients()
        strategies_df = get_strategies()
        if not strategies_df.empty and clients:
            for client in clients:
                col1, col2 = st.columns([2, 2])
                with col1:
                    st.write(client)
                with col2:
                    current_client = get_client_info(client)
                    current_strat_id = current_client.get("strategy_id", None)
                    options = strategies_df["id"].tolist()
                    options_names = strategies_df["name"].tolist()
                    selected_strat_id = st.selectbox(
                        f"Stratégie pour {client}",
                        options=options,
                        format_func=lambda x: options_names[options.index(x)] if x in options else "None",
                        index=options.index(current_strat_id) if current_strat_id in options else 0,
                        key=f"assign_{client}"
                    )
                    if st.button(f"Assigner la stratégie à {client}", key=f"assign_btn_{client}"):
                        assign_strategy_to_client(client, selected_strat_id)
        else:
            st.info("Assurez-vous qu'il existe à la fois des clients et des stratégies.")

    # Tab 2: Simulation de Stratégie
    with tabs[2]:
        st.header("Simulation de Stratégie")
        mode = st.radio("Mode de simulation", options=["Portefeuille Unique", "Portefeuilles Multiples"], key="sim_mode")
        if mode == "Portefeuille Unique":
            client_sim = st.selectbox("Sélectionner un client", get_all_clients(), key="sim_client")
            if client_sim:
                simulation_for_client_updated(client_sim)
        else:
            st.write("Simulation pour plusieurs portefeuilles (agrégés) de la même stratégie")
            strategies_df = get_strategies()
            strat_choice = st.selectbox("Sélectionnez une stratégie", strategies_df["name"].tolist(), key="multi_strat")
            selected_strategy = strategies_df[strategies_df["name"] == strat_choice].iloc[0]
            all_clients = get_all_clients()
            clients_with_strat = [c for c in all_clients if get_client_info(c).get("strategy_id") == selected_strategy["id"]]
            if not clients_with_strat:
                st.info("Aucun client n'est assigné à cette stratégie.")
            else:
                st.write("Clients assignés :", clients_with_strat)
                agg_pf = aggregate_portfolios(clients_with_strat)
                simulation_for_aggregated(agg_pf, selected_strategy)
                st.write("### Détail par action")
                stock_options = list(set(agg_pf["valeur"].tolist()).union(set(json.loads(selected_strategy["targets"]).keys())))
                selected_stock = st.selectbox("Sélectionner une action", stock_options, key="detail_stock")
                if st.button("Afficher les détails"):
                    agg_details, repartition = simulation_stock_details(selected_stock, selected_strategy, clients_with_strat)
                    st.write("#### Détail agrégé")
                    st.table(pd.DataFrame([agg_details]))
                    st.write("#### Pré‑répartition")
                    st.table(repartition)


# Expose the page function
if __name__ == "__main__":
    page_strategies_and_simulation()
