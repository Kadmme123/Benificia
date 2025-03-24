import streamlit as st

st.set_page_config(layout="wide")

from pages import (
    page_manage_clients,
    page_create_portfolio,
    page_view_client_portfolio,
    page_view_all_portfolios,
    page_inventory,
    page_market,
    page_performance_fees,
    page_strategies_and_simulation
)

def add_sidebar_logo():
    st.sidebar.image("Image-1.jpg", width=250)
    st.sidebar.title("Beneficia Capital")

def main():
    # Add logo and title to sidebar
    add_sidebar_logo()
    
    page = st.sidebar.selectbox(
        "📂 Navigation",
        [
            "Gestion des clients",
            "Créer un Portefeuille",
            "Gérer un Portefeuille",
            "Stratégies et Simulation",
            "Voir tout les portefeuilles",
            "Inventaire",
            "Marché",
            "Performance & Fees"
        ]
    )
    if page == "Gestion des clients":
        page_manage_clients()
    elif page == "Créer un Portefeuille":
        page_create_portfolio()
    elif page == "Gérer un Portefeuille":
        page_view_client_portfolio()
    elif page == "Voir tout les portefeuilles":
        page_view_all_portfolios()
    elif page == "Inventaire":
        page_inventory()
    elif page == "Marché":
        page_market()
    elif page == "Performance & Fees":
        page_performance_fees() 
    elif page == "Stratégies et Simulation":
        page_strategies_and_simulation()

if __name__ == "__main__":
    main()
