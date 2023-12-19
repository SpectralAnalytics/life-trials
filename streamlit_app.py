import pandas as pd
import streamlit as st
import numpy as np

st.set_page_config(layout="wide")

st.title("Insurance EPV calculator")

st.subheader("An App by Marcel")

st.markdown("Data gathered from US Social Security's [Actuarial Life Table](https://www.ssa.gov/oact/STATS/table4c6.html).")

life_sheets = {
    int(year): df.set_index("Exact age") for year, df in pd.read_excel("LifeTable.xlsx", sheet_name=None).items()
}

option_year = st.selectbox(
    "Select a year for period life table:",
    reversed(life_sheets.keys()),
)

life_table = life_sheets[option_year]

tab1, tab2 = st.tabs(["EPV Calculator", "Life Table"])

with tab1:
    st.subheader("Calculate Now!")

    col1, col2= st.columns(2)
    with col1:
        i = st.number_input(
            label="Interest rate:",
            min_value=0.00,
            max_value=0.30,
            value=0.05,
            step=0.01,
            placeholder="Please key in an interest rate...")
    with col2:
        g = st.selectbox(
            "Gender of insured:",
            ["Male", "Female"],
        )

    col1, col2= st.columns(2)
    with col1:
        x = st.number_input(
            label="Age insured:",
            min_value=0,
            max_value=100,
            value=20,
            step=1,
            placeholder="Please key in the age of insured...",
        )
    with col2:
        S = st.number_input(
            label="Sum insured:",
            min_value=1000,
            max_value=100000,
            step=1000,
            placeholder="Please key in the sum insured...",
        )

    q = life_table[f"{g} death probability"][x:]
    p = (1 - q).cumprod().shift(1).fillna(1)
    v = (1 + i)**-pd.Series(data=range(1, len(p)+1), index=p.index)
    A = sum(q*p*v)    
    
    p = (1 - q).cumprod()

    ddot_a = 1 + sum(p*v)
    
    st.markdown(f"$A_{{{x}}} = {A:0.08f}$ and $\ddot{{a}}_{{{x}}} = {ddot_a:0.06f}$")

    st.markdown(f"$P = S \\frac{{A_{{ {x} }}}}{{\\ddot{{a}}_{{ {x} }}}} = {S} \\frac{{ {A:0.08f} }}{{ {ddot_a:0.06f} }} = {S*A/ddot_a:0.4f}$ pure premium per year")
with tab2:
    st.subheader("Data Loaded")
    st.dataframe(life_sheets[option_year])