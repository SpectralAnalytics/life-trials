import pandas as pd
import streamlit as st
import numpy as np

st.set_page_config(layout="wide")

st.title("Life Contigent Risks Calculator")

st.subheader("An App by Marcel")

st.markdown("""
    The various exhibits contained here are implementations of the various models covered in [Actuarial Mathematics for Life Contingent Risks](https://www.cambridge.org/highereducation/books/actuarial-mathematics-for-life-contingent-risks/281DA4E8D523A6B23280ADC3D165AFDA#overview) as part of my revision on the subject.
    
    Ultimately, the various demonstrations here will be incorporated into a PyPI package that follows the idiom of [pandas](https://pandas.pydata.org/), similar to how [GeoPandas](https://geopandas.org/en/stable/) has been implemented.

    You can pick specific models to play with through the minimizable sidebar on the left.
""")
            