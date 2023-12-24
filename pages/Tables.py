import pandas as pd
import streamlit as st
import numpy as np

from scipy.integrate import RK45

st.set_page_config(layout="wide")

st.subheader("Table calculated from first principles.")

st.markdown(r"""
    Due to the lack of $\LaTeX$ support in [st.dataframe](https://docs.streamlit.io/library/api-reference/data/st.dataframe), the column headers are written in plain English.
""")

tab1, tab2 = st.tabs([
    "Standard Select and Ultimate Survival Model",
    "Standard Sickness-Death Model",
])

with tab1:
    class SSUSM:
        def __init__(self, A=0.00022, B=2.7e-6, c=1.124):
            self.A, self.B, self.c = A, B, c
    
        def mu(self, x):
            return self.A + self.B*x**self.c
    
        def p(self, x, t):
            return np.exp(-self.A*t - self.B*self.c**x*(self.c**t - 1)/np.log(self.c))
    
        def p_select(self, x, t):
            return np.exp(0.9**(2-t)*(self.A*(1-0.9**t)/np.log(0.9) + self.B*self.c**x*(self.c**t - 0.9**t)/np.log(0.9/self.c)))
    
    ssusm = SSUSM()

    x = pd.Series(data=range(18, 119))
    ssust = pd.DataFrame({
        "age": x,
        "lives at age + 2": x.map(lambda t: ssusm.p(20, max(t-18, 0)))*100000,
    })
    
    ssust["lives at [age]"] = ssust.apply(lambda row: row["lives at age + 2"]/ssusm.p_select(row["age"], 2), axis="columns")
    ssust["lives at [age] + 1"] = ssust.apply(lambda row: row["lives at [age]"]*ssusm.p_select(row["age"], 1), axis="columns")
    
    ssust = ssust[["age", "lives at [age]", "lives at [age] + 1", "lives at age + 2"]].copy()
    ssust.iloc[:2, 1:3] = np.nan
    ssust.iloc[63:, 1:3] = np.nan
    ssust = ssust.set_index("age")
    st.dataframe(ssust.style.format("{:0.3f}"))

    st.markdown(r"""
        The table above is computed from the Standard Select Survival Model with the following parameters using [NumPy](https://numpy.org/) and [pandas](https://pandas.pydata.org/):
        *  Ultimate force of mortality: $\mu_{x} = A + B c^{x}$ with $A = 2.2 \times 10^{-4}$, $B = 2.7 \times 10^{-6}$ and $c = 1.124$
        *  Select force of mortality: $\mu_{[x] + s} = 0.9^{2-s}\mu_{x+s}$ for $0 \le s < 2$
    """)

with tab2:
    class SSDM:
        def __init__(self,
            a1=4e-4, a2=5e-4,
            b1=3.47e-6, b2=7.58e-5,
            c1=0.138,c2=0.087
        ):
            self.a1, self.a2 = a1, a2
            self.b1, self.b2 = b1, b2
            self.c1, self.c2 = c1, c2
        
        def mu01(self, x):
            """
            Transition intensity from Healthy to Sick
            """
            return self.a1 + self.b1*np.exp(self.c1*x)
        
        def mu02(self, x):
            """
            Transition intensity from Healthy to Dead
            """
            return self.a2 + self.b2*np.exp(self.c2*x)
        
        def mu10(self, x):
            """
            Transition intensity from Sick to Healthy
            """
            return self.b1*np.exp(self.c1*(110 - x))
        
        def mu12(self, x):
            """
            Transition intensity from Sick to Dead
            """
            return 1.4*self.mu02(x)
        
        def fun2(self, t, y):
            """
            To compute the probability of Healthy and Sick.
            """
            return np.array([
              [ -(self.mu01(t) + self.mu02(t)), self.mu10(t)],
              [ self.mu01(t), -(self.mu12(t) + self.mu10(t))],
            ])@y
    
    ssdm = SSDM()
    
    kwargs = {
        "atol": 1e-6,
        "rtol": 1e-10,
    }
    
    rows = []
    
    for x in range(50, 81):
        row = []
        for t in [1, 10]:
            for y0 in [np.array([1.0, 0]), np.array([0, 1.0])]:
                rk45 = RK45(ssdm.fun2, x, y0, t_bound=x + t, **kwargs)
                while rk45.status != "finished": rk45.step()
                row += rk45.y.tolist()
        rows += [row]
    
    columns = [
        "0 to 0, 1 year", "0 to 1, 1 year",
        "1 to 0, 1 year", "1 to 1, 1 year",
        "0 to 0, 10 years", "0 to 1, 10 years",
        "1 to 0, 10 years", "1 to 1, 10 years",
    ]
    
    ssdt = pd.DataFrame(data=rows, index=range(50, 81), columns=columns)

    st.dataframe(ssdt[[
      "0 to 0, 1 year", "0 to 1, 1 year",
      "1 to 1, 1 year", "1 to 0, 1 year",
      "0 to 0, 10 years", "0 to 1, 10 years",
      "1 to 1, 10 years", "1 to 0, 10 years",
    ]].style.format("{:0.6f}"))

    st.markdown(r"""
        The table above is computed from the sickness-death model using [scipy.integrate.RK45](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.RK45.html) with the following transition intensities:
        *  $\mu_{x}^{01} = a_{1} + b_{1} \exp (c_{1} x)$
        *  $\mu_{x}^{02} = a_{2} + b_{2} \exp (c_{2} x)$
        *  $\mu_{x}^{10} = b_{1} \exp (c_{1} (110 - x))$
        *  $\mu_{x}^{12} = 1.4 \mu_{x}^{02}$
        
        where
        
        *  $a_{1} = 4 \times 10^{-4}$
        *  $a_{2} = 5 \times 10^{-4}$
        *  $b_{1} = 3.47 \times 10^{-6}$
        *  $b_{2} = 7.58 \times 10^{-5}$
        *  $c_{1} = 0.138$
        *  $c_{2} = 0.087$
    """)
