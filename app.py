import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# === Streamlit UI ===
st.set_page_config(page_title="📈 Nelson–Siegel Yield Curve Fitting")
st.title("📈 Nelson–Siegel Yield Curve Fitting")
st.write("Upload your yield data and fit the Nelson–Siegel model to visualize the term structure.")

# === Nelson–Siegel Function ===
def nelson_siegel(t, beta0, beta1, beta2, tau):
    t = np.array(t)
    with np.errstate(divide='ignore', invalid='ignore'):
        factor1 = (1 - np.exp(-t / tau)) / (t / tau)
        factor2 = factor1 - np.exp(-t / tau)
    return beta0 + beta1 * factor1 + beta2 * factor2

# === Loss Function ===
def objective(params, t, y):
    return np.mean((nelson_siegel(t, *params) - y) ** 2)

# === Upload CSV ===
st.subheader("📤 Upload Yield Data (CSV)")
uploaded_file = st.file_uploader("CSV Format: Maturity (Years), Yield (%)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = ["Maturity", "Yield"]
    maturities = df["Maturity"].values
    yields = df["Yield"].values

    st.subheader("📄 Uploaded Data")
    st.dataframe(df)

    # Initial Guess
    initial_params = [yields[-1], -1.0, 1.0, 1.0]

    # Optimize
    result = minimize(objective, initial_params, args=(maturities, yields), method='L-BFGS-B')
    fitted_params = result.x

    fitted_yields = nelson_siegel(maturities, *fitted_params)

    # Plot
    fig, ax = plt.subplots()
    ax.plot(maturities, yields, 'o', label="Observed")
    ax.plot(maturities, fitted_yields, '-', label="Nelson–Siegel Fit")
    ax.set_xlabel("Maturity (Years)")
    ax.set_ylabel("Yield (%)")
    ax.set_title("Nelson–Siegel Yield Curve")
    ax.legend()
    st.pyplot(fig)

    # Show parameters
    st.subheader("🧮 Fitted Parameters")
    st.write(f"β₀ (long-term rate): {fitted_params[0]:.4f}")
    st.write(f"β₁ (short-term effect): {fitted_params[1]:.4f}")
    st.write(f"β₂ (medium-term hump): {fitted_params[2]:.4f}")
    st.write(f"τ (decay factor): {fitted_params[3]:.4f}")
