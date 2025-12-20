# app.py
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from bl_core import (
    compute_task2,
    mu_o_for_target_M,
    areal_sweep_EA,
    vertical_sweep_EI,
    overall_efficiency,
    SWD_REF_FOR_M_DEFAULT
)

st.set_page_config(page_title="Buckley–Leverett Dashboard", layout="wide")


# -----------------------------
# Helpers
# -----------------------------
def fig_relperm(res2):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(res2["Sw"], res2["krw"], linewidth=2, label=r"$k_{rw}(S_w)$")
    ax.plot(res2["Sw"], res2["kro"], linewidth=2, label=r"$k_{ro}(S_w)$")
    ax.set_xlabel("Water saturation $S_w$")
    ax.set_ylabel("Relative permeability")
    ax.set_title("Relative Permeability vs Water Saturation")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    return fig


def fig_fw(res2):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(res2["Sw"], res2["fw"], linewidth=2, label=r"$f_w(S_w)$")
    ax.plot(res2["Sw"], res2["fw_tan"], "--", linewidth=2, label="Shock tangent")
    ax.scatter([res2["Swf"]], [res2["fwf"]], zorder=5, label=f"Shock (Swf={res2['Swf']:.3f})")
    ax.set_xlabel("Water saturation $S_w$")
    ax.set_ylabel("Water fractional flow $f_w$")
    ax.set_title("Fractional Flow vs Water Saturation")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    return fig


def fig_rf(res2):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(res2["PV"], res2["RF"], linewidth=2, label="RF(PV)")
    ax.axvline(res2["PV_bt"], linestyle="--", label=f"PV_bt={res2['PV_bt']:.2f}")
    ax.axhline(res2["RF_ult"], linestyle=":", label="RF ultimate (movable oil)")
    ax.set_xlabel("Injected PV")
    ax.set_ylabel("Recovery factor (RF)")
    ax.set_title("Recovery vs Injected PV (BL + Welge tail)")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    return fig


def parse_m_list(text: str):
    try:
        vals = [float(x.strip()) for x in text.split(",") if x.strip() != ""]
        vals = [v for v in vals if v > 0]
        return vals
    except Exception:
        return []


# -----------------------------
# Sidebar (Inputs + Nav)
# -----------------------------
st.sidebar.title("BL Project")
task = st.sidebar.radio(
    "Go to:",
    ["Task 1 – Inputs", "Task 2 – Buckley–Leverett", "Task 3 – Sweep Efficiency", "Task 4 – Sensitivity Study"],
)

st.sidebar.markdown("---")
st.sidebar.subheader("Rock & Fluid Inputs")

Swc = st.sidebar.number_input("Swc (connate water saturation)", min_value=0.0, max_value=0.99, value=0.20, step=0.01, format="%.4f")
Sor = st.sidebar.number_input("Sor (residual oil saturation)", min_value=0.0, max_value=0.99, value=0.30, step=0.01, format="%.4f")

mu_w = st.sidebar.number_input("μw (water viscosity, cP)", min_value=0.0001, value=1.0, step=0.1, format="%.4f")
mu_o = st.sidebar.number_input("μo (oil viscosity, cP) – base case", min_value=0.0001, value=5.0, step=0.1, format="%.4f")

alpha1 = st.sidebar.number_input("α1 (oil Corey coefficient)", min_value=0.0, value=1.0, step=0.05, format="%.4f")
alpha2 = st.sidebar.number_input("α2 (water Corey coefficient)", min_value=0.0, value=1.0, step=0.05, format="%.4f")

m = st.sidebar.number_input("m (oil Corey exponent)", min_value=0.01, value=2.0, step=0.1, format="%.4f")
n = st.sidebar.number_input("n (water Corey exponent)", min_value=0.01, value=2.0, step=0.1, format="%.4f")

st.sidebar.markdown("---")
st.sidebar.subheader("Reservoir & Flood Inputs (optional)")
phi = st.sidebar.number_input("φ (porosity)", min_value=0.001, max_value=0.9, value=0.20, step=0.01, format="%.4f")
h = st.sidebar.number_input("h (net thickness, m)", min_value=0.01, value=10.0, step=1.0, format="%.4f")
A = st.sidebar.number_input("A (pattern area, m²)", min_value=1.0, value=10000.0, step=500.0, format="%.2f")
q = st.sidebar.number_input("q (injection rate, m³/day)", min_value=0.0001, value=100.0, step=10.0, format="%.4f")

st.sidebar.markdown("---")
st.sidebar.subheader("Sweep Inputs (Task 3/4)")
M_base = st.sidebar.number_input("M (base case)", min_value=0.01, value=1.5, step=0.1, format="%.4f")
hetero = st.sidebar.number_input("Heterogeneity index (0–1 recommended)", min_value=0.0, max_value=1.0, value=0.30, step=0.05, format="%.4f")
pattern = st.sidebar.selectbox("Pattern type", ["5-spot", "9-spot", "line drive"], index=0)
PV_eval = st.sidebar.number_input("PV_eval (later PV for Task 3)", min_value=0.01, value=1.0, step=0.1, format="%.4f")

st.sidebar.markdown("---")
st.sidebar.subheader("Task 4 Settings")
SwD_ref = st.sidebar.number_input("SwD_ref used to enforce M", min_value=0.01, max_value=0.99, value=float(SWD_REF_FOR_M_DEFAULT), step=0.05, format="%.4f")
M_list_text = st.sidebar.text_input("M list (comma-separated)", value="0.5, 1, 2, 5")


# Bundle parameters for use
params = dict(
    Swc=Swc, Sor=Sor,
    mu_w=mu_w, mu_o=mu_o,
    alpha1=alpha1, alpha2=alpha2,
    m=m, n=n,
    phi=phi, h=h, A=A, q=q,
    M_base=M_base, hetero=hetero, pattern=pattern,
    PV_eval=PV_eval,
    SwD_ref=SwD_ref
)


# -----------------------------
# Cached compute (base Task 2)
# -----------------------------
@st.cache_data(show_spinner=False)
def cached_task2(Swc, Sor, mu_w, mu_o, alpha1, alpha2, m, n, PV_eval):
    return compute_task2(Swc, Sor, mu_w, mu_o, alpha1, alpha2, m, n, PV_eval=PV_eval)


# -----------------------------
# Main content
# -----------------------------
st.title("Buckley–Leverett Dashboard (Tasks 1–4)")

# Validate basic saturation window early
if (1.0 - Swc - Sor) <= 0:
    st.error("Invalid saturations: (1 − Swc − Sor) must be > 0.")
    st.stop()

# TASK 1 – Inputs (display + quick checks)
if task == "Task 1 – Inputs":
    st.subheader("Task 1 – Inputs Panel")
    st.write("All inputs are editable from the sidebar. Use the other tabs to run Tasks 2–4.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Rock & Fluid")
        st.write({
            "Swc": Swc, "Sor": Sor,
            "mu_w (cP)": mu_w, "mu_o (cP)": mu_o,
            "alpha1": alpha1, "alpha2": alpha2,
            "m": m, "n": n
        })
    with col2:
        st.markdown("### Sweep + Reservoir")
        st.write({
            "M_base": M_base, "heterogeneity": hetero, "pattern": pattern,
            "PV_eval": PV_eval,
            "phi": phi, "h (m)": h, "A (m²)": A, "q (m³/day)": q
        })

    st.info("Next: open Task 2 to generate kr, fw, and RF plots.")

# TASK 2 – Buckley–Leverett
elif task == "Task 2 – Buckley–Leverett":
    st.subheader("Task 2 – Buckley–Leverett Module")

    run = st.button("Run Task 2", type="primary")
    if run:
        try:
            res2 = cached_task2(Swc, Sor, mu_w, mu_o, alpha1, alpha2, m, n, PV_eval)
            st.session_state["res2"] = res2
        except Exception as e:
            st.error(f"Task 2 error: {e}")

    res2 = st.session_state.get("res2")
    if res2 is None:
        st.warning("Click **Run Task 2** to compute results.")
    else:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Swf", f"{res2['Swf']:.4f}")
        k2.metric("PV_bt", f"{res2['PV_bt']:.4f}")
        k3.metric("ED_bt", f"{res2['ED_bt']:.4f}")
        k4.metric("PV_end (auto)", f"{res2['PV_end']:.2f}")

        # Plot order: kr → fw → RF
        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(fig_relperm(res2))
        with c2:
            st.pyplot(fig_fw(res2))

        st.pyplot(fig_rf(res2))

# TASK 3 – Sweep Efficiency
elif task == "Task 3 – Sweep Efficiency":
    st.subheader("Task 3 – Sweep Efficiency Calculations")

    if "res2" not in st.session_state:
        st.info("Run Task 2 first so we can use PV_bt and ED_bt.")
        st.stop()

    res2 = st.session_state["res2"]
    PV_bt = res2["PV_bt"]
    ED_bt = res2["ED_bt"]

    EA_bt = areal_sweep_EA(PV_bt, M_base, pattern)
    EI_bt = vertical_sweep_EI(M_base, hetero)
    EV_bt = EA_bt * EI_bt
    E_bt = overall_efficiency(ED_bt, EA_bt, EI_bt)

    EA_ev = areal_sweep_EA(PV_eval, M_base, pattern)
    EI_ev = vertical_sweep_EI(M_base, hetero)
    EV_ev = EA_ev * EI_ev
    E_ev = overall_efficiency(ED_bt, EA_ev, EI_ev)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### At breakthrough")
        st.write({
            "PV_bt": PV_bt,
            "ED_bt": ED_bt,
            "EA_bt": EA_bt,
            "EI_bt": EI_bt,
            "EV_bt": EV_bt,
            "Overall E (ED*EA*EI)": E_bt
        })

    with col2:
        st.markdown(f"### At PV = {PV_eval:g}")
        st.write({
            "EA": EA_ev,
            "EI": EI_ev,
            "EV": EV_ev,
            "Overall E (ED*EA*EI)": E_ev
        })

    # Optional operational time estimate
    try:
        t_bt_days = PV_bt * (phi * A * h) / q
        st.caption(f"Approx breakthrough time (days) = PV_bt × (φAh)/q ≈ **{t_bt_days:.2f}** days")
    except Exception:
        pass

# TASK 4 – Sensitivity Study
elif task == "Task 4 – Sensitivity Study":
    st.subheader("Task 4 – Sensitivity Study")

    M_list = parse_m_list(M_list_text)
    if len(M_list) == 0:
        st.error("Enter a valid M list in the sidebar (e.g., 0.5,1,2,5).")
        st.stop()

    run4 = st.button("Run Task 4", type="primary")
    if run4:
        try:
            scenarios = []
            rows = []

            for M_case in M_list:
                mu_o_case = mu_o_for_target_M(alpha1, alpha2, m, n, mu_w, M_case, SwD_ref)
                r = compute_task2(Swc, Sor, mu_w, mu_o_case, alpha1, alpha2, m, n, PV_eval=PV_eval)

                # Sweep (placeholders)
                EA_bt = areal_sweep_EA(r["PV_bt"], M_case, pattern)
                EI_bt = vertical_sweep_EI(M_case, hetero)
                EV_bt = EA_bt * EI_bt
                E_bt = overall_efficiency(r["ED_bt"], EA_bt, EI_bt)

                EA_ev = areal_sweep_EA(PV_eval, M_case, pattern)
                EI_ev = vertical_sweep_EI(M_case, hetero)
                EV_ev = EA_ev * EI_ev
                E_ev = overall_efficiency(r["ED_bt"], EA_ev, EI_ev)

                RF_bt = float(np.interp(min(r["PV_bt"], r["PV"].max()), r["PV"], r["RF"]))
                RF_ev = float(np.interp(PV_eval, r["PV"], r["RF"])) if PV_eval <= r["PV"].max() else np.nan

                scenarios.append({"M": M_case, "mu_o": mu_o_case, "res": r})
                rows.append({
                    "M": M_case,
                    "heterogeneity": hetero,
                    "pattern": pattern,
                    "mu_o_used(cp)": mu_o_case,
                    "Swf": r["Swf"],
                    "PV_bt": r["PV_bt"],
                    "ED_bt": r["ED_bt"],
                    "RF_bt": RF_bt,
                    f"RF@PV={PV_eval:g}": RF_ev,
                    "EA_bt": EA_bt,
                    "EI_bt": EI_bt,
                    "EV_bt": EV_bt,
                    "Overall_E_bt": E_bt,
                    f"EA@PV={PV_eval:g}": EA_ev,
                    f"EI@PV={PV_eval:g}": EI_ev,
                    f"EV@PV={PV_eval:g}": EV_ev,
                    f"Overall_E@PV={PV_eval:g}": E_ev,
                })

            st.session_state["scenarios"] = scenarios
            st.session_state["summary_df"] = pd.DataFrame(rows)

        except Exception as e:
            st.error(f"Task 4 error: {e}")

    scenarios = st.session_state.get("scenarios")
    df = st.session_state.get("summary_df")

    if not scenarios:
        st.info("Click **Run Task 4** to compute sensitivity scenarios.")
        st.stop()

    # Plot 1: fw curves
    fig1, ax1 = plt.subplots(figsize=(8.5, 5.2))
    for sc in scenarios:
        r = sc["res"]
        ax1.plot(r["Sw"], r["fw"], linewidth=2, label=f"M={sc['M']}")
    ax1.set_xlabel("Water saturation $S_w$")
    ax1.set_ylabel("Water fractional flow $f_w$")
    ax1.set_title("Sensitivity: Fractional Flow Curves")
    ax1.grid(True)
    ax1.legend()
    fig1.tight_layout()
    st.pyplot(fig1)

    # Plot 2: RF vs PV on common PV axis
    pvbt_max = max(sc["res"]["PV_bt"] for sc in scenarios)
    PV_abs_end = min(60.0, 2.0 * max(PV_eval, pvbt_max, 1.0))
    PV_common = np.linspace(0.0, PV_abs_end, 2000)

    fig2, ax2 = plt.subplots(figsize=(8.5, 5.2))
    for sc in scenarios:
        r = sc["res"]
        RF_common = np.interp(np.minimum(PV_common, r["PV"].max()), r["PV"], r["RF"])
        ax2.plot(PV_common, RF_common, linewidth=2, label=f"M={sc['M']} (PVbt={r['PV_bt']:.2f})")
    ax2.axvline(PV_eval, linestyle="--", label=f"PV={PV_eval:g}")
    ax2.set_xlabel("Injected PV")
    ax2.set_ylabel("Recovery factor (RF)")
    ax2.set_title("Sensitivity: RF vs Injected PV (common PV axis)")
    ax2.grid(True)
    ax2.legend()
    fig2.tight_layout()
    st.pyplot(fig2)

    # Plot 3: RF vs PV/PVbt (normalized)
    fig3, ax3 = plt.subplots(figsize=(8.5, 5.2))
    PVn = np.linspace(0.0, 3.0, 1200)
    for sc in scenarios:
        r = sc["res"]
        PV_scaled = PVn * r["PV_bt"]
        RF_scaled = np.interp(np.minimum(PV_scaled, r["PV"].max()), r["PV"], r["RF"])
        ax3.plot(PVn, RF_scaled, linewidth=2, label=f"M={sc['M']} (PVbt={r['PV_bt']:.2f})")
    ax3.axvline(1.0, linestyle="--", label="PV/PVbt = 1 (Breakthrough)")
    ax3.set_xlabel("Normalized injected PV, PV/PVbt")
    ax3.set_ylabel("Recovery factor (RF)")
    ax3.set_title("Sensitivity: RF vs Normalized PV (PV/PVbt)")
    ax3.grid(True)
    ax3.legend()
    fig3.tight_layout()
    st.pyplot(fig3)

    st.markdown("### Summary Table")
    st.dataframe(df, use_container_width=True)

    # Download CSV
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download summary table as CSV",
        data=csv_bytes,
        file_name="task4_summary.csv",
        mime="text/csv"
    )
