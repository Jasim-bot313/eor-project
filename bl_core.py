# bl_core.py
import numpy as np

# =====================================================
# GLOBAL CONTROLS (fixed grids + AUTO PV)
# =====================================================
N_SW_DEFAULT = 2500
N_PV_DEFAULT = 2500

SWD_REF_FOR_M_DEFAULT = 0.5
PV_AUTO_MIN_DEFAULT = 1.0
PV_AUTO_FACTOR_DEFAULT = 2.0
PV_AUTO_CAP_DEFAULT = 60.0


def auto_pv_end(PV_bt: float, PV_eval: float | None, pv_min: float, pv_factor: float, pv_cap: float) -> float:
    targets = [pv_min, PV_bt]
    if PV_eval is not None:
        targets.append(PV_eval)
    raw = pv_factor * max(targets)
    return max(pv_min, min(raw, pv_cap))


def swd(Sw: np.ndarray, Swc: float, Sor: float) -> np.ndarray:
    denom = 1.0 - Sor - Swc
    if denom <= 0:
        raise ValueError("(1 - Sor - Swc) must be > 0.")
    return np.clip((Sw - Swc) / denom, 0.0, 1.0)


def relperm_corey(Sw: np.ndarray, Swc: float, Sor: float, alpha1: float, alpha2: float, m: float, n: float):
    SwD = swd(Sw, Swc, Sor)
    kro = alpha1 * (1.0 - SwD) ** m
    krw = alpha2 * (SwD) ** n
    return krw, kro


def dfdSw_numeric(Sw_grid: np.ndarray, fw_grid: np.ndarray) -> np.ndarray:
    return np.gradient(fw_grid, Sw_grid)


def find_shock_saturation_maxsecant(Sw_grid: np.ndarray, fw_grid: np.ndarray, Swc: float):
    """
    Textbook robust shock finder:
      maximize secant slope: fw(S)/(S - Swc)  (since fw(Swc)=0)
    Excludes boundaries to avoid collapse to endpoints.
    """
    df = dfdSw_numeric(Sw_grid, fw_grid)

    eps = 1e-6
    mask = (Sw_grid > Swc + eps) & (Sw_grid < Sw_grid.max() - eps)
    Sw2 = Sw_grid[mask]
    fw2 = fw_grid[mask]

    sec = fw2 / (Sw2 - Swc)
    i = int(np.argmax(sec))

    Swf = float(Sw2[i])
    fwf = float(np.interp(Swf, Sw_grid, fw_grid))
    fwprime = float(np.interp(Swf, Sw_grid, df))
    return Swf, fwf, fwprime


def mu_o_for_target_M(alpha1: float, alpha2: float, m: float, n: float, mu_w: float, M_target: float, SwD_ref: float):
    """
    Enforce mobility ratio at reference SwD:
      M = (krw/mu_w) / (kro/mu_o)  =>  mu_o = M * mu_w * (kro_ref/krw_ref)
    """
    krw_ref = alpha2 * (SwD_ref ** n)
    kro_ref = alpha1 * ((1.0 - SwD_ref) ** m)
    if krw_ref <= 0 or kro_ref <= 0:
        raise ValueError("Reference kr values invalid. Check alpha/m/n or SwD_ref.")
    return M_target * mu_w * (kro_ref / krw_ref)


def rf_curve_welge(PV: np.ndarray, Sw_grid: np.ndarray, fw_grid: np.ndarray, df_grid: np.ndarray,
                   Swc: float, Sor: float, Swf: float, fwprime: float):
    """
    RF(PV) = BL linear to BT + Welge tail after BT.
    Ensures continuity at PV_bt and clips to movable oil limit.
    """
    PV_bt = 1.0 / fwprime
    RF_ult = (1.0 - Sor - Swc) / (1.0 - Swc)
    RF_bt = (Swf - Swc) / (1.0 - Swc)

    RF = np.zeros_like(PV, dtype=float)

    mask_post = Sw_grid >= (Swf - 1e-12)
    Sw_post = Sw_grid[mask_post]
    df_post = df_grid[mask_post]
    fw_post = fw_grid[mask_post]

    order = np.argsort(-df_post)
    df_sorted = df_post[order]
    Sw_sorted = Sw_post[order]

    pre = PV <= PV_bt
    RF[pre] = RF_bt * (PV[pre] / PV_bt)

    post = ~pre
    if np.any(post):
        target = 1.0 / np.maximum(PV[post], 1e-30)
        tmin, tmax = float(np.min(df_sorted)), float(np.max(df_sorted))
        target = np.clip(target, tmin, tmax)

        # invert df(Sw)=target; need increasing x for interp
        Sw_p = np.interp(target, df_sorted[::-1], Sw_sorted[::-1])
        fw_p = np.interp(Sw_p, Sw_post, fw_post)

        Sw_bar = Sw_p + (1.0 - fw_p) * PV[post]
        Sw_bar = np.clip(Sw_bar, Swc, 1.0 - Sor)

        RF_post = (Sw_bar - Swc) / (1.0 - Swc)
        RF_post -= (RF_post[0] - RF_bt)  # continuity
        RF[post] = RF_post

    RF = np.clip(RF, 0.0, RF_ult)
    return RF, PV_bt, RF_bt, RF_ult


# =====================================================
# Task 3 sweep placeholders (swap with your lecture correlations if needed)
# =====================================================
def areal_sweep_EA(PV: float, M: float, pattern: str):
    pattern = (pattern or "").lower()
    pattern_factor = 1.0
    if "line" in pattern:
        pattern_factor = 0.90
    elif "9" in pattern:
        pattern_factor = 1.05
    elif "5" in pattern or "spot" in pattern:
        pattern_factor = 1.00

    EA = pattern_factor * (1.0 - np.exp(-1.1 * PV)) * (1.0 / (1.0 + 0.18 * (M - 1.0)))
    return float(np.clip(EA, 0.0, 1.0))


def vertical_sweep_EI(M: float, hetero: float):
    hetero = float(np.clip(hetero, 0.0, 1.0))
    EI = (1.0 - 0.65 * hetero) * (1.0 / (1.0 + 0.22 * (M - 1.0)))
    return float(np.clip(EI, 0.0, 1.0))


def overall_efficiency(ED: float, EA: float, EI: float):
    return float(np.clip(ED * EA * EI, 0.0, 1.0))


# =====================================================
# Task 2 core
# =====================================================
def compute_task2(
    Swc: float, Sor: float,
    mu_w: float, mu_o: float,
    alpha1: float, alpha2: float, m: float, n: float,
    PV_eval: float | None = None,
    n_sw: int = N_SW_DEFAULT,
    n_pv: int = N_PV_DEFAULT,
    pv_min: float = PV_AUTO_MIN_DEFAULT,
    pv_factor: float = PV_AUTO_FACTOR_DEFAULT,
    pv_cap: float = PV_AUTO_CAP_DEFAULT
) -> dict:
    # Sanity checks
    if not (0 <= Swc < 1) or not (0 <= Sor < 1):
        raise ValueError("Swc and Sor must be between 0 and 1.")
    if (1.0 - Swc - Sor) <= 0:
        raise ValueError("(1 - Swc - Sor) must be > 0.")
    if mu_w <= 0 or mu_o <= 0:
        raise ValueError("Viscosities must be > 0.")
    if alpha1 < 0 or alpha2 < 0 or m <= 0 or n <= 0:
        raise ValueError("alpha1/alpha2 >=0 and m/n >0 required.")

    Sw = np.linspace(Swc, 1.0 - Sor, n_sw)

    krw, kro = relperm_corey(Sw, Swc, Sor, alpha1, alpha2, m, n)
    lam_w = krw / mu_w
    lam_o = kro / mu_o
    fw = lam_w / (lam_w + lam_o + 1e-30)
    dfw = dfdSw_numeric(Sw, fw)

    Swf, fwf, fwprime = find_shock_saturation_maxsecant(Sw, fw, Swc)
    if not (Swc < Swf < (1.0 - Sor)):
        raise ValueError("Shock saturation Swf not between Swc and 1-Sor. Check inputs.")
    if fwprime <= 0:
        raise ValueError("fw'(Swf) must be positive. Check inputs.")
    if not (0.0 < fwf < 1.0):
        raise ValueError("fw(Swf) must be between 0 and 1. Check inputs.")

    PV_bt = 1.0 / fwprime
    PV_end = auto_pv_end(PV_bt, PV_eval, pv_min, pv_factor, pv_cap)

    PV = np.linspace(0.0, PV_end, n_pv)
    RF, PV_bt, RF_bt, RF_ult = rf_curve_welge(PV, Sw, fw, dfw, Swc, Sor, Swf, fwprime)

    ED_bt = (Swf - Swc) / (1.0 - Swc)

    # Tangent line: anchored at (Swc,0) and shown only up to Swf
    m_tan = fwf / (Swf - Swc)
    fw_tan = m_tan * (Sw - Swc)
    fw_tan[(Sw < Swc) | (Sw > Swf)] = np.nan
    fw_tan[(fw_tan < 0) | (fw_tan > 1)] = np.nan

    return {
        "Sw": Sw,
        "krw": krw,
        "kro": kro,
        "fw": fw,
        "dfw": dfw,
        "Swf": Swf,
        "fwf": fwf,
        "fwprime": fwprime,
        "PV": PV,
        "RF": RF,
        "PV_bt": PV_bt,
        "RF_bt": RF_bt,
        "RF_ult": RF_ult,
        "ED_bt": ED_bt,
        "fw_tan": fw_tan,
        "PV_end": PV_end,
        "mu_o_used": mu_o,
    }
