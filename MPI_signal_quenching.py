"""MPI signal quenching -- dynamic-magnetization (Neel-Brown) simulation library.

Macrospin model of the MPI signal of superparamagnetic nanoclusters: sinusoidal
drive plus a linear bias ramp, relaxation toward the Langevin equilibrium with
a field-dependent effective time  tau_eff(H) = (1/tau_B + 1/tau_N)^-1, and
point-spread-function (PSF) reconstruction by binning dM/dt against H.

  tau_B : Brownian Fokker-Planck eigenvalue solve at every field.
  tau_N : Neel Fokker-Planck eigenvalue solve where sigma_eff = sigma(1-h)^2 <= 20,
          Brown's high-barrier closed form (Eq. 14) elsewhere ("hybrid" model).

Sections A-C are the solver. The last section ("Size-distribution quadrature
and viscosity-series panel") holds the quadrature helpers used for the
polydisperse glycerol series. Run `MPI_signal_quenching_figures.py` to
reproduce the manuscript figures and the source-data workbook.
"""
import numpy as np
from scipy.signal import lfilter
# ======================================================================
# Section A -- Function Definition Segment
# Simulates Dynamic Magnetization under a continuous sinusoid and bias ramping field
# Relaxation is field dependent, tau(H), from the hybrid model (Section C):
# FPE eigenvalue solve where resolvable, Brown's Eq. 14 elsewhere.
# PSF reconstruction bins dM/dt against H separately for the positive and
# negative slew of dM/dt.
# ======================================================================
k_b = 1.380649e-23  # Boltzmann constant in J/K
u_o = 4 * np.pi * 1e-7
T = 298.0  # K. Single source for every scan: M_eq, tau_B/tau_N and the eigen solves all read this.

def langevin(x):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    small = np.abs(x) < 1e-6
    out[small] = x[small]/3.0
    out[~small] = 1/np.tanh(x[~small]) - 1/x[~small]
    return out  

def M_eq(H, V_particle=1.0, rho_iron=1.0, Ms=1.0):
    """
    Equilibrium magnetization curve.
    Ms: saturation magnetization (a.u.)
    NOTE: H must be in Tesla -- simulate_scan converts its mT input before
    calling this.
    """
    m = V_particle * rho_iron * Ms  # magnetic moment per particle (Am^2)
    return Ms * langevin(m * H / (k_b * T))  # magnetization in Am^2/kg

def excitation_waveform(H_B_range_mT, amplitude_mT, frequency_Hz,
                            n_cycles=2000, n_bias=1000):
    """
    Generates the magnetic field excitation waveform including excitation and bias waveform where
    H_total(t) = H_ramp(t) + amplitude*sin(2*pi*f*t),  H_ramp(t) = H_Bias_Start_mT + ramp_rate*t.
    Args: 
        H_B_range_mT: the bias field range i.e. 50mT implies bias field from [-50mT,50mT]
        amplitude_mT:excitation field amplitude in mT
        frequency_Hz: Excitation frequency in Hz
        n_cycles: number of samples per excitation period
        n_bias: number of bias steps in a relaxometry measurement
    Returns: 
        t_ex: Descritization of time (s)
        H_total: Total field include bias and excitation fields (mT)
        dt: timstep (s)
        ramp_rate: Bias field ramp rate (mT/s)
    """
    f = frequency_Hz
    a = amplitude_mT

    dt = 1/(n_bias*f) # Sampe Rate
    n_t = n_cycles*n_bias # Total number of samples
    t = np.arange(n_t)*dt # vectorized excitation timescale

    T_total = t[-1]
    ramp_rate = (2*H_B_range_mT)/T_total # Bias amplitude ramp rate
    H_ramp_t = -H_B_range_mT + ramp_rate*t # Linear bias field ramping across the sample field range
    excitation_field = lambda t, f, a: a*np.sin(2*np.pi*f*t) # Generation of the excitation waveform
    H_total = H_ramp_t + excitation_field(t, f, a) # Total field waveform for nanoparticle relaxometry measurement

    return t, H_total, dt,f,a, ramp_rate

def bin_average(H, values, mask, H_grid):
    """Binned average: for each H_grid cell, average all (H, value) samples with mask=True."""
    Hm, vm = H[mask], values[mask]
    edges = np.concatenate([[H_grid[0]-(H_grid[1]-H_grid[0])/2],
                             (H_grid[:-1]+H_grid[1:])/2,
                             [H_grid[-1]+(H_grid[1]-H_grid[0])/2]])
    idx = np.clip(np.digitize(Hm, edges) - 1, 0, len(H_grid)-1)
    sums = np.bincount(idx, weights=vm, minlength=len(H_grid))
    counts = np.bincount(idx, minlength=len(H_grid))
    with np.errstate(invalid='ignore'):
        avg = sums/counts
    avg[counts == 0] = np.nan
    return avg, counts

def _relax_step_timevarying(Meq, decay, efold_budget=500.0):
    """
    Vectorized solver of the linear recurrence with time-varying coefficients
        M[i] = decay[i]*M[i-1] + (1-decay[i])*Meq[i],   M[-1] := Meq[0]
    Meq: equilibrium magnetization
    decay: decay factor
    efold_budget: sets chunks to solve
    
    """
    n=len(Meq)
    decay = np.maximum(decay,1e-100)
    logdecay = np.log(decay)
    cum = np.cumsum(-logdecay) #number of efoldings
    # if decay is infinite, solve in a single chunk
    if cum[-1] <= 0:
        edges = np.array([n])
    # Otherwise, solve in chunks
    else:
        nchunks_est = int(cum[-1]/efold_budget)+1
        edges = np.searchsorted(cum, np.arange(1, nchunks_est+1)*efold_budget)
        edges = np.unique(np.clip(edges,1,n))
        if edges[-1] !=n:
            edges = np.append(edges,n)

    M = np.empty(n)
    M_prev = Meq[0] # initial magnetization in the equilibrium magnetization state

    start = 0
    # Loop over all chunks
    for end in edges:
        d=decay[start:end] #this chunks decay value
        me = Meq[start:end] #this chunks starting point magnetization
        P = np.cumprod(d)
        w =(1-d)*me/P
        S = np.cumsum(w)

        M[start:end]=P*(M_prev+S)

        M_prev=M[end-1]

        start=end #advance to the next chunk
    return M

def simulate_scan(H_B_range_mT, amplitude, frequency, tau_of_H, n_cycles=2000, n_bias=1000, settle_cycles=20,
                                     V_c=1.0,V_h=1.0, rho_iron=1.0, Ms=1.0,
                                     H_grid_points=2000):
    """Dynamic magnetization under a bias ramp plus sinusoidal drive.

    M relaxes toward M_eq(H(t)) with time constant tau_of_H(H(t)) (H in mT);
    dM/dt is then binned against H on the positive and negative slew.

    Returns: t, H_total, M_t, dMdt, H_grid, M_vs_H_pos, M_vs_H_neg, dMdt_vs_H_pos,
        dMdt_vs_H_neg, tau_t, Tx, Bx
        (all trimmed to the post-settle_cycles window. tau_t is Tau_eff(H(t)) as actually
        used to solve dM/dt. Tx is the AC-only excitation a*sin(2*pi*f*t); Bx is the
        bias-ramp-only trajectory -H_B_range_mT + ramp_rate*t; Tx + Bx == H_total.)
    """
    t,H_total, dt, f, a, ramp_rate = excitation_waveform(
            H_B_range_mT, amplitude, frequency,
            n_cycles=n_cycles, n_bias=n_bias)
    H_total_T = H_total * 1e-3

    Meq_t = M_eq(H_total_T, V_particle=V_c, rho_iron=rho_iron, Ms=Ms)

    tau_t= np.asarray(tau_of_H(H_total), dtype=float)
    if tau_t.ndim == 0:
        tau_t = np.full_like(H_total, float(tau_t))
    tau_is_constant = np.allclose(tau_t,tau_t[0])

    if tau_is_constant:
        tau0 = tau_t[0]
        lr = np.exp(-dt/tau0)
        M_t, _ = lfilter([1-lr],[1,-lr], Meq_t, zi=np.array([Meq_t[0]*lr]))
    else:
        decay = np.exp(-dt/tau_t)
        M_t = _relax_step_timevarying(Meq_t, decay)

    dMdt = np.gradient(M_t, dt)
    omega = 2*np.pi*f
    H_ref = H_total
    pos_mask = dMdt > 0          # positive / negative slew of dM/dt
    neg_mask = ~pos_mask

    # Raw Tx (AC-only) / Bx (bias-ramp-only) trajectories, same time base as H_total.
    Tx = a * np.sin(omega * t)
    Bx = -H_B_range_mT + ramp_rate * t

    settle_n = settle_cycles*n_bias
    keep = np.arange(len(t)) >= settle_n
    t, H_total, H_ref, M_t, dMdt = t[keep], H_total[keep], H_ref[keep], M_t[keep], dMdt[keep]
    pos_mask, neg_mask = pos_mask[keep], neg_mask[keep]
    tau_t, Tx, Bx = tau_t[keep], Tx[keep], Bx[keep]

    lo = -H_B_range_mT - amplitude
    hi =  H_B_range_mT + amplitude
    H_grid = np.linspace(lo, hi, H_grid_points)

    M_vs_H_pos, _    = bin_average(H_ref, M_t,  pos_mask, H_grid)
    M_vs_H_neg, _    = bin_average(H_ref, M_t,  neg_mask, H_grid)
    dMdt_vs_H_pos, _ = bin_average(H_ref, dMdt, pos_mask, H_grid)
    dMdt_vs_H_neg, _ = bin_average(H_ref, dMdt, neg_mask, H_grid)

    return t, H_total, M_t, dMdt, H_grid, M_vs_H_pos, M_vs_H_neg, dMdt_vs_H_pos, dMdt_vs_H_neg, tau_t, Tx, Bx

     
# ======================================================================
# Section B -- field-dependent relaxation model (Saayujya et al. 2024)
# Verified term-for-term against Eq. 1, 3, 4, 6, 13, 14 of Saayujya et al.
# ======================================================================
gamma = 1.76e11  # rad/(s*T), electron gyromagnetic ratio

def tau_B_zero(V_h, eta, T=T):
    """Zero-field Brownian relaxation time (Debye). Eq. 1: tau_B = 3*eta*V_h / k_B*T"""
    return 3 * eta * V_h / (k_b * T)

def tau_0_neel(Ms_vol, V_c, alpha_damp, T=T):
    """Characteristic internal diffusion time tau_0 (Eq. 4)."""
    return Ms_vol * V_c * (1 + alpha_damp**2) / (2 * k_b * T * gamma * alpha_damp)


def tau_N_field(H, Ms_vol, V_c, K_eff, alpha_damp, T=T):
    """Field-dependent Neel relaxation, Brown's high-barrier approximation (Eq. 14).
    Accurate to within 1% only for xi/sigma <~ 1.5 (h <~ 0.75) per Saayujya et al. -- for
    magnetite this is B <~ 30 mT worst-case, <~1.2 T best-case, depending on K."""
    H = np.atleast_1d(np.asarray(H, dtype=float))
    sigma = K_eff * V_c / (k_b * T)
    t0 = tau_0_neel(Ms_vol, V_c, alpha_damp, T)
    h = u_o * Ms_vol * np.abs(H) / (2 * K_eff)
    h = np.clip(h, 0, 0.9999)

    prefactor = t0 / (sigma * (1 - h**2))
    term1 = sigma**1.5 / (np.sqrt(np.pi) * (1 + sigma)) + 2**(-sigma) / 2
    # These exponentials legitimately overflow to +inf at large sigma -- the
    # np.where guards below already treat that as the (1+h)/exp_plus -> 0 limit,
    # so suppress the RuntimeWarning rather than emit it on every high-barrier
    # call. tau_N_hybrid() routes exactly the large-sigma cases here, so without
    # this the warning fires constantly.
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        exp_plus = np.exp(sigma * (1 + h)**2)
        exp_minus = np.exp(sigma * (1 - h)**2)
        rate_plus = np.where(exp_plus > 1e15, (1 + h) / exp_plus, (1 + h) / (exp_plus - 1))
        rate_minus = np.where(exp_minus > 1e15, (1 - h) / exp_minus, (1 - h) / (exp_minus - 1))
    term2 = rate_plus + rate_minus

    safe = (np.abs(term1) > 1e-300) & (np.abs(term2) > 1e-300)
    tau = np.full_like(H, 1e10)
    tau[safe] = prefactor[safe] / (term1 * term2[safe])
    return np.abs(tau)

def brownian_matrix(xi, N):
    """Tridiagonal system matrix for the Legendre coefficients of the Brownian FPE."""
    A = np.zeros((N, N))
    for kk in range(N):
        A[kk, kk] = -kk*(kk+1)
        if kk+1 < N:
            A[kk, kk+1] = -xi*kk*(kk+1)/(2*kk+3)
        if kk-1 >= 0:
            A[kk, kk-1] = xi*kk*(kk+1)/(2*kk-1)
    return A

def neel_matrix(xi, sigma, N):
    """Pentadiagonal (bandwidth 2) system matrix for the Legendre coefficients of the
    Neel FPE -- bandwidth 2 because the 2*sigma*x*W drift term couples P_l to
    P_l, P_l+2, P_l-2 (not just P_l-1/P_l+1)."""
    A = np.zeros((N, N))
    for kk in range(N):
        A[kk, kk] = -kk*(kk+1) + 2*sigma*kk*(kk+1)/((2*kk+3)*(2*kk-1))
        if kk+1 < N:
            A[kk, kk+1] = -xi*kk*(kk+1)/(2*kk+3)
        if kk-1 >= 0:
            A[kk, kk-1] = xi*kk*(kk+1)/(2*kk-1)
        if kk+2 < N:
            A[kk, kk+2] = -2*sigma*(kk+2)*(kk+1)*kk/((2*kk+5)*(2*kk+3))
        if kk-2 >= 0:
            A[kk, kk-2] = 2*sigma*(kk-1)*kk*(kk+1)/((2*kk-3)*(2*kk-1))
    return A

def slowest_tau(A, tau0):
    """The l=0 mode is always the conserved/equilibrium eigenvalue 0; the long-time
    relaxation is set by the eigenvalue with the next-largest (least negative) real part.

    Returns NaN when that eigenvalue is not resolvable in double precision. For a
    high barrier the target scales as lambda_1 ~ -2*exp(-sigma_eff) while
    ||A|| ~ N^2, so once |lambda_1| < eps*||A|| the mode is numerically lost and
    re[1] rounds to +0.0 (or goes positive). Returning -1/0 = -inf there would
    inject a NEGATIVE relaxation time downstream into tau_eff and any tau_N/tau_B
    comparison; NaN makes the failure visible instead. Use tau_N_hybrid() (Section C) to get a usable
    value in that regime.
    """
    eigvals = np.linalg.eigvals(A/(2*tau0))
    re = np.sort(np.real(eigvals))[::-1]
    if not np.isfinite(re[1]) or re[1] >= 0.0:
        return np.nan
    return -1.0/re[1]

# ----------------------------------------------------------------------
# How many Legendre coefficients does the truncation actually need?
#
# A FIXED N IS WRONG, because the coupling term in both matrices is
# xi = mu0*m*H/(kB*T) with m = Ms_vol*V_c ~ d^3. Over a 10-60 nm sweep xi spans
# more than two decades, and a value of N that is exact at the small end is
# silently wrong at the large end. Measured during development (not reproduced
# by this package) with N = 40 at 80% glycerol and
# 120 mT: a 38 nm single core returned +23% signal change where the converged
# answer is -0.9%, and a 70 nm cluster returned -86.6% against -98.4%. Both
# looked entirely plausible on a plot.
#
# CALIBRATION (tol = 1e-3 on the returned tau, against N = 440):
#   Brownian matrix, which depends on xi alone --
#       xi      1    10    30   100   300   500  1200  2000
#       N_min   8    16    32    56   100   132   208   268
#       N/sqrt(xi)  ~5.1  5.8  5.6  5.8  5.9  6.0  6.0
#   so N_min = 6*sqrt(xi) to within a few percent across three decades.
#
#   Neel matrix additionally carries sigma, but sigma at fixed xi REDUCES the
#   requirement (the barrier localises the distribution) except at small xi,
#   where the sigma term dominates and saturates near N = 36 for sigma = 20.
#   Since tau_N_hybrid only calls the eigen branch where sigma_eff <= 20, a
#   floor of 40 covers that corner, and the 6*sqrt(xi) term covers the rest.
#
# Cost is O(N^3) per LUT point, so picking N PER POINT rather than per scan
# matters: xi rises linearly with H, so the low-field half of the LUT can use
# the floor while only the top needs the large solves.
N_FLOOR = 40          # covers the sigma_eff <= 20 corner at small xi
N_CAP = 400           # beyond this the mode is lost to float64 anyway
N_SAFETY = 6.5        # calibrated 6.0, plus margin


def required_N(xi, sigma=0.0, floor=N_FLOOR, cap=N_CAP, safety=N_SAFETY):
    """Legendre truncation order needed at reduced field xi (and barrier sigma).

    Returns an int array matching xi. See the calibration note above. The sigma
    term uses sigma_eff = sigma*(1-h)^2 with h = xi/(2*sigma), NOT the bare
    sigma: in the hybrid the eigen branch is only ever entered where sigma_eff
    is small, and keying on bare sigma would inflate N by ~8x for no gain.
    """
    xi = np.abs(np.asarray(xi, dtype=float))
    n = safety * np.sqrt(xi)
    if np.any(np.asarray(sigma) > 0):
        s = np.asarray(sigma, dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            h = np.where(s > 0, xi / (2.0 * np.maximum(s, 1e-30)), 0.0)
        sigma_eff = np.where(s > 0, s * np.clip(1.0 - h, 0.0, None) ** 2, 0.0)
        n = np.maximum(n, 2.2 * sigma_eff)
    n = np.clip(np.ceil(n), floor, cap)
    return (2 * np.ceil(n / 2)).astype(int)      # keep it even


def _N_per_point(N, xi, sigma=0.0):
    """Resolve the N argument to one integer per xi. N='auto' -> required_N."""
    xi = np.atleast_1d(np.asarray(xi, dtype=float))
    if isinstance(N, str):
        if N != 'auto':
            raise ValueError(f"N must be an int or 'auto', got {N!r}")
        return required_N(xi, sigma)
    return np.full(xi.shape, int(N), dtype=int)


def tau_B_eigen(H_array, m, tau_B0, N='auto'):
    xi = u_o*m*np.abs(H_array)/(k_b*T)
    Ns = _N_per_point(N, xi)
    return np.array([slowest_tau(brownian_matrix(x, n), tau_B0)
                     for x, n in zip(np.atleast_1d(xi), Ns)])

def tau_N_eigen(H_array, m, sigma, tau_D, N='auto'):
    xi = u_o*m*np.abs(H_array)/(k_b*T)
    Ns = _N_per_point(N, xi, sigma)
    return np.array([slowest_tau(neel_matrix(x, sigma, n), tau_D)
                     for x, n in zip(np.atleast_1d(xi), Ns)])


def tau_eff(H, V_c, V_h, K_eff, Ms_vol, alpha_damp, eta, T=T, N='auto'):
    """Field-dependent relaxation times, hybrid model. H in A/m.

    Returns a 3-tuple (tau_combined, tau_B, tau_N):
        tau_combined: what drives the dynamics -- 1/(1/tau_B+1/tau_N).
        tau_B, tau_N: the component values.
    SHAPE CONTRACT: the return shape follows H's shape.
        H scalar  ->  three Python floats
        H array   ->  three arrays of H's shape
    (Internally everything is np.atleast_1d; a scalar is collapsed back so that
    scalar call sites such as `grid[i, j] = tN` or f"{tN:.5f}" work on NumPy 2.)
    """
    # Remember whether the caller passed a scalar, so the return can match.
    scalar_in = np.ndim(H) == 0

    # tau_B: the Brownian FPE has no energy barrier, so its slowest eigenvalue
    # never becomes exponentially small and the eigenvalue solve is well
    # conditioned at every field -- no fallback needed.
    # tau_N: eigenvalue where sigma_eff = sigma*(1-h)^2 <= SIGMA_EIGEN_MAX,
    # Brown Eq. 14 elsewhere. See Section C.
    H_arr = np.atleast_1d(np.asarray(H, dtype=float))
    m = Ms_vol * V_c
    tB0 = tau_B_zero(V_h, eta, T)
    tB = tau_B_eigen(H_arr, m, tB0, N=N)
    tN = tau_N_hybrid(H_arr, Ms_vol, V_c, K_eff, alpha_damp, T=T, N=N)

    combined = 1.0 / (1.0/tB + 1.0/tN)

    if scalar_in:
        # Collapse the internal atleast_1d back to plain floats; np.ravel(...)[0]
        # avoids NumPy's deprecation of implicit size-1-array-to-scalar conversion.
        return tuple(float(np.ravel(v)[0]) for v in (combined, tB, tN))

    return combined, tB, tN

# ======================================================================
# Solver presets and the tau-LUT cache
#
# 'preset' selects the NUMERICS (truncation order and LUT resolution) only --
# a preset must never change the model.
#
# 'fast' differs from 'reference' only in lut_points (800 -> 400): 0.13% on the
# quench ratio (Gd peak / bare peak) at eta = 0.89, 1.73, 4.05 and 66.6 cP,
# because tau(H) is smooth. Both choose N per LUT point (required_N). 'fast'
# also enables the exact tau-LUT cache (_cached_tau_components). The memo is exact (see _cached_tau_components).
#
# NOTE the preset deliberately does NOT touch n_cycles / n_bias / settle_cycles.
# Those live in common_scan because different panels need different
# sampling (e.g. the illustration scan uses far fewer cycles than the PSF scan), and
# silently overriding a caller's scan definition would be a trap.
# ======================================================================

RUN_PRESETS = {
    # N is chosen PER LUT POINT from xi via required_N(). This is both more
    # correct and cheaper than any fixed N: correct because the requirement
    # grows as sqrt(xi) ~ d^1.5 and a fixed N is only ever right over part of a
    # diameter sweep, cheaper because xi rises linearly with H so most of the
    # LUT stays at the floor while only the top needs the big solves.
    #
    # Validated against a fixed N = 140 at 80% glycerol, 120 mT (signal change
    # on quenching, %). These development checks include particle types beyond the
    # 35 nm cluster simulated here and are not reproduced by this package:
    #                       N = 40      auto      N = 140
    #   15 nm single core    +0.003    +0.003     +0.003
    #   26 nm single core   -59.559   -59.559    -59.559
    #   38 nm single core   +21.524    -0.937     -0.937   <-- N=40 fails
    #   50 nm single core   -25.163    -1.063     -1.063   <-- N=40 fails
    #   35 nm cluster       -58.651   -58.651    -58.651
    #   70 nm cluster       -87.890   -98.418    -98.418   <-- N=40 fails
    # auto matches the reference everywhere and runs ~3x faster than N=140 at
    # small xi, where it drops to the floor.
    'reference': dict(N='auto', lut_points=800),
    'fast':      dict(N='auto', lut_points=400),
}

_TAU_LUT_CACHE = {}


def _cached_tau_components(V_c, K_eff, Ms_vol, alpha_damp, N,
                            lut_B_max_mT, lut_points, T_K):
    """(H_lut_mT, tau_N, G) where tau_B_eigen = tau_B0(eta) * G(xi).

    Two exact identities make this cacheable across viscosity:
      (a) tau_N has no eta dependence at all;
      (b) brownian_matrix depends only on xi and slowest_tau divides by 2*tau0,
          so tau_B_eigen(H, m, tau_B0) = tau_B0 * G(xi) exactly. Verified to
          6e-7 relative (pure floating point) over eta = 1 to 66.6 cP.

    Used by run_scan when preset='fast' (use_cache): one eigendecomposition
    pass per (diameter, K_eff) then serves every viscosity.
    """
    key = (float(V_c), float(K_eff), float(Ms_vol), float(alpha_damp),
           str(N), float(lut_B_max_mT), int(lut_points), float(T_K))
    hit = _TAU_LUT_CACHE.get(key)
    if hit is not None:
        return hit

    B_lut = np.linspace(0, lut_B_max_mT, lut_points) * 1e-3
    H_lut = B_lut / u_o
    m = Ms_vol * V_c
    tN = tau_N_hybrid(H_lut, Ms_vol, V_c, K_eff, alpha_damp, T=T_K, N=N)
    G = tau_B_eigen(H_lut, m, 1.0, N=N)
    out = (B_lut * 1e3, np.asarray(tN, float), np.asarray(G, float))
    _TAU_LUT_CACHE[key] = out
    return out


def make_tau_of_H(common_particle, N='auto',
                   lut_B_max_mT=120, lut_points=800, use_cache=False):
    """Build tau(H) interpolants from a LUT over 0..lut_B_max_mT.

    Returns (tau_of_H, tau_N_of_H, tau_B_of_H):
        tau_of_H: the combined relaxation time consumed by simulate_scan.
        tau_N_of_H, tau_B_of_H: the raw Neel-only / Brownian-only relaxation time at
            the same field points, for reporting. All three come from ONE tau_eff()
            call, so the N/B breakdown adds no extra LUT cost.
    """
    if use_cache:
        # Exact viscosity factorisation -- one eigendecomposition pass per
        # (diameter, K_eff) serves every eta. See _cached_tau_components.
        H_lut_mT, tauN_lut, G = _cached_tau_components(
            common_particle['V_c'], common_particle['K_eff'],
            common_particle['Ms_vol'], common_particle['alpha_damp'], N,
            lut_B_max_mT, lut_points, T)
        tauB_lut = tau_B_zero(common_particle['V_h'],
                               common_particle['eta'], T) * G
        tau_lut = 1.0 / (1.0 / tauB_lut + 1.0 / tauN_lut)
    else:
        B_lut = np.linspace(0, lut_B_max_mT, lut_points) * 1e-3
        H_lut_Am = B_lut / u_o
        H_lut_mT = B_lut * 1e3
        tau_lut, tauB_lut, tauN_lut = tau_eff(H_lut_Am, **common_particle, N=N)
    tau_of_H   = lambda H_mT: np.interp(np.abs(np.asarray(H_mT, dtype=float)), H_lut_mT, tau_lut)
    tau_N_of_H = lambda H_mT: np.interp(np.abs(np.asarray(H_mT, dtype=float)), H_lut_mT, tauN_lut)
    tau_B_of_H = lambda H_mT: np.interp(np.abs(np.asarray(H_mT, dtype=float)), H_lut_mT, tauB_lut)
    return tau_of_H, tau_N_of_H, tau_B_of_H


def run_scan(common_particle, common_scan,
             N=None,
             lut_B_max_mT=None, lut_points=None, preset='reference',
             use_cache=None):
    """
    Returns a single dict:
        H_grid, psf_pos, psf_neg  -- field grid (mT) and binned dM/dt
        Tau_eff, Tau_N, Tau_B     -- float if constant across the scan, else array
                                     matching Tx/Bx/Rx's length.
        Tx   -- AC-only excitation, a*sin(2*pi*f*t), post-settle
        Bx   -- bias-ramp-only trajectory, -H_B_range_mT + ramp_rate*t, post-settle
                 (Tx + Bx reconstructs H_total)
        Rx   -- dict: {'M': M(t), 'dMdt': dM/dt(t)}, both post-settle
    """
    if lut_B_max_mT is None:
        lut_B_max_mT = common_scan['H_B_range_mT'] + common_scan['amplitude']

    # 'preset' fills the SOLVER knobs only; an explicit N or lut_points wins.
    if preset not in RUN_PRESETS:
        raise ValueError(f"preset must be one of {tuple(RUN_PRESETS)}, got {preset!r}")
    p = RUN_PRESETS[preset]
    if N is None:
        N = p['N']
    if lut_points is None:
        lut_points = p['lut_points']
    if use_cache is None:
        use_cache = (preset == 'fast')

    tau_of_H, tau_N_of_H, tau_B_of_H = make_tau_of_H(
        common_particle, N=N,
        lut_B_max_mT=lut_B_max_mT, lut_points=lut_points, use_cache=use_cache)

    res = simulate_scan(tau_of_H=tau_of_H, **common_scan)
    (t, H_total, M_t, dMdt, H_grid, M_vs_H_pos, M_vs_H_neg,
     dMdt_vs_H_pos, dMdt_vs_H_neg, tau_t, Tx, Bx) = res

    Tau_N_arr = np.asarray(tau_N_of_H(H_total), dtype=float)
    Tau_B_arr = np.asarray(tau_B_of_H(H_total), dtype=float)

    def _collapse(x):
        x = np.asarray(x, dtype=float)
        if x.size and np.allclose(x, x[0], equal_nan=True):
            return float(x[0])
        return x

    return dict(
        H_grid=H_grid,
        psf_pos=dMdt_vs_H_pos,
        psf_neg=dMdt_vs_H_neg,
        Tau_eff=_collapse(tau_t),
        Tau_N=_collapse(Tau_N_arr),
        Tau_B=_collapse(Tau_B_arr),
        Tx=Tx,
        Bx=Bx,
        Rx=dict(M=M_t, dMdt=dMdt),
    )

# ======================================================================
# Section C -- numerical validity of the relaxation models
#
# Section B provides two treatments of tau_N. Each is valid on a different
# domain, and they fail in different, quantifiable ways:
#
#   tau_N_field (Eq. 14)  Brown high-barrier with field. Needs sigma >> 1 AND
#                         xi/sigma <~ 1.5 (Saayujya et al.'s 1% bound, Fig. 4).
#   tau_N_eigen           Legendre-truncated FPE eigenvalue solve. Exact in
#                         principle at any field, but limited in practice by
#                         DOUBLE PRECISION. The barrier-crossing eigenvalue
#                         scales as lambda_1 ~ 2*exp(-sigma_eff) while the
#                         system matrix has ||A|| ~ N^2, so once
#                         |lambda_1| < eps*||A|| the mode is unresolvable and
#                         np.linalg.eigvals silently returns the fast diffusion
#                         mode instead -- i.e. tau_N collapses to ~tau_D.
#
# The controlling parameter for that failure is NOT sigma but the FIELD-REDUCED
# barrier sigma_eff = sigma*(1-h)^2, h = xi/(2 sigma): an applied field lowers
# the barrier, so a particle that is unresolvable at H = 0 becomes perfectly
# resolvable once h >~ 0.4. This was mapped against a symmetrised tridiagonal
# finite-difference reference solution over sigma in [5, 50] and h in [0, 0.8]:
# sigma_eff <= 20 reproduces the reference to within 1.4% in every case tested,
# and sigma_eff > 20 fails in every case tested.
#
# Conveniently, Eq. 14 and the eigenvalue solve are COMPLEMENTARY: sigma_eff > 20
# requires small h, which is exactly where xi/sigma = 2h is far inside Eq. 14's
# validity bound. tau_N_hybrid() below exploits that.
# ======================================================================

SIGMA_EIGEN_MAX = 20.0         # sigma_eff above which the eigenvalue solve loses lambda_1


def reduced_field(H, Ms_vol, K_eff):
    """h = mu_0 Ms H / (2 K_eff) = xi/(2 sigma), the field in units of the
    anisotropy field. h = 1 is where the energy barrier vanishes."""
    return u_o * Ms_vol * np.abs(np.asarray(H, dtype=float)) / (2 * K_eff)


def sigma_effective(sigma, h):
    """Field-reduced energy barrier sigma_eff = sigma*(1-h)^2, in units of kB*T.
    This is the exponent that sets the barrier-crossing rate, and therefore the
    quantity that determines whether the eigenvalue solve can resolve it."""
    h = np.clip(np.asarray(h, dtype=float), 0.0, 1.0)
    return sigma * (1.0 - h) ** 2


def eigen_is_resolvable(sigma, h, sigma_eigen_max=SIGMA_EIGEN_MAX):
    """True where tau_N_eigen can be trusted (sigma_eff <= sigma_eigen_max)."""
    return sigma_effective(sigma, h) <= sigma_eigen_max


def tau_N_hybrid(H_array, Ms_vol, V_c, K_eff, alpha_damp, T=T, N='auto',
                 sigma_eigen_max=SIGMA_EIGEN_MAX):
    """Néel relaxation time, valid at every field and every barrier height.

    Uses the FPE eigenvalue solve wherever sigma_eff = sigma*(1-h)^2 is small
    enough to be resolvable in double precision, and Brown's high-barrier
    closed form (Eq. 14) elsewhere. The two regions are complementary:

        sigma_eff <= 20   ->  eigenvalue (exact; Eq. 14 would be the approximation)
        sigma_eff >  20   ->  Eq. 14 (requires small h, so xi/sigma = 2h << 1.5,
                                      i.e. deep inside Eq. 14's 1% bound)

    """
    H_array = np.atleast_1d(np.asarray(H_array, dtype=float))
    sigma = K_eff * V_c / (k_b * T)
    tau_D = tau_0_neel(Ms_vol, V_c, alpha_damp, T)
    m = Ms_vol * V_c
    h = reduced_field(H_array, Ms_vol, K_eff)
    use_eig = eigen_is_resolvable(sigma, h, sigma_eigen_max)

    out = np.empty_like(H_array)
    if np.any(use_eig):
        out[use_eig] = tau_N_eigen(H_array[use_eig], m, sigma, tau_D, N=N)
    if np.any(~use_eig):
        out[~use_eig] = tau_N_field(H_array[~use_eig], Ms_vol, V_c, K_eff,
                                     alpha_damp, T)

    # Safety net: the sigma_eff <= 20 criterion was calibrated against a
    # finite-difference reference, but right at the boundary an individual solve
    # can still fail. Anything non-finite falls back to Eq. 14 rather than
    # propagating a NaN into tau_eff.
    bad = ~np.isfinite(out)
    if np.any(bad):
        out[bad] = tau_N_field(H_array[bad], Ms_vol, V_c, K_eff, alpha_damp, T)

    return out


# ======================================================================
# Section D -- Size-distribution quadrature and viscosity-series panel
#
# Deterministic quadrature over the diameter distribution (replaces a Monte
# Carlo) and the polydisperse signal-vs-viscosity (glycerol) calculation. Every
# scan goes through run_scan above.
# ======================================================================

# ---- convergence-verified fast settings  ----
# N='auto' -> required_N(xi) per LUT point, chosen inside the solver. It MUST
# NOT be a fixed number here: these dicts are passed straight into run_scan and
# so override whatever preset the caller asked for. A fixed N=40 is wrong above
# xi ~ 300 (roughly 35 nm for a solid magnetite core at 120 mT). See required_N().
FAST_SOLVER = dict(N='auto', lut_points=400)
FAST_SCAN = dict(n_cycles=2000, n_bias=1000, settle_cycles=1)
REF_SOLVER = dict(N='auto', lut_points=800)
REF_SCAN = dict(n_cycles=4000, n_bias=1000, settle_cycles=5)


def psf_peak(y):
    """PSF peak: the raw maximum of |y|."""
    y = np.abs(np.nan_to_num(np.asarray(y, float)))
    return float(np.max(y))


def scan_peak(V_c, V_h, K_eff, Ms_vol, alpha_damp, eta, Ms_mass, rho_iron,
              H_B_range_mT=100.0, amplitude=20.0, frequency=20000.0,
              fast=True, **over):
    """One PSF peak via production run_scan. fast=False uses the reference settings."""
    solver = dict(FAST_SOLVER if fast else REF_SOLVER)
    scan_kw = dict(FAST_SCAN if fast else REF_SCAN)
    scan_kw.update({k: over.pop(k) for k in list(over)
                    if k in ('n_cycles', 'n_bias', 'settle_cycles')})
    solver.update({k: over.pop(k) for k in list(over) if k in ('N', 'lut_points')})
    part = dict(V_c=V_c, V_h=V_h, K_eff=K_eff, Ms_vol=Ms_vol,
                alpha_damp=alpha_damp, eta=eta)
    scan = dict(H_B_range_mT=H_B_range_mT, amplitude=amplitude, frequency=frequency,
                V_c=V_c, V_h=V_h, rho_iron=rho_iron, Ms=Ms_mass,
                **scan_kw)
    r = run_scan(part, scan, **solver)
    return psf_peak(r['psf_pos'])


# ======================================================================
# size-distribution quadrature -- replaces the Monte Carlo
# ======================================================================

def size_quadrature(d0, cv, n_nodes=9, n_sigma=3.0):
    """Nodes and weights for a diameter distribution of coefficient of variation cv.

    Evenly spaced over +/- n_sigma with Gaussian weights. The integrand is
    smooth in diameter, so this converges far faster than random sampling: the
    variance comes from the same nodes instead of needing its own sampling, and
    an n-sample MC standard deviation is biased low for small n.
    """
    if cv <= 0:
        return np.array([d0]), np.array([1.0])
    z = np.linspace(-n_sigma, n_sigma, n_nodes)
    w = np.exp(-0.5 * z ** 2)
    w = w / w.sum()
    d = d0 * (1.0 + cv * z)
    return np.clip(d, 1e-9, None), w


def glycerol_panel(d0, packing_fraction, K_eff_uq, K_eff_q, Ms_vol, Ms_mass,
                   rho_iron, eta_list, t_shell=2e-9, cv=0.10, n_nodes=9,
                   alpha_damp=0.05, weighting='mass', fast=True,
                   progress=False, H_B_range_mT=100.0, amplitude=20.0,
                   frequency=20000.0, **kw):
    """Mean and s.d. of the normalised signal vs viscosity, by quadrature.

    Same output contract as compute_panel_glycerol_uncertainty:
        mean_uq, sd_uq, mean_q, sd_q   (one entry per eta in eta_list)

    WEIGHTING -- why this is mass and not number
    --------------------------------------------
    `scan_peak` returns signal PER UNIT Fe MASS, not an absolute per-particle
    amplitude: M_eq's prefactor is the mass magnetization Ms (Am^2/kg), and the
    particle moment m = V*rho*Ms enters only inside the Langevin argument, where
    it sets the field scale xi. Per-unit-mass signal still rises as d^3 (verified
    d^3.05 at 17.5 nm, d^3.09 at 25 nm) because dM/dH = Ms*L'(xi)*mu0*m/kT
    carries m through dxi/dH -- the standard MPI result.

    So each node's scan_peak is "the mV/ugFe of a MONODISPERSE sample of that
    size". Combining nodes into a polydisperse prediction therefore weights each
    by its share of the sample's IRON, not of the particle count:

        S_ens(eta) = sum_i w_i V_i S_i(eta) / sum_i w_i V_i,     V_i ~ d_i^3

    Number weighting would instead weight a 24 nm particle as much as a 45 nm
    one despite carrying ~1/6 the iron, which is not what the vial contains.
    """
    if weighting != 'mass':
        raise ValueError(f"only weighting='mass' is supported, got {weighting!r}")

    ds, w = size_quadrature(d0, cv, n_nodes)

    # K_eff may be a constant or a function of diameter; callables are
    # evaluated once per node.
    K_uq_of = K_eff_uq if callable(K_eff_uq) else (lambda _d: K_eff_uq)
    K_q_of = K_eff_q if callable(K_eff_q) else (lambda _d: K_eff_q)
    K_uq_nodes = np.array([float(K_uq_of(dd)) for dd in ds])
    K_q_nodes = np.array([float(K_q_of(dd)) for dd in ds])

    # per-unit-Fe-mass signal for a monodisperse sample at each node
    Suq = np.zeros((len(ds), len(eta_list)))
    Sq = np.zeros_like(Suq)
    for i, dd in enumerate(ds):
        V_c = (4 / 3) * np.pi * (dd / 2) ** 3 * packing_fraction
        V_h = (4 / 3) * np.pi * (dd / 2 + t_shell) ** 3
        for j, eta in enumerate(eta_list):
            Suq[i, j] = scan_peak(V_c, V_h, K_uq_nodes[i], Ms_vol, alpha_damp,
                                  eta, Ms_mass, rho_iron, H_B_range_mT,
                                  amplitude, frequency, fast=fast, **kw)
            Sq[i, j] = scan_peak(V_c, V_h, K_q_nodes[i], Ms_vol, alpha_damp,
                                 eta, Ms_mass, rho_iron, H_B_range_mT,
                                 amplitude, frequency, fast=fast, **kw)
        if progress:
            print(f"    node {i+1}/{len(ds)}  d={dd*1e9:.1f} nm  "
                  f"K_uq={K_uq_nodes[i]:.0f}  K_q={K_q_nodes[i]:.0f}", flush=True)

    mass_rel = (ds / d0) ** 3          # Fe mass of one particle, relative to d0

    W = w * mass_rel
    W = W / W.sum()
    # ensemble per-unit-Fe signal, then ONE global normalisation so the
    # unquenched peak is 1 -- the same plot convention as before, but the
    # relative amplitude between nodes now survives.
    Euq, Eq = (W[:, None] * Suq).sum(0), (W[:, None] * Sq).sum(0)
    g = Euq.max()
    g = g if g > 0 else 1.0
    mu, mq = Euq / g, Eq / g
    su = np.sqrt(np.maximum((W[:, None] * (Suq / g - mu) ** 2).sum(0), 0.0))
    sq = np.sqrt(np.maximum((W[:, None] * (Sq / g - mq) ** 2).sum(0), 0.0))

    return dict(mean_uq=mu, sd_uq=su, mean_q=mq, sd_q=sq)
