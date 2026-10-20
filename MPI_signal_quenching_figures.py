#!/usr/bin/env python3
"""Reproduce the simulated figures and source data for the MPI signal-quenching paper.

    python MPI_signal_quenching_figures.py [--outdir OUTDIR]

Writes to OUTDIR (default: the "output" folder next to this script):

  Figures (.png at 300 dpi and .pdf)
    Fig1a_relaxation_vs_diameter   relaxation time vs cluster diameter
    Fig1b_relaxation_vs_field      relaxation time vs applied field (35 nm cluster)
    Fig2ab_psf                     simulated 2-D (and 1-D) PSFs, Cluster and Cluster-Gd

  MPI_signal_quenching_panel_data.xlsx      source data, one sheet per figure panel
    Fig1a_tau_vs_d_unquenched / _quenched   tau_B, tau_N, tau_eff vs diameter, 16/84 bands
    Fig1b_tau_vs_field          tau_B, tau_N, tau_eff vs field, 16/84 bands
    Fig1c_overview              drive field, M(t), dM/dt(t)
    Fig2ab_psf_1D               1-D field-domain PSF
    Fig2c_glycerol              signal vs glycerol content
    manifest                    every setting the numbers depend on + package versions

All physics lives in MPI_signal_quenching.py. This script only sets the
parameters, calls the solver, and writes the outputs.
"""
import argparse
import os
import platform
from importlib import metadata

import matplotlib
matplotlib.use('Agg')                      
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import MPI_signal_quenching as mq
from MPI_signal_quenching import (run_scan, tau_eff, glycerol_panel,
                                  size_quadrature, k_b, u_o)

# ===========================================================================
# Simulation parameters
# ===========================================================================
# --- Physical constants ---
T = mq.T                # K. The scan path (run_scan -> M_eq, tau_B/tau_N eigen
                        # solves) reads the module's T, so there is one source.
assert T == 298.0, f"expected T = 298 K, module has {T}"
rho_magnetite = 5200    # kg/m^3
t_shell = 10e-9         # m, shell thickness (d_h = d_core + 20 nm)
eta_h20 = 1e-3          # Pa*s, water (baseline viscosity for Figs. 1a-c and 2a,b)

# --- Saturation magnetization (clusters) ---
# Ms is passed two ways: mass magnetization in emu/g = Am^2/kg (Ms_Fe_norm, for
# the scan-level equilibrium M_eq) and volumetric magnetization in A/m (Ms_vol,
# for the per-particle Neel/Brownian relaxation). Both derive from one value.
Ms_mn_clusters = 72.0                                # emu/g, measured from ICP-OES
Ms_Fe_norm_clusters = Ms_mn_clusters
Ms_vol_clusters = Ms_mn_clusters * rho_magnetite     # A/m

# --- Damping ---
alpha = 0.05            # Gilbert damping constant

# --- Effective anisotropy, K_eff (J/m^3) ---
# ACS (Cole-Cole) derived, packing fraction 0.5 for the magnetic core volume.
K_eff_uq_clusters = 2801
K_eff_q_clusters = 6300

# Quenched K_eff = unquenched K_eff * the measured cluster ratio. K_eff does not
# vary with cluster diameter.
K_Q_RATIO = K_eff_q_clusters / K_eff_uq_clusters      # 2.2492


def K_eff_at_diameter(particle, d):
    """(K_eff_uq, K_eff_q) for the particle; independent of diameter d."""
    K_uq = particle['K_eff_uq']
    return K_uq, K_uq * K_Q_RATIO


# --- Packing fraction & shared volume functions ---
packing_fraction_clusters = 0.5


def V_c_fn(d, packing_fraction):
    """Packing-fraction-corrected magnetic core volume (m^3) for diameter d (m)."""
    return (4/3) * np.pi * (d/2)**3 * packing_fraction


def V_h_fn(d, t_shell=t_shell):
    """Full hydrodynamic volume (m^3, core + shell) for diameter d (m)."""
    return (4/3) * np.pi * (d/2 + t_shell)**3


# --- Size dispersion: coefficient of variation of the core diameter ---
SIZE_CV = 0.10


def particle_cv(particle, override=None):
    """Diameter c.o.v.: explicit argument, then the particle's own 'cv' key,
    then SIZE_CV. Returns 0.0 for monodisperse."""
    if override is not None:
        return float(override)
    return float(particle.get('cv', SIZE_CV))


# --- Glycerol / viscosity series (the measured sample set) ---
GLYCEROL_PCT = np.array([0, 20, 40, 80])
GLYCEROL_ETA_CP = np.array([0.893, 1.73, 4.05, 66.6])   # measured, 25 C
GLYCEROL_ETA_PAS = GLYCEROL_ETA_CP * 1e-3

# --- Particle: the 35 nm Gd-quenchable cluster simulated in the manuscript ---
CLUSTER_35NM = dict(d=35e-9, packing_fraction=packing_fraction_clusters,
                    K_eff_uq=K_eff_uq_clusters, K_eff_q=K_eff_q_clusters,
                    Ms_vol=Ms_vol_clusters, Ms_Fe_norm=Ms_Fe_norm_clusters)

# --- Scan / drive conditions (constant across every figure) ---
frequency = 20000   # Hz
amplitude = 20      # mT, drive-field amplitude
SCAN_PARAMS = dict(H_B_range_mT=100, amplitude=amplitude, frequency=frequency,
                   n_cycles=4000, n_bias=1000, settle_cycles=5)

# --- Size-distribution quadrature ---
# Deterministic nodes evenly spaced over +/-3 sigma, weighted by the Gaussian
# pdf, then re-weighted by Fe mass (~d^3) when the ensemble signal is formed.
POLYDISPERSE_NODES = 7

# Polydisperse glycerol bars: quadrature runner and Fe-mass weighting.
GLYCEROL_RUNNER = 'fast'
GLYCEROL_WEIGHTING = 'mass'

# Solver preset for the PSF (Fig. 2a,b; 'reference': auto N, 800 LUT points).
PSF_PRESET = 'reference'


def normal_quantile(p):
    """Inverse standard-normal CDF, by bisection on math.erf."""
    import math
    p = float(p)
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0,1), got {p}")
    lo, hi = -8.0, 8.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 0.5 * (1.0 + math.erf(mid / math.sqrt(2.0))) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ===========================================================================
# Diameter and axis ranges
# ===========================================================================
SIGMA_PLOT_CAP = 25.0      # Reduced Barrier Cap
D_CEILING_NM = 50.0
D_SWEEP_COMMON = (10.0, D_CEILING_NM)


def sigma_ceiling_nm(particle, state, cap=25.0):
    """Largest core diameter [nm] at which sigma = K_eff*V_c/(kB*T) <= cap.

    Closed form because K_eff carries no diameter dependence for clusters, so
    sigma goes exactly as d^3.
    """
    d0 = particle['d']
    K0 = float(K_eff_at_diameter(particle, d0)[0 if state == 'uq' else 1])
    s0 = K0 * V_c_fn(d0, particle['packing_fraction']) / (k_b * T)
    return float(d0 * 1e9 * (cap / s0) ** (1.0 / 3.0))


# Each quench state runs to its own sigma = 25 ceiling (quenched stops first).
D_SWEEP_CLUSTER_UQ = (D_SWEEP_COMMON[0],
                      min(sigma_ceiling_nm(CLUSTER_35NM, 'uq', SIGMA_PLOT_CAP),
                          D_CEILING_NM))
D_SWEEP_CLUSTER_Q = (D_SWEEP_COMMON[0],
                     min(sigma_ceiling_nm(CLUSTER_35NM, 'q', SIGMA_PLOT_CAP),
                         D_CEILING_NM))

D_AXIS_NM = D_SWEEP_COMMON                      # shared diameter axis
B_AXIS_MT = (0.0, 50)                           # shared applied-field axis
TAU_AXIS_C_S = (1e-7, 50e-3)                    # Fig. 1a  (vs diameter)
TAU_AXIS_J_S = (1e-10, 1e-3)                    # Fig. 1b  (vs field)
x_lo, x_hi = D_AXIS_NM


def particle_d_sweep(override=None):
    """(lo, hi) diameter sweep in nm: the explicit range if given, else the
    shared (x_lo, x_hi)."""
    if override is not None:
        return (float(override[0]), float(override[1]))
    return (float(x_lo), float(x_hi))


# ===========================================================================
# Plot style
# ===========================================================================
DEFAULT_STYLE = dict(
    font_family='sans-serif',
    font_sans_serif=['Arial', 'Helvetica', 'DejaVu Sans'],
    font_size=11,
    axis_title_fontsize=16,
    axis_tick_fontsize=14,
    axes_linewidth=1.25,
    tick_width=0.9,
    line_width=1.8,
    color_glycerol_cluster='tab:blue',      # unquenched
    color_glycerol_cluster_gd='#8a8a1a',    # quenched
    color_image_ticks='0.85',
    cmap_psf='hot',
    save_dpi=300,
)

RELAX_COLOURS = {
    'cluster_uq': '#1F77B4',        # blue
    'cluster_q': '#8A8A1A',         # olive
    'tau_B_cluster': 'black',
}


def apply_global_rcparams():
    s = DEFAULT_STYLE
    plt.rcParams.update({
        'font.family': s['font_family'],
        'font.sans-serif': s['font_sans_serif'],
        'font.size': s['font_size'],
        'axes.linewidth': s['axes_linewidth'],
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.width': s['tick_width'],
        'ytick.major.width': s['tick_width'],
        'savefig.dpi': s['save_dpi'],
    })


def psf_fwhm(H, y):
    """FWHM and peak position of a 1-D PSF, by linear interpolation at half max.

    Returns (fwhm, H_peak, peak_value). Assumes H is ascending and the peak is
    interior; returns nan for the width if the curve does not fall to half max
    on both sides inside the window.
    """
    y = np.abs(np.asarray(y, dtype=float))
    H = np.asarray(H, dtype=float)
    if not np.any(np.isfinite(y)):
        return np.nan, np.nan, np.nan
    i = int(np.nanargmax(y))
    half = y[i] / 2.0
    if i == 0 or i == len(y) - 1 or y[0] > half or y[-1] > half:
        return np.nan, float(H[i]), float(y[i])
    left = np.interp(half, y[:i + 1], H[:i + 1])
    right = np.interp(half, y[i:][::-1], H[i:][::-1])
    return float(right - left), float(H[i]), float(y[i])


# ===========================================================================
# Fig. 1c -- overview: drive, M(t), dM/dt(t)  (data only; in the workbook)
# ===========================================================================
def compute_panel_overview_data(particle=CLUSTER_35NM):
    """Illustration-scan magnetization traces, quenched vs unquenched (monodisperse)."""
    V_c = V_c_fn(particle['d'], particle['packing_fraction'])
    V_h = V_h_fn(particle['d'])

    quenched_particle = dict(V_c=V_c, V_h=V_h, K_eff=particle['K_eff_q'],
                             Ms_vol=particle['Ms_vol'], alpha_damp=alpha, eta=eta_h20)
    unquenched_particle = dict(V_c=V_c, V_h=V_h, K_eff=particle['K_eff_uq'],
                               Ms_vol=particle['Ms_vol'], alpha_damp=alpha, eta=eta_h20)

    illustration_scan = dict(H_B_range_mT=0, amplitude=amplitude, frequency=frequency,
                             n_cycles=20, n_bias=400, settle_cycles=2,
                             V_c=V_c, V_h=V_h, rho_iron=rho_magnetite, Ms=particle['Ms_Fe_norm'])

    res_q_cl = run_scan(quenched_particle, illustration_scan)
    res_uq_cl = run_scan(unquenched_particle, illustration_scan)

    H_t_cl = res_q_cl['Tx'] + res_q_cl['Bx']
    M_q_cl, dMdt_q_cl = res_q_cl['Rx']['M'], res_q_cl['Rx']['dMdt']
    M_uq_cl, dMdt_uq_cl = res_uq_cl['Rx']['M'], res_uq_cl['Rx']['dMdt']

    dt = 1/(illustration_scan['n_bias']*illustration_scan['frequency'])
    t_us = np.arange(len(H_t_cl))*dt*1e6

    M_scale = max(np.max(np.abs(M_q_cl)), np.max(np.abs(M_uq_cl)))
    M_q_norm, M_uq_norm = M_q_cl/M_scale, M_uq_cl/M_scale

    dMdt_scale = max(np.max(np.abs(dMdt_q_cl)), np.max(np.abs(dMdt_uq_cl)))
    dMdt_q_norm, dMdt_uq_norm = dMdt_q_cl/dMdt_scale, dMdt_uq_cl/dMdt_scale

    return dict(H_t_cl=H_t_cl, M_uq_norm=M_uq_norm, M_q_norm=M_q_norm,
                t_us=t_us, dMdt_uq_norm=dMdt_uq_norm, dMdt_q_norm=dMdt_q_norm)


# ===========================================================================
# Fig. 2a,b -- simulated PSFs
# ===========================================================================
def compute_panel_psf_data(particle=CLUSTER_35NM, cv=None,
                           n_nodes=None, preset=None):
    """1-D field-domain PSFs + 2-D spatial PSFs.

    SIZE DISPERSION (cv, default SIZE_CV). The 1-D PSF is the Fe-mass-weighted
    SUM of the per-size PSF curves,

        p_ens(H) = sum_k w_k p_k(H),     w_k = m_k / sum_l m_l,  m_k ~ d_k^3

    Set cv=0 for the monodisperse curve.

    2-D RENDERING: an isotropic Gaussian whose FWHM matches the 1-D PSF's, per
    state, amplitude set by the 1-D peak.
    """
    cv = particle_cv(particle, cv)
    n_nodes = n_nodes or POLYDISPERSE_NODES
    preset = preset or PSF_PRESET

    if cv and cv > 0:
        ds, w_num = size_quadrature(particle['d'], cv, n_nodes)
    else:
        ds, w_num = np.array([particle['d']]), np.array([1.0])
    mass_rel = (ds / particle['d']) ** 3          # Fe mass of one particle
    W = w_num * mass_rel
    W = W / W.sum()
    print(f"Fig. 2a,b: preset={preset}  "
          f"cv={cv}  nodes={len(ds)}")

    H_grid = None
    psf_uq = psf_q = None
    for dk, wk in zip(ds, W):
        V_c_k = V_c_fn(dk, particle['packing_fraction'])
        V_h_k = V_h_fn(dk)
        base_k = dict(V_c=V_c_k, V_h=V_h_k, Ms_vol=particle['Ms_vol'],
                      alpha_damp=alpha, eta=eta_h20)
        # per-NODE K_eff (constant for cluster presets)
        K_uq_k, K_q_k = K_eff_at_diameter(particle, dk)
        psf_scan = dict(SCAN_PARAMS, V_c=V_c_k, V_h=V_h_k, rho_iron=rho_magnetite,
                        Ms=particle['Ms_Fe_norm'])
        res_uq_psf = run_scan(dict(base_k, K_eff=K_uq_k),
                              psf_scan, preset=preset)
        res_q_psf = run_scan(dict(base_k, K_eff=K_q_k),
                             psf_scan, preset=preset)
        if H_grid is None:
            H_grid = np.asarray(res_uq_psf['H_grid'], dtype=float)
            psf_uq = np.zeros_like(H_grid)
            psf_q = np.zeros_like(H_grid)
        psf_uq = psf_uq + wk * np.nan_to_num(np.asarray(res_uq_psf['psf_pos'], dtype=float))
        psf_q = psf_q + wk * np.nan_to_num(np.asarray(res_q_psf['psf_pos'], dtype=float))

    H_LIM_MT = 40.0
    keep = np.abs(H_grid) <= H_LIM_MT
    H_grid = H_grid[keep]
    psf_uq = psf_uq[keep]
    psf_q = psf_q[keep]

    peak = max(np.abs(psf_uq).max(), np.abs(psf_q).max())
    psf_uq = psf_uq / peak
    psf_q = psf_q / peak

    Gx = 5.7      # T/m
    Gy = 5.7      # T/m

    x_max_mm = H_LIM_MT / Gx/2
    y_max_mm = x_max_mm

    n_pix = 300
    x_mm = np.linspace(-x_max_mm, x_max_mm, n_pix)
    y_mm = np.linspace(-y_max_mm, y_max_mm, n_pix)
    X_mm, Y_mm = np.meshgrid(x_mm, y_mm)

    fwhm_uq, Hpk_uq, amp_uq = psf_fwhm(H_grid, psf_uq)
    fwhm_q, Hpk_q, amp_q = psf_fwhm(H_grid, psf_q)

    # Isotropic Gaussian of matching FWHM, centred at the origin. Widths are
    # converted mT -> mm through the gradient.
    k_fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0))       # FWHM = k * sigma

    def _gauss2d(fwhm_mT, amp):
        if not np.isfinite(fwhm_mT) or fwhm_mT <= 0:
            return np.full_like(X_mm, np.nan)
        sx = (fwhm_mT / Gx) / k_fwhm
        sy = (fwhm_mT / Gy) / k_fwhm
        return amp * np.exp(-0.5 * ((X_mm / sx) ** 2 + (Y_mm / sy) ** 2))

    PSF2D_uq = _gauss2d(fwhm_uq, amp_uq)
    PSF2D_q = _gauss2d(fwhm_q, amp_q)

    print(f"  FWHM: unquenched {fwhm_uq:.3f} mT ({fwhm_uq/Gx:.3f} mm), "
          f"quenched {fwhm_q:.3f} mT ({fwhm_q/Gx:.3f} mm); "
          f"H_peak {Hpk_uq:.2f} / {Hpk_q:.2f} mT")

    return dict(H_grid=H_grid, psf_uq=psf_uq, psf_q=psf_q,
                PSF2D_uq=PSF2D_uq, PSF2D_q=PSF2D_q,
                x_max_mm=x_max_mm, y_max_mm=y_max_mm, H_LIM_MT=H_LIM_MT)


def plot_panel_psf(fig, gs_block, data):
    """Fig. 2a,b: 2-D spatial PSFs (top) + 1-D field-domain PSFs (bottom),
    unquenched (left) vs quenched (right)."""
    s = DEFAULT_STYLE
    inner_gs = gs_block.subgridspec(2, 2, hspace=0.3, wspace=0.3)

    H_LIM_MT = data['H_LIM_MT']
    extent = [-data['x_max_mm'], data['x_max_mm'], -data['y_max_mm'], data['y_max_mm']]
    vmax = 1.0

    ax_2d_uq = fig.add_subplot(inner_gs[0, 0])
    ax_2d_uq.imshow(data['PSF2D_uq'], extent=extent, origin='lower', cmap=s['cmap_psf'],
                    vmin=0, vmax=vmax, aspect='equal')
    ax_2d_uq.set_xlabel('x (mm)', fontsize=s['axis_title_fontsize'])
    ax_2d_uq.set_ylabel('y (mm)', fontsize=s['axis_title_fontsize'])
    ax_2d_uq.tick_params(color=s['color_image_ticks'], which='both', labelsize=s['axis_tick_fontsize'])

    ax_2d_q = fig.add_subplot(inner_gs[0, 1])
    im_q = ax_2d_q.imshow(data['PSF2D_q'], extent=extent, origin='lower', cmap=s['cmap_psf'],
                          vmin=0, vmax=vmax, aspect='equal')
    ax_2d_q.set_xlabel('x (mm)', fontsize=s['axis_title_fontsize'])
    ax_2d_q.set_ylabel('y (mm)', fontsize=s['axis_title_fontsize'])
    ax_2d_q.tick_params(color=s['color_image_ticks'], which='both', labelsize=s['axis_tick_fontsize'])

    cbar = fig.colorbar(im_q, ax=[ax_2d_uq, ax_2d_q], shrink=0.8, aspect=25, pad=0.02,
                        label='Signal (a.u.)', ticks=[0, 0.25, 0.5, 0.75, 1.0])
    cbar.outline.set_linewidth(s['axes_linewidth'])
    cbar.ax.tick_params(width=s['tick_width'], labelsize=s['axis_tick_fontsize'])
    cbar.set_label('Signal (a.u.)', fontsize=s['axis_title_fontsize'])

    ax_1d_uq = fig.add_subplot(inner_gs[1, 0])
    ax_1d_uq.plot(data['H_grid'], data['psf_uq'], color=s['color_glycerol_cluster'], lw=s['line_width'])
    ax_1d_uq.set_xlabel('H (mT)', fontsize=s['axis_title_fontsize'])
    ax_1d_uq.set_ylabel('dM/dt (a.u.)', fontsize=s['axis_title_fontsize'])
    ax_1d_uq.set_xlim(-H_LIM_MT, H_LIM_MT)
    ax_1d_uq.set_ylim(0, 1.1)
    ax_1d_uq.set_yticks([0, 0.5, 1.0])
    ax_1d_uq.tick_params(labelsize=s['axis_tick_fontsize'], width=s['tick_width'])

    ax_1d_q = fig.add_subplot(inner_gs[1, 1])
    ax_1d_q.plot(data['H_grid'], data['psf_q'], color=s['color_glycerol_cluster_gd'], lw=s['line_width'])
    ax_1d_q.set_xlabel('H (mT)', fontsize=s['axis_title_fontsize'])
    ax_1d_q.set_xlim(-H_LIM_MT, H_LIM_MT)
    ax_1d_q.set_ylim(0, 1.1)
    ax_1d_q.set_yticks([0, 0.5, 1.0])
    ax_1d_q.tick_params(labelsize=s['axis_tick_fontsize'], width=s['tick_width'])

    return dict(ax_2d_uq=ax_2d_uq, ax_2d_q=ax_2d_q, ax_1d_uq=ax_1d_uq, ax_1d_q=ax_1d_q, cbar=cbar)


# ===========================================================================
# Fig. 1a -- zero-field relaxation time vs cluster diameter
# ===========================================================================
def compute_panel_relaxation_data(particle=CLUSTER_35NM, d_range_nm=None):
    """tau_B, tau_N, tau_eff vs core diameter at H = 0 for the unquenched and
    quenched K_eff cases in `particle`. `particle` supplies K_eff_uq / K_eff_q /
    Ms_vol / packing_fraction; its own `d` is not used here -- the sweep range
    comes from d_range_nm (default: the shared diameter axis)."""
    Ms_vol = particle['Ms_vol']
    packing_fraction = particle['packing_fraction']

    _d_lo, _d_hi = particle_d_sweep(d_range_nm)
    d_core_nm = np.linspace(_d_lo, _d_hi, 100)
    d_core_m = d_core_nm * 1e-9

    # K_eff is constant for a cluster (its crystallites do not grow with the envelope)
    K_uq_vs_d = np.broadcast_to(K_eff_at_diameter(particle, d_core_m)[0],
                                d_core_m.shape)
    K_q_vs_d = np.broadcast_to(K_eff_at_diameter(particle, d_core_m)[1],
                               d_core_m.shape)

    V_core_vals = V_c_fn(d_core_m, packing_fraction)
    V_h_vals = V_h_fn(d_core_m, t_shell)

    def sweep_diameter(K_eff_relax):
        """tau_B, tau_N, tau_eff vs diameter. K_eff may be a scalar or one value
        per diameter."""
        K_arr = np.broadcast_to(np.asarray(K_eff_relax, float), d_core_m.shape)
        tau_B_list, tau_N_list, tau_eff_list = [], [], []
        for Vc, Vh, K_i in zip(V_core_vals, V_h_vals, K_arr):
            particle_i = dict(V_c=Vc, V_h=Vh, K_eff=K_i, Ms_vol=Ms_vol,
                              alpha_damp=alpha, eta=eta_h20)
            combined, tB, tN = tau_eff(0.0, **particle_i, T=T)
            tau_B_list.append(tB)
            tau_N_list.append(tN)
            tau_eff_list.append(combined)
        return np.array(tau_B_list), np.array(tau_N_list), np.array(tau_eff_list)

    tau_B_uq, tau_N_uq, tau_eff_uq = sweep_diameter(K_uq_vs_d)
    tau_B_q, tau_N_q, tau_eff_q = sweep_diameter(K_q_vs_d)

    return dict(d_core_nm=d_core_nm,
                tau_B_uq=tau_B_uq, tau_N_uq=tau_N_uq, tau_eff_uq=tau_eff_uq,
                tau_B_q=tau_B_q, tau_N_q=tau_N_q, tau_eff_q=tau_eff_q,
                K_eff_uq_vs_d=np.asarray(K_uq_vs_d).copy(),
                K_eff_q_vs_d=np.asarray(K_q_vs_d).copy())


def compute_panel_relaxation_uncertainty(particle=CLUSTER_35NM, sigma_frac=None,
                                         percentiles=(16, 84), d_range_nm=None):
    """Exact percentile bands on tau_B, tau_N and tau_eff vs core diameter,
    propagating a Gaussian core-DIAMETER distribution (sigma_frac defaults to
    SIZE_CV; unrelated to the barrier height sigma, despite the shared name).

    tau_B, tau_N and tau_eff are all strictly INCREASING in diameter, and for a
    strictly monotonic map the p-th percentile of tau is simply tau at the p-th
    percentile of d. So the band needs exactly three evaluations per diameter --
    the mean and the two quantile edges -- instead of a Monte Carlo sample.
    Monotonicity is ASSERTED over the plotted range, since the shortcut fails
    silently otherwise.
    """
    sigma_frac = particle_cv(particle, sigma_frac)

    Ms_vol = particle['Ms_vol']
    packing_fraction = particle['packing_fraction']

    _d_lo, _d_hi = particle_d_sweep(d_range_nm)
    d_core_nm = np.linspace(_d_lo, _d_hi, 100)

    def tau_at(d_nm, state):
        """tau at diameter d_nm in quench `state` ('uq'|'q'); K_eff re-evaluated
        AT THAT DIAMETER."""
        d_m = d_nm * 1e-9
        K_eff_relax = K_eff_at_diameter(particle, d_m)[0 if state == 'uq' else 1]
        Vc = V_c_fn(d_m, packing_fraction)
        Vh = V_h_fn(d_m, t_shell)
        particle_i = dict(V_c=Vc, V_h=Vh, K_eff=K_eff_relax, Ms_vol=Ms_vol,
                          alpha_damp=alpha, eta=eta_h20)
        combined, tB, tN = tau_eff(0.0, **particle_i, T=T)
        return tB, tN, combined

    def exact_band(state):
        """Exact percentile band: tau evaluated at the diameter quantiles."""
        z_lo, z_hi = (normal_quantile(p / 100.0) for p in percentiles)
        out = []
        for z in (z_lo, z_hi):
            d_edge = np.clip(d_core_nm * (1.0 + sigma_frac * z), 0.5, None)
            vals = np.array([tau_at(dd, state) for dd in d_edge])
            out.append(vals)              # (100, 3) -> tB, tN, eff
        lo, hi = out
        return (lo[:, 0], hi[:, 0], lo[:, 1], hi[:, 1], lo[:, 2], hi[:, 2])

    # The monotonicity the shortcut relies on, checked on a coarse grid for both states.
    for st_chk in ('uq', 'q'):
        probe = np.array([tau_at(dd, st_chk) for dd in d_core_nm[::10]])
        for j, nm in enumerate(('tau_B', 'tau_N', 'tau_eff')):
            if not np.all(np.diff(probe[:, j]) > 0):
                raise AssertionError(
                    f"{nm} is not monotonically increasing in diameter in "
                    f"the {st_chk} state; the exact percentile band is invalid here.")

    tB_lo_uq, tB_hi_uq, tN_lo_uq, tN_hi_uq, eff_lo_uq, eff_hi_uq = exact_band('uq')
    tB_lo_q, tB_hi_q, tN_lo_q, tN_hi_q, eff_lo_q, eff_hi_q = exact_band('q')

    return dict(d_core_nm=d_core_nm,
                tau_B_lo_uq=tB_lo_uq, tau_B_hi_uq=tB_hi_uq,
                tau_N_lo_uq=tN_lo_uq, tau_N_hi_uq=tN_hi_uq,
                tau_eff_lo_uq=eff_lo_uq, tau_eff_hi_uq=eff_hi_uq,
                tau_B_lo_q=tB_lo_q, tau_B_hi_q=tB_hi_q,
                tau_N_lo_q=tN_lo_q, tau_N_hi_q=tN_hi_q,
                tau_eff_lo_q=eff_lo_q, tau_eff_hi_q=eff_hi_q,
                sigma_frac=sigma_frac, percentiles=percentiles, band='exact')


def plot_panel_relaxation_combined(fig, gs_block, entries):
    """Fig. 1a -- zero-field tau vs core diameter on ONE axes.

    `entries` is (label, data, uncertainty, colour_key, states, draw_tau_B).
    Colour = quench state, line style = quantity (tau_B dashed, tau_N dotted,
    tau_eff solid), tau_B drawn once in black. Bands are the exact 16/84 band
    from the diameter c.o.v. Seconds on a log axis; tau_f = 1/(2f) marked.
    """
    s = DEFAULT_STYLE
    ax = fig.add_subplot(gs_block)
    for entry in entries:
        lbl, rel, unc, ckey = entry[:4]
        _states = entry[4] if len(entry) > 4 else ('uq', 'q')
        _draw_B = entry[5] if len(entry) > 5 else True
        d = np.asarray(rel['d_core_nm'], dtype=float)
        if _draw_B:
            ax.plot(d, rel['tau_B_uq'], ls='--', color='k', lw=1.6, zorder=2)
            ax.fill_between(d, unc['tau_B_lo_uq'], unc['tau_B_hi_uq'],
                            color='k', alpha=0.12, lw=0, zorder=1)
        for st in ('uq', 'q'):
            if st not in _states:
                continue
            col = RELAX_COLOURS[f'{ckey}_{st}']
            for key, ls, z, lw in (('tau_eff', '-', 3, 1.8),
                                   ('tau_N', ':', 4, 2.2)):
                ax.plot(d, rel[f'{key}_{st}'], ls=ls, color=col, lw=lw, zorder=z)
                # the upper edge leaves the frame before it becomes
                # unavailable, so a large finite stand-in lets the band run
                # off the top continuously
                _hi = np.nan_to_num(np.asarray(unc[f'{key}_hi_{st}'], float),
                                    nan=TAU_AXIS_C_S[1] * 1e3,
                                    posinf=TAU_AXIS_C_S[1] * 1e3)
                ax.fill_between(d, unc[f'{key}_lo_{st}'], _hi,
                                color=col, alpha=0.15, lw=0, zorder=z - 1)

    tau_f = 1.0 / (2.0 * frequency)
    ax.axhline(tau_f, color='0.45', ls=(0, (4, 3)), lw=1.1, zorder=1)
    ax.annotate(r'$\tau_f = 1/2f$', xy=(D_AXIS_NM[0], tau_f),
                xytext=(4, 4), textcoords='offset points', ha='left',
                va='bottom', fontsize=16, color='0.35')

    ax.set_yscale('log')
    ax.set_xlim(*D_AXIS_NM)
    ax.set_ylim(*TAU_AXIS_C_S)
    ax.set_xlabel('Cluster Diameter (nm)', fontsize=s['axis_title_fontsize'])
    ax.set_ylabel('Relaxation Time (s)', fontsize=s['axis_title_fontsize'])
    ax.grid(alpha=0.2, which='both')
    ax.tick_params(labelsize=s['axis_tick_fontsize'], width=s['tick_width'])
    for sp in ax.spines.values():
        sp.set_linewidth(s['axes_linewidth'])
    return ax


# ===========================================================================
# Fig. 1b -- relaxation time vs applied field, 35 nm cluster
# ===========================================================================
def compute_panel_relaxation_vs_field(particle=CLUSTER_35NM,
                                      B_max_mT=50, n_points=121, eta=None,
                                      cv=None):
    """tau_B, tau_N, tau_eff vs applied field at the particle's own diameter,
    with exact 16/84 bands (tau evaluated at the diameter quantiles)."""
    eta_use = eta_h20 if eta is None else eta
    d_m = particle['d']
    Ms_vol = particle['Ms_vol']

    # LINEAR field axis over the MPI-relevant range
    B_mT = np.linspace(0.0, B_max_mT, n_points)
    H_Am = B_mT * 1e-3 / u_o

    cv = particle_cv(particle, cv)
    z_lo, z_hi = (normal_quantile(p / 100.0) for p in (16.0, 84.0))
    d_edges = [d_m * (1.0 + cv * z_lo), d_m, d_m * (1.0 + cv * z_hi)]

    out = {'B_mT': B_mT}
    for si, st in ((0, 'uq'), (1, 'q')):
        # index 0 = lower edge, 1 = mean, 2 = upper edge
        tB = np.empty((3, B_mT.size)); tN = np.empty_like(tB); te = np.empty_like(tB)
        for e, d_e in enumerate(d_edges):
            V_c_e = V_c_fn(d_e, particle['packing_fraction'])
            V_h_e = V_h_fn(d_e, t_shell)
            # K_eff at the EDGE diameter, not the nominal one
            K = K_eff_at_diameter(particle, d_e)[si]
            for j, H in enumerate(H_Am):
                combined, b, n = tau_eff(H, V_c_e, V_h_e, K, Ms_vol, alpha,
                                         eta_use, T=T)
                tB[e, j], tN[e, j], te[e, j] = b, n, combined
        out[f'tau_B_{st}'] = tB[1];   out[f'tau_B_{st}_lo'] = tB[0];   out[f'tau_B_{st}_hi'] = tB[2]
        out[f'tau_N_{st}'] = tN[1];   out[f'tau_N_{st}_lo'] = tN[0];   out[f'tau_N_{st}_hi'] = tN[2]
        out[f'tau_eff_{st}'] = te[1]; out[f'tau_eff_{st}_lo'] = te[0]; out[f'tau_eff_{st}_hi'] = te[2]
    return out


def h_unity_field_mT(particle, state):
    """Field (mT) at which the Stoner-Wohlfarth barrier collapses (h = 1):
    B = 2*K_eff/Ms_vol. Reported, not drawn."""
    K = float(K_eff_at_diameter(particle, particle['d'])[0 if state == 'uq' else 1])
    return 2.0 * K / particle['Ms_vol'] * 1e3


def plot_panel_relaxation_vs_field(fig, gs_block, label, data, colour_key,
                                   tauB_colour_key):
    """Fig. 1b -- tau vs applied field for ONE sample, both quench states.
    tau_B is drawn once (it carries no K_eff dependence at fixed diameter)."""
    s = DEFAULT_STYLE
    ax = fig.add_subplot(gs_block)
    B = np.asarray(data['B_mT'], dtype=float)
    for st in ('uq', 'q'):
        col = RELAX_COLOURS[f'{colour_key}_{st}']
        for key, ls, z, lw in (('tau_B', '--', 2, 1.6),
                               ('tau_eff', '-', 3, 1.8),
                               ('tau_N', ':', 4, 2.2)):
            if key == 'tau_B':
                if st == 'q':
                    continue                       # state-independent
                c_use = RELAX_COLOURS[tauB_colour_key]
            else:
                c_use = col
            ax.plot(B, data[f'{key}_{st}'], ls=ls, color=c_use, lw=lw, zorder=z)
            ax.fill_between(B, data[f'{key}_{st}_lo'], data[f'{key}_{st}_hi'],
                            color=c_use, alpha=0.16, lw=0, zorder=z - 1)

    tau_f = 1.0 / (2.0 * frequency)
    ax.axhline(tau_f, color='0.45', ls=(0, (4, 3)), lw=1.1, zorder=1)
    ax.annotate(r'$\tau_f = 1/2f$', xy=(B_AXIS_MT[1], tau_f),
                xytext=(-4, 4), textcoords='offset points', ha='right',
                va='bottom', fontsize=16, color='0.35')

    ax.set_yscale('log')
    ax.set_xlim(*B_AXIS_MT)
    ax.set_ylim(*TAU_AXIS_J_S)
    ax.set_xlabel('Applied Field (mT)', fontsize=s['axis_title_fontsize'])
    ax.set_ylabel('Relaxation Time (s)', fontsize=s['axis_title_fontsize'])
    ax.grid(alpha=0.2, which='both')
    ax.tick_params(labelsize=s['axis_tick_fontsize'], width=s['tick_width'])
    for sp in ax.spines.values():
        sp.set_linewidth(s['axes_linewidth'])
    ax.set_title(label, fontsize=s['axis_title_fontsize'],
                 loc='left')
    return ax


# ===========================================================================
# Fig. 2c -- signal vs glycerol content (data only; in the workbook)
# ===========================================================================
def compute_panel_glycerol_data(particle=CLUSTER_35NM):
    """Simulated MPI signal vs glycerol content, quenched/unquenched,
    monodisperse (deterministic, no size distribution)."""
    V_c_particle = V_c_fn(particle['d'], particle['packing_fraction'])
    V_h_particle = V_h_fn(particle['d'])

    glycerol_pct = GLYCEROL_PCT
    eta_Pas = GLYCEROL_ETA_PAS

    signal_max_uq = np.zeros_like(eta_Pas)   # unquenched
    signal_max_q = np.zeros_like(eta_Pas)    # quenched

    for i, eta_i in enumerate(eta_Pas):
        unquenched_particle = dict(V_c=V_c_particle, V_h=V_h_particle, K_eff=particle['K_eff_uq'],
                                   Ms_vol=particle['Ms_vol'], alpha_damp=alpha, eta=eta_i)
        quenched_particle = dict(V_c=V_c_particle, V_h=V_h_particle, K_eff=particle['K_eff_q'],
                                 Ms_vol=particle['Ms_vol'], alpha_damp=alpha, eta=eta_i)

        scan = dict(SCAN_PARAMS, V_c=V_c_particle, V_h=V_h_particle,
                    rho_iron=rho_magnetite, Ms=particle['Ms_Fe_norm'])

        res_uq = run_scan(unquenched_particle, scan)
        res_q = run_scan(quenched_particle, scan)

        signal_max_uq[i] = np.nanmax(np.abs(res_uq['psf_pos']))
        signal_max_q[i] = np.nanmax(np.abs(res_q['psf_pos']))

    return dict(glycerol_pct=glycerol_pct, signal_max_uq=signal_max_uq,
                signal_max_q=signal_max_q)


def compute_panel_glycerol_uncertainty(particle=CLUSTER_35NM, sigma_frac=None,
                                       n_nodes=None, weighting=None):
    """Mean and s.d. of the normalised signal vs glycerol content, propagating a
    Gaussian diameter distribution (sigma = sigma_frac * particle['d'], default
    SIZE_CV) by deterministic quadrature. Fe-mass weighted. This is the series
    plotted as bars with error bars in Fig. 2c."""
    sigma_frac = particle_cv(particle, sigma_frac)
    n_nodes = n_nodes or POLYDISPERSE_NODES
    weighting = weighting or GLYCEROL_WEIGHTING
    eta_Pas = GLYCEROL_ETA_PAS
    # K_eff as a FUNCTION of the node diameter (constant for clusters)
    P = glycerol_panel(
        particle['d'], particle['packing_fraction'],
        lambda dd: K_eff_at_diameter(particle, dd)[0],
        lambda dd: K_eff_at_diameter(particle, dd)[1],
        particle['Ms_vol'], particle['Ms_Fe_norm'], rho_magnetite,
        list(eta_Pas), t_shell=t_shell, cv=sigma_frac, n_nodes=n_nodes,
        alpha_damp=alpha, fast=True, weighting=weighting,
        H_B_range_mT=SCAN_PARAMS['H_B_range_mT'],
        amplitude=SCAN_PARAMS['amplitude'], frequency=SCAN_PARAMS['frequency'])
    print(f"  glycerol uncertainty: runner={GLYCEROL_RUNNER!r}, {n_nodes} quadrature nodes, "
          f"cv={sigma_frac}, weighting={weighting!r}")
    return dict(mean_uq=P['mean_uq'], sd_uq=P['sd_uq'],
                mean_q=P['mean_q'], sd_q=P['sd_q'])


# ===========================================================================
# Output helpers
# ===========================================================================
def savefig_panel(outdir, stem, dpi=300):
    """Save the current figure as .pdf + .png and report the paths."""
    pdf = os.path.join(outdir, f"{stem}.pdf")
    png = os.path.join(outdir, f"{stem}.png")
    plt.savefig(pdf, bbox_inches='tight')
    plt.savefig(png, dpi=dpi, bbox_inches='tight')
    plt.close(plt.gcf())
    print(f"  saved: {pdf}  |  {png}")


def package_versions():
    """(name, version) for the interpreter and every package the run imports."""
    rows = [('python', platform.python_version()),
            ('platform', platform.platform())]
    for pkg in ('numpy', 'scipy', 'matplotlib', 'pandas', 'openpyxl'):
        try:
            rows.append((pkg, metadata.version(pkg)))
        except metadata.PackageNotFoundError:
            rows.append((pkg, 'not installed'))
    return rows


def panel_settings_id():
    """Everything that changes the numbers."""
    return dict(
        T=T, alpha=alpha, t_shell=t_shell, rho_magnetite=rho_magnetite,
        eta_h20=eta_h20, frequency=frequency, amplitude=amplitude,
        scan=dict(SCAN_PARAMS),
        size_cv=SIZE_CV, nodes=POLYDISPERSE_NODES,
        glycerol_runner=GLYCEROL_RUNNER, glycerol_weighting=GLYCEROL_WEIGHTING,
        K_eff_uq=K_eff_uq_clusters, K_eff_q=K_eff_q_clusters,
    )


# ===========================================================================
# Main
# ===========================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--outdir',
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output'),
                    help='directory for figures and the workbook '
                         '(default: "output" next to this script)')
    args = ap.parse_args(argv)
    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)
    print(f'Writing outputs to {outdir}')
    apply_global_rcparams()

    print(f"Sweeps: unquenched {D_SWEEP_CLUSTER_UQ[0]:.1f}-{D_SWEEP_CLUSTER_UQ[1]:.1f} nm "
          f"(sigma = {SIGMA_PLOT_CAP:g} at {sigma_ceiling_nm(CLUSTER_35NM, 'uq', SIGMA_PLOT_CAP):.1f} nm), "
          f"quenched {D_SWEEP_CLUSTER_Q[0]:.1f}-{D_SWEEP_CLUSTER_Q[1]:.1f} nm "
          f"(sigma = {SIGMA_PLOT_CAP:g} at {sigma_ceiling_nm(CLUSTER_35NM, 'q', SIGMA_PLOT_CAP):.1f} nm)")

    # ---- Fig. 1c (workbook only) ----
    print('\n[Fig. 1c] overview')
    overview_data = compute_panel_overview_data()

    # ---- Fig. 2a,b: PSFs (figure + workbook) ----
    print('\n[Fig. 2a,b] PSF')
    psf_data = compute_panel_psf_data()
    fig = plt.figure(figsize=(8, 7), layout='constrained')
    gs = fig.add_gridspec(1, 1)
    plot_panel_psf(fig, gs[0, 0], psf_data)
    savefig_panel(outdir, 'Fig2ab_psf')

    # ---- Fig. 1a: tau vs diameter (figure + workbook). Each quench state runs to
    #      its own sigma = 25 ceiling, so each has its own diameter grid. ----
    print('\n[Fig. 1a] relaxation time vs diameter')
    relaxation = {}
    for name, rng in (('uq', D_SWEEP_CLUSTER_UQ), ('q', D_SWEEP_CLUSTER_Q)):
        relaxation[name] = (
            compute_panel_relaxation_data(particle=CLUSTER_35NM, d_range_nm=rng),
            compute_panel_relaxation_uncertainty(particle=CLUSTER_35NM, d_range_nm=rng))
    (relaxation_data, uncertainty_data) = relaxation['uq']
    (relaxation_data_q, uncertainty_data_q) = relaxation['q']
    print(f"  band={uncertainty_data['band']}, cv={float(uncertainty_data['sigma_frac'])}, "
          f"percentiles={tuple(uncertainty_data['percentiles'])}")

    fig = plt.figure(figsize=(6.4, 4.6), layout='constrained')
    gs = fig.add_gridspec(1, 1)
    plot_panel_relaxation_combined(
        fig, gs[0, 0],
        [('Cluster', relaxation_data, uncertainty_data, 'cluster', ('uq',), True),
         ('Cluster', relaxation_data_q, uncertainty_data_q, 'cluster', ('q',), False)])
    savefig_panel(outdir, 'Fig1a_relaxation_vs_diameter')

    # ---- Fig. 1b: tau vs field (figure + workbook) ----
    print('\n[Fig. 1b] relaxation time vs applied field')
    relax_field_cluster = compute_panel_relaxation_vs_field(particle=CLUSTER_35NM)
    fig = plt.figure(figsize=(6.4, 4.6), layout='constrained')
    plot_panel_relaxation_vs_field(fig, fig.add_gridspec(1, 1)[0, 0],
                                   f"{CLUSTER_35NM['d']*1e9:.0f} nm cluster",
                                   relax_field_cluster, 'cluster', 'tau_B_cluster')
    savefig_panel(outdir, 'Fig1b_relaxation_vs_field')
    print(f"  35 nm cluster: h=1 at {h_unity_field_mT(CLUSTER_35NM, 'uq'):.1f} mT unquenched, "
          f"{h_unity_field_mT(CLUSTER_35NM, 'q'):.1f} mT quenched")

    # ---- Fig. 2c (workbook only) ----
    print('\n[Fig. 2c] signal vs glycerol content')
    glycerol_data = compute_panel_glycerol_data(particle=CLUSTER_35NM)
    glycerol_uncertainty = compute_panel_glycerol_uncertainty(particle=CLUSTER_35NM)

    # ---- Workbook export ----
    excel_path = os.path.join(outdir, 'MPI_signal_quenching_panel_data.xlsx')
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:

        # Fig. 1c: drive field, M(t), dM/dt(t) on one shared time index
        pd.DataFrame({
            't_us': overview_data['t_us'],
            'H_t_cl_mT': overview_data['H_t_cl'],
            'M_unquenched_norm': overview_data['M_uq_norm'],
            'M_quenched_norm': overview_data['M_q_norm'],
            'dMdt_unquenched_norm': overview_data['dMdt_uq_norm'],
            'dMdt_quenched_norm': overview_data['dMdt_q_norm'],
        }).to_excel(writer, sheet_name='Fig1c_overview', index=False)

        # Fig. 2a,b: 1-D field-domain PSF
        pd.DataFrame({
            'H_mT': psf_data['H_grid'],
            'psf_unquenched': psf_data['psf_uq'],
            'psf_quenched': psf_data['psf_q'],
        }).to_excel(writer, sheet_name='Fig2ab_psf_1D', index=False)

        # Fig. 1a: one sheet per quench state (each has its own diameter grid)
        for name, key in (('unquenched', 'uq'), ('quenched', 'q')):
            rel, unc = relaxation[key]
            pd.DataFrame({
                'd_core_nm': rel['d_core_nm'],
                'tau_B_unquenched_s': rel['tau_B_uq'],
                'tau_N_unquenched_s': rel['tau_N_uq'],
                'tau_eff_unquenched_s': rel['tau_eff_uq'],
                'tau_B_quenched_s': rel['tau_B_q'],
                'tau_N_quenched_s': rel['tau_N_q'],
                'tau_eff_quenched_s': rel['tau_eff_q'],
                'tau_eff_uq_lo': unc['tau_eff_lo_uq'], 'tau_eff_uq_hi': unc['tau_eff_hi_uq'],
                'tau_eff_q_lo': unc['tau_eff_lo_q'],   'tau_eff_q_hi': unc['tau_eff_hi_q'],
                'tau_N_uq_lo': unc['tau_N_lo_uq'],     'tau_N_uq_hi': unc['tau_N_hi_uq'],
                'tau_N_q_lo': unc['tau_N_lo_q'],       'tau_N_q_hi': unc['tau_N_hi_q'],
                'tau_B_uq_lo': unc['tau_B_lo_uq'],     'tau_B_uq_hi': unc['tau_B_hi_uq'],
                'tau_B_q_lo': unc['tau_B_lo_q'],       'tau_B_q_hi': unc['tau_B_hi_q'],
                'K_eff_uq_J_m3': rel['K_eff_uq_vs_d'],
                'K_eff_q_J_m3': rel['K_eff_q_vs_d'],
            }).to_excel(writer, sheet_name=f'Fig1a_tau_vs_d_{name}', index=False)

        # Fig. 1b: tau vs applied field, 35 nm cluster
        jd = relax_field_cluster
        pd.DataFrame({
            'B_mT': jd['B_mT'],
            'tau_B_unquenched_s': jd['tau_B_uq'],
            'tau_N_unquenched_s': jd['tau_N_uq'],
            'tau_eff_unquenched_s': jd['tau_eff_uq'],
            'tau_B_quenched_s': jd['tau_B_q'],
            'tau_N_quenched_s': jd['tau_N_q'],
            'tau_eff_quenched_s': jd['tau_eff_q'],
            'tau_B_uq_lo': jd['tau_B_uq_lo'],       'tau_B_uq_hi': jd['tau_B_uq_hi'],
            'tau_N_uq_lo': jd['tau_N_uq_lo'],       'tau_N_uq_hi': jd['tau_N_uq_hi'],
            'tau_eff_uq_lo': jd['tau_eff_uq_lo'],   'tau_eff_uq_hi': jd['tau_eff_uq_hi'],
            'tau_B_q_lo': jd['tau_B_q_lo'],         'tau_B_q_hi': jd['tau_B_q_hi'],
            'tau_N_q_lo': jd['tau_N_q_lo'],         'tau_N_q_hi': jd['tau_N_q_hi'],
            'tau_eff_q_lo': jd['tau_eff_q_lo'],     'tau_eff_q_hi': jd['tau_eff_q_hi'],
        }).to_excel(writer, sheet_name='Fig1b_tau_vs_field', index=False)

        # Fig. 2c: 35 nm cluster signal vs glycerol content
        pd.DataFrame({
            'glycerol_pct': glycerol_data['glycerol_pct'],
            'viscosity_cP': GLYCEROL_ETA_CP,
            'viscosity_Pa_s': GLYCEROL_ETA_PAS,
            'signal_max_unquenched': glycerol_data['signal_max_uq'],
            'signal_max_quenched': glycerol_data['signal_max_q'],
            'signal_avg_norm_unquenched': glycerol_uncertainty['mean_uq'],
            'signal_avg_norm_quenched': glycerol_uncertainty['mean_q'],
            'signal_errorbar_norm_unquenched': glycerol_uncertainty['sd_uq'],
            'signal_errorbar_norm_quenched': glycerol_uncertainty['sd_q'],
        }).to_excel(writer, sheet_name='Fig2c_glycerol', index=False)

        # manifest: every setting the numbers depend on, plus the software versions
        _rows = [('relaxation_model', 'hybrid')]
        for _k, _v in panel_settings_id().items():
            _rows.append((_k, repr(_v)))
        for _k, _v in (('D_SWEEP_CLUSTER_UQ', D_SWEEP_CLUSTER_UQ),
                       ('D_SWEEP_CLUSTER_Q', D_SWEEP_CLUSTER_Q),
                       ('B_AXIS_MT', B_AXIS_MT),
                       ('TAU_AXIS_C_S', TAU_AXIS_C_S),
                       ('TAU_AXIS_J_S', TAU_AXIS_J_S)):
            _rows.append((_k, repr(_v)))
        _rows.append(('preset_CLUSTER_35NM', repr(dict(CLUSTER_35NM))))
        for _k, _v in package_versions():
            _rows.append((f'software_{_k}', _v))
        pd.DataFrame(_rows, columns=['setting', 'value']).to_excel(
            writer, sheet_name='manifest', index=False)

    print(f'\nSaved source data to {excel_path}')
    for _sn in pd.ExcelFile(excel_path).sheet_names:
        print('   ', _sn)


if __name__ == '__main__':
    main()
