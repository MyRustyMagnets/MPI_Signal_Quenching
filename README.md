# MPI signal quenching: dynamic-magnetization (Néel–Brown) simulation

Code and source data for the simulated panels of *"Modulating Néel Relaxation in Iron Oxide Nanocrystal Clusters to Develop Smart Magnetic Particle Imaging Probes for in vivo Biological Sensing*
(submitted to *Nature Materials"*). A macrospin model of the magnetic particle imaging
(MPI) signal of superparamagnetic iron-oxide nanoclusters, with and without a
Gd³⁺-shell-induced increase in effective anisotropy ("quenching"). It computes
relaxation times, the dynamic magnetization under a sinusoidal drive plus a
linear bias ramp, and the resulting point-spread functions (PSFs).

DOI of the archived release: **[10.5281/zenodo.23522547](https://doi.org/10.5281/zenodo.23522547)**

## Files

| File | Role |
|---|---|
| `MPI_signal_quenching.py` | Solver library: excitation waveform, Langevin equilibrium magnetization, field-dependent Brownian/Néel relaxation (Fokker–Planck eigenvalue solves plus Brown's high-barrier form), dynamic magnetization, PSF reconstruction, size-distribution quadrature. |
| `MPI_signal_quenching_figures.py` | Sets particle and scan parameters, calls the solver, writes the figures and the workbook. |
| `MPI_signal_quenching_panel_data.xlsx` | Source data of the simulated panels, as produced by the script. |
| `requirements.txt` | Pinned package versions. |

## Run

```bash
pip install -r requirements.txt
python MPI_signal_quenching_figures.py            # writes to the output folder next to the script
python MPI_signal_quenching_figures.py --outdir some/dir
```

There is no input data and no random number generator: the run is
deterministic and recomputes everything from the parameters in
`MPI_signal_quenching_figures.py`.

## Outputs

Figures (`.png` at 300 dpi and `.pdf`) in `output/` (next to the script by default):

| Manuscript | File stem | Content |
|---|---|---|
| Fig. 1a | `Fig1a_relaxation_vs_diameter` | τ_B, τ_N, τ_eff vs cluster diameter at zero field, exact 16/84 % bands from a 10 % diameter c.o.v. |
| Fig. 1b | `Fig1b_relaxation_vs_field` | τ_B, τ_N, τ_eff vs applied field, 35 nm cluster |
| Fig. 2a,b | `Fig2ab_psf` | Simulated 2-D PSF images (top) and 1-D field-domain PSFs (bottom), Cluster (unquenched) and Cluster-Gd (quenched) |

Workbook `output/MPI_signal_quenching_panel_data.xlsx`, one sheet per figure
panel (Fig. 1c and Fig. 2c are in the workbook only):

| Sheet | Manuscript | Columns |
|---|---|---|
| `Fig1a_tau_vs_d_unquenched`, `Fig1a_tau_vs_d_quenched` | Fig. 1a | diameter, τ_B/τ_N/τ_eff (s), 16/84 % bands, K_eff. One sheet per quench state, each runs to its own σ = 25 ceiling |
| `Fig1b_tau_vs_field` | Fig. 1b | field (mT), τ_B/τ_N/τ_eff (s) with 16/84 % bands |
| `Fig1c_overview` | Fig. 1c | t (µs), H (mT), normalized M(t) and dM/dt(t), unquenched and quenched |
| `Fig2ab_psf_1D` | Fig. 2a,b | H (mT), normalized 1-D PSF, unquenched and quenched |
| `Fig2c_glycerol` | Fig. 2c | glycerol %, viscosity (cP and Pa·s), monodisperse peak signal, size-averaged normalized signal and s.d. |
| `manifest` | | every setting the numbers depend on, and the software versions of the run |

File and sheet names follow the figure numbers of the submitted manuscript. In the
figures, "Cluster" is the unquenched particle and "Cluster-Gd" the Gd-quenched one.

## Main inputs

All set at the top of `MPI_signal_quenching_figures.py`:

| Quantity | Value |
|---|---|
| Temperature | 298 K |
| Drive | 20 mT amplitude, 20 kHz, ±100 mT bias sweep, 4000 cycles |
| Magnetization M_s | 72 emu/g (× 5200 kg/m³ = 3.74 × 10⁵ A/m) |
| Effective anisotropy K_eff | 2801 J/m³ unquenched, 6300 J/m³ quenched (ACS-derived) |
| Packing fraction | 0.5 |
| Gilbert damping α | 0.05 |
| Shell thickness | 10 nm (d_h = d_core + 20 nm) (DLS-derived) |
| Size dispersion | 10 % c.o.v. of core diameter; 7 deterministic quadrature nodes (TEM-derived) |
| Viscosity series | glycerol 0/20/40/80 % → 0.893, 1.73, 4.05, 66.6 cP (from measured viscosity values) |
| PSF gradient | 5.7 T/m (Derived from Momentum MPI imaging system|

## Model

`H(t) = H_ramp(t) + A·sin(2πft)`; the magnetization relaxes toward the Langevin
equilibrium `M_eq(H)` with the field-dependent time `τ_eff(H) = (1/τ_B + 1/τ_N)⁻¹`.

- **τ_B**: Brownian Fokker–Planck eigenvalue solve at every field.
- **τ_N**: Néel Fokker–Planck eigenvalue solve wherever the field-reduced barrier
  σ_eff = σ(1−h)² ≤ 20, and Brown's high-barrier closed form (Eq. 14) elsewhere.
  The Legendre truncation order is chosen per field point from ξ.
- **PSF**: dM/dt binned against H for the positive slew; the polydisperse PSF is
  the Fe-mass-weighted sum of per-diameter PSFs over the quadrature nodes. The 2-D
  image is an isotropic Gaussian whose FWHM matches the 1-D PSF.

`tau_B/tau_N` lookup tables are memoized in memory within a run when the
`fast` solver preset is used (polydisperse glycerol series). This is an exact
identity (τ_N has no viscosity dependence; τ_B,eigen = τ_B0·G(ξ)), not a result
cache, and nothing is written to disk.

## Known limitations

- The 100 mT bias excursion lies outside the validated range of Brown's Eq. 14
  for these clusters (its ξ/σ = 1.5 bound is reached at 11–25 mT). τ_N is negligible
  there and the PSF is dominated by the low-field region.
- The model assumes coherent (macrospin) reversal and an easy axis aligned with
  the drive, which requires the particles to reorient within the scan (true for the
  liquid samples simulated; not for immobilized samples).

## Software environment

Figures and workbook were produced with:

| Package | Version |
|---|---|
| Python | 3.13.9 |
| NumPy | 2.5.2 |
| SciPy | 1.18.1 |
| Matplotlib | 3.11.1 |
| pandas | 3.0.5 |
| openpyxl | 3.1.5 |

The `manifest` sheet of the workbook records the versions of the run that wrote it.
`savefig` output depends on available fonts (Arial → Helvetica → DejaVu Sans).

## Code availability statement (suggested)

> The code used to simulate the relaxation times, dynamic magnetization and
> point-spread functions, and to generate the simulated panels of Figs. 1 and 2,
> is available at [GITHUB URL] and archived on Zenodo (DOI: [ZENODO DOI]) under
> the MIT license. The simulation uses Python 3.13.9 with NumPy 2.5.2, SciPy 1.18.1,
> Matplotlib 3.11.1, pandas 3.0.5 and openpyxl 3.1.5.

## AI-assistance disclosure

An AI assistant (Claude, Anthropic) was used to refactor the analysis code from a
Jupyter notebook into standalone scripts, remove unused code, and draft this README for open access.
All simulation methods, parameters and results are those of the authors, and the
authors verified that the packaged code reproduces the original outputs. Furthermore,
AI was used to aid in debugging the original code.

## Citation

Please cite the manuscript and the archived release (see DOI above).

## License

MIT, see `LICENSE`. Copyright (c) 2026 The Board of Trustees of the Leland Stanford
Junior University. The code is authored by Jacob Bryan-Kim; for academic purposes,
cite the manuscript and the archived release (see Citation).

## Acknowledgements
The work described was supported, in part, by NIH awards R01CA271530-01A1, R01CA243033,1S10OD028536-01. 
J.T.B is supported under the National Science Foundation Graduate Research Fellowship Program under Grant 
No DGE-2146755 and a Chevron Stanford Graduate Fellowship. Its contents are solely the responsibility of 
the authors and do not necessarily represent the official views of the NCRR, the National Institutes of 
Health, NSF, or SGF program. Part of this work was performed at the Stanford Nano Shared Facilities (SNSF), 
supported by the National Science Foundation under award ECCS-2026822. We thank Emily Chen for helpful 
discussions on lanthanide magnetism. 
