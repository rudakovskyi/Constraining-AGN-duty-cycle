# Constraining the AGN Duty Cycle with Network Analysis


This project studies whether **network (graph) analysis** of the spatial distribution of low-excitation radio galaxies (LERGs) can constrain their **duty cycle** — the fraction of galaxies that are active as radio sources — beyond what the radio luminosity function (RLF) alone can determine.

## Simple LERG model

- **Hosts.** LERGs are assigned to **red, central** galaxies.
- **Radio luminosity.** Each host receives a 150 MHz luminosity drawn from a **log-normal** distribution whose mean scales with stellar mass, $\langle \log_{10} L_{150} \rangle = \log_{10} L_0 + \alpha \, \log_{10}(M_\star / M_{\rm norm})$, with a **fixed dispersion** $\sigma_{\log L} = 0.5$ dex.
- **Duty cycle.** Two parametrizations are compared:
  - **(a) constant:** $F_{\rm duty} = F_0$;
  - **(b) mass-dependent power law:** $F_{\rm duty}(M_\star) = F_0  \left( M_\star / (10^{11}~M_\odot/h) \right)^{\beta}$.

## Data

**LERG luminosity function — the fit target.**
The 150 MHz LERG RLF from Kondapally et al. 2022 ([arXiv:2204.07588](https://arxiv.org/abs/2204.07588)), derived from the LOFAR Two-metre Sky Survey (LoTSS) Deep Fields. The machine-readable luminosity functions are available at [rohitk-10/AGN_LF_Kondapally22](https://github.com/rohitk-10/AGN_LF_Kondapally22). We fit the quiescent-host LERG LF in the $0.5 < z \le 1.0$ bin.

**Galaxy catalogue — for generating the LERG mocks.**
Red central galaxies from the Euclid Flagship (v2) mock catalogue ([Euclid Collaboration 2025, A&A, 697, A5](https://ui.adsabs.harvard.edu/abs/2025A%26A...697A...5E/abstract)). The host sample is extracted from the CosmoHub platform ([cosmohub.pic.es](https://cosmohub.pic.es)) with the query in [`euclid_fs2_lerg_hosts.sql`](https://github.com/rudakovskyi/Constraining-AGN-duty-cycle/blob/main/euclid_fs2_lerg_hosts.sql) — selecting observed $z \le 1.0$, $\log_{10}(M_\star/M_\odot) \ge 9.0$, and declination $\le 12^\circ$.


## Repository structure

.
├── testing_AGN_mass_dependent_duty_cycle_network_analysis.ipynb   # main analysis

├── rlf_fit_ref.py              # RLF loading, binned model, chi-square

├── radiogalaxies_duty_ref.py   # duty-cycle laws and luminosity assignment

├── generate_network.py         # lightcone graphs and centrality metrics

├── euclid_fs2_lerg_hosts.sql   # CosmoHub query for the host catalogue

└── README.md
