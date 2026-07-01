-- Euclid Flagship v2 mock (photo-z table): candidate LERG-host galaxy selection
-- Cuts: observed redshift <= 1.0, log(M*/Msun) >= 9.0, dec <= 12 deg
-- Table: euclid_fs2_mock_dr_v1_1_phz

SELECT
  `kind`,
  `galaxy_id`,
  `halo_id`,
  `true_redshift_gal`,
  `r_gal`,
  `color_kind`,
  `log_luminosity_r01`,
  `log_stellar_mass`,
  `lm_halo`,
  `metallicity`,
  `log_sfr`,
  `x_gal`,
  `y_gal`,
  `z_gal`,
  `d4000_n_blue_abs`,
  `d4000_n_red_abs`,
  `g01r01_hod`,
  `ra_gal`,
  `dec_gal`,
  `abs_mag_r01`
FROM
  euclid_fs2_mock_dr_v1_1_phz
WHERE
  observed_redshift_gal <= 1.0
  AND log_stellar_mass >= 9.0
  AND dec_gal <= 12;
