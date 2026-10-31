# Results

Written by `backtest.py` and `forecast.py`. Re-run the pipeline to refresh it.

Test period: 2026-04-25 to 2026-10-25. Origins: 00:00, 06:00, 12:00, 18:00 UTC daily. Emissions are computed from actual intensity.

## Forecast accuracy on the test period (gCO2/kWh)

| horizon | model | MAE | RMSE | bias | n |
|---|---|---|---|---|---|
| 6h | lightgbm | 18.49 | 23.44 | 5.06 | 8831 |
| 6h | ridge | 19.58 | 24.39 | 5.18 | 8831 |
| 6h | official_forecast | 10.78 | 16.11 | -1.91 | 8831 |
| 6h | naive_7d | 46.24 | 58.40 | -0.05 | 8831 |
| 6h | naive_24h | 37.14 | 48.43 | 0.34 | 8831 |
| 12h | lightgbm | 21.06 | 26.36 | 7.03 | 8831 |
| 12h | ridge | 23.11 | 28.81 | 5.37 | 8831 |
| 12h | official_forecast | 10.78 | 16.11 | -1.91 | 8831 |
| 12h | naive_7d | 46.24 | 58.40 | -0.05 | 8831 |
| 12h | naive_24h | 37.14 | 48.43 | 0.34 | 8831 |
| 24h | lightgbm | 19.77 | 25.01 | 7.24 | 8831 |
| 24h | ridge | 21.66 | 27.41 | 5.33 | 8831 |
| 24h | official_forecast | 10.78 | 16.11 | -1.91 | 8831 |
| 24h | naive_7d | 46.24 | 58.40 | -0.05 | 8831 |
| 24h | naive_24h | 37.14 | 48.43 | 0.34 | 8831 |
| 48h | lightgbm | 21.51 | 27.28 | 5.73 | 8831 |
| 48h | ridge | 24.19 | 30.48 | 5.07 | 8831 |
| 48h | official_forecast | 10.78 | 16.11 | -1.91 | 8831 |
| 48h | naive_7d | 46.24 | 58.40 | -0.05 | 8831 |


## Scheduling backtest

`mean_kg` is the average emissions of one job run. Savings are relative to starting immediately or to a randomly chosen start inside the window. `share_of_oracle_saving_pct` is the part of the perfect-hindsight saving that was captured.

### EV charge (4 h, 7 kW)

| window_h | strategy | mean_kg | saving_vs_immediate_pct | saving_vs_window_mean_pct | share_of_oracle_saving_pct |
|---|---|---|---|---|---|
| 6 | immediate | 3.306 | 0.000 | -0.057 | 0.000 |
| 6 | window_mean | 3.304 | 0.057 | 0.000 | 0.785 |
| 6 | naive_7d | 3.103 | 6.124 | 6.071 | 84.104 |
| 6 | official_forecast | 3.086 | 6.641 | 6.587 | 91.195 |
| 6 | lightgbm | 3.088 | 6.572 | 6.518 | 90.249 |
| 6 | oracle | 3.065 | 7.282 | 7.229 | 100.000 |
| 12 | immediate | 3.308 | 0.000 | 0.441 | 0.000 |
| 12 | window_mean | 3.323 | -0.443 | 0.000 | -2.036 |
| 12 | naive_7d | 2.767 | 16.366 | 16.735 | 75.302 |
| 12 | official_forecast | 2.626 | 20.613 | 20.962 | 94.841 |
| 12 | lightgbm | 2.666 | 19.391 | 19.746 | 89.218 |
| 12 | oracle | 2.589 | 21.734 | 22.079 | 100.000 |
| 24 | immediate | 3.311 | 0.000 | 0.552 | 0.000 |
| 24 | window_mean | 3.330 | -0.555 | 0.000 | -1.498 |
| 24 | naive_7d | 2.448 | 26.063 | 26.471 | 70.339 |
| 24 | official_forecast | 2.138 | 35.426 | 35.783 | 95.608 |
| 24 | lightgbm | 2.207 | 33.360 | 33.728 | 90.033 |
| 24 | oracle | 2.084 | 37.053 | 37.401 | 100.000 |
| 48 | immediate | 3.319 | 0.000 | 0.520 | 0.000 |
| 48 | window_mean | 3.336 | -0.522 | 0.000 | -1.086 |
| 48 | naive_7d | 2.372 | 28.528 | 28.899 | 59.285 |
| 48 | official_forecast | 1.789 | 46.084 | 46.364 | 95.769 |
| 48 | lightgbm | 1.868 | 43.716 | 44.008 | 90.849 |
| 48 | oracle | 1.722 | 48.120 | 48.389 | 100.000 |

Origins evaluated per window (h): {6: 735, 12: 734, 24: 732, 48: 728}

### ML training run (6 h, 20 kW)

| window_h | strategy | mean_kg | saving_vs_immediate_pct | saving_vs_window_mean_pct | share_of_oracle_saving_pct |
|---|---|---|---|---|---|
| 12 | immediate | 14.280 | 0.000 | -0.052 | 0.000 |
| 12 | window_mean | 14.272 | 0.052 | 0.000 | 0.342 |
| 12 | naive_7d | 12.733 | 10.829 | 10.783 | 71.177 |
| 12 | official_forecast | 12.228 | 14.369 | 14.324 | 94.441 |
| 12 | lightgbm | 12.345 | 13.550 | 13.505 | 89.060 |
| 12 | oracle | 12.107 | 15.215 | 15.171 | 100.000 |
| 24 | immediate | 14.296 | 0.000 | -0.086 | 0.000 |
| 24 | window_mean | 14.284 | 0.086 | 0.000 | 0.263 |
| 24 | naive_7d | 11.040 | 22.774 | 22.707 | 69.466 |
| 24 | official_forecast | 9.824 | 31.285 | 31.226 | 95.430 |
| 24 | lightgbm | 10.099 | 29.357 | 29.296 | 89.546 |
| 24 | oracle | 9.609 | 32.784 | 32.726 | 100.000 |
| 48 | immediate | 14.326 | 0.000 | -0.162 | 0.000 |
| 48 | window_mean | 14.303 | 0.161 | 0.000 | 0.356 |
| 48 | naive_7d | 10.530 | 26.498 | 26.379 | 58.459 |
| 48 | official_forecast | 8.125 | 43.289 | 43.197 | 95.503 |
| 48 | lightgbm | 8.548 | 40.334 | 40.238 | 88.984 |
| 48 | oracle | 7.832 | 45.328 | 45.239 | 100.000 |

Origins evaluated per window (h): {12: 734, 24: 732, 48: 728}

### Nightly ETL job (2 h, 5 kW)

| window_h | strategy | mean_kg | saving_vs_immediate_pct | saving_vs_window_mean_pct | share_of_oracle_saving_pct |
|---|---|---|---|---|---|
| 6 | immediate | 1.197 | 0.000 | -1.295 | 0.000 |
| 6 | window_mean | 1.182 | 1.278 | 0.000 | 8.606 |
| 6 | naive_7d | 1.056 | 11.805 | 10.663 | 79.487 |
| 6 | official_forecast | 1.041 | 12.987 | 11.860 | 87.447 |
| 6 | lightgbm | 1.042 | 12.957 | 11.830 | 87.245 |
| 6 | oracle | 1.019 | 14.851 | 13.749 | 100.000 |
| 12 | immediate | 1.198 | 0.000 | -0.987 | 0.000 |
| 12 | window_mean | 1.186 | 0.978 | 0.000 | 3.545 |
| 12 | naive_7d | 0.950 | 20.659 | 19.876 | 74.926 |
| 12 | official_forecast | 0.894 | 25.400 | 24.664 | 92.122 |
| 12 | lightgbm | 0.912 | 23.841 | 23.090 | 86.468 |
| 12 | oracle | 0.868 | 27.573 | 26.858 | 100.000 |
| 24 | immediate | 1.199 | 0.000 | -0.846 | 0.000 |
| 24 | window_mean | 1.189 | 0.839 | 0.000 | 2.040 |
| 24 | naive_7d | 0.852 | 28.904 | 28.302 | 70.283 |
| 24 | official_forecast | 0.733 | 38.840 | 38.322 | 94.442 |
| 24 | lightgbm | 0.763 | 36.361 | 35.822 | 88.414 |
| 24 | oracle | 0.706 | 41.126 | 40.627 | 100.000 |
| 48 | immediate | 1.202 | 0.000 | -0.879 | 0.000 |
| 48 | window_mean | 1.191 | 0.872 | 0.000 | 1.704 |
| 48 | naive_7d | 0.832 | 30.772 | 30.163 | 60.139 |
| 48 | official_forecast | 0.620 | 48.416 | 47.963 | 94.622 |
| 48 | lightgbm | 0.649 | 45.979 | 45.504 | 89.858 |
| 48 | oracle | 0.587 | 51.168 | 50.739 | 100.000 |

Origins evaluated per window (h): {6: 735, 12: 734, 24: 732, 48: 728}
