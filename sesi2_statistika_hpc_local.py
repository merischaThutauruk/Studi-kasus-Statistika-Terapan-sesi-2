import math
import multiprocessing as mp
import random
import statistics
import time

def monte_carlo_building_worker(task_params):
    """Menjalankan simulasi stokastik Monte Carlo untuk satu gedung pada kombinasi

    Faktor A (Kebijakan HVAC: STANDAR vs ECO_SMART) dan Faktor B (Zona: LAB_SAINS vs KELAS_UMUM).
    """
    rep_id, policy, zone, n_monte_carlo_runs = task_params
    rng = random.Random(rep_id * 7919)

    energy_samples = []
    flops_count = 0

    for _ in range(n_monte_carlo_runs):
        daily_kwh = 0.0
        for hour in range(24):
            temp_out = (
                27.0
                + 6.5 * math.sin((hour - 6) * math.pi / 12.0)
                + rng.gauss(0, 1.2)
            )
            occupants = (
                rng.randint(40, 180) if (8 <= hour <= 17) else rng.randint(0, 15)
            )

            base_zone_load = 18.5 if zone == "LAB_SAINS" else 6.0

            if policy == "ECO_SMART":
                setpoint = 25.0
                active_factor = 1.0 if (9 <= hour <= 16) else 0.15
            else:
                setpoint = 20.0
                active_factor = 1.0 if (7 <= hour <= 20) else 0.40

            delta_t = max(0.0, temp_out - setpoint)
            cooling_load = (
                (delta_t * 3.2) + (occupants * 0.09)
            ) * active_factor
            daily_kwh += base_zone_load + cooling_load
            flops_count += 25

        energy_samples.append(daily_kwh)

    return {
        "rep_id": rep_id,
        "policy": policy,
        "zone": zone,
        "mean_daily_kwh": statistics.mean(energy_samples),
        "flops": flops_count,
    }


def run_parallel_monte_carlo_and_anova():
    print("=" * 85)
    print("TAHAP 1: SIMULASI MONTE CARLO PARALEL (HPC) & BENCHMARK KECEPATAN FLOPS")
    print("=" * 85)

    policies = ["STANDAR", "ECO_SMART"]
    zones = ["LAB_SAINS", "KELAS_UMUM"]
    replications_per_cell = 15  
    mc_iterations = 800

    tasks = []
    rep_counter = 1
    for p in policies:
        for z in zones:
            for _ in range(replications_per_cell):
                tasks.append((rep_counter, p, z, mc_iterations))
                rep_counter += 1

    t0 = time.perf_counter()
    _ = [monte_carlo_building_worker(t) for t in tasks]
    seq_time = time.perf_counter() - t0

    n_cores = min(mp.cpu_count(), 8)
    t1 = time.perf_counter()
    with mp.Pool(processes=n_cores) as pool:
        results = pool.map(monte_carlo_building_worker, tasks)
    par_time = time.perf_counter() - t1

    total_flops = sum(r["flops"] for r in results)
    mflops = (total_flops / par_time) / 1e6
    print(f"Total Iterasi Monte Carlo     : {len(tasks) * mc_iterations:,} simulasi")
    print(f"Waktu Sekuensial (1 Core)     : {seq_time:.4f} detik")
    print(f"Waktu Paralel ({n_cores} Cores)       : {par_time:.4f} detik (Speedup: {seq_time/par_time:.2f}x)")
    print(f"Performa Komputasi Paralel    : {mflops:.2f} MFLOPS\n")

    print("=" * 85)
    print("TAHAP 2: TABEL TWO-WAY ANOVA (FAKTOR A: KEBIJAKAN ENERGI, FAKTOR B: ZONA GEDUNG)")
    print("=" * 85)

    N = len(results)
    grand_mean = statistics.mean(r["mean_daily_kwh"] for r in results)

    mean_A = {
        p: statistics.mean(
            r["mean_daily_kwh"] for r in results if r["policy"] == p
        )
        for p in policies
    }
    mean_B = {
        z: statistics.mean(
            r["mean_daily_kwh"] for r in results if r["zone"] == z
        )
        for z in zones
    }

    mean_AB = {}
    for p in policies:
        for z in zones:
            mean_AB[(p, z)] = statistics.mean(
                r["mean_daily_kwh"]
                for r in results
                if r["policy"] == p and r["zone"] == z
            )

    n_B = len(zones) * replications_per_cell
    n_A = len(policies) * replications_per_cell
    ssa = sum(n_B * ((mean_A[p] - grand_mean) ** 2) for p in policies)
    ssb = sum(n_A * ((mean_B[z] - grand_mean) ** 2) for z in zones)

    ssab = 0.0
    for p in policies:
        for z in zones:
            ssab += replications_per_cell * (
                (mean_AB[(p, z)] - mean_A[p] - mean_B[z] + grand_mean) ** 2
            )

    sse = sum(
        (r["mean_daily_kwh"] - mean_AB[(r["policy"], r["zone"])]) ** 2
        for r in results
    )
    sst = ssa + ssb + ssab + sse

    df_a = len(policies) - 1
    df_b = len(zones) - 1
    df_ab = df_a * df_b
    df_e = N - (len(policies) * len(zones))

    msa, msb, msab, mse = ssa / df_a, ssb / df_b, ssab / df_ab, sse / df_e
    f_a, f_b, f_ab = msa / mse, msb / mse, msab / mse

    f_crit = 4.013

    print(f"{'SUMBER KERAGAMAN (SOURCE)':<28} | {'SS':<12} | {'df':<4} | {'MS':<12} | {'F-Hitung':<10} | {'KEPUTUSAN (alpha=0.05)'}")
    print("-" * 96)
    print(f"{'Faktor A (Kebijakan HVAC)':<28} | {ssa:<12.2f} | {df_a:<4d} | {msa:<12.2f} | {f_a:<10.2f} | {'Signifikan (Tolak H0)' if f_a > f_crit else 'Tidak Signifikan'}")
    print(f"{'Faktor B (Zona Bangunan)':<28} | {ssb:<12.2f} | {df_b:<4d} | {msb:<12.2f} | {f_b:<10.2f} | {'Signifikan (Tolak H0)' if f_b > f_crit else 'Tidak Signifikan'}")
    print(f"{'Interaksi (A x B)':<28} | {ssab:<12.2f} | {df_ab:<4d} | {msab:<12.2f} | {f_ab:<10.2f} | {'Signifikan (Tolak H0)' if f_ab > f_crit else 'Tidak Signifikan'}")
    print(f"{'Galat (Within Error)':<28} | {sse:<12.2f} | {df_e:<4d} | {mse:<12.2f} | {'-':<10} |")
    print(f"{'Total':<28} | {sst:<12.2f} | {N-1:<4d} | {'-':<12} | {'-':<10} |\n")

def time_series_pm25_and_bootstrap():
    print("=" * 85)
    print("TAHAP 3: PERAMALAN DERET WAKTU PM 2.5 & ESTIMASI INTERVAL KEPERCAYAAN BOOTSTRAP")
    print("=" * 85)

    random.seed(101)
    n_days = 60
    pm25_series = []
    for t in range(n_days):
        seasonal_comp = 14.0 * math.sin(2.0 * math.pi * t / 7.0)
        trend_comp = 0.18 * t
        noise = random.gauss(0, 4.5)
        val = (
            42.0
            + trend_comp
            + seasonal_comp
            + (0.45 * pm25_series[t - 1] if t > 0 else 0)
            + noise
        )
        pm25_series.append(val)

    s = 7
    diff_series = [pm25_series[t] - pm25_series[t - s] for t in range(s, n_days)]

    y_curr = diff_series[1:]
    y_lag1 = diff_series[:-1]
    mean_c, mean_l = statistics.mean(y_curr), statistics.mean(y_lag1)

    phi_1 = sum(
        (y_curr[i] - mean_c) * (y_lag1[i] - mean_l) for i in range(len(y_curr))
    ) / sum((y_lag1[i] - mean_l) ** 2 for i in range(len(y_lag1)))
    intercept = mean_c - phi_1 * mean_l

    residuals = [
        y_curr[i] - (intercept + phi_1 * y_lag1[i]) for i in range(len(y_curr))
    ]

    n_bootstrap = 1000
    print(f"Parameter Terestimasi SARIMA(1,0,0)(0,1,0)_7 -> Phi_1 = {phi_1:.4f}, Intercept = {intercept:.4f}")
    print(f"Menjalankan {n_bootstrap} replikasi non-parametrik Bootstrap untuk 95% Confidence Interval...\n")

    print(f"{'HORIZON RAMALAN':<18} | {'PREDIKSI TITIK PM2.5':<22} | {'95% BOOTSTRAP CI [LOWER, UPPER]':<32} | {'KATEGORI UDARA'}")
    print("-" * 95)

    last_diff = diff_series[-1]
    for h in range(1, 8):
        seasonal_base = pm25_series[n_days - s + (h - 1)]
        pred_diff = intercept + phi_1 * last_diff
        point_forecast = seasonal_base + pred_diff

        boot_forecasts = []
        for _ in range(n_bootstrap):
            sampled_res = random.choice(residuals)
            boot_forecasts.append(point_forecast + sampled_res)

        boot_forecasts.sort()
        ci_lower = boot_forecasts[int(0.025 * n_bootstrap)]
        ci_upper = boot_forecasts[int(0.975 * n_bootstrap)]

        status = (
            "Tidak Sehat (Waspada Asma)"
            if point_forecast > 65.0
            else "Sedang / Aman"
        )
        print(
            f"Hari ke-+{h:<9} | {point_forecast:<14.2f} µg/m³ | [{ci_lower:6.2f}, {ci_upper:6.2f}] µg/m³          | {status}"
        )
        last_diff = pred_diff


if __name__ == "__main__":
    run_parallel_monte_carlo_and_anova()
    time_series_pm25_and_bootstrap()