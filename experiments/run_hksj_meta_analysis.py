"""HKSJ-adjusted random-effects meta-analysis with full heterogeneity diagnostics."""
import json
import math
import numpy as np
from scipy import stats

def wilson_ci(p, n, z=1.96):
    """Wilson score confidence interval for a proportion."""
    denom = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    margin = z * math.sqrt((p*(1-p) + z**2/(4*n)) / n) / denom
    return max(0, center - margin), min(1, center + margin)

def run_meta_analysis():
    # Suite data: F1 estimates and sample sizes
    suites = [
        {"name": "Suite B (CEGAR ablation)", "f1": 0.966, "n": 32},
        {"name": "Suite C (external bugs)", "f1": 0.875, "n": 34},
        {"name": "Suite D (standard)", "f1": 0.917, "n": 50},
        {"name": "Deep composition", "f1": 1.000, "n": 25},
    ]
    
    k = len(suites)
    
    # Within-study variances (using Wilson CI-based variance)
    for s in suites:
        lo, hi = wilson_ci(s["f1"], s["n"])
        s["ci_lo"] = lo
        s["ci_hi"] = hi
        s["se"] = (hi - lo) / (2 * 1.96)
        s["var"] = s["se"]**2
    
    # --- Fixed-effect estimate ---
    w_fe = [1.0/s["var"] for s in suites]
    W_fe = sum(w_fe)
    theta_fe = sum(w*s["f1"] for w, s in zip(w_fe, suites)) / W_fe
    se_fe = math.sqrt(1.0 / W_fe)
    
    # --- Cochran's Q ---
    Q = sum(w * (s["f1"] - theta_fe)**2 for w, s in zip(w_fe, suites))
    df = k - 1
    p_Q = 1 - stats.chi2.cdf(Q, df)
    
    # --- I² ---
    I2 = max(0, (Q - df) / Q) * 100 if Q > df else 0.0
    
    # --- H² ---
    H2 = Q / df if df > 0 else 1.0
    
    # --- DerSimonian-Laird τ² ---
    C = W_fe - sum(w**2 for w in w_fe) / W_fe
    tau2_dl = max(0, (Q - df) / C)
    
    # --- Random-effects weights ---
    w_re = [1.0/(s["var"] + tau2_dl) for s in suites]
    W_re = sum(w_re)
    theta_re = sum(w*s["f1"] for w, s in zip(w_re, suites)) / W_re
    se_re_dl = math.sqrt(1.0 / W_re)
    
    # --- DL confidence interval ---
    ci_dl_lo = theta_re - 1.96 * se_re_dl
    ci_dl_hi = theta_re + 1.96 * se_re_dl
    
    # --- HKSJ adjustment ---
    # HKSJ replaces se_re with: sqrt( (1/(k*(k-1))) * sum(w_i * (theta_i - theta_re)^2) )
    # This is the Hartung-Knapp-Sidik-Jonkman variance estimator
    q_hksj = sum(w * (s["f1"] - theta_re)**2 for w, s in zip(w_re, suites))
    var_hksj = q_hksj / (k * (k - 1))
    se_hksj = math.sqrt(var_hksj)
    
    # HKSJ uses t-distribution with k-1 df
    t_crit = stats.t.ppf(0.975, df=k-1)
    ci_hksj_lo = theta_re - t_crit * se_hksj
    ci_hksj_hi = theta_re + t_crit * se_hksj
    
    # --- Prediction interval ---
    # PI = theta_re ± t_{k-2, 0.975} * sqrt(se_re^2 + tau2)
    if k > 2:
        t_pred = stats.t.ppf(0.975, df=k-2)
        se_pred = math.sqrt(se_re_dl**2 + tau2_dl)
        pi_lo = theta_re - t_pred * se_pred
        pi_hi = theta_re + t_pred * se_pred
    else:
        pi_lo, pi_hi = float('nan'), float('nan')
    
    # --- Leave-one-out sensitivity ---
    leave_one_out = []
    for i in range(k):
        sub = [s for j, s in enumerate(suites) if j != i]
        w_sub_fe = [1.0/s["var"] for s in sub]
        W_sub_fe = sum(w_sub_fe)
        theta_sub_fe = sum(w*s["f1"] for w, s in zip(w_sub_fe, sub)) / W_sub_fe
        Q_sub = sum(w*(s["f1"]-theta_sub_fe)**2 for w, s in zip(w_sub_fe, sub))
        C_sub = W_sub_fe - sum(w**2 for w in w_sub_fe)/W_sub_fe
        tau2_sub = max(0, (Q_sub - (k-2)) / C_sub)
        w_sub_re = [1.0/(s["var"] + tau2_sub) for s in sub]
        W_sub_re = sum(w_sub_re)
        theta_sub_re = sum(w*s["f1"] for w, s in zip(w_sub_re, sub)) / W_sub_re
        se_sub = math.sqrt(1.0/W_sub_re)
        leave_one_out.append({
            "excluded": suites[i]["name"],
            "pooled_f1": round(theta_sub_re, 4),
            "ci_lo": round(theta_sub_re - 1.96*se_sub, 4),
            "ci_hi": round(min(1.0, theta_sub_re + 1.96*se_sub), 4),
            "tau2": round(tau2_sub, 6),
            "I2_pct": round(max(0, (Q_sub-(k-2))/Q_sub)*100 if Q_sub > k-2 else 0, 1)
        })
    
    # --- Funnel plot data ---
    funnel_data = [{"f1": s["f1"], "se": s["se"], "name": s["name"]} for s in suites]
    
    results = {
        "experiment": "hksj_adjusted_meta_analysis",
        "timestamp": "2026-02-25T15:30:00Z",
        "k": k,
        "total_n": sum(s["n"] for s in suites),
        "per_suite": [{
            "name": s["name"], "f1": s["f1"], "n": s["n"],
            "se": round(s["se"], 4),
            "ci_95": [round(s["ci_lo"], 4), round(s["ci_hi"], 4)]
        } for s in suites],
        "heterogeneity": {
            "cochrans_Q": round(Q, 4),
            "Q_df": df,
            "Q_p_value": round(p_Q, 4),
            "I_squared_pct": round(I2, 1),
            "I_squared_interpretation": "low" if I2 < 25 else "moderate" if I2 < 75 else "high",
            "H_squared": round(H2, 4),
            "tau_squared_DL": round(tau2_dl, 6),
            "tau_DL": round(math.sqrt(tau2_dl), 4)
        },
        "dersimonian_laird": {
            "pooled_f1": round(theta_re, 4),
            "se": round(se_re_dl, 4),
            "ci_95": [round(max(0, ci_dl_lo), 4), round(min(1, ci_dl_hi), 4)]
        },
        "hksj_adjusted": {
            "pooled_f1": round(theta_re, 4),
            "se_hksj": round(se_hksj, 4),
            "t_critical": round(t_crit, 4),
            "df": k - 1,
            "ci_95": [round(max(0, ci_hksj_lo), 4), round(min(1, ci_hksj_hi), 4)],
            "note": "HKSJ uses t-distribution with k-1 df instead of normal, producing wider and more conservative CIs at small k"
        },
        "prediction_interval": {
            "pi_95": [round(max(0, pi_lo), 4), round(min(1, pi_hi), 4)],
            "interpretation": "Expected F1 range for a new, unobserved evaluation suite"
        },
        "leave_one_out_sensitivity": leave_one_out,
        "funnel_plot_data": funnel_data,
        "forest_plot_data": [
            {"label": s["name"], "estimate": s["f1"], "ci_lo": round(s["ci_lo"],4), 
             "ci_hi": round(s["ci_hi"],4), "weight_pct": round(100*w/W_re, 1), "n": s["n"]}
            for w, s in zip(w_re, suites)
        ] + [{
            "label": "Pooled (HKSJ)", "estimate": round(theta_re, 4),
            "ci_lo": round(max(0, ci_hksj_lo), 4), "ci_hi": round(min(1, ci_hksj_hi), 4),
            "weight_pct": 100.0, "n": sum(s["n"] for s in suites)
        }],
        "comparison": {
            "dl_ci_width": round(ci_dl_hi - ci_dl_lo, 4),
            "hksj_ci_width": round(ci_hksj_hi - ci_hksj_lo, 4),
            "hksj_wider": bool((ci_hksj_hi - ci_hksj_lo) > (ci_dl_hi - ci_dl_lo)),
            "note": "HKSJ CI is expected to be wider than DL CI at small k, providing more honest uncertainty quantification"
        }
    }
    
    with open("experiments/hksj_meta_analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"DL:   F1={theta_re:.4f}, 95% CI [{ci_dl_lo:.4f}, {min(1,ci_dl_hi):.4f}]")
    print(f"HKSJ: F1={theta_re:.4f}, 95% CI [{max(0,ci_hksj_lo):.4f}, {min(1,ci_hksj_hi):.4f}]")
    print(f"PI:   [{max(0,pi_lo):.4f}, {min(1,pi_hi):.4f}]")
    print(f"I²={I2:.1f}%, Q={Q:.4f} (p={p_Q:.4f}), τ²={tau2_dl:.6f}")
    print(f"\nLeave-one-out:")
    for loo in leave_one_out:
        print(f"  Excl {loo['excluded']}: F1={loo['pooled_f1']:.4f} [{loo['ci_lo']:.4f}, {loo['ci_hi']:.4f}]")
    
    return results

if __name__ == "__main__":
    run_meta_analysis()
