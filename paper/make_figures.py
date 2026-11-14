"""Publication figures for the P05 JBHI manuscript.

Reads only Director-audited aggregate outputs (T05b numbers, T06, T06-HF1, T06b).
No waveforms, IDs or record-level data are written. Run with envs/pn_b python -I.
Palette: first slots of the dataviz reference categorical palette (fixed order).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy.stats import beta

import argparse
import yaml
ROOT = Path(__file__).resolve().parents[1]
PTH = yaml.safe_load((ROOT / "configs" / "paths.yaml").read_text(encoding="utf-8"))
OUT = (ROOT / str(PTH["outputs"])).resolve()
_parser = argparse.ArgumentParser()
_parser.add_argument("--out", default="paper_outputs/")
FIG = Path(_parser.parse_args().out)
FIG.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5,
    "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.5, "lines.linewidth": 1.6, "pdf.fonttype": 42,
    "savefig.dpi": 600,
})
COL1, COL2 = 3.5, 7.16  # IEEE column / page width (inches)


def cp(k, n, conf=0.95):
    a = 1 - conf
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return lo, hi


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------- Fig. 1
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(COL2, 3.0))
    ax.set_xlim(0, 100); ax.set_ylim(0, 60); ax.axis("off")

    def box(x, y, w, h, title, body, edge=INK2, fill="#f4f3f0"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.0",
                                    fc=fill, ec=edge, lw=0.9))
        ax.text(x + w / 2, y + h - 1.6, title, ha="center", va="top", fontsize=7.2,
                fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h - 5.6, body, ha="center", va="top", fontsize=6.4,
                color=INK2, linespacing=1.3)

    PAD = 0.3  # FancyBboxPatch pad: the drawn border lies PAD outside (x, y, w, h)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8, shrinkA=0, shrinkB=0,
                                    mutation_scale=7))

    def path(*pts):
        """Orthogonal connector: plain segments, arrowhead on the last one, ending on a box border."""
        for (xa, ya), (xb, yb) in zip(pts[:-2], pts[1:-1]):
            ax.plot([xa, xb], [ya, yb], color=INK2, lw=0.8, solid_capstyle="butt")
        arrow(*pts[-2], *pts[-1])

    box(0.5, 28, 16, 20, "VTaC v1.1", "5,037 VT alarms\n2,260 recordings\n1,441 true VT\nannotator votes")
    split = [(46, "Fitting set", "2,830 alarms\n1,265 recordings", INK2),
             (32, "Calibration set", "1,230 alarms, 543 rec.\n188 rec. with true VT", BLUE),
             (18, "Validation", "495 alarms, 226 rec.\ngates G1, G2", INK2),
             (4, "Test (sealed)", "482 alarms, 226 rec.\nopened once", ORANGE)]
    trunk_x = 18.6
    src_y = 28 + 20 / 2  # VTaC box mid-height
    ax.plot([16.5 + PAD, trunk_x], [src_y, src_y], color=INK2, lw=0.8)
    ax.plot([trunk_x, trunk_x], [4 + 5.75, 46 + 5.75], color=INK2, lw=0.8, solid_capstyle="butt")
    for y, t, b, e in split:
        box(20.5, y, 21, 11.5, t, b, edge=e)
        arrow(trunk_x, y + 5.75, 20.5 - PAD, y + 5.75)

    box(45, 40, 21, 16, "ResNet-1D, 5 seeds", "ECG1, ECG2, PPG, ABP\n+ availability masks\n10 s before onset")
    box(46, 17, 19, 16, "Record-level CRC", "largest λ with\n(ΣLⱼ(λ)+1)/(n+1) ≤ α\nα = 0.02 primary", edge=BLUE)
    split_right = 20.5 + 21 + PAD
    arrow(split_right, 46 + 5.75, 45 - PAD, 46 + 5.75)            # fitting set -> ResNet
    path((split_right, 32 + 5.75), (43.6, 32 + 5.75), (43.6, 17 + 8), (46 - PAD, 17 + 8))  # calibration -> CRC
    arrow(55.5, 40 - PAD, 55.5, 17 + 16 + PAD)                     # ResNet -> CRC

    box(68.5, 22, 14.5, 20, "Lock file", "thresholds A–E\n25 ckpt hashes\nanalysis list\none-shot flag")
    arrow(46 + 19 + PAD, 17 + 8, 68.5 - PAD, 17 + 8)               # CRC -> lock file

    box(86, 35, 13.5, 16, "Internal test", "VTaC official\nsplit, 482\nalarms", edge=ORANGE)
    box(86, 12, 13.5, 16, "External", "2015 Challenge\n341 VT alarms\nno recalibration", edge=ORANGE)
    fork_x = 84.5
    ax.plot([68.5 + 14.5 + PAD, fork_x], [32, 32], color=INK2, lw=0.8)
    ax.plot([fork_x, fork_x], [12 + 8, 35 + 8], color=INK2, lw=0.8, solid_capstyle="butt")
    arrow(fork_x, 35 + 8, 86 - PAD, 35 + 8)                         # lock -> internal test
    arrow(fork_x, 12 + 8, 86 - PAD, 12 + 8)                         # lock -> external

    ax.text(46, 9.5, "Post-hoc analyses (planned after test opening): calibration size and PAC;\n"
                     "modality and acquisition-profile shift; annotator reference standard;\n"
                     "prevalence shift; alarm burden per 1,000 VT alarms.",
            fontsize=6.4, color=INK2, va="top", linespacing=1.3)
    save(fig, "fig1_pipeline")


# ---------------------------------------------------------------- Fig. 2
def fig_crc_ltt():
    # Audited T05b / T06 aggregates: (suppression, k missed, n true)
    rows = {
        "VTaC test": [("CRC α=0.01", .145, 0, 137), ("CRC α=0.02", .554, 1, 137),
                      ("CRC α=0.05", .635, 4, 137), ("LTT", .739, 5, 137)],
        "Challenge 2015 (external)": [("CRC α=0.01", .135, 0, 89), ("CRC α=0.02", .619, 2, 89),
                                      ("CRC α=0.05", .718, 3, 89), ("LTT", .794, 6, 89)],
    }
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.5), sharey=True)
    for ax, (title, pts) in zip(axes, rows.items()):
        crc = [p for p in pts if p[0].startswith("CRC")]
        offs = {"CRC α=0.01": (5, 4), "CRC α=0.02": (-38, 9), "CRC α=0.05": (-47, 24), "LTT": (6, -3)}
        if title.startswith("Challenge"):
            offs.update({"CRC α=0.02": (-42, 4), "CRC α=0.05": (5, -13)})
        ax.plot([p[1] * 100 for p in crc], [p[2] / p[3] * 100 for p in crc], color=BLUE, lw=1.2, zorder=1)
        for lab, s, k, n in pts:
            lo, hi = cp(k, n)
            c = ORANGE if lab == "LTT" else BLUE
            f = k / n
            ax.errorbar(s * 100, f * 100, yerr=[[(f - lo) * 100], [(hi - f) * 100]], fmt=("s" if lab == "LTT" else "o"), ms=5,
                        color=c, mec="white", mew=1.0, elinewidth=1.0, capsize=2, zorder=3)
            ax.annotate(lab.replace("CRC ", ""), (s * 100, f * 100), xytext=offs[lab],
                        textcoords="offset points", fontsize=7, color=INK)
        ax.axhline(5, color=INK2, lw=0.8, ls="--")
        ax.text(1, 5.3, "5%", fontsize=7, color=INK2)
        ax.set_title(title, loc="left")
        ax.set_xlim(0, 90); ax.set_ylim(-0.5, 15)
        ax.set_xlabel("False alarms suppressed (%)")
    axes[0].set_ylabel("True VT alarms suppressed, FNR (%)")
    from matplotlib.lines import Line2D
    axes[1].legend(handles=[Line2D([], [], marker="o", color=BLUE, label="CRC (record-level FNR)", mec="white"),
                            Line2D([], [], marker="s", color=ORANGE, ls="", label="LTT (P(true | suppressed) ≤ 5%)", mec="white")],
                   loc="upper left", frameon=False)
    fig.tight_layout()
    save(fig, "fig2_crc_vs_ltt")


# ---------------------------------------------------------------- Fig. 3
def fig_stability():
    d = pd.read_parquet(OUT / "t06_hf1" / "A1_bootstrap.parquet")
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL2, 2.45))
    ens = d[d.model == "ensemble"].groupby("n_pos")["suppression"]
    q = ens.quantile([.025, .5, .975]).unstack() * 100
    a.fill_between(q.index, q[0.025], q[0.975], color=BLUE, alpha=.18, lw=0, label="Ensemble, 95% range")
    a.plot(q.index, q[0.5], "o-", color=BLUE, ms=4, mec="white", label="Ensemble, median")
    for i in range(5):
        s = d[(d.model == f"seed{i}") & (d.n_pos == 188)]["suppression"] * 100
        x = 188 + 7 + i * 4
        a.errorbar(x, s.median(), yerr=[[s.median() - s.quantile(.025)], [s.quantile(.975) - s.median()]],
                   fmt="o", ms=3, color=INK2, elinewidth=.8, capsize=1.5, label="Single seeds (n = 188)" if i == 0 else None)
    p = d[d.model == "ensemble"].groupby("n_pos")["suppression"].apply(lambda v: (v < .30).mean())
    for n, v in p.items():
        a.text(n, 3, f"{v:.0%}", ha="center", fontsize=6.8, color=INK2)
    a.text(50, 8, "share of draws with suppression < 30%", ha="left", fontsize=6.6, color=INK2)
    a.set_xlim(25, 220); a.set_ylim(0, 85)
    a.set_xticks([50, 100, 150, 188])
    a.set_xlabel("Positive calibration recordings (bootstrap)")
    a.set_ylabel("Validation false alarms suppressed (%)")
    a.set_title("(a) Benefit of the locked procedure, α = 0.02", loc="left")
    a.legend(loc="upper left", frameon=False, fontsize=6.8)

    for alpha, c, lab in ((.02, BLUE, "α = 0.02"), (.05, AQUA, "α = 0.05")):
        # n chosen so that alpha*(n+1) is an integer (upper envelope of the saw-tooth)
        k = np.arange(1, int(alpha * 2001) + 1)
        ns = np.round(k / alpha - 1).astype(int)
        keep = ns >= 49
        k, ns = k[keep], ns[keep]
        k = np.floor(alpha * (ns + 1) + 1e-9).astype(int)
        lo, md, hi = (beta.ppf(qq, k, ns + 1 - k) * 100 for qq in (.05, .5, .95))
        b.fill_between(ns, lo, hi, color=c, alpha=.16, lw=0)
        b.plot(ns, md, color=c, ls=("-" if alpha == 0.02 else (0, (5, 2))), label=f"{lab}: median, 5–95%")
        b.axhline(alpha * 100, color=c, lw=.7, ls=":")
    q188 = beta.ppf([.05, .5, .95], 3, 186) * 100
    b.errorbar(188, q188[1], yerr=[[q188[1] - q188[0]], [q188[2] - q188[1]]], fmt="D", ms=4,
               color=INK, capsize=2, elinewidth=.9, label="this study (n = 188, k* = 3)")
    b.axvline(188, color=INK2, lw=.8, ls=":")
    b.text(180, 9.3, "n = 188", fontsize=6.8, color=INK2, ha="right")
    b.set_xscale("log"); b.set_xlim(50, 2000); b.set_ylim(0, 10)
    b.set_xticks([50, 100, 188, 500, 1000, 2000]); b.set_xticklabels(["50", "100", "188", "500", "1000", "2000"])
    b.set_xlabel("Positive calibration recordings (log scale)")
    b.set_ylabel("Miss probability of deployed threshold (%)")
    b.set_title("(b) Beta approximation, binary record loss", loc="left")
    b.legend(loc="upper right", frameon=False, fontsize=6.8)
    fig.tight_layout()
    save(fig, "fig3_stability")


# ---------------------------------------------------------------- Fig. 4
def fig_prevalence():
    d = pd.read_parquet(OUT / "t06" / "A4_prevalence.parquet")
    cohorts = [("test", "VTaC test"), ("c2015", "Challenge 2015")]
    fig, axes = plt.subplots(2, 2, figsize=(COL2, 3.9), sharex=True)
    for j, (co, name) in enumerate(cohorts):
        for i, (col, ylab) in enumerate((("fnr", "FNR (%)"), ("p_true_given_suppressed", "P(true | suppressed) (%)"))):
            ax = axes[i, j]
            for meth, c in (("CRC", BLUE), ("LTT", ORANGE)):
                g = d[(d.cohort == co) & (d.method == meth)].groupby("target_prevalence")[col]
                q = g.quantile([.025, .5, .975]).unstack() * 100
                x = q.index * 100
                ax.fill_between(x, q[0.025], q[0.975], color=c, alpha=.16, lw=0)
                ax.plot(x, q[0.5], "o-" if meth == "CRC" else "s--", color=c, ms=3.5, mec="white",
                        label=("CRC, α = 0.02" if meth == "CRC" else "LTT, target 5%"))
            if col == "p_true_given_suppressed":
                ax.axhline(5, color=ORANGE, lw=.8, ls="--")
                ax.text(11, 5.6, "LTT target", fontsize=6.6, color=INK2)
            ax.set_ylim(0, 20 if col == "fnr" else 18)
            if i == 0:
                ax.set_title(f"{'(a)' if j == 0 else '(b)'} {name} (post-hoc)", loc="left")
            if j == 0:
                ax.set_ylabel(ylab)
            if i == 1:
                ax.set_xlabel("True VT prevalence after resampling (%)")
    axes[0, 0].legend(loc="upper left", frameon=False)
    fig.tight_layout()
    save(fig, "fig4_prevalence")


# ---------------------------------------------------------------- Fig. 5
def fig_profiles():
    tr = pd.read_parquet(OUT / "t06b" / "trained_cross_profile.parquet")
    pc = pd.read_parquet(OUT / "t06b" / "profile_crc.parquet")

    def get(df, **kw):
        m = np.ones(len(df), bool)
        for k, v in kw.items():
            m &= (df[k] == v).to_numpy()
        r = df[m]
        assert len(r) == 1, kw
        return r.iloc[0]

    rows = []
    for src, tgt in (("P1", "P2"), ("P2", "P1")):
        r = get(pc, method="cross_profile_crc", source_profile=src, evaluated_profile=tgt, cohort="val", alpha=0.05)
        rows.append(("Mixed", f"cal. {src} → val. {tgt}", r.fnr, r.fnr_ci_lo, r.fnr_ci_hi, BLUE, "o"))
    lo, hi = cp(3, 89)
    rows.append(("Mixed", "locked → C2015", 3 / 89, lo, hi, BLUE, "o"))
    for src, tgt, col, mk in (("P1", "P2", ORANGE, "s"), ("P2", "P1", AQUA, "^")):
        for co, lab in (("val", f"val. {tgt}"), ("c2015", "C2015")):
            r = get(tr, method="profile_only_train", source_profile=src, cohort=co, alpha=0.05, model="ensemble")
            rows.append((f"{src} only", f"cal. {src} → {lab}", r.fnr, r.fnr_ci_lo, r.fnr_ci_hi, col, mk))

    fig, ax = plt.subplots(figsize=(COL1, 2.7))
    y = np.arange(len(rows))[::-1]
    for yy, (grp, lab, f, l, h, c, mk) in zip(y, rows):
        ax.errorbar(f * 100, yy, xerr=[[(f - l) * 100], [(h - f) * 100]], fmt=mk, ms=(5 if mk == "^" else 4.5), color=c,
                    mec="white", mew=.9, elinewidth=1.0, capsize=2)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{g}: {lab}" for g, lab, *_ in rows], fontsize=6.8)
    ax.axvline(5, color=INK2, lw=.8, ls="--")
    ax.text(5.4, y[0] + .45, "α = 0.05", fontsize=6.8, color=INK2)
    ax.set_xlim(-1, 30); ax.set_xticks(range(0, 31, 5))
    ax.set_xlabel("FNR, true VT alarms suppressed (%), 95% CI")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "fig5_profiles")


if __name__ == "__main__":
    fig_pipeline(); fig_crc_ltt(); fig_stability(); fig_prevalence(); fig_profiles()
    print("FIGURES_OK", sorted(p.name for p in FIG.iterdir()))
