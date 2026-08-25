"""Generates the KS(rho) vs max(g) comparison chart embedded in
Kreisselmeier-Steinhauser.pptx. Standalone -- rerun after editing to
refresh optimization/artifacts/ks_vs_max.png before rebuilding the deck.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK_PRIMARY = "#1B1F2A"
INK_SECONDARY = "#555D6B"
INK_MUTED = "#898781"
ACCENT = "#2F6FED"
AXIS = "#C3C2B7"
DEFAULT_RHO = 0.15  # matches stress_ks_rho_mpa default used across the project


def ks_stress(sigma, rho):
    sigma = np.asarray(sigma, dtype=float)
    sigma_max = sigma.max()
    return sigma_max + np.log(np.sum(np.exp(rho * (sigma - sigma_max)))) / rho


def main():
    rng = np.random.default_rng(0)
    sigma = np.concatenate([
        rng.normal(60, 12, 400),   # bulk of the mesh, low stress
        rng.normal(150, 15, 20),   # a stress-concentration region
    ])
    sigma = np.clip(sigma, 5, None)
    sigma_max = sigma.max()

    x_lo, x_hi, label_x = 0.01, 3.0, 3.0
    rho_range = np.logspace(np.log10(x_lo), np.log10(x_hi), 200)
    ks_vals = np.array([ks_stress(sigma, r) for r in rho_range])
    ks_at_default = ks_stress(sigma, DEFAULT_RHO)

    fig, ax = plt.subplots(figsize=(11.0, 2.96), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(rho_range, ks_vals, color=ACCENT, linewidth=3, solid_capstyle="round", zorder=3)
    ax.axhline(sigma_max, color=INK_SECONDARY, linewidth=2, linestyle=(0, (4, 3)), zorder=2)

    ax.scatter([DEFAULT_RHO], [ks_at_default], s=36, color=ACCENT, zorder=4)

    ax.set_xscale("log")
    ax.set_xlim(x_lo, x_hi)
    ax.set_xticks([0.01, 0.1, 1])
    ymin, ymax = sigma_max - 5, ks_vals[0] * 1.02
    ax.set_ylim(ymin, ymax)

    ax.text(label_x, ks_vals[-1] + (ymax - ymin) * 0.10, "KS(ρ)",
            color=ACCENT, fontsize=18, fontweight="bold", ha="right", va="bottom")
    ax.text(label_x, sigma_max - (ymax - ymin) * 0.10, "max(gᵢ)",
            color=INK_SECONDARY, fontsize=18, fontweight="bold", ha="right", va="top")
    ax.text(DEFAULT_RHO, ks_at_default + (ymax - ymin) * 0.12, "ρ=0.15",
            color=INK_MUTED, fontsize=13, ha="center", va="bottom")

    ax.set_xlabel("ρ  (1/MPa)", color=INK_MUTED, fontsize=14, labelpad=6)
    ax.set_ylabel("MPa", color=INK_MUTED, fontsize=14, labelpad=6)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS)
        ax.spines[spine].set_linewidth(1.5)

    ax.tick_params(colors=INK_MUTED, labelsize=13, length=4)
    ax.grid(axis="y", color="#E1E0D9", linewidth=0.8, zorder=0)

    fig.tight_layout(pad=0.5)
    fig.savefig("optimization/artifacts/ks_vs_max.png", dpi=200, transparent=False)
    print("wrote optimization/artifacts/ks_vs_max.png")


if __name__ == "__main__":
    main()
