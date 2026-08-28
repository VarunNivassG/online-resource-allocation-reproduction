from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def plot_regret(df: pd.DataFrame, output: str | Path, title: str = "Regret vs Horizon") -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    grouped = df.groupby(["algorithm", "T"], as_index=False)["regret"].mean()
    for algorithm, g in grouped.groupby("algorithm"):
        g = g.sort_values("T")
        ax.plot(g["T"], g["regret"], marker="o", label=algorithm)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Horizon T")
    ax.set_ylabel("Regret")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_frequency(df: pd.DataFrame, output: str | Path) -> None:
    """
    Generate one frequency-vs-regret plot for each
    (distribution, T) combination.

    Each plot contains:
        - algorithm1
        - algorithm2
        - first_order

    Regret is averaged across seeds/trials.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    styles = {
    "algorithm1": {"linestyle": "-", "marker": "o"},
    "algorithm2": {"linestyle": "--", "marker": "s"},
    "first_order": {"linestyle": "-.", "marker": "^"},
    }

    # Average regret across seeds/trials
    grouped = (
        df.groupby(
            ["distribution", "T", "f", "algorithm"],
            as_index=False
        )["regret"]
        .mean()
    )

    # Generate one figure for each distribution + horizon
    for (distribution, T), g in grouped.groupby(
        ["distribution", "T"]
    ):
        fig, ax = plt.subplots(figsize=(8, 5))

        for algorithm in [
            "algorithm1",
            "algorithm2",
            "first_order",
        ]:
            h = g[g["algorithm"] == algorithm].sort_values("f")

            if h.empty:
                continue
            
            style = styles[algorithm]

            ax.plot(
                h["f"],
                h["regret"],
                linewidth=2,
                marker=style["marker"],
                linestyle=style["linestyle"],
                label=algorithm,
            )

        ax.set_xscale("log")
        ax.set_yscale("linear")

        ax.set_xlabel("Re-solving interval f")
        ax.set_ylabel("Mean Cumulative regret")

        ax.set_title(
            f"Regret vs Re-solving Frequency — "
            f"{distribution.capitalize()}, T={T}"
        )

        ax.grid(
            True,
            which="both",
            alpha=0.25
        )

        ax.legend()

        fig.tight_layout()

        filename = (
            f"frequency_"
            f"{distribution}_"
            f"T{T}.png"
        )

        fig.savefig(
            output / filename,
            dpi=180,
            bbox_inches="tight"
        )

        plt.close(fig)