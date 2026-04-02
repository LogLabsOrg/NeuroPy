"""
╔══════════════════════════════════════════════════════════════════════════╗
║               NeuroPy — Backend Visualización                           ║
║               Gráficas de métricas de entrenamiento                     ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

from typing import Dict, List, Optional

try:
    import matplotlib.pyplot as plt
    import matplotlib.style as mstyle
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════
# GRAFICADOR PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════

NEUROPY_COLORS = {
    "loss":          "#E8593C",
    "val_loss":      "#F0997B",
    "accuracy":      "#1D9E75",
    "val_accuracy":  "#5DCAA5",
    "f1":            "#7F77DD",
    "val_f1":        "#AFA9EC",
    "precision":     "#378ADD",
    "recall":        "#85B7EB",
    "default":       "#888780",
}


def plot_metrics(metrics: List[str], history: Dict[str, List[float]],
                 title: Optional[str] = None, save_path: Optional[str] = None):
    """
    Grafica las métricas del historial de entrenamiento.

    Args:
        metrics:   Lista de métricas a graficar (ej: ["loss", "accuracy"])
        history:   Diccionario con listas de valores por época
        title:     Título del gráfico
        save_path: Si se especifica, guarda la imagen en esa ruta
    """
    if not MPL_AVAILABLE:
        _print_ascii_chart(metrics, history)
        return

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 4))
    if n_metrics == 1:
        axes = [axes]

    fig.patch.set_facecolor("#1a1a2e")

    for ax, metric in zip(axes, metrics):
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="#c2c0b6")
        ax.spines[:].set_color("#444441")

        # Métrica de entrenamiento
        if metric in history:
            epochs = list(range(1, len(history[metric]) + 1))
            color  = NEUROPY_COLORS.get(metric, NEUROPY_COLORS["default"])
            ax.plot(epochs, history[metric], color=color,
                    linewidth=2, label=metric, marker="o", markersize=3)

        # Métrica de validación
        val_key = f"val_{metric}"
        if val_key in history:
            epochs  = list(range(1, len(history[val_key]) + 1))
            v_color = NEUROPY_COLORS.get(val_key, "#aaaaaa")
            ax.plot(epochs, history[val_key], color=v_color,
                    linewidth=2, linestyle="--", label=val_key, marker="s", markersize=3)

        ax.set_xlabel("Época", color="#c2c0b6", fontsize=10)
        ax.set_ylabel(metric.capitalize(), color="#c2c0b6", fontsize=10)
        ax.set_title(metric.capitalize(), color="#ffffff", fontsize=12, fontweight="bold")
        ax.legend(facecolor="#2a2a4e", labelcolor="#c2c0b6", fontsize=9)
        ax.grid(True, color="#333355", linewidth=0.5, alpha=0.5)

    overall_title = title or "NeuroPy — Métricas de entrenamiento"
    fig.suptitle(overall_title, color="#ffffff", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[NeuroPy] Gráfica guardada → {save_path}")

    plt.show()
    print(f"[NeuroPy] plot {', '.join(metrics)}")


def _print_ascii_chart(metrics: List[str], history: Dict[str, List[float]]):
    """Gráfica en ASCII para cuando matplotlib no está disponible."""
    print("\n[NeuroPy] plot (matplotlib no instalado — mostrando en texto)\n")
    for metric in metrics:
        if metric not in history:
            continue
        values = history[metric]
        print(f"  {metric.upper()}")
        max_v = max(values) if values else 1
        min_v = min(values) if values else 0
        rng   = max_v - min_v or 1
        rows  = 8
        for row in range(rows, -1, -1):
            threshold = min_v + (row / rows) * rng
            line = f"  {threshold:6.3f} │"
            for v in values:
                line += "█" if v >= threshold else " "
            print(line)
        print("         └" + "─" * len(values))
        print(f"          {'Épocas':^{len(values)}}\n")
