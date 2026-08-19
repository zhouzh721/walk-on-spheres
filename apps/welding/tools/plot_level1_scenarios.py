#!/usr/bin/env python3
"""Create two Chinese Level-1 scenario overview figures from solver CSV output."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
import numpy as np


APP_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = APP_DIR / "results" / "data"
FIGURE_DIR = APP_DIR / "results" / "figures"

INK = "#172033"
MUTED = "#657083"
GRID = "#DDE3EC"
BLUE = "#2F6B9A"
ORANGE = "#D8752B"
TEAL = "#24827A"
CARD = "#F4F7FB"
TIME_CARD = "#E8F3F1"


def load_csv(name: str) -> np.ndarray:
    path = DATA_DIR / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True, dtype=float))


def load_summary(name: str) -> np.void:
    path = DATA_DIR / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.genfromtxt(path, delimiter=",", names=True, dtype=float)


def as_grid(field: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.unique(field["x"])
    y = np.unique(field["y"])
    values = np.full((x.size, y.size), np.nan, dtype=float)
    values[np.searchsorted(x, field["x"]), np.searchsorted(y, field["y"])] = field[name]
    if np.isnan(values).any():
        raise ValueError(f"incomplete grid for {name}")
    return x, y, values


def configure_style() -> None:
    font_path = Path("/mnt/c/Windows/Fonts/msyh.ttc")
    if font_path.is_file():
        font_manager.fontManager.addfont(font_path)
        chinese_font = font_manager.FontProperties(fname=font_path).get_name()
    else:
        chinese_font = "DejaVu Sans"
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [chinese_font, "DejaVu Sans", "sans-serif"],
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 9,
            "axes.labelcolor": INK,
            "axes.edgecolor": "#9AA5B5",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(-0.08, 1.08, label, transform=axis.transAxes, fontsize=15,
              fontweight="bold", va="top", ha="left", color=INK)


def style_spatial_axis(axis: plt.Axes) -> None:
    axis.set_xlabel("x（mm）")
    axis.set_ylabel("y（mm）")
    axis.set_aspect("equal")
    axis.tick_params(length=3, width=0.7)
    for spine in axis.spines.values():
        spine.set_linewidth(0.8)


def add_colorbar(fig: plt.Figure, image, axis: plt.Axes, label: str) -> None:
    colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.035)
    colorbar.ax.set_title(label, color=MUTED, fontsize=8, pad=6)
    colorbar.outline.set_linewidth(0.6)
    colorbar.ax.tick_params(length=2.5, width=0.6)


def condition_card(axis: plt.Axes, title: str, lines: list[str]) -> None:
    """Draw problem parameters in a dedicated panel, never over the heatmap."""
    axis.set_axis_off()
    axis.add_patch(
        FancyBboxPatch(
            (0.01, 0.02), 0.98, 0.96,
            boxstyle="round,pad=0.02,rounding_size=0.035",
            transform=axis.transAxes, linewidth=0.8,
            edgecolor="#D7DEE9", facecolor=CARD,
        )
    )
    axis.text(0.06, 0.86, title, transform=axis.transAxes, fontsize=10,
              fontweight="bold", va="top", color=INK)
    axis.text(0.06, 0.68, "\n".join(lines), transform=axis.transAxes,
              fontsize=7.7, linespacing=1.30, va="top", color=MUTED)


def metric_card(axis: plt.Axes, title: str, value: str, detail: str,
                accent: str = BLUE, facecolor: str = CARD) -> None:
    axis.set_axis_off()
    axis.add_patch(
        FancyBboxPatch(
            (0.01, 0.05), 0.98, 0.90,
            boxstyle="round,pad=0.015,rounding_size=0.035",
            transform=axis.transAxes, linewidth=0.8,
            edgecolor="#D7DEE9", facecolor=facecolor,
        )
    )
    axis.add_patch(
        FancyBboxPatch(
            (0.06, 0.79), 0.18, 0.045,
            boxstyle="round,pad=0.0,rounding_size=0.02",
            transform=axis.transAxes, linewidth=0, facecolor=accent,
        )
    )
    axis.text(0.06, 0.68, title, transform=axis.transAxes, color=MUTED,
              fontsize=9, va="top")
    axis.text(0.06, 0.43, value, transform=axis.transAxes, color=INK,
              fontsize=18, fontweight="bold", va="center")
    axis.text(0.06, 0.15, detail, transform=axis.transAxes, color=MUTED,
              fontsize=7.7, va="bottom")


def make_canvas(title: str, subtitle: str, badge: str):
    fig = plt.figure(figsize=(15.6, 8.775), facecolor="white")
    layout = fig.add_gridspec(
        4, 10, height_ratios=(0.12, 0.38, 0.27, 0.23),
        left=0.04, right=0.98, bottom=0.045, top=0.975,
        wspace=0.86, hspace=0.58,
    )
    title_axis = fig.add_subplot(layout[0, :])
    title_axis.set_axis_off()
    title_axis.text(0.0, 0.76, title, fontsize=22, fontweight="bold",
                    va="center", ha="left", color=INK)
    title_axis.text(0.0, 0.20, subtitle, fontsize=10.5,
                    va="center", ha="left", color=MUTED)
    title_axis.text(
        1.0, 0.64, badge, transform=title_axis.transAxes,
        ha="right", va="center", fontsize=10, fontweight="bold", color=TEAL,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#E8F3F1",
              "edgecolor": "#B8D9D4", "linewidth": 0.8},
    )
    condition_layout = layout[1, 0:2].subgridspec(
        2, 1, height_ratios=(0.54, 0.46), hspace=0.52,
    )
    condition_axis = fig.add_subplot(condition_layout[0, 0])
    source_axis = fig.add_subplot(condition_layout[1, 0])
    map_axes = [fig.add_subplot(layout[1, start:start + 2])
                for start in (2, 4, 6, 8)]
    line_axes = [fig.add_subplot(layout[2, 0:5]),
                 fig.add_subplot(layout[2, 5:10])]
    metric_layout = layout[3, :].subgridspec(1, 5, wspace=0.18)
    metric_axes = [fig.add_subplot(metric_layout[0, index]) for index in range(5)]
    return fig, condition_axis, source_axis, map_axes, line_axes, metric_axes


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    base = FIGURE_DIR / stem
    png_path = base.with_suffix(".png")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    print(png_path)
    plt.close(fig)


def line_comparison(axis: plt.Axes, coordinate: np.ndarray,
                    wos_values: np.ndarray, reference_values: np.ndarray,
                    standard_error: np.ndarray, title: str, xlabel: str,
                    reference_label: str) -> None:
    lower = wos_values - 1.96 * standard_error
    upper = wos_values + 1.96 * standard_error
    axis.fill_between(coordinate, lower, upper, color=BLUE, alpha=0.17,
                      linewidth=0, label="WoS 95% Monte Carlo 区间")
    axis.plot(coordinate, reference_values, color=ORANGE, linewidth=2.0,
              label=reference_label)
    axis.plot(coordinate, wos_values, "o-", color=BLUE, linewidth=1.3,
              markersize=2.8, markevery=2, label="WoS")
    axis.set_title(title, loc="left", pad=9)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("温升 ΔT（K）")
    axis.set_xlim(float(coordinate.min()), float(coordinate.max()))
    axis.grid(True, color=GRID, linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18),
                ncol=3, frameon=False, fontsize=8)


def plot_manufactured() -> None:
    field = load_csv("level1_manufactured_field.csv")
    summary = load_summary("level1_manufactured_summary.csv")
    x, y, source = as_grid(field, "source")
    _, _, wos_field = as_grid(field, "temperature_rise")
    _, _, exact = as_grid(field, "exact")
    _, _, error = as_grid(field, "error")
    _, _, standard_error = as_grid(field, "standard_error")
    x_mm = 1000.0 * x
    y_mm = 1000.0 * y

    fig, condition_axis, source_axis, map_axes, line_axes, cards = make_canvas(
        "Level 1A｜制造解稳态 Poisson 验证",
        "零温升边界 + 已知正弦体热源；用解析解检验 WoS 的场精度与统计误差",
        "解析解验证 · PASS",
    )

    condition_card(
        condition_axis, "问题设置（稳态，无时间初值）",
        ["板面 120 × 80 mm；网格 49 × 33", "四边界：ΔT = 0 K",
         "每点 6000 次随机游走；ε = 8 μm", "固定随机种子：20260818"],
    )
    normalized_source = source / np.max(np.abs(source))
    source_image = source_axis.pcolormesh(
        x_mm, y_mm, normalized_source.T, shading="auto", cmap="YlOrRd",
        vmin=0.0, vmax=1.0,
    )
    source_axis.set_title("归一化体热源", fontsize=10, pad=4)
    style_spatial_axis(source_axis)
    panel_label(source_axis, "a")
    add_colorbar(fig, source_image, source_axis, "f / fmax")

    maximum = max(float(wos_field.max()), float(exact.max()))
    error_limit = max(float(np.max(np.abs(error))), np.finfo(float).eps)
    map_specs = (
        (wos_field, "WoS 温升场", "inferno", 0.0, maximum, "温升（K）"),
        (exact, "解析解", "inferno", 0.0, maximum, "温升（K）"),
        (error, "误差：WoS − 解析解", "RdBu_r", -error_limit, error_limit,
         "误差（K）"),
        (standard_error, "WoS 标准误差", "magma", 0.0, None, "SE（K）"),
    )
    for index, (values, title, cmap, vmin, vmax, label) in enumerate(map_specs):
        axis = map_axes[index]
        image = axis.pcolormesh(x_mm, y_mm, values.T, shading="auto",
                                cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(title)
        style_spatial_axis(axis)
        panel_label(axis, chr(ord("b") + index))
        add_colorbar(fig, image, axis, label)

    centre_i = x.size // 2
    centre_j = y.size // 2
    line_comparison(line_axes[0], x_mm, wos_field[:, centre_j],
                    exact[:, centre_j], standard_error[:, centre_j],
                    "纵向中心线｜y = 40 mm", "x（mm）", "解析解")
    line_comparison(line_axes[1], y_mm, wos_field[centre_i, :],
                    exact[centre_i, :], standard_error[centre_i, :],
                    "横向中心线｜x = 60 mm", "y（mm）", "解析解")
    panel_label(line_axes[0], "f")
    panel_label(line_axes[1], "g")

    metric_card(cards[0], "中心温升", f"{float(summary['center_mean']):.2f} K",
                "解析值 100.00 K", BLUE)
    metric_card(cards[1], "全场 RMSE", f"{float(summary['field_rmse']):.3f} K",
                "内部网格点；越低越好", ORANGE)
    metric_card(cards[2], "最大绝对误差",
                f"{float(summary['field_max_absolute_error']):.3f} K",
                "全场最不利点", ORANGE)
    metric_card(cards[3], "中心标准误",
                f"{float(summary['center_standard_error']):.3f} K",
                "Monte Carlo 标准误", TEAL)
    metric_card(cards[4], "求解耗时", f"{float(summary['solve_seconds']):.3f} s",
                "仅计 WoS 场求解；单 MPI rank", TEAL, TIME_CARD)

    fig.text(0.98, 0.012, "数据：固定种子 20260818；全部网格点均用于场误差与标准误差展示",
             ha="right", va="bottom", fontsize=7.5, color=MUTED)
    save_figure(fig, "level1a_manufactured_scenario")


def plot_gaussian(
    field_name: str = "level1_gaussian_field.csv",
    summary_name: str = "level1_gaussian_summary.csv",
    validation_name: str = "level1_gaussian_validation.csv",
    figure_stem: str = "level1b_gaussian_scenario",
    title: str = "Level 1B｜固定 Gaussian 焊接热源",
    subtitle: str = (
        "有限薄板上的定常局部热输入；功率归一化后，用有限差分参考场验证 WoS"
    ),
    badge: str = "全部验证门限 · PASS",
    condition_title: str = "问题设置（稳态，无时间初值）",
    baseline_summary: np.void | None = None,
    baseline_validation: np.void | None = None,
) -> None:
    field = load_csv(field_name)
    reference = load_csv("level1_gaussian_reference.csv")
    summary = load_summary(summary_name)
    validation = load_summary(validation_name)
    x, y, heat_source = as_grid(field, "volumetric_power_density")
    _, _, wos_rise = as_grid(field, "temperature_rise")
    _, _, standard_error = as_grid(field, "standard_error")
    _, _, reference_rise = as_grid(reference, "reference_temperature_rise")
    error = wos_rise - reference_rise
    x_mm = 1000.0 * x
    y_mm = 1000.0 * y
    fig, condition_axis, source_axis, map_axes, line_axes, cards = make_canvas(
        title,
        subtitle,
        badge,
    )

    condition_card(
        condition_axis, condition_title,
        ["板 120 × 80 × 6 mm；k = 35 W/(m·K)",
         "四边界：T = 293.15 K", "电功率 800 W；效率 75%；吸收 600 W",
         "中心 (60, 40) mm；σx = σy = 6 mm"],
    )
    source_image = source_axis.pcolormesh(
        x_mm, y_mm, (heat_source / 1e6).T, shading="auto", cmap="YlOrRd",
        vmin=0.0,
    )
    source_axis.plot(60.0, 40.0, marker="x", color="#083B66",
                     markersize=8, markeredgewidth=1.5)
    source_axis.set_title("Gaussian 体热源", fontsize=10, pad=4)
    style_spatial_axis(source_axis)
    panel_label(source_axis, "a")
    add_colorbar(fig, source_image, source_axis, "q'''（MW/m³）")

    maximum = max(float(wos_rise.max()), float(reference_rise.max()))
    error_limit = max(float(np.max(np.abs(error))), np.finfo(float).eps)
    map_specs = (
        (wos_rise, "WoS 温升场", "inferno", 0.0, maximum, "温升（K）"),
        (reference_rise, "有限差分参考场", "inferno", 0.0, maximum, "温升（K）"),
        (error, "误差：WoS − FD", "RdBu_r", -error_limit, error_limit,
         "误差（K）"),
        (standard_error, "WoS 标准误差", "magma", 0.0, None, "SE（K）"),
    )
    for index, (values, title, cmap, vmin, vmax, label) in enumerate(map_specs):
        axis = map_axes[index]
        image = axis.pcolormesh(x_mm, y_mm, values.T, shading="auto",
                                cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(title)
        style_spatial_axis(axis)
        panel_label(axis, chr(ord("b") + index))
        add_colorbar(fig, image, axis, label)

    centre_i = x.size // 2
    centre_j = y.size // 2
    line_comparison(line_axes[0], x_mm, wos_rise[:, centre_j],
                    reference_rise[:, centre_j], standard_error[:, centre_j],
                    "纵向中心线｜y = 40 mm", "x（mm）", "有限差分参考")
    line_comparison(line_axes[1], y_mm, wos_rise[centre_i, :],
                    reference_rise[centre_i, :], standard_error[centre_i, :],
                    "横向中心线｜x = 60 mm", "y（mm）", "有限差分参考")
    panel_label(line_axes[0], "f")
    panel_label(line_axes[1], "g")

    peak_detail = f"峰值温升 {float(summary['peak_temperature_rise']):.2f} K"
    rmse_detail = f"绝对 RMSE {float(validation['field_rmse']):.03f} K"
    peak_se_detail = "Monte Carlo 标准误"
    time_detail = "仅计 WoS 场求解；单 MPI rank"
    if baseline_summary is not None and baseline_validation is not None:
        se_reduction = 100.0 * (
            1.0
            - float(summary["peak_standard_error"])
            / float(baseline_summary["peak_standard_error"])
        )
        time_increase = 100.0 * (
            float(summary["solve_seconds"])
            / float(baseline_summary["solve_seconds"])
            - 1.0
        )
        peak_detail += "；峰值位于热源中心"
        rmse_detail += (
            f"；Green {float(baseline_validation['field_rmse']):.03f} K"
        )
        peak_se_detail = (
            f"Green {float(baseline_summary['peak_standard_error']):.2f} K；"
            f"降低 {se_reduction:.1f}%"
        )
        time_detail = (
            f"Green {float(baseline_summary['solve_seconds']):.3f} s；"
            f"增加 {time_increase:.1f}%"
        )

    metric_card(cards[0], "峰值温度", f"{float(summary['peak_temperature']):.2f} K",
                peak_detail, BLUE)
    metric_card(cards[1], "全场归一化 RMSE",
                f"{100.0 * float(validation['field_normalized_rmse']):.3f}%",
                rmse_detail, ORANGE)
    metric_card(cards[2], "峰值标准误",
                f"{float(summary['peak_standard_error']):.2f} K",
                peak_se_detail, TEAL)
    metric_card(cards[3], "功率守恒误差",
                f"{float(summary['power_relative_error']):.2e}",
                "积分功率 600.000 W", ORANGE)
    metric_card(cards[4], "求解耗时", f"{float(summary['solve_seconds']):.3f} s",
                time_detail, TEAL, TIME_CARD)

    fig.text(0.98, 0.012,
             f"网格 49 × 33；每点 6000 次随机游走；3SE 覆盖率 "
             f"{100.0 * float(validation['within_three_standard_errors_fraction']):.2f}%",
             ha="right", va="bottom", fontsize=7.5, color=MUTED)
    save_figure(fig, figure_stem)


def plot_gaussian_mis() -> None:
    baseline_summary = load_summary("level1_gaussian_summary.csv")
    baseline_validation = load_summary("level1_gaussian_validation.csv")
    plot_gaussian(
        field_name="level1_gaussian_mis_field.csv",
        summary_name="level1_gaussian_mis_summary.csv",
        validation_name="level1_gaussian_mis_validation.csv",
        figure_stem="level1b_gaussian_mis_scenario",
        title="Level 1B｜固定 Gaussian 热源 · Green–Source MIS",
        subtitle=(
            "全域 Gaussian 源 proposal 与球内 Green proposal 混合；"
            "在保持无偏的同时降低温度场 Monte Carlo 方差"
        ),
        badge="MIS 验证门限 · PASS",
        condition_title="问题设置｜Green–Source MIS（β = 0.5）",
        baseline_summary=baseline_summary,
        baseline_validation=baseline_validation,
    )


def main() -> None:
    configure_style()
    plot_manufactured()
    plot_gaussian()
    plot_gaussian_mis()


if __name__ == "__main__":
    main()
