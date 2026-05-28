"""Generates PNG plots from preprocessed directivity visualization data."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap


@dataclass
class VisualizerConfig:
    input_npz: Path = Path(
        "pressure_data_formatted.npz"
    )
    output_horizontal_png: Path = Path("horizontal_isobar.png")
    output_vertical_png: Path = Path("vertical_isobar.png")
    output_impedance_png: Path = Path("acoustic_impedance.png")
    colorbar_tick_step_db: float = 3.0
    figure_width_in: float = 11.0
    figure_height_in: float = 6.0
    figure_dpi: int = 160
    isobar_interp_angle_factor: int = 2
    isobar_interp_freq_factor: int = 3
    custom_colors: tuple[str, ...] = (
        "#00008F",
        "#0000FF",
        "#006FFF",
        "#00DFFF",
        "#4FFFBF",
        "#BFFF4F",
        "#FFDF00",
        "#FF6F00",
        "#FF0000",
        "#8F0000",
    )

    def __init__(self, args):
        # self.colorbar_tick_step_db = args.colorbar_tick_step_db
        # self.figure_width_in = args.figure_width_in
        # self.figure_height_in = args.figure_height_in
        # self.figure_dpi = args.figure_dpi
        # self.isobar_interp_angle_factor = args.isobar_interp_angle_factor
        self.isobar_interp_freq_factor = args.isobar_interp_freq_factor
        self.output_horizontal_png = Path(args.horizontal_isobar_output) # Path("output/horizontal_isobar.png")
        self.output_vertical_png = Path(args.vertical_isobar_output)     # Path("output/vertical_isobar.png")
        self.output_impedance_png = Path(args.acoustic_impedance_output) # Path("output/acoustic_impedance.png")


class Visualizer(object):
    def __init__(self, conf: VisualizerConfig):
        self.conf = conf

    def load_data(self, prepped_data_file: str) -> dict[str, np.ndarray]:
        npz_path = Path(prepped_data_file)
        if not npz_path.exists():
            raise FileNotFoundError(f"File not found: {npz_path}")

        with np.load(npz_path) as data:
            required = {
                "isobar_angle_deg",
                "isobar_freq_hz",
                "horizontal_isobar_db",
                "vertical_isobar_db",
                "impedance_freq_hz",
                "impedance_real",
                "impedance_imag",
                "clip_min_db",
                "clip_max_db",
            }
            missing = required - set(data.files)
            if missing:
                raise ValueError(f"Visualization NPZ missing arrays: {sorted(missing)}")

            return { k: data[k] for k in data.files }

    def __build_db_tick_values(self, min_db: float, max_db: float, step_db: float) -> np.ndarray:
        if step_db <= 0:
            raise ValueError("colorbar_tick_step_db must be > 0")

        start = np.ceil(min_db / step_db) * step_db
        end = np.floor(max_db / step_db) * step_db

        if end < start:
            return np.array([min_db, max_db], dtype=float)

        return np.arange(start, end + 0.5 * step_db, step_db, dtype=float)

    def __setup_log_frequency_axis(self, ax: plt.Axes) -> None:
        tickvals = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
        ticktext = ["20", "50", "100", "200", "500", "1k", "2k", "5k", "10k", "20k"]

        ax.set_xscale("log")
        ax.set_xlim(200, 20000)
        ax.set_xticks(tickvals)
        ax.set_xticklabels(ticktext)

    def __upsample_isobar_grid(
        self,
        angle_deg: np.ndarray,
        freqs_hz: np.ndarray,
        spl_matrix: np.ndarray,
        angle_factor: int,
        freq_factor: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if angle_factor < 1 or freq_factor < 1:
            raise ValueError("Interpolation factors must be >= 1")

        if angle_factor == 1 and freq_factor == 1:
            return angle_deg, freqs_hz, spl_matrix

        n_angles = angle_deg.shape[0]
        n_freqs = freqs_hz.shape[0]

        dense_angle_count = (n_angles - 1) * angle_factor + 1
        dense_freq_count = (n_freqs - 1) * freq_factor + 1

        dense_angles = np.linspace(float(angle_deg[0]), float(angle_deg[-1]), dense_angle_count)
        dense_log_freqs = np.linspace(np.log10(float(freqs_hz[0])), np.log10(float(freqs_hz[-1])), dense_freq_count)
        source_log_freqs = np.log10(freqs_hz)

        freq_upsampled = np.empty((n_angles, dense_freq_count), dtype=float)
        for row_idx in range(n_angles):
            freq_upsampled[row_idx, :] = np.interp(dense_log_freqs, source_log_freqs, spl_matrix[row_idx, :])

        dense_spl = np.empty((dense_angle_count, dense_freq_count), dtype=float)
        for col_idx in range(dense_freq_count):
            dense_spl[:, col_idx] = np.interp(dense_angles, angle_deg, freq_upsampled[:, col_idx])

        return dense_angles, np.power(10.0, dense_log_freqs), dense_spl

    def __save_isobar_plot(
        self,
        output_path: Path,
        angle_deg: np.ndarray,
        freqs_hz: np.ndarray,
        spl_matrix: np.ndarray,
        title: str,
        colors: tuple[str, ...],
        clip_min_db: float,
        clip_max_db: float,
        colorbar_tick_step_db: float,
        figure_width_in: float,
        figure_height_in: float,
        figure_dpi: int,
        isobar_interp_angle_factor: int,
        isobar_interp_freq_factor: int,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        boundaries = np.linspace(clip_min_db, clip_max_db, len(colors) + 1)
        cmap = ListedColormap(list(colors))
        norm = BoundaryNorm(boundaries, cmap.N)
        cbar_ticks = self.__build_db_tick_values(clip_min_db, clip_max_db, colorbar_tick_step_db)

        plot_angles, plot_freqs, plot_spl = self.__upsample_isobar_grid(
            angle_deg=angle_deg,
            freqs_hz=freqs_hz,
            spl_matrix=spl_matrix,
            angle_factor=isobar_interp_angle_factor,
            freq_factor=isobar_interp_freq_factor,
        )
        plot_spl = np.clip(plot_spl, clip_min_db, clip_max_db)

        fig, ax = plt.subplots(figsize=(figure_width_in, figure_height_in), dpi=figure_dpi)
        mesh = ax.pcolormesh(plot_freqs, plot_angles, plot_spl, cmap=cmap, norm=norm, shading="gouraud")

        self.__setup_log_frequency_axis(ax)
        ax.set_ylim(-180, 180)
        ax.set_yticks(np.arange(-180, 181, 30))
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Angle (deg)")
        ax.set_title(title)
        ax.grid(which="major", color="#808080", linewidth=0.8)

        cbar = fig.colorbar(mesh, ax=ax)
        cbar.set_label("Normalized SPL (dB)")
        cbar.set_ticks(cbar_ticks)

        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)

    def __save_impedance_plot(
        self,
        output_path: Path,
        impedance_freq_hz: np.ndarray,
        impedance_real: np.ndarray,
        impedance_imag: np.ndarray,
        figure_width_in: float,
        figure_height_in: float,
        figure_dpi: int,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(figure_width_in, figure_height_in), dpi=figure_dpi)

        ax.plot(impedance_freq_hz, impedance_real, linewidth=2, label="Z real")
        ax.plot(impedance_freq_hz, impedance_imag, linewidth=2, linestyle="--", label="Z imag")

        self.__setup_log_frequency_axis(ax)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Acoustic Impedance (Pa·s/m³)")
        ax.set_title("Acoustic Impedance")
        ax.grid(which="major", color="#808080", linewidth=0.8)
        ax.legend()

        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)

    def generate_plots(self, dataset: dict[str, np.ndarray]) -> dict[str, str]:
        cfg = self.conf
        angle_deg = dataset["isobar_angle_deg"].astype(float)
        freqs_hz = dataset["isobar_freq_hz"].astype(float)
        horizontal_spl = dataset["horizontal_isobar_db"].astype(float)
        vertical_spl = dataset["vertical_isobar_db"].astype(float)
        impedance_freq_hz = dataset["impedance_freq_hz"].astype(float)
        impedance_real = dataset["impedance_real"].astype(float)
        impedance_imag = dataset["impedance_imag"].astype(float)

        clip_min_db = float(dataset["clip_min_db"])
        clip_max_db = float(dataset["clip_max_db"])

        self.__save_isobar_plot(
            output_path=cfg.output_horizontal_png,
            angle_deg=angle_deg,
            freqs_hz=freqs_hz,
            spl_matrix=horizontal_spl,
            title="Horizontal Isobar",
            colors=cfg.custom_colors,
            clip_min_db=clip_min_db,
            clip_max_db=clip_max_db,
            colorbar_tick_step_db=cfg.colorbar_tick_step_db,
            figure_width_in=cfg.figure_width_in,
            figure_height_in=cfg.figure_height_in,
            figure_dpi=cfg.figure_dpi,
            isobar_interp_angle_factor=cfg.isobar_interp_angle_factor,
            isobar_interp_freq_factor=cfg.isobar_interp_freq_factor,
        )

        self.__save_isobar_plot(
            output_path=cfg.output_vertical_png,
            angle_deg=angle_deg,
            freqs_hz=freqs_hz,
            spl_matrix=vertical_spl,
            title="Vertical Isobar",
            colors=cfg.custom_colors,
            clip_min_db=clip_min_db,
            clip_max_db=clip_max_db,
            colorbar_tick_step_db=cfg.colorbar_tick_step_db,
            figure_width_in=cfg.figure_width_in,
            figure_height_in=cfg.figure_height_in,
            figure_dpi=cfg.figure_dpi,
            isobar_interp_angle_factor=cfg.isobar_interp_angle_factor,
            isobar_interp_freq_factor=cfg.isobar_interp_freq_factor,
        )

        self.__save_impedance_plot(
            output_path=cfg.output_impedance_png,
            impedance_freq_hz=impedance_freq_hz,
            impedance_real=impedance_real,
            impedance_imag=impedance_imag,
            figure_width_in=cfg.figure_width_in,
            figure_height_in=cfg.figure_height_in,
            figure_dpi=cfg.figure_dpi,
        )

        return {
            "horizontal_isobar_png": str(cfg.output_horizontal_png),
            "vertical_isobar_png": str(cfg.output_vertical_png),
            "acoustic_impedance_png": str(cfg.output_impedance_png),
        }
