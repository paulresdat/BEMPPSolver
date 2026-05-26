from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from utils.log import Log


@dataclass
class PrepConfig:
    input_polar_npz: Path = Path("pressure_data.npz")
    output_npz: Path = Path("pressure_data_formatted.npz")
    # these values are defaulted as constants and are overwritten, these are here for posterity
    # best not to change these original values, they represent the intended starting point.
    min_db: float = -30.0   #minimum dB for clipping SPL data
    max_db: float = 0.0     #maximum dB for clipping SPL data
    isobar_angle_samples_smooth: int = 250
    isobar_freq_samples_smooth: int = 500
    isobar_octave_smooth_fraction: int | float | None = 24  #fractional octave smoothing for plots
    horizontal_reference_angle_deg: float = 10              #normalization angle for horizontal plane
    vertical_reference_angle_deg: float = 10                #normalization angle for vertical plane

    def __init__(self, args):
        self.input_polar_npz = Path(args.input_polar_npz + ".npz")
        self.output_npz = Path(args.output_npz + ".npz")
        self.min_db = args.min_db
        self.max_db = args.max_db
        self.isobar_angle_samples_smooth = args.isobar_angle_samples_smooth
        self.isobar_freq_samples_smooth = args.isobar_freq_samples_smooth
        self.isobar_octave_smooth_fraction = args.isobar_octave_smooth_fraction
        self.horizontal_reference_angle_deg = args.horizontal_reference_angle_deg
        self.vertical_reference_angle_deg = args.vertical_reference_angle_deg


class VisualizationPrep(object):
    def __init__(self, config: PrepConfig, log: Log):
        self.conf = config
        self.log = log

    def __load_polar_npz(self, file_path: Path) -> dict[str, np.ndarray]:
        if not file_path.exists():
            raise FileNotFoundError(f"Polar file not found: {file_path}")

        with np.load(file_path) as data:
            required = {
                "freq_hz",
                "polar_angle_deg",
                "horizontal_spl_norm_db",
                "vertical_spl_norm_db",
                "impedance_freq_hz",
                "impedance_real",
                "impedance_imag",
            }
            missing = required - set(data.files)
            if missing:
                raise ValueError(f"Polar NPZ missing arrays: {sorted(missing)}")

            freq_hz = np.asarray(data["freq_hz"], dtype=float)
            angles_deg = np.asarray(data["polar_angle_deg"], dtype=float)
            horizontal = np.asarray(data["horizontal_spl_norm_db"], dtype=float)
            vertical = np.asarray(data["vertical_spl_norm_db"], dtype=float)
            z_freq = np.asarray(data["impedance_freq_hz"], dtype=float)
            z_real = np.asarray(data["impedance_real"], dtype=float)
            z_imag = np.asarray(data["impedance_imag"], dtype=float)

        if horizontal.ndim != 2 or vertical.ndim != 2:
            raise ValueError("horizontal/vertical SPL arrays must be 2D (n_freq, n_angles).")
        if horizontal.shape != vertical.shape:
            raise ValueError("horizontal_spl_norm_db and vertical_spl_norm_db must have matching shapes.")
        if horizontal.shape[0] != freq_hz.size:
            raise ValueError("horizontal/vertical SPL first axis must match freq_hz length.")
        if horizontal.shape[1] != angles_deg.size:
            raise ValueError("polar_angle_deg length must match SPL second axis.")
        if z_freq.size != z_real.size or z_real.size != z_imag.size:
            raise ValueError("Impedance arrays must have matching lengths.")

        return {
            "freq_hz": freq_hz,
            "polar_angle_deg": angles_deg,
            "horizontal_spl_norm_db": horizontal,
            "vertical_spl_norm_db": vertical,
            "impedance_freq_hz": z_freq,
            "impedance_real": z_real,
            "impedance_imag": z_imag,
        }


    def __fractional_octave_smooth(
        self,
        spl_matrix: np.ndarray,
        freqs: np.ndarray,
        fraction: int | float | None,
    ) -> np.ndarray:
        if not fraction or fraction <= 0:
            return spl_matrix

        if freqs.ndim != 1 or freqs.size < 2:
            return spl_matrix

        log2_freqs = np.log2(freqs)
        half_band = 1.0 / (2.0 * float(fraction))

        smoothed = np.empty_like(spl_matrix)
        for i in range(freqs.size):
            mask = np.abs(log2_freqs - log2_freqs[i]) <= half_band
            smoothed[:, i] = np.mean(spl_matrix[:, mask], axis=1)
        return smoothed


    def __normalize_plane_to_reference_angle(
        self,
        spl_matrix: np.ndarray,
        angles_deg: np.ndarray,
        reference_angle_deg: float,
    ) -> np.ndarray:
        if spl_matrix.ndim != 2:
            raise ValueError("spl_matrix must be 2D with shape (n_angles, n_freq).")

        if angles_deg.ndim != 1 or angles_deg.size != spl_matrix.shape[0]:
            raise ValueError("angles_deg must be 1D and match spl_matrix first axis.")

        if angles_deg.size < 2:
            return spl_matrix

        reference_wrapped = ((float(reference_angle_deg) + 180.0) % 360.0) - 180.0

        if np.isclose(angles_deg[0], -180.0) and np.isclose(angles_deg[-1], 180.0):
            interp_angles = angles_deg[:-1]
            interp_matrix = spl_matrix[:-1, :]
        else:
            interp_angles = angles_deg
            interp_matrix = spl_matrix

        if interp_angles.size < 2:
            return spl_matrix

        angles_ext = np.concatenate([interp_angles - 360.0, interp_angles, interp_angles + 360.0])
        out = np.empty_like(spl_matrix)

        for i in range(spl_matrix.shape[1]):
            values = interp_matrix[:, i]
            values_ext = np.concatenate([values, values, values])
            ref_db = np.interp(reference_wrapped, angles_ext, values_ext)
            out[:, i] = spl_matrix[:, i] - ref_db

        return out


    def __interpolate_isobar_heatmap(
        self,
        angles_deg: np.ndarray,
        freqs: np.ndarray,
        spl_matrix: np.ndarray,
        angle_samples: int | None,
        freq_samples: int | None,
        fill_db: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if angle_samples is None and freq_samples is None:
            return angles_deg, freqs, spl_matrix

        log_freqs = np.log10(freqs)
        target_angles = np.linspace(angles_deg.min(), angles_deg.max(), angle_samples or angles_deg.size)
        target_log_freqs = np.linspace(log_freqs.min(), log_freqs.max(), freq_samples or freqs.size)

        interpolator = RegularGridInterpolator(
            (angles_deg, log_freqs),
            spl_matrix,
            bounds_error=False,
            fill_value=fill_db,
        )

        angle_grid, log_freq_grid = np.meshgrid(target_angles, target_log_freqs, indexing="ij")
        query = np.column_stack([angle_grid.ravel(), log_freq_grid.ravel()])
        spl_interp = interpolator(query).reshape(angle_grid.shape)

        return target_angles, np.power(10.0, target_log_freqs), spl_interp.astype(np.float32, copy=False)

    def prepare(self):
        cfg = self.conf
        polar = self.__load_polar_npz(cfg.input_polar_npz)

        freq_hz = polar["freq_hz"].astype(np.float32, copy=False)
        base_angles_deg = polar["polar_angle_deg"].astype(np.float32, copy=False)

        horizontal_raw = np.clip(polar["horizontal_spl_norm_db"], cfg.min_db, cfg.max_db).astype(np.float32, copy=False)
        vertical_raw = np.clip(polar["vertical_spl_norm_db"], cfg.min_db, cfg.max_db).astype(np.float32, copy=False)

        horizontal = horizontal_raw.T
        vertical = vertical_raw.T

        horizontal = self.__normalize_plane_to_reference_angle(
            horizontal,
            base_angles_deg,
            cfg.horizontal_reference_angle_deg,
        )
        vertical = self.__normalize_plane_to_reference_angle(
            vertical,
            base_angles_deg,
            cfg.vertical_reference_angle_deg,
        )

        horizontal = self.__fractional_octave_smooth(horizontal, freq_hz, cfg.isobar_octave_smooth_fraction)
        vertical = self.__fractional_octave_smooth(vertical, freq_hz, cfg.isobar_octave_smooth_fraction)

        isobar_angles, isobar_freqs, horizontal_interp = self.__interpolate_isobar_heatmap(
            base_angles_deg,
            freq_hz,
            horizontal,
            cfg.isobar_angle_samples_smooth,
            cfg.isobar_freq_samples_smooth,
            cfg.min_db,
        )
        _, _, vertical_interp = self.__interpolate_isobar_heatmap(
            base_angles_deg,
            freq_hz,
            vertical,
            cfg.isobar_angle_samples_smooth,
            cfg.isobar_freq_samples_smooth,
            cfg.min_db,
        )

        z_freq = polar["impedance_freq_hz"].astype(np.float32, copy=False)
        z_real = polar["impedance_real"].astype(np.float32, copy=False)
        z_imag = polar["impedance_imag"].astype(np.float32, copy=False)

        np.savez_compressed(
            cfg.output_npz,
            freq_hz=freq_hz,
            polar_angle_deg=base_angles_deg,
            horizontal_spl_norm_db=horizontal_raw,
            vertical_spl_norm_db=vertical_raw,
            isobar_angle_deg=isobar_angles.astype(np.float32, copy=False),
            isobar_freq_hz=isobar_freqs.astype(np.float32, copy=False),
            horizontal_isobar_db=horizontal_interp,
            vertical_isobar_db=vertical_interp,
            impedance_freq_hz=z_freq,
            impedance_real=z_real,
            impedance_imag=z_imag,
            clip_min_db=np.float32(cfg.min_db),
            clip_max_db=np.float32(cfg.max_db),
            horizontal_reference_angle_deg=np.float32(cfg.horizontal_reference_angle_deg),
            vertical_reference_angle_deg=np.float32(cfg.vertical_reference_angle_deg),
        )
