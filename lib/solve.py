from dataclasses import dataclass
from typing import List, Sequence, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils.log import Log
import json
import bempp_cl.api
import numpy as np
import meshio
import os
import multiprocessing as mp


@dataclass
class SimulationConfig:
    # These are defined as constants in the pipeline script.  They are here for verbosity
    # but are always overruled by the constants in the main file.
    mesh_file: str
    sound_speed: float = 343.0      # m/s
    rho: float = 1.21               # kg/m^3
    distance: float = 2.0           # meters
    observation_axial_offset_m: float = 0.116  # meters; shifts polar origin along +Z axis
    polar_angle_step_deg: float = 2.5  # angular precision for polar sampling
    polar_angle_min_deg: float = -180
    polar_angle_max_deg: float = 180
    freq_min: float = 200.0
    freq_max: float = 20000.0
    freq_count: int = 72
    tag_throat: int = 6             # Mesh physical tag index for the disc representing the compression driver
    scale_factor: float = 0.001     # Mesh should be scaled to mm
    use_burton_miller: bool = True  # Use Burton-Miller formulation to mitigate fictitious resonances
    workers: int = 3

    def __init__(self, args):
        if args:
            self.mesh_file = args.clean_mesh_output
            self.output_file = args.solution_output
            self.sound_speed = args.sound_speed
            self.rho = args.rho
            self.distance = args.distance
            self.observation_axial_offset_m = args.observation_axial_offset_m
            self.polar_angle_step_deg = args.polar_angle_step_deg
            self.polar_angle_min_deg = args.polar_angle_min_deg
            self.polar_angle_max_deg = args.polar_angle_max_deg
            self.freq_min = args.freq_min
            self.freq_max = args.freq_max
            self.freq_count = args.freq_count
            self.tag_throat = args.tag_throat
            self.scale_factor = args.scale_factor
            self.use_burton_miller = args.use_burton_miller
            self.workers = args.workers
            # BEMPP Device Configuration, may need to put these somewhere else
            bempp_cl.api.BOUNDARY_OPERATOR_DEVICE_TYPE = "cpu"
            bempp_cl.api.POTENTIAL_OPERATOR_DEVICE_TYPE = "cpu"
            bempp_cl.api.DEFAULT_PRECISION = 'single'
            bempp_cl.api.DEFAULT_DEVICE_INTERFACE = 'numba'

    def __str__(self):
        return json.dumps(self.__dict__)

    # Output controls
    # output_npz_base_path: str = "pressure_data"

class HornBEMSolver:
    grid: bempp_cl.api.Grid

    def __init__(self, config: SimulationConfig, log: Log):
        self.cfg = config
        self.log = log

        self.log.console("Loading mesh", {"mesh_file": self.cfg.mesh_file})
        self.grid, self.physical_tags = self._load_mesh()
        
        # Setup Spaces
        # P1: Continuous linear elements (for Pressure)
        # DP0: Discontinuous constant elements (for Velocity/Flux)
        self.p1_space = bempp_cl.api.function_space(self.grid, "P", 1)
        self.dp0_space = bempp_cl.api.function_space(self.grid, "DP", 0)
        
        # Pre-compute Geometry info
        self._setup_driver_geometry()
        self._setup_polar_evaluation_points()
        
        # Pre-compute Identity Operator (Frequency Independent)
        self.lhs_identity = bempp_cl.api.operators.boundary.sparse.identity(
            self.p1_space, self.p1_space, self.p1_space
        )
        self.rhs_identity = bempp_cl.api.operators.boundary.sparse.identity(
            self.dp0_space, self.p1_space, self.p1_space
        )

        # Create Unit Velocity Excitation (to scale later)
        self.unit_velocity_fun = self._create_unit_velocity()

    def _load_mesh(self) -> Tuple[bempp_cl.api.Grid, np.ndarray]:
        #Load mesh and extract physical tags.
        mesh_data = meshio.read(self.cfg.mesh_file)
        vertices = mesh_data.points * self.cfg.scale_factor
        
        # Handle meshio cell key variations
        if 'triangle' in mesh_data.cells_dict:
            elements = mesh_data.cells_dict['triangle']
            tri_key = 'triangle'
        elif 'triangle3' in mesh_data.cells_dict:
            elements = mesh_data.cells_dict['triangle3']
            tri_key = 'triangle3'
        else:
            raise ValueError("No triangular elements found in mesh.")

        physical_tags = None
        for key in mesh_data.cell_data_dict:
            if 'gmsh:physical' in key and tri_key in mesh_data.cell_data_dict[key]:
                physical_tags = mesh_data.cell_data_dict[key][tri_key]
                break
        
        if physical_tags is None:
            raise ValueError("No physical tags found in mesh.")

        grid = bempp_cl.api.Grid(vertices.T, elements.T)
        return grid, physical_tags

    def _setup_driver_geometry(self):
        #Identify throat elements for impedance calculation
        # In DP0, DOFs map 1:1 to elements
        self.driver_dofs = [
            i for i in range(self.dp0_space.global_dof_count) 
            if self.physical_tags[i] == self.cfg.tag_throat
        ]

        if len(self.driver_dofs) == 0:
            raise ValueError(
                f"No throat elements found for tag_throat={self.cfg.tag_throat}. "
                "Check mesh physical tags."
            )

        self.enclosure_dofs = [
            i for i in range(self.dp0_space.global_dof_count)
            if self.physical_tags[i] != self.cfg.tag_throat
        ]
        
        # Geometry for impedance integration
        self.throat_element_areas = self.grid.volumes[self.driver_dofs]
        self.throat_p1_dofs = self.p1_space.local2global[self.driver_dofs]
        self.log.console("driven service identified", {"elements": len(self.driver_dofs)})
        self.log.console("Enclosure service identified", {"elements": len(self.enclosure_dofs)})
        # print(f"Driven surface identified with {len(self.driver_dofs)} elements. "
        #       f"Enclosure identified with {len(self.enclosure_dofs)} elements.")

    def _create_unit_velocity(self):
        #Create a normal velocity boundary condition with magnitude 1.0 on the throat.
        coeffs = np.zeros(self.dp0_space.global_dof_count, dtype=np.complex128)
        coeffs[self.driver_dofs] = 1.0
        return bempp_cl.api.GridFunction(self.dp0_space, coefficients=coeffs)

    def _setup_polar_evaluation_points(self):
        #Generate horizontal and vertical polar evaluation points.
        step = float(self.cfg.polar_angle_step_deg)
        if step <= 0:
            raise ValueError("polar_angle_step_deg must be positive.")

        angle_min = float(self.cfg.polar_angle_min_deg)
        angle_max = float(self.cfg.polar_angle_max_deg)
        if angle_min < -180.0 or angle_max > 180.0:
            raise ValueError("polar angle range must stay within [-180, 180] degrees.")
        if angle_max < angle_min:
            raise ValueError("polar_angle_max_deg must be >= polar_angle_min_deg.")
        if not (angle_min <= 0.0 <= angle_max):
            raise ValueError("polar angle range must include 0 degrees for on-axis normalization.")

        self.polar_angles_deg = np.arange(angle_min, angle_max + 0.5 * step, step, dtype=np.float32)
        self.polar_angles_deg = np.clip(self.polar_angles_deg, angle_min, angle_max)
        angles_rad = np.deg2rad(self.polar_angles_deg.astype(float))

        x_h = np.sin(angles_rad)
        y_h = np.zeros_like(x_h)
        z_h = np.cos(angles_rad)

        x_v = np.zeros_like(angles_rad)
        y_v = np.sin(angles_rad)
        z_v = np.cos(angles_rad)

        r_dist = float(self.cfg.distance)
        axial_offset_m = float(self.cfg.observation_axial_offset_m)
        axial_shift = np.array([[0.0], [0.0], [axial_offset_m]], dtype=float)

        self.horizontal_eval_points = r_dist * np.vstack([x_h, y_h, z_h]) + axial_shift
        self.vertical_eval_points = r_dist * np.vstack([x_v, y_v, z_v]) + axial_shift
        self.on_axis_idx = int(np.argmin(np.abs(self.polar_angles_deg)))

    def solve_sweep(self) -> Tuple[list, np.ndarray]:
        # What's interesting about this piece of code is that it's using a logarithmic function to
        # create a sequence of frequencies to test by RATIO, not hz count.  To best measure
        # frequency response, as the octaves go up, the number of htz to jump to also go up.
        # This is best modeled by a logarithmic function.
        frequencies = np.logspace(
            np.log10(self.cfg.freq_min),
            np.log10(self.cfg.freq_max),
            self.cfg.freq_count
        )

        worker_count = self._resolve_worker_count(len(frequencies))
        print(
            f"Starting solver: {len(frequencies)} frequencies "
            f"using {worker_count} worker{'s' if worker_count != 1 else ''}."
        )

        if worker_count == 1:
            return self.solve_frequencies(frequencies, show_progress=True)

        return self._solve_sweep_parallel(frequencies, worker_count)

    def solve_frequencies(self, frequencies: Sequence[float], show_progress: bool = True) -> Tuple[list, np.ndarray]:
        frequencies = np.asarray(frequencies, dtype=float)
        freq_count = len(frequencies)
        self.log.console("total frequencies", { "count": freq_count })

        results_polar = []
        results_imp = []
        for i, freq in enumerate(frequencies):
            res_h, res_v, res_z = self._solve_single_frequency(freq)
            results_polar.append((freq, res_h, res_v))
            results_imp.append(res_z)
            self.log.console("freq_count", { "increment": i+1, "total": freq_count })
            # if show_progress:
            #     print(f"[{i+1}/{len(frequencies)}] {freq:.1f} Hz")

        imp_matrix = np.asarray(results_imp, dtype=np.float32)
        return results_polar, imp_matrix

    def _resolve_worker_count(self, frequency_count: int) -> int:
        if self.cfg.workers < 1:
            raise ValueError("workers must be >= 1.")

        return min(self.cfg.workers, max(1, frequency_count), os.cpu_count() or 1)

    def _solve_sweep_parallel(self, frequencies: np.ndarray, worker_count: int) -> Tuple[list, np.ndarray]:
        chunks = _split_frequencies_evenly(frequencies, worker_count)
        ctx = mp.get_context("spawn")
        chunk_results = {}
        completed = 0

        with ProcessPoolExecutor(max_workers=len(chunks), mp_context=ctx) as executor:
            futures = {
                executor.submit(_solve_frequency_chunk, self.cfg, chunk.tolist()): index
                for index, chunk in enumerate(chunks)
            }

            for future in as_completed(futures):
                index = futures[future]
                polar_chunk, imp_chunk = future.result()
                chunk_results[index] = (polar_chunk, imp_chunk)
                completed += len(polar_chunk)
                # this needs to be tested
                print(f"[{completed}/{len(frequencies)}] completed worker chunk {index + 1}/{len(chunks)}")

        polar_results = []
        imp_results = []
        for index in range(len(chunks)):
            polar_chunk, imp_chunk = chunk_results[index]
            polar_results.extend(polar_chunk)
            imp_results.append(imp_chunk)

        imp_matrix = np.vstack(imp_results).astype(np.float32, copy=False)
        return polar_results, imp_matrix

    def _solve_single_frequency(self, freq):
        omega = 2 * np.pi * freq
        k = omega / self.cfg.sound_speed
        
        # 1. Update Boundary Conditions
        # v = 1 m/s. Use normal velocity on the throat only.
        velocity_fun = self.unit_velocity_fun
        neumann_fun = 1j * self.cfg.rho * omega * velocity_fun

        # 2. Assemble Operators
        dlp = bempp_cl.api.operators.boundary.helmholtz.double_layer(
            self.p1_space, self.p1_space, self.p1_space, k
        )
        slp = bempp_cl.api.operators.boundary.helmholtz.single_layer(
            self.dp0_space, self.p1_space, self.p1_space, k
        )

        # 3. Formulate LHS and RHS
        if self.cfg.use_burton_miller:
            hyp = bempp_cl.api.operators.boundary.helmholtz.hypersingular(
                self.p1_space, self.p1_space, self.p1_space, k
            )
            adlp = bempp_cl.api.operators.boundary.helmholtz.adjoint_double_layer(
                self.dp0_space, self.p1_space, self.p1_space, k
            )
            # Exterior Neumann, Burton-Miller (BEMPP sign conventions)
            # Note that BEMPP negates the hypersingular operator
            coupling = 1j / k
            lhs = 0.5 * self.lhs_identity - dlp - coupling * -hyp
            rhs = (-slp - coupling * (adlp + 0.5 * self.rhs_identity)) * neumann_fun
        else:
            # Exterior Neumann (classical)
            lhs = dlp - 0.5 * self.lhs_identity
            rhs = slp * neumann_fun

        # 4. Solve System
        dirichlet_fun, info = bempp_cl.api.linalg.gmres(lhs, rhs, tol=1E-3)
        if info != 0:
            self.log.warning("Warning: Solver did not converge", { "freq": f"{freq:.1f}Hz" })

        # 5. Post-Processing
        z_data = self._calculate_impedance(freq, dirichlet_fun)
        horizontal_spl = self._evaluate_field(self.horizontal_eval_points, k, dirichlet_fun, neumann_fun, omega)
        vertical_spl = self._evaluate_field(self.vertical_eval_points, k, dirichlet_fun, neumann_fun, omega)
        horizontal_spl_norm, vertical_spl_norm = self._normalize_polar_to_on_axis(horizontal_spl, vertical_spl)
        
        return horizontal_spl_norm, vertical_spl_norm, z_data

    def _calculate_impedance(self, freq, dirichlet_fun):
        # Pressure at local P1 dofs of throat elements.
        # Do not index with raw mesh vertex ids: P1 global dof numbering may differ.
        p_at_vertices = dirichlet_fun.coefficients[self.throat_p1_dofs]
        p_avg = np.mean(p_at_vertices, axis=1)
        
        # Force = Integral(p dS) ~ sum(p_avg * area)
        total_force = np.sum(p_avg * self.throat_element_areas) * 10
        
        # Z = Force / Velocity (v=1)
        return [freq, np.real(total_force)/2, -np.imag(total_force)/2]

    def _evaluate_field(self, points, k, dirichlet_fun, neumann_fun, omega):
        slp_pot = bempp_cl.api.operators.potential.helmholtz.single_layer(
            self.dp0_space, points, k, # device_interface="opencl"
        )
        dlp_pot = bempp_cl.api.operators.potential.helmholtz.double_layer(
            self.p1_space, points, k, # device_interface="opencl"
        )

        p_field = (dlp_pot * dirichlet_fun - slp_pot * neumann_fun).ravel()
        
        # Convert to SPL
        # Ref pressure = 20e-6 Pa
        return 20 * np.log10(np.abs(p_field) / 20e-6)

    def _normalize_polar_to_on_axis(self, horizontal_spl, vertical_spl):
        on_axis_ref = horizontal_spl[self.on_axis_idx]
        return horizontal_spl - on_axis_ref, vertical_spl - on_axis_ref

    def save_outputs(self, polar_results, imp_matrix):
        base = self.cfg.output_file # output_npz_base_path

        freqs = np.array([freq for freq, _, _ in polar_results], dtype=np.float32)
        horizontal_spl = np.vstack([h_spl for _, h_spl, _ in polar_results]).astype(np.float32, copy=False)
        vertical_spl = np.vstack([v_spl for _, _, v_spl in polar_results]).astype(np.float32, copy=False)
        z_freq_hz = imp_matrix[:, 0].astype(np.float32, copy=False)
        z_real = imp_matrix[:, 1].astype(np.float32, copy=False)
        z_imag = imp_matrix[:, 2].astype(np.float32, copy=False)

        np.savez_compressed(
            f"{base}.npz",
            freq_hz=freqs,
            polar_angle_deg=self.polar_angles_deg.astype(np.float32, copy=False),
            horizontal_spl_norm_db=horizontal_spl,
            vertical_spl_norm_db=vertical_spl,
            impedance_freq_hz=z_freq_hz,
            impedance_real=z_real,
            impedance_imag=z_imag,
            observation_axial_offset_m=np.float32(self.cfg.observation_axial_offset_m),
        )
        self.log.console("Saved solved file", {"filename": f"{base}.npz"})


def _split_frequencies_evenly(frequencies: np.ndarray, worker_count: int) -> List[np.ndarray]:
    if worker_count <= 1 or len(frequencies) == 0:
        return [frequencies]

    return [chunk for chunk in np.array_split(frequencies, worker_count) if len(chunk) > 0]

def _solve_frequency_chunk(config: SimulationConfig, frequencies: Sequence[float]):
    solver = HornBEMSolver(config)
    return solver.solve_frequencies(np.asarray(frequencies, dtype=float), show_progress=False)
