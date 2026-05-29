import os
import argparse
import meshio
import time
from lib.clean import CleanMesh, MeshArgs, MeshioStatistic
from lib.solve import HornBEMSolver, SimulationConfig
from lib.prep import PrepConfig, VisualizationPrep
from lib.visual import VisualizerConfig, Visualizer
from utils.log import Log
from utils.signalr import Signalr
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import asdict

# These are constants as part of the program defaults for all stages of the pipeline
# They work as defaults to configurable arguments, arguents should be thorought and complete
CLEAN_MERGE_TOL=1e-9
CLEAN_AREA_TOL=0.0
CLEAN_WRITE_BINARY=False
MESH_FILE_FORMAT="gmsh22"

SOUND_SPEED: float = 343.0      # m/s
RHO: float = 1.21               # kg/m^3
DISTANCE: float = 2.0           # meters
OBSERVATION_AXIAL_OFFSET_M: float = 0.116  # meters; shifts polar origin along +Z axis
POLAR_ANGLE_STEP_DEG: float = 2.5  # angular precision for polar sampling
POLAR_ANGLE_MIN_DEG: float = -180
POLAR_ANGLE_MAX_DEG: float = 180
FREQ_MIN: float = 200.0
FREQ_MAX: float = 20000.0
FREQ_COUNT: int = 72
# THIS SHOULD BE REMOVED AND DEFINED PROGRAMMATICALLY
TAG_THROAT: int = 6             # Mesh physical tag index for the disc representing the compression driver
SCALE_FACTOR: float = 0.001     # Mesh should be scaled to mm
USE_BURTON_MILLER: bool = True  # Use Burton-Miller formulation to mitigate fictitious resonances

# this is something different, may need to rethink how configs here are set for ease of configuration
WORKERS: int = 3

BEMPP_BOUNDARY_OPERATOR_DEVICE_TYPE: str = "cpu"
BEMPP_POTENTIAL_OPERATOR_DEVICE_TYPE: str = "cpu"
BEMPP_DEFAULT_PRECISION: str = "single"
BEMPP_DEFAULT_DEVICE_INTERFACE: str = "numba"

# hmm
# input_polar_npz: Path = Path("pressure_data.npz")
# output_npz: Path = Path("pressure_data_formatted.npz")

PREP_MIN_DB: float = -30.0   #minimum dB for clipping SPL data
PREP_MAX_DB: float = 0.0     #maximum dB for clipping SPL data

# visual preparation requirements
ISOBAR_ANGLE_SAMPLES_SMOOTH: int = 250
ISOBAR_FREQ_SAMPLES_SMOOTH: int = 500
ISOBAR_OCTAVE_SMOOTH_FRACTION: int | float | None = 24  #fractional octave smoothing for plots
HORIZONTAL_REFERENCE_ANGLE_DEG: float = 10              #normalization angle for horizontal plane
VERTICAL_REFERENCE_ANGLE_DEG: float = 10                #normalization angle for vertical plane

# visualization constant defaults
ISOBAR_INTERP_ANGLE_FACTOR: int = 2
ISOBAR_INTERP_FREQ_FACTOR: int = 3

COLORBAR_TICK_STEP_DB: float = 3.0
FIGURE_WIDTH_IN: float = 11.0
FIGURE_HEIGHT_IN: float = 6.0
FIGURE_DPI: int = 160


def main():
    parser = argparse.ArgumentParser(
        description="BEMPPSolver Pipeline Arguments - There are 4 main sequences: --stats, --clean, --solve, --visualize, " \
        "each with their own set of sub tags.  Most of them have defaults.  Console the readme and docs for more information")
    # global arguments that determine what's going to be done
    parser.add_argument("--stats", action="store_true", help="Grab some quick stats on a mesh")
    parser.add_argument("--clean", action="store_true", help="Clean/stitch a triangle .msh surface mesh for BEM.")
    parser.add_argument("--solve", action="store_true", help="")
    parser.add_argument("--visualize", action="store_true", help="")

    # for all processing
    parser.add_argument("--job-id", action="store", help="The job id for output -- for machine consumption and artifact naming")
    # thinking about this, seems intuitive
    # parser.add_argument("--output-dir", action="store", help="")

    # stats related arguments
    parser.add_argument("--mesh", action="store", help="Mesh for analysis")

    # clean related arguments
    parser.add_argument("--dirty-mesh-input", nargs="?", help="Input .msh file")
    parser.add_argument("--clean-mesh-output", nargs="?", help="Output cleaned .msh file")
    parser.add_argument(
        "--merge-tol",
        type=float,
        default=CLEAN_MERGE_TOL,
        help="Vertex merge tolerance in mesh units (default: 1e-9)",
    )
    parser.add_argument(
        "--area-tol",
        type=float,
        default=CLEAN_AREA_TOL,
        help="Area tolerance for removing tiny triangles in mesh units^2 (default: 0.0)",
    )
    parser.add_argument(
        "--binary",
        action="store_true",
        default=CLEAN_WRITE_BINARY,
        help="Write binary .msh (default is ASCII gmsh22 for compatibility)",
    )
    parser.add_argument(
        "--output-mesh-type",
        action="store",
        default=MESH_FILE_FORMAT,
        help="MESH type output: default is gmsh22"
    )

    # solver related arguments
    # requires "clean-mesh-output"
    parser.add_argument("--sound-speed", type=float, action="store", default=SOUND_SPEED, help="")
    parser.add_argument("--rho", type=float, action="store", default=RHO, help="")
    parser.add_argument("--distance", type=float, action="store", default=DISTANCE, help="")
    parser.add_argument("--observation-axial-offset-m", type=float, action="store", default=OBSERVATION_AXIAL_OFFSET_M, help="")
    parser.add_argument("--polar-angle-step-deg", type=float, action="store", default=POLAR_ANGLE_STEP_DEG, help="")
    parser.add_argument("--polar-angle-min-deg", type=float, action="store", default=POLAR_ANGLE_MIN_DEG, help="")
    parser.add_argument("--polar-angle-max-deg", type=float, action="store", default=POLAR_ANGLE_MAX_DEG, help="")
    parser.add_argument("--freq-min", type=float, action="store", default=FREQ_MIN, help="")
    parser.add_argument("--freq-max", type=float, action="store", default=FREQ_MAX, help="")
    parser.add_argument("--freq-count", type=int, action="store", default=FREQ_COUNT, help="")
    parser.add_argument("--tag-throat", type=int, action="store", default=TAG_THROAT, help="")
    parser.add_argument("--scale-factor", type=float, action="store", default=SCALE_FACTOR, help="")
    parser.add_argument("--use-burton-miller", action="store_true", default=USE_BURTON_MILLER, help="")
    parser.add_argument("--workers", type=int, action="store", default=WORKERS, help="")

    parser.add_argument("--bempp-boundary-operator-device-type", action="store", default=BEMPP_BOUNDARY_OPERATOR_DEVICE_TYPE, help="")
    parser.add_argument("--bempp-potential-operator-device-type", action="store", default=BEMPP_POTENTIAL_OPERATOR_DEVICE_TYPE, help="")
    parser.add_argument("--bempp-default-precision", action="store", default=BEMPP_DEFAULT_PRECISION, help="")
    parser.add_argument("--bempp-default-device-interface", action="store", default=BEMPP_DEFAULT_DEVICE_INTERFACE, help="")

    parser.add_argument("--solution-output", action="store", help="The solved data (does not need extension specified)")


    # preparation and visualization related arguments
    parser.add_argument("--input-polar-npz", action="store", help="Path to the input solver NPZ file",)
    parser.add_argument("--output-npz", action="store", help="Path to the output formatted NPZ file",)

    parser.add_argument(
        "--min-db",
        type=float,
        default=PREP_MIN_DB,
        help="Minimum dB clipping value",
    )
    parser.add_argument(
        "--max-db",
        type=float,
        default=PREP_MAX_DB,
        help="Maximum dB clipping value",
    )
    parser.add_argument(
        "--isobar-angle-samples-smooth",
        type=int,
        default=ISOBAR_ANGLE_SAMPLES_SMOOTH,
        help="Number of smoothed angular samples for isobar interpolation",
    )
    parser.add_argument(
        "--isobar-freq-samples-smooth",
        type=int,
        default=ISOBAR_FREQ_SAMPLES_SMOOTH,
        help="Number of smoothed frequency samples for isobar interpolation",
    )
    parser.add_argument(
        "--isobar-octave-smooth-fraction",
        type=float,
        default=ISOBAR_OCTAVE_SMOOTH_FRACTION,
        help="Fractional-octave smoothing denominator; use 0 to disable",
    )
    parser.add_argument(
        "--horizontal-reference-angle-deg",
        type=float,
        default=HORIZONTAL_REFERENCE_ANGLE_DEG,
        help="Horizontal reference angle for normalization",
    )
    parser.add_argument(
        "--vertical-reference-angle-deg",
        type=float,
        default=VERTICAL_REFERENCE_ANGLE_DEG,
        help="Vertical reference angle for normalization",
    )

    parser.add_argument(
        "--isobar-interp-freq-factor",
        type=int,
        default=ISOBAR_INTERP_FREQ_FACTOR,
        help="Frequency interpolation factor for isobar smoothing (>=1)",
    )


    parser.add_argument("--output-horizontal-isobar", type=str, action="store", help="")
    parser.add_argument("--output-vertical-isobar", type=str, action="store", help="")
    parser.add_argument("--output-acoustic-impedance", type=str, action="store", help="")

    # script configuration specifics
    parser.add_argument("--hub-connection", type=str, default="http://localhost:5000/mainhub")
    args = parser.parse_args()

    signalr = None
    if args.hub_connection:
        signalr = Signalr()
        signalr.start(args.hub_connection)

    log = Log(signalr=signalr)

    if not args.job_id:
        log.error("fatal", "error: job id not set")
        exit(1)
    else:
        log.job_id = args.job_id
        log.console("log", "running job")

    if not args.clean and not args.solve and not args.visualize and not args.stats:
        log.error("fatal", "error: arguments not set")
        exit(1)

    if args.stats:
        log.persist("stats")
        if not args.mesh:
            log.error("fatal", "mesh /loc/name.msh is required")
            exit(1)
        # time to clean
        meshStat = MeshioStatistic()
        meshArgs = MeshArgs(args)
        mesh = MeshioWrapper.read(meshArgs.mesh)
        _, _, _, stats = meshStat.stats(mesh, meshArgs.area_tolerance)
        log.console("status", "mesh stats", asdict(stats))

    if args.clean:
        log.persist("clean")
        # time to clean
        meshArgs = MeshArgs(args)
        if not meshArgs.dirty_mesh_input:
            log.error("fatal", "The input mesh for cleaning up is required")
            exit(1)
        mesh = MeshioWrapper.read(meshArgs.dirty_mesh_input)
        clean = CleanMesh(MeshioStatistic(), meshArgs)
        out_mesh, changes, stats_before, stats_after = clean.clean_mesh(mesh)
        log.console("status", "clean results", {"changes": changes, "stats_before": asdict(stats_before), "stats_after": asdict(stats_after)})

        MeshioWrapper.write(args.clean_mesh_output, out_mesh, args.output_mesh_type, args.binary)
        log.console("status", "Wrote cleaned mesh: ", {"mesh": args.clean_mesh_output})

        if stats_after.boundary_edges > 0:
            log.warning("log", "mesh still has open edges. This usually means real holes (not just unstitched seams)")

    if args.solve:
        if not args.clean_mesh_output:
            log.error("fatal", "a cleaned mesh is reuired for solving")
            exit(1)
        if not args.solution_output:
            log.error("fatal", "no solution output file specified")
            exit(1)
        t_start = time.time()
        config = SimulationConfig(args)
        solver = HornBEMSolver(config, log)
        polar_results, imp_matrix = solver.solve_sweep()

        # Save Results
        solver.save_outputs(polar_results, imp_matrix)
        
        print(f"Total Analysis Time: {time.time() - t_start:.2f}s")
        print("Analysis Complete.")

    if args.visualize:
        log.persist("visual")
        # both prep and output is here
        if not args.input_polar_npz and not args.solution_output:
            log.error("fatal", "solution output is not specified and is required")
            exit(1)
        elif not args.input_polar_npz and args.solution_output:
            args.input_polar_npz = args.solution_output
        
        if not os.path.exists(args.input_polar_npz + ".npz"):
            log.error("fatal", "the solution file was not generated, which means it hasn't been solved yet")
            exit(1)

        if not args.output_npz:
            log.error("fatal", "the output solution file name for prepared data has not been specified")
            exit(1)

        if not args.output_horizontal_isobar:
            log.error("fatal", "horizontal isobar output filename is required")
            exit(1)

        if not args.output_vertical_isobar:
            log.error("fatal", "vertical isobar output filename is required")
            exit(1)

        if not args.output_acoustic_impedance:
            log.error("fatal", "impedance acoustic output filename is required")
            exit(1)

        conf = PrepConfig(args)
        prep = VisualizationPrep(conf, log)
        prep.prepare()

        conf = VisualizerConfig(args)
        vis = Visualizer(conf)

        dataset = vis.load_data(args.output_npz + ".npz")
        outputs = vis.generate_plots(dataset)
        print("Generated PNG plots:")
        for name, path in outputs.items():
            print(f"  - {name}: {path}")


class MeshioWrapper:
    @staticmethod
    def read(meshio_loc: str):
        with redirect_stdout(open(os.devnull, 'w')), redirect_stderr(open(os.devnull, 'w')):
            mesh = meshio.read(meshio_loc)
            return mesh
    
    @staticmethod
    def write(meshio_loc: str, mesh, file_format: str, binary: bool):
        with redirect_stdout(open(os.devnull, 'w')), redirect_stderr(open(os.devnull, 'w')):
            meshio.write(meshio_loc, mesh, file_format=file_format, binary=binary)


if __name__ == '__main__':
    main()