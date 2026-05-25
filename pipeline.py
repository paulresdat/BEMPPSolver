import os
import argparse
import meshio
import time
from lib.clean import CleanMesh, MeshArgs, MeshioStatistic
from lib.solve import HornBEMSolver, SimulationConfig
from lib.log import Log
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


def main():
    parser = argparse.ArgumentParser(description="Splux Pipeline Arguments")
    # global arguments that determine what's going to be done
    parser.add_argument("--stats", action="store_true", help="Grab some quick stats on a mesh")
    parser.add_argument("--clean", action="store_true", help="Clean/stitch a triangle .msh surface mesh for BEM.")
    parser.add_argument("--solve", action="store_true", help="")
    parser.add_argument("--visualize", action="store_true", help="")

    # for all processing
    parser.add_argument("--job-id", action="store", help="The job id for output -- for machine consumption and artifact naming")

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

    # preparation and visualization related arguments

    args = parser.parse_args()

    log = Log()

    if not args.job_id:
        log.error("error: job id not set")
        exit(1)
    else:
        log.job_id = args.job_id
        log.console("running job")

    if not args.clean and not args.solve and not args.visualize and not args.stats:
        log.error("error: arguments not set")
        exit(1)

    if args.stats:
        log.persist("stats")
        if not args.mesh:
            log.error("mesh /loc/name.msh is required")
            exit(1)
        # time to clean
        meshStat = MeshioStatistic()
        meshArgs = MeshArgs(args)
        mesh = MeshioWrapper.read(meshArgs.mesh)
        _, _, _, stats = meshStat.stats(mesh, meshArgs.area_tolerance)
        log.console("mesh stats", asdict(stats))

    if args.clean:
        log.persist("clean")
        # time to clean
        meshArgs = MeshArgs(args)
        if not meshArgs.dirty_mesh_input:
            log.error("The input mesh for cleaning up is required")
            exit(1)
        mesh = MeshioWrapper.read(meshArgs.dirty_mesh_input)
        clean = CleanMesh(MeshioStatistic(), meshArgs)
        out_mesh, changes, stats_before, stats_after = clean.clean_mesh(mesh)
        log.console("clean results", {"changes": changes, "stats_before": asdict(stats_before), "stats_after": asdict(stats_after)})

        MeshioWrapper.write(args.clean_mesh_output, out_mesh, args.output_mesh_type, args.binary)
        log.console("Wrote cleaned mesh: ", {"mesh": args.clean_mesh_output})

        if stats_after.boundary_edges > 0:
            log.warning("mesh still has open edges. This usually means real holes (not just unstitched seams)")

    if args.solve:
        if not args.clean_mesh_output:
            log.error("a cleaned mesh is reuired for solving")
            exit(1)
        t_start = time.time()
        config = SimulationConfig(args)
        solver = HornBEMSolver(config, log)
        polar_results, imp_matrix = solver.solve_sweep()

        # Save Results
        solver.save_outputs(polar_results, imp_matrix)
        
        print(f"Total Analysis Time: {time.time() - t_start:.2f}s")
        print("Analysis Complete.")
        pass

    if args.visualize:
        # both prep and output is here
        pass


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