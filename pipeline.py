import os
import argparse
import json
import meshio
from datetime import datetime
from pathlib import Path
from utils.clean import CleanMesh, MeshArgs, MeshioStatistic
from contextlib import redirect_stdout, redirect_stderr
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import asdict

CLEAN_MERGE_TOL=1e-9
CLEAN_AREA_TOL=0.0
CLEAN_WRITE_BINARY=False


class Log:
    persist_log: bool = False
    logname: Optional[str] = None
    job_id: Optional[str] = None
    log_loc: Optional[str] = None
    def __init__(self, log_location: Optional[str] = None):
        self.job_id = None
        self.log_loc = log_location
        pass

    @staticmethod
    def log(message: str, level: str = "info", job_id: Optional[str] = None, args: Optional[Any] = None) -> str:
        timestamp = datetime.now().astimezone().isoformat()
        return json.dumps({"level": level, "job_id": job_id, "timestamp": timestamp, "message": message, "args": args})

    def console(self, message: str, args: Optional[Any] = None):
        log = self.log(message, "info", self.job_id, args)
        self.__write(log)
    
    def error(self, message: str, args: Optional[Any] = None):
        log = self.log(message, "error", self.job_id, args)
        self.__write(log)

    def __write(self, log: str):
        print(log)
        if self.persist:
            logname = str(self.logname)
            if os.path.exists(logname):
                with open(logname, "a") as f:
                    f.write(log + "\n")

    def persist(self, log_type: str):
        self.persit = True
        if self.logname is None:
            log_loc = self.log_loc if self.log_loc else "logs/"
            self.logname = log_loc + "/" + log_type + "-" + str(self.job_id)
            path = Path(self.logname).resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("")


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

    # solver related arguments

    # preparation related arguments

    # visualization related arguments

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
            log.error(args.job_id, "mesh /loc/name.msh is required")
            exit(1)
        # time to clean
        meshStat = MeshioStatistic(args.mesh)
        meshArgs = MeshArgs(args)
        # clean = CleanMesh(meshStat, cleanMeshArgs)
        mesh = MeshioWrapper.read(meshArgs.mesh)
        _, _, _, stats = meshStat.stats(mesh, meshArgs.area_tolerance)
        log.console("mesh stats", asdict(stats))

    if args.clean:
        # time to clean
        # cleanMeshArgs = CleanMeshArgs(args)
        # if not cleanMeshArgs.dirty_mesh_input:
        #     print("The input mesh for cleaning up is required")
        #     exit(1)
        # clean = CleanMesh(cleanMeshArgs)
        # # :/
        # with redirect_stdout(open(os.devnull, 'w')), redirect_stderr(open(os.devnull, 'w')):
        #     mesh = meshio.read(clean.config.dirty_mesh_input)
        # _, _, _, stats = clean.stats(mesh)
        # log.console("dirty mesh stats", asdict(stats))
        pass


class MeshioWrapper:
    @staticmethod
    def read(meshio_loc: str):
        with redirect_stdout(open(os.devnull, 'w')), redirect_stderr(open(os.devnull, 'w')):
            mesh = meshio.read(meshio_loc)
            return mesh


if __name__ == '__main__':
    main()