import argparse

CLEAN_MERGE_TOL=1e-9
CLEAN_AREA_TOL=0.0
CLEAN_WRITE_BINARY=False

def main():
        parser = argparse.ArgumentParser(description="Clean/stitch a triangle .msh surface mesh for BEM.")
        parser.add_argument("input_msh", nargs="?", help="Input .msh file")
        parser.add_argument("output_msh", nargs="?", help="Output cleaned .msh file")
        parser.add_argument("job-id", action="store", help="The job id for output -- for machine consumption and artifact naming")
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

        args = parser.parse_args()



if __name__ == '__main__':
    main()