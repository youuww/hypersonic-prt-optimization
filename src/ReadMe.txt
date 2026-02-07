================================================================================
  src/ — Source Code & Runtime Files
================================================================================

STRUCTURE
---------
  src/
  ├── run_optimization.py        # Main entry point (Optimization Loop)
  ├── su2_interface.py           # SU2 Solver Interface & File Management
  ├── generate_ramp.py           # Mesh generator for compression ramp (WIP)
  ├── mesh_flatplate_turb_545x385.su2   # Mesh file (must be here — SU2 runs from src/)
  └── mesh_ramp_15deg.su2               # Ramp mesh (future work)

HOW TO RUN
----------
  cd src/
  python run_optimization.py

NOTES
-----
- SU2 runs from this directory, so mesh files (.su2) must live here.
- Results are saved under results/<geometry>_<N>iter_<yymmdd>/ (see results/README.txt).
- Post-processing scripts are in post_processing/ (GIFs, AIAA plot).

================================================================================
