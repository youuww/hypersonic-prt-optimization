================================================================================
  results/ — Folder layout and rules (so we don't forget in 2 weeks)
================================================================================

STRUCTURE (one folder per run)
-----------------------------
  results/
  ├── <geometry>_<N>iter_<yymmdd>/     e.g. turb_SA_flatplate_M14Tw018_5iter_260207
  │   ├── Pr_0.5000/                   (flow.dat, plot, etc. per Pr_t)
  │   ├── Pr_0.5660/
  │   ├── optimization_log.csv
  │   └── optimization_convergence.png
  ├── <geometry>_<N>iter_<yymmdd>_<HHMM>/   (if same name exists, time is appended)
  └── ...

RULES
-----
1. Each run of run_optimization.py creates ONE folder. Name = config stem + number of iterations + date (yymmdd). Example: turb_SA_flatplate_M14Tw018_5iter_260207.

2. The code never deletes or empties results/. Old run folders stay unless you remove them yourself.

3. Post-processing (GIFs, AIAA plot) automatically uses the LATEST run folder (the one whose name contains "iter" and has the newest modification time). If optimization_log.csv exists directly in results/, that legacy layout is used instead.

4. For important runs you want to keep long-term: copy the whole run folder to My_Results/ (or elsewhere). My_Results/ is manual and not touched by the code.

5. To see which run a folder is from: read the folder name (geometry, iteration count, date). optimization_log.csv inside the folder has the full history of that run.

================================================================================
