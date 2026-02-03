├── config/               # Base SU2 configuration templates
├── data/                 # DNS Validation datasets (Ground Truth)
├── src/
│   ├── run_optimization.py   # Main entry point (Optimization Loop)
│   ├── su2_interface.py      # Solver Interface & File Management Class
├── results/              # Auto-generated simulation outputs
├── README.md             # Documentation
└── requirements.txt      # Python dependencies


The tool successfully identifies the optimal $Pr_t$ for Mach 14 flow.
Blue Line: Optimization Path (History)
Green Star: Global Minimum Found
Red X: Simulation Crashes (Handled automatically)