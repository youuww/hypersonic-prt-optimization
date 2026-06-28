import pandas as pd
import numpy as np
import subprocess
import os
import re
import shutil
from pathlib import Path
from typing import Optional

from case_config import FlowCondition

# --- Force non-interactive backend for WSL ---
import matplotlib
matplotlib.use('Agg') # Must be before importing pyplot
import matplotlib.pyplot as plt

# --- Path Handling ---
SCRIPT_DIR = Path(__file__).resolve().parent # Absolute Location
BASE_CFG = SCRIPT_DIR.parent / "config" / "turb_SA_flatplate_M14Tw018.cfg"
DNS_FILE = SCRIPT_DIR.parent / "data" / "DNS Dataset.csv"
RESULTS_DIR = SCRIPT_DIR.parent / "results"

class SU2Interface:
    def __init__(
        self,
        flow: Optional[FlowCondition] = None,
        base_config: Path = BASE_CFG,
        dns_csv: Optional[Path] = DNS_FILE,
        num_cores: int = 4,
    ):
        self.SCRIPT_DIR = SCRIPT_DIR
        self.base_config = base_config
        self.num_cores = num_cores
        self.RESULTS_DIR = RESULTS_DIR

        # Flow condition — defaults to the unified DNS-consistent Mach 14 case.
        # generate_config ALWAYS injects Mach, T_inf, P_inf, Re, T_wall and the
        # mesh from this FlowCondition, so the SU2 freestream matches it exactly
        # (in particular, the derived Re drives SU2's freestream density).
        self.flow = flow or FlowCondition.dns_M14Tw018()

        # DNS reference data — FlowCondition.dns_data_path takes priority
        dns_path = self.flow.dns_data_path or dns_csv
        if dns_path and Path(dns_path).exists():
            self.dns_data = pd.read_csv(dns_path)
            self.dns_u = self.dns_data.iloc[:, 0].values
            self.dns_t = self.dns_data.iloc[:, 1].values
            self._has_dns = True
        else:
            self.dns_u = None
            self.dns_t = None
            self._has_dns = False

        # Physics derived from flow condition (replaces old hardcoded values)
        self.T_INF = self.flow.t_inf
        self.U_INF = self.flow.u_inf

        # Analysis location (downstream station for profile extraction)
        self.X_STATION = 1.5   # [m]
        self.X_TOLERANCE = 0.005

        # Simulation settings (overridden per-run in generate_config)
        self.ITERATIONS = 51
        self.SAVE_FREQ = 10

    def generate_config(self, pr_t: float, run_id: str) -> Path:
        """Generates a temporary SU2 config with injected Pr_t and flow conditions.

        All freestream parameters (Mach, T_inf, P_inf, derived Re, T_wall and
        mesh) are injected from ``self.flow`` so the SU2 freestream is always
        consistent with the FlowCondition.  ``REYNOLDS_NUMBER`` is the derived
        ``flow.re`` — SU2 reconstructs the freestream density from it.
        """
        new_cfg = SCRIPT_DIR / f"run_{run_id}.cfg"

        # Freestream overrides — always injected from the FlowCondition.
        flow_overrides: dict[str, str] = {
            "MACH_NUMBER":            str(self.flow.mach),
            "FREESTREAM_TEMPERATURE": str(self.flow.t_inf),
            "FREESTREAM_PRESSURE":    str(self.flow.p_inf),
            "REYNOLDS_NUMBER":        str(self.flow.re),
            "MARKER_ISOTHERMAL":      f"( wall, {self.flow.t_wall:.1f} )",
            "MESH_FILENAME":          str(self.flow.mesh_file),
        }

        with open(self.base_config, 'r') as f:
            lines = f.readlines()
        
        with open(new_cfg, 'w') as f:
            for line in lines:
                if "PRANDTL_TURB" in line:
                    f.write(f"PRANDTL_TURB= {pr_t}\n")

                elif "RESTART_SOL" in line:
                    f.write("RESTART_SOL= NO\n")

                elif "OUTPUT_WRT_FREQ" in line:
                    f.write(f"OUTPUT_WRT_FREQ= {self.SAVE_FREQ}\n")
                elif line.strip().startswith("ITER="):
                    f.write(f"ITER= {self.ITERATIONS}\n")

                elif "OUTPUT_FILES" in line:
                    f.write("OUTPUT_FILES= (RESTART, PARAVIEW, TECPLOT_ASCII)\n")
                elif "VOLUME_FILENAME" in line:
                    f.write("VOLUME_FILENAME= flow\n")
                elif "CONV_FILENAME" in line:
                    f.write(f"CONV_FILENAME= history\n")
                elif "RESTART_FILENAME" in line:
                    f.write("RESTART_FILENAME= restart_flow\n")

                else:
                    # --- Flow condition injection ---
                    # Always override freestream params from the FlowCondition
                    # so the SU2 freestream matches the case exactly.
                    key = line.split("=")[0].strip()
                    if key in flow_overrides:
                        f.write(f"{key}= {flow_overrides[key]}\n")
                    else:
                        f.write(line)
        return new_cfg

    def run_su2(self, cfg_file):
        """Executes SU2 with MPI support."""
        print(f"--> Running SU2 (Cores: {self.num_cores}) | Config: {cfg_file}")
        command = ["SU2_CFD", cfg_file] # Serial run: SU2_CFD config.cfg
        if self.num_cores > 1:
            # Parallel run: mpirun -n 4 SU2_CFD config.cfg
            command = ["mpirun", "-n", str(self.num_cores), "SU2_CFD", cfg_file]            
        try:
            subprocess.run(command, check=True, stdout= subprocess.DEVNULL) # stdout= subprocess.DEVNULL
            return True
        except subprocess.CalledProcessError:
            print("!!! Simulation Crashed.")
            return False

    def load_tecplot_data(self, filename_base="flow"):
        """Helper to load and clean Tecplot data"""
        filename = Path(filename_base + ".dat")

        # Build the full path
        dat_file = self.SCRIPT_DIR / filename
        
        if not dat_file.exists(): 
            print(f"!!! Warning: Could not find data file at: {dat_file}")
            return None

        with open(dat_file, 'r') as f:
            lines = f.readlines()
        
        header_rows = 0
        col_names = []
        for i, line in enumerate(lines):
            if "VARIABLES" in line: col_names = re.findall(r'"(.*?)"', line)
            if "ZONE" in line: header_rows = i + 1; break
            
        if not col_names: return None


        df = pd.read_csv(dat_file, skiprows=header_rows, sep='\s+', names=col_names, on_bad_lines='skip')
        
        rename_map = {}
        for col in df.columns:
            c = col.lower()
            if 'x' == c or "coordinatex" in c: rename_map[col] = 'x'
            if "temperature" in c: rename_map[col] = 'T'
            if "velocity" in c and ("_x" in c or "x" in c) and "y" not in c: rename_map[col] = 'u'
            if "momentum" in c and ("_x" in c or "x" in c) and "y" not in c: rename_map[col] = 'mom_x'
            if "density" in c: rename_map[col] = 'rho'
        
        df.rename(columns=rename_map, inplace=True)
        
        if 'u' not in df.columns and 'mom_x' in df.columns:
            df['u'] = df['mom_x'] / df['rho']
            
        return df

    def calculate_loss(self, filename_base):
        """Extracts profile at X_STATION and computes RMSE vs DNS."""
        if not self._has_dns:
            print("!!! No DNS reference data — cannot compute loss.")
            return 999.0
        try:
            df = self.load_tecplot_data(filename_base)
            if df is None: return 999.0

            # Filter Slice
            slice_df = df[ (df['x'] > self.X_STATION - self.X_TOLERANCE) & 
                           (df['x'] < self.X_STATION + self.X_TOLERANCE) ].copy()
            if slice_df.empty: return 999.0

            # Normalize
            slice_df['u_norm'] = slice_df['u'] / self.U_INF
            slice_df['t_norm'] = slice_df['T'] / self.T_INF
            slice_df = slice_df.sort_values(by='u_norm').drop_duplicates(subset='u_norm')

            # Interp & RMSE
            t_dns_interp = np.interp(slice_df['u_norm'], self.dns_u, self.dns_t)
            error = np.sqrt(((slice_df['t_norm'] - t_dns_interp) ** 2).mean())
            return error
        except Exception as e:
            print(f"!!! Error calculating loss: {e}")
            return 999.0

    def plot_results(self, filename_base, pr_val):
        """Generates T-U profile plot. Overlays DNS data when available."""
        try:
            df = self.load_tecplot_data(filename_base)
            if df is None: return

            slice_df = df[ (df['x'] > self.X_STATION - self.X_TOLERANCE) & 
                           (df['x'] < self.X_STATION + self.X_TOLERANCE) ].copy()
            if slice_df.empty: return

            slice_df['u_norm'] = slice_df['u'] / self.U_INF
            slice_df['t_norm'] = slice_df['T'] / self.T_INF
            slice_df = slice_df.sort_values(by='u_norm')

            plt.figure(figsize=(10, 6), dpi=300)
            if self._has_dns:
                plt.plot(self.dns_u, self.dns_t, 'k.', label='DNS Data', markersize=8)
            plt.plot(slice_df['u_norm'], slice_df['t_norm'], 'r-', linewidth=2, label=f'SU2 (Pr_t={pr_val})')
            
            label = getattr(self.flow, 'label', f'Mach {self.flow.mach}')
            plt.title(f'Boundary Layer T-U Profile ({label})\n'
                      f'Pr_t = {pr_val}, x = {self.X_STATION} m')
            plt.xlabel('u / u_inf')
            plt.ylabel('T / T_inf')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plot_name = f"plot_Pr{pr_val}.png"
            plt.savefig(plot_name)
            plt.close()
            print(f"   [Plot] Saved: {plot_name}")
            
        except Exception as e:
            print(f"!!! Plotting error: {e}")

    def organize_files(self, pr_val):
        """Moves simulation output files into a dedicated folders."""
        # 1. Define folder name (e.g., ../results/Pr_0.5)
        folder_name = self.RESULTS_DIR / f"Pr_{pr_val}"
        
        if not folder_name.exists():
            print(f"   [Org] Created folder: {folder_name}")
        folder_name.mkdir(parents=True, exist_ok=True)

        # 2. List of files to move (Data files + The Plot)
        files_to_move = [
            "flow.dat", 
            "flow.vtu", 
            "surface_flow.vtu",
            "surface_flow.csv",
            "history.csv",
            "restart_flow.dat",
            f"plot_Pr{pr_val}.png"
        ]

        print(f"   [Org]  Moving files to: {folder_name}/")

        # 3. Move files loop
        for filename in files_to_move:
            src = SCRIPT_DIR / filename # File Source
            dst = folder_name / filename # File Destination
                
            if src.exists():
                if dst.exists():
                    dst.unlink()
                
                # Move the file
                shutil.move(str(src), str(dst))
                print(f"       -> Moved: {filename}")


    def cleanup(self, run_id):
        # Script Directory
        files_to_remove = [
            self.SCRIPT_DIR / f"run_{run_id}.cfg",
            self.SCRIPT_DIR / "flow.dat"
        ]

        for file_path in files_to_remove:
            file_path.unlink(missing_ok=True)
