import numpy as np

def generate_ramp_mesh(filename="mesh_ramp_15deg.su2"):
    # --- Geometry Parameters ---
    L_PLATE = 1.0      # אורך הפלטה הישרה
    L_RAMP = 0.5       # אורך הרמפה
    H_DOMAIN = 0.8     # גובה הדומיין
    RAMP_ANGLE_DEG = 15.0  # <--- שנה כאן את הזווית לניסויים!
    
    # --- Grid Resolution ---
    NX_PLATE = 200     # תאים באזור הישר
    NX_RAMP = 150      # תאים באזור הרמפה
    NY = 120           # תאים לגובה (מספיק ל-Boundary Layer)
    
    # --- Boundary Layer Clustering ---
    # אנחנו צריכים תאים דקים מאוד ליד הקיר
    STRETCH_FACTOR = 2.5 # חוזק הדחיסה לקיר (Tanh stretching)

    print(f"--> Generating Ramp Mesh: {RAMP_ANGLE_DEG} deg")
    
    # 1. יצירת נקודות בציר X
    x_plate = np.linspace(-L_PLATE, 0, NX_PLATE + 1)
    x_ramp = np.linspace(0, L_RAMP, NX_RAMP + 1)[1:] # משמיטים את ה-0 כדי לא לכפול
    x_coords = np.concatenate([x_plate, x_ramp])
    nx_total = len(x_coords)
    
    # 2. יצירת נקודות בציר Y (עם דחיסה לקיר)
    # שימוש בפונקציית Tanh ליצירת תאים צפופים למטה ודלילים למעלה
    eta = np.linspace(0, 1, NY + 1)
    y_dist = 1.0 + np.tanh(STRETCH_FACTOR * (eta - 1.0)) / np.tanh(STRETCH_FACTOR)
    
    # 3. בניית הרשת הדו-ממדית
    ramp_angle_rad = np.radians(RAMP_ANGLE_DEG)
    points = []
    
    # רשימות לשמירת האלמנטים של הגבולות (Markers)
    # הפורמט: 3 (Line), Node1, Node2
    marker_wall = []
    marker_farfield = []
    marker_inlet = []
    marker_outlet = []
    
    # Nodes Indexing helper
    def get_node_idx(i, j):
        return j * nx_total + i

    # יצירת הנקודות (Nodes)
    for j in range(NY + 1):
        for i in range(nx_total):
            x = x_coords[i]
            
            # חישוב גובה הרצפה המקומי
            if x <= 0:
                y_wall = 0.0
            else:
                y_wall = x * np.tan(ramp_angle_rad)
            
            # חישוב ה-Y הסופי ע"י אינטרפולציה בין הרצפה לתקרה
            # y = y_wall + distribution * (H_top - y_wall)
            y = y_wall + y_dist[j] * (H_DOMAIN - y_wall)
            
            points.append(f"{x:.6f} {y:.6f}")

    # 4. יצירת האלמנטים (Quads) וחיבור ה-Markers
    elems = []
    for j in range(NY):
        for i in range(nx_total - 1):
            # הגדרת 4 פינות של כל תא
            n0 = get_node_idx(i, j)
            n1 = get_node_idx(i+1, j)
            n2 = get_node_idx(i+1, j+1)
            n3 = get_node_idx(i, j+1)
            
            # ב-SU2 אלמנט מרובע מוגדר כ: 9 n0 n1 n2 n3
            elems.append(f"9 {n0} {n1} {n2} {n3}")

            # --- Boundary Markers Logic ---
            # Wall (השורה התחתונה j=0)
            if j == 0:
                marker_wall.append(f"3 {n0} {n1}")
            
            # Farfield (השורה העליונה j=NY-1) -> Top boundary
            if j == NY - 1:
                marker_farfield.append(f"3 {n3} {n2}") # שים לב לכיוון
            
            # Inlet (העמודה השמאלית i=0)
            if i == 0:
                marker_inlet.append(f"3 {n0} {n3}") # ורטיקלי
                
            # Outlet (העמודה הימנית i=nx-2)
            if i == nx_total - 2:
                marker_outlet.append(f"3 {n1} {n2}")

    # 5. כתיבת הקובץ בפורמט SU2 Native
    with open(filename, "w") as f:
        f.write(f"NDIME= 2\n")
        f.write(f"NELEM= {len(elems)}\n")
        for e in elems:
            f.write(f"{e}\n")
            
        f.write(f"POIN= {len(points)}\n")
        for p in points:
            f.write(f"{p}\n")
            
        f.write(f"NMARK= 4\n")
        
        f.write(f"MARKER_TAG= wall\n")
        f.write(f"MARKER_ELEMS= {len(marker_wall)}\n")
        for m in marker_wall: f.write(f"{m}\n")

        f.write(f"MARKER_TAG= farfield\n")
        f.write(f"MARKER_ELEMS= {len(marker_farfield)}\n")
        for m in marker_farfield: f.write(f"{m}\n")
        
        f.write(f"MARKER_TAG= inlet\n")
        f.write(f"MARKER_ELEMS= {len(marker_inlet)}\n")
        for m in marker_inlet: f.write(f"{m}\n")

        f.write(f"MARKER_TAG= outlet\n")
        f.write(f"MARKER_ELEMS= {len(marker_outlet)}\n")
        for m in marker_outlet: f.write(f"{m}\n")

    print(f"--> Done! Created: {filename}")
    print(f"    Grid Size: {nx_total} x {NY+1}")

if __name__ == "__main__":
    generate_ramp_mesh()