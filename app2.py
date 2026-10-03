import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.interpolate as interpolate
from scipy.ndimage import gaussian_filter1d
import streamlit as st
from matplotlib.animation import FuncAnimation
import cv2

# Web page configuration
st.set_page_config(page_title="3D Track Designer & Simulator", layout="wide")
plt.style.use('dark_background')

STABLE_LABEL   = "[STABLE]: RIDER ON ASPHALT & WITHIN SAFETY MARGIN"
ALERT_LABEL    = "[RED ALERT]: VELOCITY / GRIP LIMIT BREACH!"
OFFTRACK_LABEL = "[RED ALERT]: OFF-TRACK BREACH!"

# --- Session State Initialization ---
if 'phase' not in st.session_state:
    st.session_state.phase = 'upload'
if 'raw_geometry' not in st.session_state:
    st.session_state.raw_geometry = None
if 'custom_elev' not in st.session_state:
    st.session_state.custom_elev = {0.0: 0.0}
if 'custom_bank' not in st.session_state:
    st.session_state.custom_bank = {0.0: 0.0}
if 'track_length' not in st.session_state:
    st.session_state.track_length = 5000.0

# --- Geometry and Topography Extraction ---
def extract_track_data_from_image(image_bytes, use_heatmap, min_val, max_val, target_length, extraction_method, smooth_factor):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img_color = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    if img_color is None:
        return None, None

    img_h, img_w = img_color.shape[:2]

    if "Color Masking" in extraction_method:
        hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
        lower_color = np.array([15, 50, 50])
        upper_color = np.array([150, 255, 255])
        mask = cv2.inRange(hsv, lower_color, upper_color)
        
        bridge_kernel = np.ones((30, 30), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, bridge_kernel)
        
        clean_kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, clean_kernel)
        thresh = mask

    elif "Mathematical" in extraction_method:
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    else:
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)
        kernel_close = np.ones((7, 7), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)
        kernel_open = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open)


    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        return None, None

    valid_contours = []
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        
        if w > (img_w * 0.92) and h > (img_h * 0.92):
            continue
        if w > (img_w * 0.98) or h > (img_h * 0.98):
            continue
        if area < 500:
            continue
            
        valid_contours.append(cnt)
        
    if not valid_contours:
        return None, None

    best_contour = max(valid_contours, key=cv2.contourArea)
    wp_raw = best_contour.reshape(-1, 2).astype(float)
    
    idx = np.linspace(0, len(wp_raw) - 1, 600, dtype=int)
    wp = wp_raw[idx]
    
    dists = np.sum(np.diff(wp, axis=0)**2, axis=1)
    valid_idx = np.concatenate(([True], dists > 1.0))
    wp = wp[valid_idx]
    
    if len(wp) < 10:
        return None, None

    lowest_y_idx = np.argmax(wp[:, 1]) 
    
    next_idx = (lowest_y_idx + 1) % len(wp)
    prev_idx = (lowest_y_idx - 1) % len(wp)
    if wp[next_idx, 0] < wp[prev_idx, 0]:
        wp = wp[::-1]
        lowest_y_idx = len(wp) - 1 - lowest_y_idx

    wp = np.roll(wp, -lowest_y_idx, axis=0)
    
    extracted_elevations = {0.0: 0.0}
    if use_heatmap:
        hsv_img = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
        extracted_elevations.clear()
        
        for i, pt in enumerate(wp):
            px, py = int(pt[0]), int(pt[1])
            
            if 0 <= py < hsv_img.shape[0] and 0 <= px < hsv_img.shape[1]:
                hue = hsv_img[py, px, 0]
                if hue > 120: 
                    hue = 120
                
                norm_val = 1.0 - (hue / 120.0)
                elev = min_val + norm_val * (max_val - min_val)
                
                percent = i / float(len(wp))
                s_pos = percent * target_length
                extracted_elevations[s_pos] = round(elev, 2)

    wp[:, 1] = -wp[:, 1]
    
    tck_base, _ = interpolate.splprep([wp[:,0], wp[:,1]], s=smooth_factor, per=True)
    u_dense = np.linspace(0, 1, 30000)
    x_smooth, y_smooth = interpolate.splev(u_dense, tck_base)

    return (x_smooth, y_smooth), extracted_elevations

# --- Build Full Physics Model ---
def build_physics_model():
    x_smooth, y_smooth = st.session_state.raw_geometry
    
    dx_sm = np.diff(x_smooth, prepend=x_smooth[-1])
    dy_sm = np.diff(y_smooth, prepend=y_smooth[-1])
    ds_sm = np.sqrt(dx_sm**2 + dy_sm**2)
    raw_len = np.sum(ds_sm)
    
    target_len = st.session_state.track_length
    scale_factor = target_len / raw_len

    x_track = x_smooth * scale_factor
    y_track = y_smooth * scale_factor

    dx = np.gradient(x_track)
    dy = np.gradient(y_track)
    ds = np.sqrt(dx**2 + dy**2)
    s_arr = np.concatenate([[0], np.cumsum(ds)[:-1]])
    num_pts = len(x_track)

    tx = dx / ds
    ty = dy / ds
    nx = -ty
    ny = tx

    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    kappa_raw = (dx * ddy - dy * ddx) / (ds**3 + 1e-9)
    kappa_arr = gaussian_filter1d(kappa_raw, sigma=200, mode='wrap')

    track_w = 12.0 
    bike_w  = 0.8  

    x_outer = x_track + (track_w / 2.0) * nx
    y_outer = y_track + (track_w / 2.0) * ny
    x_inner = x_track - (track_w / 2.0) * nx
    y_inner = y_track - (track_w / 2.0) * ny

    u_fine = np.linspace(0, 1, num_pts)
    n_ideal = -4.2 * np.sin(4 * u_fine * 2 * np.pi) * np.cos(3 * u_fine * 2 * np.pi)
    
    x_ideal = x_track + n_ideal * nx
    y_ideal = y_track + n_ideal * ny

    elev_keys = sorted(list(st.session_state.custom_elev.keys()))
    elev_vals = [st.session_state.custom_elev[k] for k in elev_keys]
    bank_keys = sorted(list(st.session_state.custom_bank.keys()))
    bank_vals = [st.session_state.custom_bank[k] for k in bank_keys]

    if elev_keys[-1] < target_len:
        elev_keys.append(target_len)
        elev_vals.append(elev_vals[0])
    if bank_keys[-1] < target_len:
        bank_keys.append(target_len)
        bank_vals.append(bank_vals[0])

    f_elev = interpolate.interp1d(elev_keys, elev_vals, kind='linear', fill_value="extrapolate")
    f_bank = interpolate.interp1d(bank_keys, bank_vals, kind='linear', fill_value="extrapolate")

    h_grid_full = f_elev(s_arr)
    bank_arr = f_bank(s_arr)

    dh_ds = np.gradient(h_grid_full, s_arr)
    slope_arr = np.degrees(np.arctan(dh_ds))

    return (x_track, y_track, x_outer, y_outer, x_inner, y_inner, 
            x_ideal, y_ideal, nx, ny, num_pts, s_arr, slope_arr, 
            bank_arr, h_grid_full, kappa_arr, track_w, bike_w, n_ideal, target_len)


# ==========================================
# PHASE 1: UPLOAD
# ==========================================
if st.session_state.phase == 'upload':
    st.title("3D Motorcycle Track Simulator & Physics Analyzer 🏍️🏁")
    
    st.markdown("""
    ### Welcome! 
    This application transforms any 2D race track layout into an interactive **3D physical simulation environment**. 
    
    #### 💡 Pro Tip: Using AI & Online Data for Topography Maps
    You can easily generate color-coded topographical track maps using **Artificial Intelligence (AI)** combined with real-world public data available on the internet:
    1. Ask an AI assistant (like ChatGPT or Claude) or search online for the elevation profile and corner banking data of your desired circuit.
    2. Prompt an AI image generator or use Python/Matplotlib to map those elevation values into a continuous color gradient (Hue heatmap: Blue for low/descents, Red for high/climbs).
    3. Upload that colored map here and check the **"Extract Topography from Colors (Heatmap)"** box in the sidebar to automatically reconstruct real-world 3D elevations!
    """)
    
    st.markdown("#### 🏁 Track Reference Example: Mugello Circuit (Topographical Heatmap)")
    st.info("The system reads the color spectrum below (Blue-to-Red heatmap) to reconstruct continuous 3D elevations automatically:")
    
    col_ex1, col_ex2, col_ex3 = st.columns([1, 2, 1])
    with col_ex2:
        img_filename = "track_example.png"
        if os.path.exists(img_filename):
            st.image(img_filename, caption="Mugello Circuit - Elevation Heatmap Layout", use_container_width=True)
        else:
            st.warning("`track_example.png` not found in root directory. Please upload any track layout image below to begin.")

    st.markdown("---")
    
    st.sidebar.markdown("### Handling Logos / Map Noise")
    extraction_method = st.sidebar.selectbox("Track Extraction Method", 
        ["1. Standard Extraction (Smart Adaptive Mode)", 
         "2. Color Masking (Ignore Red/White Logos)", 
         "3. Mathematical Smoothing (Adjust Spline 's')"])
    
    smooth_factor = 3.0
    if "Mathematical" in extraction_method:
        smooth_factor = st.sidebar.slider("Spline Smoothing Factor (s)", 0.0, 500.0, 5.0, step=1.0)
        st.sidebar.info("Higher values mathematically force the function to smooth over anomalies.")

    st.sidebar.markdown("### Color Topography")
    use_color_extraction = st.sidebar.checkbox("Extract Topography from Colors (Heatmap)", value=False)
    
    heatmap_min = 0.0
    heatmap_max = 10.0
    if use_color_extraction:
        heatmap_min = st.sidebar.number_input("Blue (Minimum) Elevation Value (m)", value=3.0)
        heatmap_max = st.sidebar.number_input("Red (Maximum) Elevation Value (m)", value=8.0)

    st.sidebar.markdown("### Base Track Parameters")
    temp_length = st.sidebar.number_input("Total Track Length (m)", value=st.session_state.track_length, step=100.0)
    
    uploaded_file = st.file_uploader("Upload your track layout image (PNG, JPG, JPEG)", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        image_bytes = uploaded_file.read()
        raw_geom, extracted_elevs = extract_track_data_from_image(
            image_bytes, 
            use_heatmap=use_color_extraction, 
            min_val=heatmap_min, 
            max_val=heatmap_max,
            target_length=temp_length,
            extraction_method=extraction_method,
            smooth_factor=smooth_factor
        )
        
        if raw_geom is None:
            st.error("Failed to extract geometry. The image might be too complex or empty.")
        else:
            st.session_state.raw_geometry = raw_geom
            st.session_state.track_length = temp_length
            
            if use_color_extraction and extracted_elevs:
                st.session_state.custom_elev = extracted_elevs
                st.success("Geometry and continuous color topography extracted successfully!")
            else:
                st.success("Geometry extracted successfully!")
                
            if st.button("Proceed to Track Design", type="primary"):
                st.session_state.phase = 'design'
                st.rerun()

# ==========================================
# PHASE 2: DESIGN (ELEVATION & BANKING)
# ==========================================
elif st.session_state.phase == 'design':
    st.title("Step 2: Track Design & Topography Review")
    
    if st.sidebar.button("Back to Upload"):
        st.session_state.phase = 'upload'
        st.rerun()
        
    st.sidebar.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("**Navigate the track to review or edit topography waypoints:**")
        design_s = st.slider("Position on Track (m)", 0.0, st.session_state.track_length, 0.0, step=1.0)
        
        x_sm, y_sm = st.session_state.raw_geometry
        dx_sm = np.diff(x_sm, prepend=x_sm[-1])
        dy_sm = np.diff(y_sm, prepend=y_sm[-1])
        ds_sm = np.sqrt(dx_sm**2 + dy_sm**2)
        raw_len = np.sum(ds_sm)
        scale_f = st.session_state.track_length / raw_len
        x_d = x_sm * scale_f
        y_d = y_sm * scale_f
        
        idx_d = int((design_s / st.session_state.track_length) * (len(x_d) - 1))
        
        fig_map, ax_map = plt.subplots(figsize=(8, 6), facecolor='#0b0e14')
        ax_map.set_facecolor='#0b0e14'
        ax_map.plot(x_d, y_d, color='#475569', lw=2.0, label='Centerline')
        ax_map.plot([x_d[idx_d]], [y_d[idx_d]], 'ro', ms=10, label=f'Current Pos: {design_s:.0f}m')
        ax_map.set_aspect('equal')
        ax_map.axis('off')
        ax_map.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='white')
        st.pyplot(fig_map)

    with col2:
        st.markdown("### Data at Current Position")
        
        elev_keys = sorted(list(st.session_state.custom_elev.keys()))
        elev_vals = [st.session_state.custom_elev[k] for k in elev_keys]
        if elev_keys[-1] < st.session_state.track_length:
            elev_keys.append(st.session_state.track_length)
            elev_vals.append(elev_vals[0])
        f_e = interpolate.interp1d(elev_keys, elev_vals, kind='linear', fill_value="extrapolate")
        current_est_elev = float(f_e(design_s))
        
        input_elev = st.number_input(f"Elevation at {design_s:.0f}m", value=current_est_elev, step=0.5)
        input_bank = st.number_input(f"Banking at {design_s:.0f}m", value=0.0, step=0.5)
        
        if st.button("Save Waypoint"):
            st.session_state.custom_elev[design_s] = input_elev
            st.session_state.custom_bank[design_s] = input_bank
            st.success(f"Saved data at {design_s:.0f}m")
            
        st.markdown(f"Total Waypoints: {len(st.session_state.custom_elev)}")
        
        if st.button("Clear All Waypoints"):
            st.session_state.custom_elev = {0.0: 0.0}
            st.session_state.custom_bank = {0.0: 0.0}
            st.rerun()

        st.markdown("---")
        if st.button("Confirm Track Design & Run Simulation", type="primary"):
            st.session_state.phase = 'simulate'
            st.rerun()

# ==========================================
# PHASE 3: SIMULATION
# ==========================================
elif st.session_state.phase == 'simulate':
    st.sidebar.title("Simulation Control Panel")
    if st.sidebar.button("Back to Track Design"):
        st.session_state.phase = 'design'
        st.rerun()
    
    st.sidebar.markdown("---")
    
    (x_track, y_track, x_outer, y_outer, x_inner, y_inner, 
     x_ideal, y_ideal, nx, ny, num_pts, s_arr, slope_arr, 
     bank_arr, h_grid_full, kappa_arr, track_w, bike_w, n_ideal, t_len) = build_physics_model()

    s_val = st.sidebar.slider('Track Distance s (m)', 0.0, float(t_len), 0.0, step=1.0)
    n_val = st.sidebar.slider('Lateral Offset n (m)', -8.0, 8.0, 0.0, step=0.1)
    V_kmh = st.sidebar.slider('Rider Speed V (km/h)', 30.0, 300.0, 140.0, step=1.0)
    user_theta_deg = st.sidebar.slider('Manual Lean θ (deg)', -60.0, 60.0, 0.0, step=0.1)

    st.sidebar.markdown("---")
    run_animation = st.sidebar.button("Generate 3D Animation Loop")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Window 1: Custom 2D Track Map")
        fig_2d, ax_2d = plt.subplots(figsize=(10, 8.5), facecolor='#0b0e14')
        ax_2d.set_facecolor='#0b0e14'
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

        ax_2d.fill(np.concatenate([x_outer, x_inner[::-1]]), 
                   np.concatenate([y_outer, y_inner[::-1]]), 
                   color='#1e293b', alpha=0.9)

        ax_2d.plot(x_outer, y_outer, color='#475569', lw=1.5, ls='--')
        ax_2d.plot(x_inner, y_inner, color='#475569', lw=1.5, ls='--')
        ax_2d.plot(x_track, y_track, color='#334155', lw=1.0, ls=':')
        ax_2d.plot(x_ideal, y_ideal, color='#38bdf8', lw=2.5, label='Calculated Optimal Trajectory')

        idx = int((s_val / t_len) * (num_pts - 1))
        idx = np.clip(idx, 0, num_pts - 1)
        ax_2d.plot([x_track[idx] + n_val * nx[idx]], [y_track[idx] + n_val * ny[idx]], 'ro', ms=12, mec='#ffffff', mew=2.5, label='Rider Position', zorder=10)

        ax_2d.set_aspect('equal')
        ax_2d.axis('off')
        ax_2d.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='white')
        st.pyplot(fig_2d)

    def draw_3d_scene(ax_3d, base_s, base_n, current_V, current_theta, anim_frame=None, show_future_path=True):
        idx_start = int((base_s / t_len) * (num_pts - 1))
        idx_start = np.clip(idx_start, 0, num_pts - 1)

        slope_deg = float(slope_arr[idx_start])
        bank_deg  = float(bank_arr[idx_start])
        elev_curr = float(h_grid_full[idx_start])
        
        kappa_curr_3d = -kappa_arr[idx_start]
        R_curr = 1.0 / (abs(kappa_curr_3d) + 1e-5)
        
        g = 9.81
        V_ms = current_V / 3.6
        mu_val = 1.15
        
        theta_req_rad = np.arctan((V_ms**2 * kappa_curr_3d) / g)
        theta_req_deg = np.degrees(theta_req_rad)
        
        if abs(current_theta) > 0.1:
            effective_bike_lean = current_theta
        else:
            effective_bike_lean = theta_req_deg

        theta_max_deg = np.degrees(np.arctan(mu_val))
        V_max_kmh = np.sqrt(mu_val * g * R_curr) * 3.6

        rad_p = np.radians(slope_deg)
        rad_b = np.radians(bank_deg)
        
        def transform_terrain(X, Y):
            cos_p, sin_p = np.cos(rad_p), np.sin(rad_p)
            cos_b, sin_b = np.cos(rad_b), np.sin(rad_b)
            tX = X * cos_b - Y * sin_b
            tY = X * sin_b + Y * cos_b
            tZ = tY * sin_p
            return tX, tY, tZ

        horizon_dist = 60.0 if anim_frame is None else 120.0
        num_h = 50 if anim_frame is None else 100
        ds_step = horizon_dist / num_h
        s_ahead = np.linspace(0, horizon_dist, num_h)
        
        dpsi_road = 0.0
        x_centerline_ahead = [0.0]
        y_centerline_ahead = [0.0]
        
        dpsi_bike = 0.0
        n_curr = base_n
        x_actual_ahead = [n_curr]
        dpsi_bike_arr = [0.0]

        for i in range(num_h):
            ds_curr = s_ahead[i]
            idx_h = int(((base_s + ds_curr) % t_len / t_len) * (num_pts - 1))
            
            if i > 0:
                k_road_3d = -kappa_arr[idx_h]
                dpsi_road += k_road_3d * ds_step
                x_centerline_ahead.append(x_centerline_ahead[-1] + np.sin(dpsi_road) * ds_step)
                y_centerline_ahead.append(y_centerline_ahead[-1] + np.cos(dpsi_road) * ds_step)
                
                if abs(current_theta) < 0.1:
                    k_bike_3d = k_road_3d
                else:
                    k_bike_3d = (g * np.tan(np.radians(current_theta))) / (V_ms**2 + 1e-5)
                
                dpsi_bike += k_bike_3d * ds_step
                psi_rel = dpsi_bike - dpsi_road
                n_curr -= np.sin(psi_rel) * ds_step
                x_actual_ahead.append(n_curr)
                dpsi_bike_arr.append(psi_rel)

        xc_arr = np.array(x_centerline_ahead)
        yc_arr = np.array(y_centerline_ahead)
        
        x_road_left = []
        x_road_right = []
        x_ideal_ahead = []
        
        dpsi_road_temp = 0.0
        for i in range(num_h):
            idx_h = int(((base_s + s_ahead[i]) % t_len / t_len) * (num_pts - 1))
            if i > 0:
                dpsi_road_temp += (-kappa_arr[idx_h]) * ds_step
                
            x_road_left.append(xc_arr[i] - track_w/2.0 * np.cos(dpsi_road_temp))
            x_road_right.append(xc_arr[i] + track_w/2.0 * np.cos(dpsi_road_temp))
            x_ideal_ahead.append(xc_arr[i] - n_ideal[idx_h] * np.cos(dpsi_road_temp))
            
        x_road_left = np.array(x_road_left)
        x_road_right = np.array(x_road_right)
        x_ideal_ahead = np.array(x_ideal_ahead)
        x_actual_ahead = np.array(x_actual_ahead)
        y_ahead = yc_arr

        grid_y = np.linspace(0, horizon_dist, 6)
        grid_x = np.linspace(-10, 10, 6)
        GX, GY = np.meshgrid(grid_x, grid_y)
        tGX, tGY, tGZ = transform_terrain(GX, GY)
        ax_3d.plot_wireframe(tGX, tGY, tGZ, color='#0284c7', alpha=0.20, lw=0.6, linestyle=':')

        Y_mesh = np.tile(y_ahead, (2, 1)).T
        X_mesh = np.column_stack([x_road_left, x_road_right])
        tr_X, tr_Y, tr_Z = transform_terrain(X_mesh, Y_mesh)
        ax_3d.plot_surface(tr_X, tr_Y, tr_Z, color='#1e293b', alpha=0.92, edgecolor='#334155', lw=0.3)
        
        b1_x, b1_y, b1_z = transform_terrain(x_road_left, y_ahead)
        b2_x, b2_y, b2_z = transform_terrain(x_road_right, y_ahead)
        ax_3d.plot(b1_x, b1_y, b1_z, color='#64748b', lw=1.5, ls='--')
        ax_3d.plot(b2_x, b2_y, b2_z, color='#64748b', lw=1.5, ls='--')

        id_x, id_y, id_z = transform_terrain(x_ideal_ahead, y_ahead)
        ax_3d.plot(id_x, id_y, id_z + 0.05, color='#38bdf8', lw=3.0)
        
        is_off_track_now = (base_n < -track_w/2.0 + bike_w/2.0) or (base_n > track_w/2.0 - bike_w/2.0)
        off_left_mask  = x_actual_ahead > (track_w/2.0 - bike_w/2.0)
        off_right_mask = x_actual_ahead < (-track_w/2.0 + bike_w/2.0)
        off_track_ahead_mask = off_left_mask | off_right_mask
        is_off_track_ahead = np.any(off_track_ahead_mask)
        is_speed_breach = current_V > V_max_kmh or abs(effective_bike_lean) > theta_max_deg
        
        actual_color = '#ef4444' if (is_off_track_now or is_off_track_ahead or is_speed_breach) else '#22c55e'
        
        if show_future_path:
            dpsi_temp = 0.0
            act_x_world = []
            for i in range(num_h):
                if i > 0:
                    dpsi_temp += (-kappa_arr[int(((base_s + s_ahead[i]) % t_len / t_len) * (num_pts - 1))]) * ds_step
                act_x_world.append(xc_arr[i] - x_actual_ahead[i] * np.cos(dpsi_temp))
            
            act_x, act_y, act_z = transform_terrain(np.array(act_x_world), y_ahead)
            ax_3d.plot(act_x, act_y, act_z + 0.08, color=actual_color, lw=3.5, ls='-')

        curr_idx = anim_frame if anim_frame is not None else 0
        
        dpsi_bike_frame = 0.0
        for i in range(curr_idx + 1):
            if i > 0:
                dpsi_bike_frame += (-kappa_arr[int(((base_s + s_ahead[i]) % t_len / t_len) * (num_pts - 1))]) * ds_step
                
        bx_cart = xc_arr[curr_idx] - x_actual_ahead[curr_idx] * np.cos(dpsi_bike_frame)
        by_cart = yc_arr[curr_idx]
        byaw = dpsi_bike_arr[curr_idx]
        
        cx_base = xc_arr[curr_idx]
        cy_base = yc_arr[curr_idx]

        rad_l = np.radians(effective_bike_lean)
        
        def transform_bike(x, y, z):
            cos_l, sin_l = np.cos(rad_l), np.sin(rad_l)
            xr1 = x * cos_l + z * sin_l
            zr1 = -x * sin_l + z * cos_l
            yr1 = y

            cos_yaw, sin_yaw = np.cos(byaw), np.sin(byaw)
            xr_yaw = xr1 * cos_yaw - yr1 * sin_yaw
            yr_yaw = xr1 * sin_yaw + yr1 * cos_yaw

            xr_pos = xr_yaw + bx_cart
            yr_pos = yr_yaw + by_cart
            zr_pos = zr1

            cos_p, sin_p = np.cos(rad_p), np.sin(rad_p)
            yr_pitch = yr_pos * cos_p - zr_pos * sin_p
            zr_pitch = yr_pos * sin_p + zr_pos * cos_p

            cos_b, sin_b = np.cos(rad_b), np.sin(rad_b)
            xr_bank = xr_pos * cos_b - yr_pitch * sin_b
            yr_bank = xr_pos * sin_b + yr_pitch * cos_b
            zr_bank = zr_pitch

            return xr_bank, yr_bank, zr_bank

        c_white, c_red, c_gold, c_tire = '#f8fafc', '#ef4444', '#eab308', '#111827'
        t_wheel = np.linspace(0, 2*np.pi, 24)
        r_w = 0.80 
        
        for w_off in np.linspace(-0.25, 0.25, 6):
            xr_t, yr_t, zr_t = transform_bike(r_w * np.sin(t_wheel), -1.5 + w_off, r_w + r_w * np.cos(t_wheel))
            ax_3d.plot(xr_t, yr_t, zr_t, color=c_tire, lw=4.5)
            xf_t, yf_t, zf_t = transform_bike(r_w * np.sin(t_wheel), 1.8 + w_off, r_w + r_w * np.cos(t_wheel))
            ax_3d.plot(xf_t, yf_t, zf_t, color=c_tire, lw=4.0)

        fx1, fy1, fz1 = transform_bike(-0.25, 1.8, 0.6)
        fx2, fy2, fz2 = transform_bike(-0.25, 1.4, 1.8)
        ax_3d.plot([fx1, fx2], [fy1, fy2], [fz1, fz2], color=c_gold, lw=5.0)
        
        fx3, fy3, fz3 = transform_bike(0.25, 1.8, 0.6)
        fx4, fy4, fz4 = transform_bike(0.25, 1.4, 1.8)
        ax_3d.plot([fx3, fx4], [fy3, fy4], [fz3, fz4], color=c_gold, lw=5.0)

        nose_verts = [
            [transform_bike(0.0, 2.4, 1.4), transform_bike(-0.7, 1.4, 1.5), transform_bike(0.0, 1.4, 1.9)],
            [transform_bike(0.0, 2.4, 1.4), transform_bike(0.7, 1.4, 1.5), transform_bike(0.0, 1.4, 1.9)],
            [transform_bike(-0.75, 0.6, 1.5), transform_bike(0.75, 0.6, 1.5), transform_bike(0.6, -0.4, 1.3), transform_bike(-0.6, -0.4, 1.3)],
            [transform_bike(-0.4, -1.2, 1.5), transform_bike(0.4, -1.2, 1.5), transform_bike(0.0, -2.0, 1.8)]
        ]
        ax_3d.add_collection3d(Poly3DCollection(nose_verts, facecolors=c_white, edgecolors='#cbd5e1', alpha=0.95))

        stripe_verts = [
            [transform_bike(-0.72, 1.1, 1.2), transform_bike(-0.75, 0.5, 1.1), transform_bike(-0.70, 0.5, 0.95), transform_bike(-0.65, 1.1, 1.05)],
            [transform_bike(0.72, 1.1, 1.2), transform_bike(0.75, 0.5, 1.1), transform_bike(0.70, 0.5, 0.95), transform_bike(0.65, 1.1, 1.05)]
        ]
        ax_3d.add_collection3d(Poly3DCollection(stripe_verts, facecolors=c_red, edgecolors=c_red, alpha=0.9))

        rx, ry, rz = transform_bike(0.0, 0.1, 2.3)
        ax_3d.scatter([rx], [ry], [rz], color='#f8fafc', s=280, depthshade=False, edgecolors='#38bdf8', linewidths=2)

        cx_t, cy_t, _ = transform_terrain(np.array([cx_base]), np.array([cy_base]))
        c_x = cx_t[0]
        c_y = cy_t[0]
        ax_3d.set_xlim(c_x - 10.0, c_x + 10.0)
        ax_3d.set_ylim(c_y - 4.0, c_y + 50.0)
        ax_3d.set_zlim(-1.0, 10.0)

        ax_3d.text2D(0.02, 0.95, "SYNCHRONIZED 3D MOTORCYCLE & HORIZON DYNAMICS", transform=ax_3d.transAxes, 
                     color='#f8fafc', fontsize=11, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.4', facecolor='#0f172a', edgecolor='#334155', alpha=0.9))

        n_display = x_actual_ahead[curr_idx]
        
        if is_off_track_now:
            alert_msg = f"[RED ALERT]: OFF-TRACK BREACH! (Position n={n_display:+.1f}m Exceeds Asphalt)"
            alert_color = '#ef4444'
        elif is_off_track_ahead and show_future_path:
            first_off_i = np.where(off_track_ahead_mask)[0][0]
            dist_off = s_ahead[first_off_i]
            alert_msg = f"[RED ALERT]: OFF-TRACK BREACH! (Leaves Asphalt {dist_off:.0f}m Ahead)"
            alert_color = '#ef4444'
        elif is_speed_breach:
            alert_msg = f"[RED ALERT]: VELOCITY / GRIP LIMIT BREACH! (Speed {current_V:.0f} > Limit {V_max_kmh:.0f})"
            alert_color = '#ef4444'
        else:
            alert_msg = STABLE_LABEL
            alert_color = '#22c55e'

        ax_3d.text2D(0.02, 0.75, alert_msg, transform=ax_3d.transAxes, color=alert_color, fontsize=10, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.4', facecolor='#0f172a', edgecolor='#334155', alpha=0.9))

        hud_s = (base_s + s_ahead[curr_idx]) % t_len
        telemetry_str = (
            f"Distance: {hud_s:.0f}m / {t_len:.0f}m   |   Elevation: {elev_curr:.1f}m   |   Curve Radius: {R_curr:.1f}m\n"
            f"Speed: {current_V:.0f} km/h (Limit: {V_max_kmh:.0f})   |   Bike Lean (θ): {effective_bike_lean:+.1f}° (Req: {theta_req_deg:+.1f}°)\n"
            f"Lateral Offset (n): {n_display:+.1f} m   |   Slope: {slope_deg:+.1f}°   |   Bank: {bank_deg:+.1f}°"
        )
        ax_3d.text2D(0.02, 0.85, telemetry_str, transform=ax_3d.transAxes, color='#38bdf8', fontsize=9,
                     bbox=dict(boxstyle='round,pad=0.4', facecolor='#0f172a', edgecolor='#334155', alpha=0.9))

        return is_off_track_now, is_off_track_ahead, is_speed_breach, elev_curr, R_curr, V_max_kmh, effective_bike_lean, theta_req_deg, slope_deg, bank_deg


    with col2:
        st.subheader("Window 2: 3D Motorcycle Dynamics")
        
        if run_animation:
            with st.spinner("Generating 3D animation loop... Please wait a few moments."):
                fig_anim = plt.figure(figsize=(9, 7), facecolor='#0b0e14')
                ax_anim = fig_anim.add_subplot(111, projection='3d', facecolor='#0b0e14')
                
                num_frames = 40
                
                def update(frame):
                    ax_anim.clear()
                    draw_3d_scene(ax_anim, s_val, n_val, V_kmh, user_theta_deg, anim_frame=frame, show_future_path=False)
                    ax_anim.view_init(elev=16, azim=-90)
                    ax_anim.set_box_aspect((1.0, 1.8, 0.45))
                    ax_anim.set_axis_off()

                anim = FuncAnimation(fig_anim, update, frames=num_frames, interval=66, blit=False)
                gif_path = "simulation_loop.gif"
                anim.save(gif_path, writer='pillow', fps=15)
                plt.close(fig_anim)
                st.image(gif_path, use_container_width=True)
        else:
            fig_3d = plt.figure(figsize=(11, 8.5), facecolor='#0b0e14')
            ax_3d = fig_3d.add_subplot(111, projection='3d', facecolor='#0b0e14')
            plt.subplots_adjust(left=0.04, right=0.96, top=0.96, bottom=0.05)
            
            is_off_now, is_off_ahead, is_speed_b, elev, r_c, v_max, eff_lean, req_lean, slp, bnk = draw_3d_scene(ax_3d, s_val, n_val, V_kmh, user_theta_deg, show_future_path=True)
            
            ax_3d.view_init(elev=16, azim=-90)
            ax_3d.set_box_aspect((1.0, 1.8, 0.45))
            ax_3d.set_axis_off()
            st.pyplot(fig_3d)

    if not run_animation:
        st.markdown("### Real-Time Telemetry Data")
        if is_off_now or is_off_ahead:
            st.error("OFF-TRACK BREACH! The vehicle is outside track limits.")
        elif is_speed_b:
            st.error(f"VELOCITY / GRIP LIMIT BREACH! Speed {V_kmh:.0f} km/h exceeds the limit of {v_max:.0f} km/h.")
        else:
            st.success("STABLE: Vehicle is on the asphalt and within performance limits.")

        st.markdown(f"""
        - **Distance:** {s_val:.0f} m out of {t_len:.0f} m | **Elevation:** {elev:.1f} m | **Curve Radius:** {r_c:.1f} m
        - **Current Speed:** {V_kmh:.0f} km/h | **Max Cornering Speed:** {v_max:.0f} km/h
        - **Lean Angle:** {eff_lean:+.1f}° | **Required Angle:** {req_lean:+.1f}°
        - **Lateral Offset (n):** {n_val:+.1f} m | **Slope:** {slp:+.1f}° | **Bank (Camber):** {bnk:+.1f}°
        """)