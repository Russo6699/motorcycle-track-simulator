import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.interpolate as interpolate
from scipy.ndimage import gaussian_filter1d

# Attempt to import OpenCV for image extraction
try:
    import cv2
except ImportError:
    print("ERROR: OpenCV is not installed. Please run 'pip install opencv-python' in your terminal and try again.")
    sys.exit()

plt.style.use('dark_background')

STABLE_LABEL   = "[STABLE]: RIDER ON ASPHALT & WITHIN SAFETY MARGIN"
ALERT_LABEL    = "[RED ALERT]: VELOCITY / GRIP LIMIT BREACH!"
OFFTRACK_LABEL = "[RED ALERT]: OFF-TRACK BREACH!"

# --- 1. Computer Vision: Extract Geometry Directly from 'images.png' ---
img_path = 'images.png'
if not os.path.exists(img_path):
    print(f"ERROR: Cannot find '{img_path}'. Make sure it is in the same folder as this script: {os.getcwd()}")
    sys.exit()

# Read the image in grayscale
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# The track is black on a white background. Threshold to invert it (Track=255, BG=0)
_, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

# Extract contours from the image
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Find the largest contour (which is the track itself)
best_contour = max(contours, key=cv2.contourArea)
wp_raw = best_contour.reshape(-1, 2).astype(float)

# In images, Y goes downwards. Invert Y to standard Cartesian coordinates.
wp_raw[:, 1] = -wp_raw[:, 1]

# Downsample the raw pixel contour to ~500 points to smooth out pixel "stairs"
idx = np.linspace(0, len(wp_raw) - 1, 500, dtype=int)
wp = wp_raw[idx]

# Align the Start/Finish line and direction
# Mugello's main straight is at the bottom (lowest Y). We find the lowest Y to orient.
lowest_y_idx = np.argmin(wp[:, 1])

# Ensure correct clockwise sequence (Moving Left to Right on the bottom straight)
next_idx = (lowest_y_idx + 15) % len(wp)
prev_idx = (lowest_y_idx - 15) % len(wp)
if wp[next_idx, 0] < wp[prev_idx, 0]:
    wp = wp[::-1] # Reverse array if going the wrong way
    lowest_y_idx = len(wp) - 1 - lowest_y_idx

# Roll array so index 0 is on the main straight (Start/Finish line approx)
wp = np.roll(wp, -lowest_y_idx, axis=0)
wp = np.roll(wp, -len(wp)//20, axis=0) # Shift slightly right to middle of the straight

# Create high-density, smooth track data (30,000 points) based EXACTLY on the image
tck_base, _ = interpolate.splprep([wp[:,0], wp[:,1]], s=3, per=True) # Slight smoothing of pixel edges
u_dense = np.linspace(0, 1, 30000)
x_smooth, y_smooth = interpolate.splev(u_dense, tck_base)

# Scale track perimeter EXACTLY to 5,245 meters
dx_sm = np.diff(x_smooth, prepend=x_smooth[-1])
dy_sm = np.diff(y_smooth, prepend=y_smooth[-1])
ds_sm = np.sqrt(dx_sm**2 + dy_sm**2)
track_len = np.sum(ds_sm)
scale_factor = 5245.0 / track_len

x_track = x_smooth * scale_factor
y_track = y_smooth * scale_factor

# Calculate exact physics and distances
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
kappa_arr = gaussian_filter1d(kappa_raw, sigma=200, mode='wrap') # Smooth kappa for reliable 3D physics

track_w = 12.0 
bike_w  = 0.8  
offtrack_limit_n = (track_w - bike_w) / 2.0 

x_outer = x_track + (track_w / 2.0) * nx
y_outer = y_track + (track_w / 2.0) * ny
x_inner = x_track - (track_w / 2.0) * nx
y_inner = y_track - (track_w / 2.0) * ny

# Ideal & Practical Trajectories
u_fine = np.linspace(0, 1, num_pts)
n_ideal = -4.2 * np.sin(4 * u_fine * 2 * np.pi) * np.cos(3 * u_fine * 2 * np.pi)
x_ideal = x_track + n_ideal * nx
y_ideal = y_track + n_ideal * ny

n_pract = n_ideal + 1.8 * np.cos(6 * u_fine * 2 * np.pi)
x_pract = x_track + n_pract * nx
y_pract = y_track + n_pract * ny

# --- 2. Digitized Elevation Profile Sequence ---
elevation_points_s = np.array([0, 300, 800, 1150, 1400, 2000, 2400, 2900, 3400, 4000, 4700, 5245], dtype=float)
elevation_points_h = np.array([256.0, 267.0, 268.0, 285.0, 281.0, 262.0, 243.0, 272.0, 271.0, 260.0, 247.0, 257.0], dtype=float)

tck_elev = interpolate.splrep(elevation_points_s, elevation_points_h, s=0, per=True)
h_grid_full = interpolate.splev(s_arr, tck_elev)

dh_ds = np.gradient(h_grid_full, s_arr)
slope_arr = np.degrees(np.arctan(dh_ds))
bank_arr  = 6.0 * np.cos(4 * u_fine * 2 * np.pi)

# --- 3. Window 1: 2D Track Map Window (PERFECT IMAGE EXTRACTION & ANNOTATION) ---
fig_2d = plt.figure("Window 1: Mugello Circuit 2D Track Map", figsize=(10, 8.5), facecolor='#0b0e14')
ax_2d = fig_2d.add_subplot(111, facecolor='#0b0e14')
plt.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.20)

ax_2d.fill(np.concatenate([x_outer, x_inner[::-1]]), 
           np.concatenate([y_outer, y_inner[::-1]]), 
           color='#1e293b', alpha=0.9, label='12m Track Asphalt')

ax_2d.plot(x_outer, y_outer, color='#475569', lw=1.5, ls='--', label='Track Limits')
ax_2d.plot(x_inner, y_inner, color='#475569', lw=1.5, ls='--')
ax_2d.plot(x_track, y_track, color='#334155', lw=1.0, ls=':', label='Centerline')

ax_2d.plot(x_ideal, y_ideal, color='#38bdf8', lw=2.5, label='Ideal Optimal Trajectory')
ax_2d.plot(x_pract, y_pract, color='#f59e0b', lw=2.0, ls='-.', label='Practical Trajectory')

rider_marker, = ax_2d.plot([x_pract[0]], [y_pract[0]], 'ro', ms=12, mec='#ffffff', mew=2.5, label='Rider Position', zorder=10)

# Beautifully annotated labels pointing EXACTLY to the apexes using connecting lines
# Format: (Percent on track, "Label Name", Offset_X, Offset_Y, horizontal_align, vertical_align)
turns_labels = [
    (0.00, "Main Straight", 0, -250, 'center', 'top'),
    (0.19, "T1: San Donato", 250, 0, 'left', 'center'),
    (0.28, "T2/T3: Luco", 150, 200, 'left', 'bottom'),
    (0.38, "T4/T5: Materassi", 0, 200, 'center', 'bottom'),
    (0.45, "T6/T7: Casanova", 0, -200, 'center', 'top'),
    (0.53, "T8/T9: Arrabbiata", -150, 250, 'right', 'bottom'),
    (0.64, "T10/T11: Scarperia", -200, 200, 'right', 'bottom'),
    (0.72, "T12: Correntaio", 200, -150, 'left', 'top'),
    (0.82, "T13/T14: Biondetti", -150, 200, 'right', 'bottom'),
    (0.93, "T15: Bucine", -300, 0, 'right', 'center')
]

for percent, name, off_x, off_y, ha_val, va_val in turns_labels:
    idx_t = int(percent * num_pts) % num_pts
    x_pt = x_track[idx_t]
    y_pt = y_track[idx_t]
    
    # Draw line with text exactly pointing to the coordinate
    ax_2d.annotate(
        name,
        xy=(x_pt, y_pt),
        xytext=(x_pt + off_x, y_pt + off_y),
        color='#cbd5e1', fontsize=9, fontweight='bold',
        ha=ha_val, va=va_val,
        arrowprops=dict(arrowstyle="-", color='#475569', lw=1.5, alpha=0.9)
    )

# Force map to fill the entire window area symmetrically
ax_2d.set_xlim(np.min(x_outer) - 350, np.max(x_outer) + 350)
ax_2d.set_ylim(np.min(y_outer) - 350, np.max(y_outer) + 350)

ax_2d.set_title("MUGELLO CIRCUIT - DIRECT IMAGE EXTRACTION (1:1)\n(Click anywhere on map or use sliders below)", color='#f8fafc', fontsize=12, fontweight='bold')
ax_2d.set_aspect('equal')
ax_2d.axis('off')
ax_2d.legend(loc='lower center', facecolor='#1e293b', edgecolor='#334155', fontsize=9, ncol=3, bbox_to_anchor=(0.5, -0.05))

# SLIDERS in Window 1
ax_s1 = plt.axes([0.15, 0.10, 0.70, 0.025], facecolor='#1e293b')
ax_n1 = plt.axes([0.15, 0.05, 0.70, 0.025], facecolor='#1e293b')

slider_s1 = Slider(ax_s1, 'Track Distance s (m)', 0.0, 5245.0, valinit=600.0, valfmt='%.0f m', color='#38bdf8')
slider_n1 = Slider(ax_n1, 'Lateral Offset n (m)', -8.0, 8.0, valinit=1.8, valfmt='%.1f m', color='#f59e0b')


# --- 4. Window 2: 3D Motorcycle & Horizon Trajectory Dynamics Window ---
fig_3d = plt.figure("Window 2: 3D Motorcycle & Horizon Trajectory Dynamics", figsize=(11, 8.5), facecolor='#0b0e14')
ax_3d = fig_3d.add_subplot(111, projection='3d', facecolor='#0b0e14')
plt.subplots_adjust(left=0.04, right=0.96, top=0.96, bottom=0.22)

# SLIDERS in Window 2
ax_s2     = plt.axes([0.15, 0.14, 0.70, 0.022], facecolor='#1e293b')
ax_n2     = plt.axes([0.15, 0.10, 0.70, 0.022], facecolor='#1e293b')
ax_v2     = plt.axes([0.15, 0.06, 0.70, 0.022], facecolor='#1e293b')
ax_theta2 = plt.axes([0.15, 0.02, 0.70, 0.022], facecolor='#1e293b')

slider_s2     = Slider(ax_s2,     'Track Distance s (m)', 0.0, 5245.0, valinit=600.0, valfmt='%.0f m', color='#38bdf8')
slider_n2     = Slider(ax_n2,     'Lateral Offset n (m)', -8.0, 8.0, valinit=1.8, valfmt='%.1f m', color='#f59e0b')
slider_v2     = Slider(ax_v2,     'Rider Speed V (km/h)', 30.0, 280.0, valinit=140.0, valfmt='%.0f km/h', color='#10b981')
slider_theta2 = Slider(ax_theta2, 'Manual Lean θ (deg)', -60.0, 60.0, valinit=0.0, valfmt='%.1f deg', color='#a855f7')

poly_list = []
line_list = []
scatter_list = []

txt_3d_title = ax_3d.text2D(0.02, 0.95, "SYNCHRONIZED 3D MOTORCYCLE & HORIZON DYNAMICS", transform=ax_3d.transAxes, 
                           color='#f8fafc', fontsize=11, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.4', facecolor='#0f172a', edgecolor='#334155', alpha=0.9))

txt_telemetry = ax_3d.text2D(0.02, 0.85, "", transform=ax_3d.transAxes, color='#38bdf8', fontsize=9,
                            bbox=dict(boxstyle='round,pad=0.4', facecolor='#0f172a', edgecolor='#334155', alpha=0.9))

txt_alert     = ax_3d.text2D(0.02, 0.75, "", transform=ax_3d.transAxes, color='#ef4444', fontsize=10, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.4', facecolor='#0f172a', edgecolor='#334155', alpha=0.9))

def draw_3d_scene(ax, s_val, n_val, V_kmh, user_theta_deg):
    global poly_list, line_list, scatter_list
    
    for p in poly_list:
        try: p.remove()
        except: pass
    poly_list.clear()

    for l in line_list:
        try: l.remove()
        except: pass
    line_list.clear()

    for s in scatter_list:
        try: s.remove()
        except: pass
    scatter_list.clear()

    idx_start = int((s_val / 5245.0) * (num_pts - 1))
    idx_start = np.clip(idx_start, 0, num_pts - 1)

    slope_deg = slope_arr[idx_start]
    bank_deg  = bank_arr[idx_start]
    elev_curr = h_grid_full[idx_start]
    kappa_curr = kappa_arr[idx_start]
    R_curr = 1.0 / (abs(kappa_curr) + 1e-5)
    
    g = 9.81
    V_ms = V_kmh / 3.6
    mu_val = 1.15
    
    theta_req_rad = np.arctan((V_ms**2 * kappa_curr) / g)
    theta_req_deg = np.degrees(theta_req_rad)
    
    if abs(user_theta_deg) > 0.1:
        effective_bike_lean = user_theta_deg
    else:
        effective_bike_lean = theta_req_deg

    theta_max_deg = np.degrees(np.arctan(mu_val))
    V_max_kmh = np.sqrt(mu_val * g * R_curr) * 3.6

    rad_p = np.radians(slope_deg)
    rad_b = np.radians(bank_deg)

    grid_y = np.linspace(0, 50, 6)
    grid_x = np.linspace(-10, 10, 6)
    GX, GY = np.meshgrid(grid_x, grid_y)
    GZ = np.zeros_like(GX)
    
    horizon_grid = ax.plot_wireframe(GX, GY, GZ, color='#0284c7', alpha=0.20, lw=0.6, linestyle=':')
    poly_list.append(horizon_grid)

    horizon_dist = 60.0
    num_h = 50
    s_ahead = np.linspace(0, horizon_dist, num_h)
    ds_step = s_ahead[1] - s_ahead[0]
    
    dpsi_road = 0.0
    x_centerline_ahead = [0.0]
    y_centerline_ahead = [0.0]
    
    kappa_bike = (g * np.tan(np.radians(effective_bike_lean))) / (V_ms**2 + 1e-5)
    dpsi_bike = 0.0
    x_actual_ahead = [n_val]
    
    x_road_left = []
    x_road_right = []
    x_ideal_ahead = []

    for i in range(num_h):
        ds_curr = s_ahead[i]
        idx_h = int(((s_val + ds_curr) % 5245.0 / 5245.0) * (num_pts - 1))
        
        if i > 0:
            k_h = kappa_arr[idx_h]
            dpsi_road += k_h * ds_step
            x_c_new = x_centerline_ahead[-1] + np.sin(dpsi_road) * ds_step
            y_c_new = y_centerline_ahead[-1] + np.cos(dpsi_road) * ds_step
            x_centerline_ahead.append(x_c_new)
            y_centerline_ahead.append(y_c_new)
            
            dpsi_bike += kappa_bike * ds_step
            x_act_new = x_actual_ahead[-1] + np.sin(dpsi_bike) * ds_step
            x_actual_ahead.append(x_act_new)

        xc = x_centerline_ahead[-1]
        n_id = n_ideal[idx_h]
        
        x_left  = xc - track_w/2.0 * np.cos(dpsi_road)
        x_right = xc + track_w/2.0 * np.cos(dpsi_road)
        
        x_road_left.append(x_left)
        x_road_right.append(x_right)
        x_ideal_ahead.append(xc + n_id * np.cos(dpsi_road))

    x_road_left = np.array(x_road_left)
    x_road_right = np.array(x_road_right)
    x_ideal_ahead = np.array(x_ideal_ahead)
    x_actual_ahead = np.array(x_actual_ahead)
    y_ahead = np.array(y_centerline_ahead)
    
    def transform_terrain(X, Y):
        cos_p, sin_p = np.cos(rad_p), np.sin(rad_p)
        cos_b, sin_b = np.cos(rad_b), np.sin(rad_b)
        tX = X * cos_b - Y * sin_b
        tY = X * sin_b + Y * cos_b
        tZ = tY * sin_p
        return tX, tY, tZ

    Y_mesh = np.tile(y_ahead, (2, 1)).T
    X_mesh = np.column_stack([x_road_left, x_road_right])
    tr_X, tr_Y, tr_Z = transform_terrain(X_mesh, Y_mesh)
    road_surf = ax.plot_surface(tr_X, tr_Y, tr_Z, color='#1e293b', alpha=0.92, edgecolor='#334155', lw=0.3)
    poly_list.append(road_surf)
    
    for p_i in range(0, num_h, 8):
        px, py, pz = transform_terrain(x_road_left[p_i], y_ahead[p_i])
        pillar, = ax.plot([px, px], [py, py], [0, pz], color='#475569', lw=1.0, ls=':')
        line_list.append(pillar)

    b1_x, b1_y, b1_z = transform_terrain(x_road_left, y_ahead)
    b2_x, b2_y, b2_z = transform_terrain(x_road_right, y_ahead)
    l_bound1, = ax.plot(b1_x, b1_y, b1_z, color='#64748b', lw=1.5, ls='--')
    l_bound2, = ax.plot(b2_x, b2_y, b2_z, color='#64748b', lw=1.5, ls='--')
    line_list.extend([l_bound1, l_bound2])

    id_x, id_y, id_z = transform_terrain(x_ideal_ahead, y_ahead)
    l_ideal_3d, = ax.plot(id_x, id_y, id_z + 0.05, color='#38bdf8', lw=3.0, label='Ideal Path')
    
    is_off_track_now = (n_val < x_road_left[0] + bike_w/2.0) or (n_val > x_road_right[0] - bike_w/2.0)
    off_left_mask  = x_actual_ahead < (x_road_left + bike_w/2.0)
    off_right_mask = x_actual_ahead > (x_road_right - bike_w/2.0)
    off_track_ahead_mask = off_left_mask | off_right_mask
    
    is_off_track_ahead = np.any(off_track_ahead_mask)
    is_off_track = is_off_track_now or is_off_track_ahead
    is_speed_breach = V_kmh > V_max_kmh or abs(effective_bike_lean) > theta_max_deg
    
    actual_color = '#ef4444' if (is_off_track or is_speed_breach) else '#22c55e'
    act_x, act_y, act_z = transform_terrain(x_actual_ahead, y_ahead)
    l_actual_3d, = ax.plot(act_x, act_y, act_z + 0.08, color=actual_color, lw=3.5, ls='-', label='Actual Path')
    line_list.extend([l_ideal_3d, l_actual_3d])

    rad_l = np.radians(effective_bike_lean)
    
    def transform_bike(x, y, z):
        cos_l, sin_l = np.cos(rad_l), np.sin(rad_l)
        xr1 = x * cos_l + z * sin_l
        zr1 = -x * sin_l + z * cos_l
        xr1 = xr1 + n_val
        cos_p, sin_p = np.cos(rad_p), np.sin(rad_p)
        yr2 = y * cos_p - zr1 * sin_p
        zr2 = y * sin_p + zr1 * cos_p
        cos_b, sin_b = np.cos(rad_b), np.sin(rad_b)
        xr3 = xr1 * cos_b - yr2 * sin_b
        yr3 = xr1 * sin_b + yr2 * cos_b
        zr3 = zr2
        return xr3, yr3, zr3

    c_white = '#f8fafc'
    c_red   = '#ef4444'
    c_gold  = '#eab308'
    c_tire  = '#111827'
    t_wheel = np.linspace(0, 2*np.pi, 24)
    r_w = 0.80 
    
    for w_off in np.linspace(-0.25, 0.25, 6):
        xr_t, yr_t, zr_t = transform_bike(r_w * np.sin(t_wheel), -1.5 + w_off, r_w + r_w * np.cos(t_wheel))
        l1, = ax.plot(xr_t, yr_t, zr_t, color=c_tire, lw=4.5)
        line_list.append(l1)

        xf_t, yf_t, zf_t = transform_bike(r_w * np.sin(t_wheel), 1.8 + w_off, r_w + r_w * np.cos(t_wheel))
        l2, = ax.plot(xf_t, yf_t, zf_t, color=c_tire, lw=4.0)
        line_list.append(l2)

    fx1, fy1, fz1 = transform_bike(-0.25, 1.8, 0.6)
    fx2, fy2, fz2 = transform_bike(-0.25, 1.4, 1.8)
    l_fork1, = ax.plot([fx1, fx2], [fy1, fy2], [fz1, fz2], color=c_gold, lw=5.0)
    line_list.append(l_fork1)
    
    fx3, fy3, fz3 = transform_bike(0.25, 1.8, 0.6)
    fx4, fy4, fz4 = transform_bike(0.25, 1.4, 1.8)
    l_fork2, = ax.plot([fx3, fx4], [fy3, fy4], [fz3, fz4], color=c_gold, lw=5.0)
    line_list.append(l_fork2)

    nose_verts = [
        [transform_bike(0.0, 2.4, 1.4), transform_bike(-0.7, 1.4, 1.5), transform_bike(0.0, 1.4, 1.9)],
        [transform_bike(0.0, 2.4, 1.4), transform_bike(0.7, 1.4, 1.5), transform_bike(0.0, 1.4, 1.9)],
        [transform_bike(-0.75, 0.6, 1.5), transform_bike(0.75, 0.6, 1.5), transform_bike(0.6, -0.4, 1.3), transform_bike(-0.6, -0.4, 1.3)],
        [transform_bike(-0.4, -1.2, 1.5), transform_bike(0.4, -1.2, 1.5), transform_bike(0.0, -2.0, 1.8)]
    ]
    fairing = Poly3DCollection(nose_verts, facecolors=c_white, edgecolors='#cbd5e1', alpha=0.95)
    ax.add_collection3d(fairing)
    poly_list.append(fairing)

    stripe_verts = [
        [transform_bike(-0.72, 1.1, 1.2), transform_bike(-0.75, 0.5, 1.1), transform_bike(-0.70, 0.5, 0.95), transform_bike(-0.65, 1.1, 1.05)],
        [transform_bike(0.72, 1.1, 1.2), transform_bike(0.75, 0.5, 1.1), transform_bike(0.70, 0.5, 0.95), transform_bike(0.65, 1.1, 1.05)]
    ]
    stripes = Poly3DCollection(stripe_verts, facecolors=c_red, edgecolors=c_red, alpha=0.9)
    ax.add_collection3d(stripes)
    poly_list.append(stripes)

    rx, ry, rz = transform_bike(0.0, 0.1, 2.3)
    sc = ax.scatter([rx], [ry], [rz], color='#f8fafc', s=280, depthshade=False, edgecolors='#38bdf8', linewidths=2)
    scatter_list.append(sc)

    if is_off_track_now:
        alert_msg = f"[RED ALERT]: OFF-TRACK BREACH! (Position n={n_val:+.1f}m Exceeds Asphalt)"
        txt_alert.set_color('#ef4444')
    elif is_off_track_ahead:
        first_off_i = np.where(off_track_ahead_mask)[0][0]
        dist_off = s_ahead[first_off_i]
        alert_msg = f"[RED ALERT]: OFF-TRACK BREACH! (Leaves Asphalt {dist_off:.0f}m Ahead)"
        txt_alert.set_color('#ef4444')
    elif is_speed_breach:
        alert_msg = f"[RED ALERT]: VELOCITY / GRIP LIMIT BREACH! (Speed {V_kmh:.0f} > Limit {V_max_kmh:.0f})"
        txt_alert.set_color('#ef4444')
    else:
        alert_msg = STABLE_LABEL
        txt_alert.set_color('#22c55e')

    txt_alert.set_text(alert_msg)

    telemetry_str = (
        f"Distance: {s_val:.0f}m / 5245m   |   Elevation: {elev_curr:.1f}m   |   Curve Radius: {R_curr:.1f}m\n"
        f"Speed: {V_kmh:.0f} km/h (Limit: {V_max_kmh:.0f})   |   Bike Lean (θ): {effective_bike_lean:+.1f}° (Req: {theta_req_deg:+.1f}°)\n"
        f"Lateral Offset (n): {n_val:+.1f} m   |   Slope: {slope_deg:+.1f}°   |   Bank: {bank_deg:+.1f}°"
    )
    txt_telemetry.set_text(telemetry_str)

ax_3d.view_init(elev=16, azim=-90)
ax_3d.set_xlim(-10.0, 10.0)
ax_3d.set_ylim(-4.0, 50.0)
ax_3d.set_zlim(-1.0, 10.0)
ax_3d.set_box_aspect((1.0, 1.8, 0.45))
ax_3d.set_axis_off()

# --- Synchronization Callbacks ---
is_updating = False

def sync_sliders_from_w1(val):
    global is_updating
    if is_updating: return
    is_updating = True
    
    s_val = slider_s1.val
    n_val = slider_n1.val
    
    slider_s2.set_val(s_val)
    slider_n2.set_val(n_val)
    
    idx = int((s_val / 5245.0) * (num_pts - 1))
    rider_marker.set_data([x_track[idx] + n_val * nx[idx]], [y_track[idx] + n_val * ny[idx]])
    
    draw_3d_scene(ax_3d, s_val, n_val, slider_v2.val, slider_theta2.val)
    fig_2d.canvas.draw_idle()
    fig_3d.canvas.draw_idle()
    is_updating = False

def sync_sliders_from_w2(val):
    global is_updating
    if is_updating: return
    is_updating = True
    
    s_val = slider_s2.val
    n_val = slider_n2.val
    
    slider_s1.set_val(s_val)
    slider_n1.set_val(n_val)
    
    idx = int((s_val / 5245.0) * (num_pts - 1))
    rider_marker.set_data([x_track[idx] + n_val * nx[idx]], [y_track[idx] + n_val * ny[idx]])
    
    draw_3d_scene(ax_3d, s_val, n_val, slider_v2.val, slider_theta2.val)
    fig_2d.canvas.draw_idle()
    fig_3d.canvas.draw_idle()
    is_updating = False

def on_map_click(event):
    if event.inaxes == ax_2d:
        click_x, click_y = event.xdata, event.ydata
        if click_x is not None and click_y is not None:
            dists = np.sqrt((x_track - click_x)**2 + (y_track - click_y)**2)
            closest_idx = np.argmin(dists)
            s_clicked = s_arr[closest_idx]
            slider_s1.set_val(s_clicked)

fig_2d.canvas.mpl_connect('button_press_event', on_map_click)
slider_s1.on_changed(sync_sliders_from_w1)
slider_n1.on_changed(sync_sliders_from_w1)
slider_s2.on_changed(sync_sliders_from_w2)
slider_n2.on_changed(sync_sliders_from_w2)
slider_v2.on_changed(sync_sliders_from_w2)
slider_theta2.on_changed(sync_sliders_from_w2)

draw_3d_scene(ax_3d, slider_s1.val, slider_n1.val, slider_v2.val, slider_theta2.val)

if __name__ == '__main__':
    plt.show(block=True)