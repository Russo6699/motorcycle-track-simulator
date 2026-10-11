# 3D Motorcycle Track Simulator & Physics Analyzer 🏍️🏁

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://motorcycle-track-simulator-wzdthvpljduvafazu7rts8.streamlit.app/)

Welcome to the **3D Motorcycle Track Simulator**. Mark any road on an interactive map, and the application rebuilds it as a 3D track using **real** elevation and slope data. It then traces the exact road from satellite imagery and computes the fastest line and speed profile for a real motorcycle model. Finally, you can ride the result in 3D with a live telemetry display.

> **No fabricated data.** All terrain, road and vehicle values come from real sources: elevation models, OpenStreetMap, satellite imagery and manufacturer specifications. If a source cannot be reached, the app reports it instead of filling in invented values.

## 🎯 Purpose of the Project

The goal of this tool is to bridge the gap between a road drawn on a 2D map and dynamic 3D physics modelling. It is designed for riders, racing enthusiasts, track designers and engineering students who want to understand how trajectory, speed, slope and corner radius affect the physical limits of a motorcycle on a *real* road.

The app answers questions such as:

* Where is the fastest line?
* How fast can each corner be taken?
* Which limit is active at each point: grip, lean angle, engine power, braking, wheelie or roll rate?
* How much time does the ideal line gain over riding the centre of the lane?

## 🕹️ How to Use the Simulator

No installation is required. Launch the application directly from your browser using the badge above.
The interface is in Hebrew; the button names below are translated.

1. **Step 1: Mark the road**
   * Navigate the map like Google Maps. You can also search for a place, or use **My location**; the browser asks for permission first.
   * Click to place the **start point**, any number of **via points**, and the **end point**. The points are connected along real roads (OSRM routing). Drag a point to move it, or right‑click a point to delete it.
   * Alternatively, load a recorded **GPX** track.
2. **Step 2: Trace the road from satellite imagery** *(recommended)*
   * Click **"Improve the road copy from satellite imagery"**.
   * The asphalt is located in the satellite image along the marked route. The road is then replaced by a smooth B‑spline sampled at **50,000 equidistant points**, and the road width is measured from the image.
   * The traced centre line and edges are drawn in blue on the map. A checkbox switches back to the plain map line at any time.
3. **Step 3: Real elevation and the 3D road**
   * Choose the elevation source and click **"Compute 3D route"**.
   * Bridges and tunnels are corrected using OpenStreetMap, and the grade is computed over an adjustable window.
   * The road is shown in 3D at true scale. Vertical exaggeration is optional.
4. **Step 4: Motorcycle and ideal line**
   * Pick a real motorcycle model. Power, weight and wheelbase are filled in from the manufacturer specifications, with the source shown.
   * Set the tyre grip, the riding corridor (your own lane, or the full road width for a closed course), the line method and a comparison line.
   * Click **"Compute ideal line and speeds"**. The line is drawn on the map and in 3D, coloured by speed, lean or grip use. A table compares the lap time of every line.
5. **Step 5: Ride along**
   * Ride the ideal line in 3D with play, pause and seek, and choose a chase, rider or top‑down camera.
   * The HUD shows:
     * speed
     * lean angle (gauge)
     * g‑force
     * distance and time
     * the active limit
     * the time gap to the comparison line
     * the next corner: distance, apex speed, lean and radius
6. **Export**
   * The route can be downloaded as **GPX**, **CSV**, **GeoJSON** or an **OBJ 3D model**.
   * The ideal line can be downloaded as a CSV.

## 🧮 The Mathematics & Physics Engine

### 1. Road Geometry from Satellite Imagery (Computer Vision & Spline Interpolation)

A route from a map is a polyline with sparse vertices. Often there is only one vertex every 30–250 m, joined by straight segments. Its position can also be several metres off the real asphalt; on the test road it was up to 14 m away, cutting through a forest. Read literally, every vertex is a sharp "kink". The physics would then react to corners that do not exist, which shows up as a zigzag in the ideal line.

The road is therefore extracted from satellite imagery, following the same computer‑vision and B‑spline approach as the original version of this project:

1. **Unrolled corridor:**
   * Imagery (Esri World Imagery, zoom 18, ≈0.5 m per pixel) is resampled along the route.
   * Each row is a station every 1 m along the road. Each column is a lateral offset every 0.25 m, across ±28 m.
2. **OpenCV analysis:**
   * The image is converted to the Lab colour space.
   * The road colour is learned from the image itself, under the map line, as a 1 km running median. This adapts to new or old asphalt, sun and shade, and is robust to stretches where the map line misses the road.
   * Sobel gradients give the edges.
3. **Scoring and Viterbi dynamic programming:**
   * For every station, each candidate centre $c$ and half‑width $h$ is scored by three terms: asphalt‑likeness inside the band, contrast against the shoulders, and edge strength at both borders.
   * A Viterbi pass chooses the most consistent centre and width along the whole route. This bridges short gaps under trees and shadows.
   * A second pass re‑centres the image on the road found and searches again with a locally learned colour model.
4. **Bridging the hidden parts:**
   * At junctions and lay‑bys only one edge of the road is visible. In tunnels (OpenStreetMap tags) the road cannot be seen at all.
   * In those places the centre is interpolated from the reliable stretches on both sides.
   * The road width is measured only where both edges are visible.
5. **Smoothing B‑spline:**
   * `scipy.interpolate.splprep` fits a parametric curve $(x(u), y(u))$ through the detected centre points.
   * The smoothing factor is set from the *measured* detection noise: $s = m\,\sigma^2$, where $\sigma$ is a robust (MAD) estimate of the noise.
   * The curve is sampled at 50,000 equidistant points.
6. **Curvature:** evaluated analytically on the spline:

$$\kappa = \frac{x' y'' - y' x''}{(x'^2 + y'^2)^{3/2}}$$

This step runs entirely in the browser with **Pyodide**, using `numpy`, `scipy` and `opencv-python`. No server is involved; the first run downloads about 30 MB of Python packages.

### 2. Real Elevation, Bridges & Tunnels

Elevation is sampled along the road (every 10 m on routes up to 30 km, every 30 m with Open‑Meteo) from the selected source. The sources are listed in the 🌐 Data Sources section below.

* **Bridges and tunnels:** an elevation model measures the ground, not the bridge deck or the tunnel tube. Where OpenStreetMap marks a bridge or tunnel, the elevation is replaced by a straight grade between the two ends.
* **Grade:** the grade is computed over an adjustable window. The vertical curvature $\kappa_v$ is computed from the smoothed profile.

### 3. Track Kinematics (Frenet‑Serret Frame)

The road is analysed as a spatial curve with arc length $s$, unit tangent $\vec{T}$ and normal $\vec{N}$. A riding line is described by its lateral offset $n(s)$ from the centre line.

* **Centre‑line denoising.** Before the physics, the centre line is lightly denoised in three passes:
  1. a Whittaker smoother (8 m);
  2. the minimum‑curvature curve that stays inside a ±tol tube around the result (tol = 0.75 m for a map line, 0.3 m for a satellite‑traced road);
  3. a second Whittaker pass.
* **Honest path curvature.** The curvature of a riding line is measured over ±2 m, so a kinked line cannot hide its kinks.
* **Gradual lean changes.** To represent the fact that a rider cannot snap into a lean instantly, a Gaussian filter (`gaussian_filter1d`, default $\sigma_\kappa$ = 6 m) is applied to $\kappa$. The effective curvature is the larger of the raw and the smoothed value, so a real corner is never "smoothed away".

### 4. Reference Racing Lines

Four reference lines are computed for comparison. The fastest of them is also the starting point of the optimisation in section 5.

1. **Corridor centre:** the middle of the allowed riding corridor.
2. **Hyperbolic offset mapping (tanh line):**
   * A wide look‑ahead Gaussian on the curvature represents the rider's forward vision.
   * The lateral offset follows a bounded hyperbolic tangent:

$$n_{ideal} = M \cdot \tanh(\kappa_{smoothed} \cdot C)$$

   * $M$ is the maximum allowed lateral deviation (corridor and safety margin) and $C$ is the curve‑sensitivity multiplier.
   * This naturally produces the classic **"Out‑In‑Out"** line.
3. **Minimum curvature:** the line that minimises $\sum \kappa^2$ inside the corridor.
4. **Maximum radius:** the line with the largest possible radius in each corner.

### 5. Minimum‑Time Line (Time‑Integral Optimisation)

The fastest line minimises the total time along the road. In curvilinear coordinates around the centre line with curvature $\kappa_c$, the time is:

$$T = \int_0^{L} \frac{\sqrt{(1 - n\,\kappa_c)^2 + n'^2}}{v(s)}\,ds$$

**Speed profile.** For every candidate line, $v(s)$ is the fastest feasible speed profile. It is found with quasi‑steady forward (acceleration) and backward (braking) passes, subject to:

* **Track limits:** $n_{min}(s) \le n(s) \le n_{max}(s)$. The road width, minus half the motorcycle width and a safety margin, sets these limits. The corridor is either your own lane or the full width.
* **Lean angle and tyre grip:**

$$\theta_{req} = \arctan\left(\frac{V^2 \cdot \kappa}{g_{eff}}\right) \le \min\left(\theta_{max},\ \arctan \mu\right)$$

* **Friction circle:** $\left(\frac{a_{long}}{\mu g_{eff}}\right)^2 + \left(\frac{a_{lat}}{\mu g_{eff}}\right)^2 \le 1$. It is checked at both ends of every step.
* **Engine power:** $F_{drive} \le \eta P / v$, together with aerodynamic drag $\tfrac12 \rho C_dA\,v^2$, rolling resistance and grade.
* **Wheelie and stoppie:**
  * acceleration: $a \le g_{eff}\, L f_F / h$
  * braking: $a \le g_{eff}\, L (1-f_F) / h$
  * where $L$ is the wheelbase, $f_F$ the front weight share and $h$ the height of the centre of gravity.
* **Roll rate:** the lean cannot change arbitrarily fast. The constraint is $\frac{v^3\,|\kappa'|\cos^2\theta}{g} \le \omega_{max}$, with a default of 90°/s.
* **Slope and crests:** $g_{eff} = g\cos\varphi + v^2\kappa_v$. Crests unload the tyres and dips load them.
* **Optional caps:** OpenStreetMap speed limits and a manual speed cap.

**Optimiser.**
* The search starts from the fastest reference line.
* Smooth, raised‑cosine "bumps" sized to the corridor are applied, and a change is kept only if it lowers $T$.
* There is **no artificial smoothness penalty** in the objective. If weaving is genuinely faster somewhere, the optimiser is free to find it; only the physical limits above constrain the line.
* After the run, each constraint is checked and reported (✓/✗) next to the result.

### 6. 3D Terrain Transformation

The 2D path is lifted into 3D using the real elevation profile.

* The road surface is built as a ribbon. Its width comes from the satellite measurement or from OpenStreetMap tags, otherwise from a default value.
* The 3D view uses true scale by default, so lateral offsets are not exaggerated. Vertical exaggeration is optional, for display only.
* During ride‑along, the camera follows the computed line, heading and lean angle.

## 🌐 Data Sources

| Data | Source |
|---|---|
| Map tiles | Esri (streets, satellite, topographic), OpenTopoMap, OpenStreetMap |
| Routing | OSRM (`router.project-osrm.org`, `routing.openstreetmap.de`) |
| Place search | Nominatim (OpenStreetMap) |
| Elevation | AWS Terrain Tiles (SRTM); Copernicus GLO‑90 via Open‑Meteo; Copernicus GLO‑30 via OpenTopography (free API key); GPX elevation |
| Bridges, tunnels, width, lanes, speed limits | OpenStreetMap via the Overpass API |
| Satellite imagery | Esri World Imagery (zoom 18, ≈0.5 m/px) |
| Python in the browser | Pyodide 0.27 (numpy, scipy, opencv-python) |
| Motorcycle specifications | Manufacturer data, linked in the app for each model |

## 📁 Project Files

| File | Purpose |
|---|---|
| `index.html` | The whole web application: map, 3D view, charts, ride‑along, UI |
| `motosim.js` | Physics engine: centre line, reference lines, speed profile, minimum‑time optimiser, constraint checks |
| `satcorridor.js` | Samples the satellite imagery along the route and runs `roadfit.py` in the browser |
| `roadfit.py` | Road extraction (OpenCV + Viterbi) and the `splprep` B‑spline with 50,000 equidistant points |
| `app2.py` | Streamlit entry point: serves the application full‑page (keeps the Streamlit link above working) |
| `requirements.txt` | Python packages for the Streamlit deployment |

**Running locally:**
* With Streamlit: `streamlit run app2.py`.
* Without Streamlit: serve the folder with any static web server and open `index.html`, for example `python -m http.server`, then `http://localhost:8000`.

## ⚠️ Known Limitations

* **Satellite tracing:**
  * Where deep tree shadow covers part of the road, only the sunlit part is found, and the centre can be 2–3 m off there.
  * Junctions and tunnels are bridged from the neighbouring stretches.
  * Imagery may be older than the current road layout.
  * The step is limited to 80 km per run.
* **Elevation:** elevation models measure the ground at 30–90 m resolution. Road cuts and fills narrower than that are smoothed out.
* **OpenStreetMap:** the public Overpass servers are sometimes overloaded. Bridge, tunnel and speed‑limit data are then unavailable, and the app says so.
* **Vehicle model:** the physics is a quasi‑steady single‑track model with lean, grip, power, aerodynamic and load‑transfer limits. It is not a full multibody tyre model.
