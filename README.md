# 3D Motorcycle Track Simulator & Physics Analyzer 🏍️🏁

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://motorcycle-track-simulator-wzdthvpljduvafazu7rts8.streamlit.app/)

Welcome to the **3D Motorcycle Track Simulator**. This web-based application allows users to upload any 2D track map, automatically extract its physical geometry using computer vision, design 3D topographical features (elevation and banking), and run a synchronized 3D physics simulation to evaluate motorcycle dynamics, lean angles, and cornering limits.

## 🎯 Purpose of the Project

The goal of this tool is to bridge the gap between static 2D track layouts and dynamic 3D physics modeling. It is designed for racing enthusiasts, track designers, and engineering students who want to understand how trajectory, speed, slope, and corner radius affect the physical limits of a motorcycle. By mathematically extracting the ideal racing line, the simulator accurately predicts whether a rider at a given speed will hold the curve or breach the traction/track limits.

## 🕹️ How to Use the Simulator

No installation is required. You can launch the application directly from your browser using the badge above.

1. **Step 1: Upload Geometry**
   * Upload an image of a race track (JPG/PNG).
   * The system will use Computer Vision to extract the centerline.
   * *Tip:* Adjust the **Adaptive Thresholding** or **Spline Smoothing** factor if the track has a noisy background (like map labels or watermarks) to ensure a perfectly connected geometry.
2. **Step 2: Topography & Track Design**
   * Navigate along the extracted track length using the slider.
   * Inject real-world physics by assigning **Elevation** (meters) and **Banking/Camber** (degrees) at specific distance waypoints. The engine will smoothly interpolate the terrain between your points.
3. **Step 3: 3D Simulation & Telemetry**
   * Set your rider's speed ($V$) and offset from the center ($n$).
   * Select a manual lean angle, or leave it at $0^\circ$ for the **Auto-Lean** algorithm to perfectly track the corner.
   * Generate the 3D Animation to watch the motorcycle navigate the custom topography, or analyze real-time telemetry (radius, required lean, speed limits) to optimize the racing line.

## 🧮 The Mathematics & Physics Engine

This simulator relies heavily on applied mathematics, signal processing, and differential geometry to construct the physical environment and evaluate the dynamics.

### 1. Geometry Extraction & Spline Interpolation
The raw pixels from the image are converted into continuous parametric equations using B-Splines:
We apply `scipy.interpolate.splprep` to generate a parametric curve $(x(u), y(u))$. This mathematical smoothing prevents abrupt pixel-level jitter from generating infinite physical forces during the simulation.

### 2. Track Kinematics (Frenet-Serret Frame)
To calculate physical forces, the track is analyzed as a spatial curve. We calculate the unit tangent vector $\vec{T}$ and the normal vector $\vec{N}$ at any given path distance $s$.
The instantaneous curvature $\kappa$ is calculated using the standard differential formula:
$$\kappa = \frac{x' y'' - y' x''}{(x'^2 + y'^2)^{3/2}}$$
*Note: To simulate real-world vehicle behavior (where riders cannot instantly snap into a lean angle), a Gaussian filter (`scipy.ndimage.gaussian_filter1d`) is applied to $\kappa$ to represent smooth weight transitions into the apex.*

### 3. Ideal Racing Line Optimization (Apex Calculation)
The simulator automatically calculates the most efficient racing line (the "ideal trajectory") for **any** given track geometry. Instead of rigidly following the geometric centerline, the algorithm maximizes the cornering radius to allow for higher speeds. It achieves this by calculating an optimal lateral track offset ($n_{ideal}$):
1. **Look-ahead Curvature:** A wide Gaussian filter is applied to the raw curvature array to simulate a rider's forward vision and turn anticipation.
2. **Hyperbolic Offset Mapping:** The lateral deviation from the centerline is determined using a bounded Hyperbolic Tangent function:
$$n_{ideal} = M \cdot \tanh(\kappa_{smoothed} \cdot C)$$
*(Where $M$ is the maximum allowed track width deviation/safety margin, and $C$ is the curve sensitivity multiplier).*

This mathematical approach naturally generates the classic **"Out-In-Out"** racing line: forcing the trajectory to the outside edge before the turn, clipping the inside apex at the point of maximum curvature, and running wide on the exit.

### 4. Motorcycle Dynamics & Lean Angle Constraint
To negotiate a curve of radius $R$ ($R = 1/\kappa$) at velocity $V$, the motorcycle must bank to balance the centrifugal force with gravity ($g$). The required lean angle $\theta$ relative to the track surface is modeled as:
$$\theta_{req} = \arctan\left(\frac{V^2 \cdot \kappa}{g}\right)$$
The system constantly evaluates the tire grip limit ($\mu$). If $\theta_{req} > \arctan(\mu)$, the engine triggers a **Velocity / Grip Limit Breach**.

### 5. 3D Terrain Transformation
The 2D path is mapped into a dynamic 3D space using sequenced rotation matrices. For a given track segment with pitch angle $p$ (derived from the elevation gradient) and bank angle $b$, the local coordinates $(x, y, z)$ are transformed via:
1. **Yaw** (Heading direction)
2. **Pitch** (Track incline/decline)
3. **Bank** (Track camber)

This ensures the 3D motorcycle model and the grid mesh visually match the calculated physical telemetry perfectly.