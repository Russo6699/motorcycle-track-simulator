# 🏍️ 3D Motorcycle Racing Simulator

An interactive web-based simulator built with Python and Streamlit that allows users to upload 2D circuit maps, design custom 3D topography, calculate optimal physics-based racing lines, and generate 3D animations of a motorcycle navigating the track.

## 🌟 Project Overview

This tool is designed for motorsport enthusiasts, engineers, and developers interested in vehicle dynamics and trajectory optimization. By simply uploading an image of a track, the application uses advanced computer vision to extract the circuit's geometry. Users can then inject custom elevation and banking data to create a fully realized 3D environment and visualize how a motorcycle behaves under real physics constraints.

## ✨ Key Features

1. **Intelligent Track Extraction**: 
   - Upload any PNG/JPG track layout.
   - Utilizes OpenCV with Adaptive Thresholding and morphological noise filtering to handle complex map backgrounds, logos, and varying colors.
2. **Custom Topography Design**: 
   - Interactive waypoint system to define elevation (meters) and banking/camber (degrees) at any point along the track.
   - Smooth spline interpolation automatically builds continuous 3D hills and dips.
3. **Physics-Based Trajectory Engine**: 
   - Calculates the optimal racing line based on local cornering radii.
   - Computes maximum safe velocity and required lean angles using real-world gravity and tire friction coefficients.
4. **3D Animation Generation**: 
   - Dynamically renders a 3D motorcycle model navigating the custom track.
   - Exports the synchronized physics simulation as a visually stunning GIF.
   - Real-time HUD displaying telemetry data (Speed, Lean Angle, Distance, Elevation).

## 🛠️ Tech Stack

* **Frontend/UI**: [Streamlit](https://streamlit.io/)
* **Computer Vision**: [OpenCV](https://opencv.org/) (`opencv-python-headless`)
* **Math & Physics**: [NumPy](https://numpy.org/), [SciPy](https://scipy.org/)
* **3D Visualization**: [Matplotlib](https://matplotlib.org/)

## 🚀 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Russo6699/motorcycle-track-simulator.git
   cd motorcycle-track-simulator
   ```

2. **Create a virtual environment (Optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🎮 Usage

Start the application by running the following command in your terminal:

```bash
streamlit run app.py
```

### The 3-Phase Workflow:
1. **Upload Phase**: Choose an extraction method (Standard, Color Masking, or Math Smoothing) and upload your track image.
2. **Design Phase**: Use the slider to travel along the extracted center-line and save elevation/banking data at specific waypoints. Click "Confirm Track Design" when finished.
3. **Simulation Phase**: Adjust rider speed, manual lean overrides, and lateral offsets. View the static trajectory prediction or click "Generate 3D Animation Loop" to create your customized riding GIF.

## 📝 License
This project is open-source and available under the [MIT License](LICENSE).