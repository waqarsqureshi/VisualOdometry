# `README.md`


# 🚴‍♂️ Visual Odometry + Accelerometer + GPS Fusion

This project performs **Visual Odometry (VO)** combined with **Low-Pass Filtered Accelerometer data** and **GPS measurements** using a **Kalman Filter**.

It supports:
- Real-time simulation
- Pause/Resume
- GPS hopping detection and skipping
- Frame-by-frame motion processing
- Saving output trajectories and error plots

---

## 📋 Algorithm Overview

### 1. Data Loading
- Load GPS (`GPS_*.csv`), Accelerometer (`ACL_*.csv`), and Magnetometer (`MAG_*.csv`) CSV files.
- Open the video file (`vo.mp4`) captured during movement.

### 2. Preprocessing
- Project GPS (latitude, longitude) into **local X/Z meters** relative to the starting position.
- Apply **Low-Pass Filter** to accelerometer readings (smoothing vibrations, alpha = 0.1).
- Interpolate GPS readings to match the video frame rate (auto-detected from video metadata).

### 3. Initialization
- Set up **Kalman Filter** with state `[X, Z, Vx, Vz]`.
- Define matrices `F`, `H`, `Q`, `R`, and initial covariance `P`.
- Initialize feature detection (Good Features to Track) and Optical Flow parameters for Visual Odometry.

### 4. Per-Frame Processing
For each video frame:
1. **Feature tracking** using Optical Flow.
2. **Visual Odometry**:
   - Estimate relative translation (dx, dz) from Essential Matrix recovery.
3. **Kalman Prediction**:
   - Predict next position and velocity.
4. **Motion Update**:
   - Update position based on VO translation.
   - Update velocity and position using **filtered accelerometer** (integrated over dt).
5. **GPS Correction**:
   - Calculate GPS jump distance from previous frame.
   - If GPS jump is **below threshold** (e.g., 2 meters), **apply Kalman update** using GPS.
   - If GPS jump is **too large**, **skip correction** (trust VO + Accel only).

### 5. Visualization
- Display live:
  - Current video frame
  - Real-time mini-map (VO + GPS paths)
- Overlay live text:
  - FPS
  - Current VO (X, Z)
  - Current GPS (X, Z)
  - GPS jump distance
  - Whether GPS correction was applied or skipped
- Pause (`p`) and Resume (`p`) anytime during simulation.
- Quit (`q`) simulation anytime.

### 6. Outputs
- Save CSV: `./output/trajectory_comparison.csv`
- Save Plots:
  - `trajectory_plot.png` (VO vs GPS path)
  - `error_plot.png` (frame-wise error)
- Print final statistics:
  - Average Error
  - Maximum Error
  - Minimum Error

---

## 📦 Folder Structure

| Path | Purpose |
|:---|:---|
| `vo.py` | Main processing script |
| `./data/` | Input files folder |
| `./output/` | Outputs (CSV, plots) |

---

## 📂 Required Input Files (inside `./data/`)

| File | Description |
|:---|:---|
| `vo.mp4` | Front-facing video captured during movement |
| `GPS_20250216_163034_data.csv` | GPS latitude/longitude and timestamp |
| `ACL_20250216_163034_data.csv` | Accelerometer readings (X, Y, Z) |
| `MAG_20250216_163034_data.csv` | Magnetometer readings (optional) |

---

## 🛠 Requirements

- Python 3.7 or higher
- Install dependencies:

```bash
pip install opencv-python numpy pandas matplotlib geopy
```

---

## 🕹️ Usage Instructions

To start the real-time simulation:

```bash
python vo.py
```

During simulation:

| Key | Action |
|:---|:---|
| `p` | Pause or Resume simulation |
| `q` | Quit simulation immediately |

During Pause:
- "PAUSED" will blink on the screen.
- Press `p` again to resume.
- Press `q` to quit from pause.

---

## 📈 Outputs

Saved automatically in `./output/`:

| File | Description |
|:---|:---|
| `trajectory_comparison.csv` | Frame-by-frame VO+GPS data and error |
| `trajectory_plot.png` | 2D plot of VO vs GPS paths |
| `error_plot.png` | Error vs Frame plot |

---

## 🎯 Important Features

- **Low-Pass Filter** reduces accelerometer vibration noise.
- **GPS Hopping Detection**:
  - If GPS suddenly moves too far (> 2 meters/frame), correction is skipped.
  - Protects trajectory from GPS glitches.
- **Accelerometer Smoothing** and Integration.
- **Real-time visual feedback** with FPS, X/Z positions, GPS jump distance.
- **Pause and Resume** feature for controlled observation.

---

# ✅ Project Ready!

---

## 🚀 Notes for Future Improvements (optional)

- Fuse Magnetometer to correct heading.
- Implement full 3D VO (X, Y, Z).
- Use IMU preintegration for even better accelerometer modeling.
- Dynamic thresholding based on detected speed.

---

# 📢 End of README
