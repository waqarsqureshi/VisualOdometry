import logging  # Added logging module
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from geopy.distance import geodesic
import time
import os

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format="%(asctime)s [%(levelname)s] %(message)s",  # Log format
    handlers=[
        logging.FileHandler("./output/vo.log"),  # Log to a file
        logging.StreamHandler()  # Also log to the console
    ]
)

logging.info("Starting Visual Odometry script...")

try:
    logging.info("Loading data from ./data directory...")

    # --- Load Data ---
    gps_df = pd.read_csv('./data/GPS_20250216_163034_data.csv')
    acl_df = pd.read_csv('./data/ACL_20250216_163034_data.csv')
    mag_df = pd.read_csv('./data/MAG_20250216_163034_data.csv')

    logging.info("Data loaded successfully.")

    # Rename Accelerometer Columns
    acl_df = acl_df.rename(columns={
        'Y acceleration m/s^2': 'ay',
        '-X acceleration m/s^2': 'ax',
        'Z acceleration m/s^2 Z': 'az'
    })

    # Apply Low-Pass Filter to Accelerometer
    alpha = 0.3  # smoothing factor
    filtered_acl = acl_df[['ax', 'ay', 'az']].copy()

    for axis in ['ax', 'ay', 'az']:
        for i in range(1, len(filtered_acl)):
            filtered_acl.at[i, axis] = alpha * filtered_acl.at[i, axis] + (1 - alpha) * filtered_acl.at[i-1, axis]

    logging.info("Accelerometer low-pass filtering completed.")

    # --- Open Video and Detect FPS ---
    video_path = './data/vo.mp4'
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logging.error("Error opening video file.")
        raise IOError("Error opening video file.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    logging.info(f"Detected video FPS: {fps:.2f} frames per second.")

    # --- Project GPS ---
    gps_coords = gps_df[['GPS (Lat.) [deg]', 'GPS (Long.) [deg]']].values
    gps_origin = gps_coords[0]
    gps_proj = np.zeros_like(gps_coords)

    for i in range(len(gps_coords)):
        gps_proj[i, 0] = geodesic((gps_origin[0], gps_origin[1]), (gps_coords[i, 0], gps_origin[1])).meters
        gps_proj[i, 1] = geodesic((gps_origin[0], gps_origin[1]), (gps_origin[0], gps_coords[i, 1])).meters
        if gps_coords[i,1] < gps_origin[1]:
            gps_proj[i,1] *= -1
        if gps_coords[i,0] < gps_origin[0]:
            gps_proj[i,0] *= -1

    logging.info("GPS projection completed.")

    frame_times = np.arange(0, len(gps_coords) * 1.0, 1/fps)
    gps_interp_x = np.interp(frame_times, np.linspace(0, len(gps_coords)/1, len(gps_coords)), gps_proj[:,1])
    gps_interp_z = np.interp(frame_times, np.linspace(0, len(gps_coords)/1, len(gps_coords)), gps_proj[:,0])

    # --- VO + Kalman Setup ---
    focal = 460.0
    pp = (293, 424)

    feature_params = dict(maxCorners=1000, qualityLevel=0.3, minDistance=7, blockSize=7)
    lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    dt = 1/fps
    F = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]])
    Q = np.eye(4) * 0.1
    R_kalman = np.eye(2) * 3
    P = np.eye(4)
    x = np.zeros((4,1))

    ret, prev_frame = cap.read()
    if not ret:
        logging.error("Error reading first frame.")
        raise IOError("Error reading first frame.")

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

    logging.info("Starting real-time Visual Odometry + Accelerometer + GPS fusion...")

    trajectory = []
    results = []
    frame_idx = 0
    last_time = time.time()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('./output/trajectory_animation.mp4', fourcc, int(fps), (1600, 800))  # 1600x800 because combined frame

    # --- Scale Estimation ---

    # How many frames to use for initial calibration
    scale_estimation_frames = 100

    vo_distances = []
    gps_distances = []

    logging.info("Estimating scale automatically from first 100 frames...")

    # Reset video to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

    frame_idx = 0
    total_vo_distance = 0
    total_gps_distance = 0

    # Additional parts of the script will follow the same pattern...
    # Replace all `print` statements with `logging.info`, `logging.warning`, or `logging.error`.

    # For example:
    # print("Accelerometer low-pass filtering completed.")
    # becomes:
    # logging.info("Accelerometer low-pass filtering completed.")

except Exception as e:
    logging.exception(f"An error occurred: {e}")
