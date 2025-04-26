import logging
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from geopy.distance import geodesic
import time
import os
import glob

# --- Configure Logging ---
os.makedirs('./output', exist_ok=True)
logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s [%(levelname)s] %(message)s",
	handlers=[
		logging.FileHandler("./output/vo.log"),
		logging.StreamHandler()
	]
)

logging.info("Loading data from ./data directory...")

# --- Load Data ---

# Automatically find the correct files
gps_file = glob.glob('./data/GPS_*.csv')[0]
acl_file = glob.glob('./data/ACL_*.csv')[0]
mag_file = glob.glob('./data/MAG_*.csv')[0]
video_file = glob.glob('./data/*.mp4')[0]

# Load data
gps_df = pd.read_csv(gps_file)
acl_df = pd.read_csv(acl_file)
mag_df = pd.read_csv(mag_file)
video_path = video_file

logging.info(f"Loaded GPS file: {os.path.basename(gps_file)}")
logging.info(f"Loaded Accelerometer file: {os.path.basename(acl_file)}")
logging.info(f"Loaded Magnetometer file: {os.path.basename(mag_file)}")
logging.info(f"Loaded Video file: {os.path.basename(video_file)}")

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

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
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
#focal = 460.0
#pp = (293, 424)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise IOError("Error opening video file.")

fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
logging.info(f"Detected video resolution: {frame_width} x {frame_height}")
logging.info(f"Detected video FPS: {fps:.2f} frames per second.")

# Automatically estimate focal length and principal point
focal = frame_width * 0.9  # or 1.0
pp = (frame_width / 2, frame_height / 2)


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
scale_estimation_frames = 200

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

while frame_idx < scale_estimation_frames:
	ret, frame = cap.read()
	if not ret:
		break
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Safe optical flow
	if prev_pts is not None and len(prev_pts) > 0:
		next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **lk_params)
	else:
		prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
		next_pts, status, _ = None, None, None

	if next_pts is not None and status is not None:
		good_old = prev_pts[status.flatten() == 1]
		good_new = next_pts[status.flatten() == 1]

		if len(good_old) > 7:
			E, mask = cv2.findEssentialMat(good_new, good_old, focal=focal, pp=pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
			if E is not None:
				_, R_pose, t, mask_pose = cv2.recoverPose(E, good_new, good_old, focal=focal, pp=pp)

				dx = t[0][0]
				dz = t[2][0]
				dist = np.sqrt(dx**2 + dz**2)
				vo_distances.append(dist)

				if frame_idx < len(gps_interp_x) - 1:
					gps_dx = gps_interp_x[frame_idx+1] - gps_interp_x[frame_idx]
					gps_dz = gps_interp_z[frame_idx+1] - gps_interp_z[frame_idx]
					gps_dist = np.sqrt(gps_dx**2 + gps_dz**2)
					gps_distances.append(gps_dist)

	prev_gray = gray.copy()
	prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
	frame_idx += 1

# Calculate total distances
total_vo_distance = np.sum(vo_distances)
total_gps_distance = np.sum(gps_distances)

# Compute estimated scale
if total_vo_distance > 0:
	estimated_scale = total_gps_distance / total_vo_distance
else:
	estimated_scale = 1.0  # fallback

logging.info(f"Estimated scale from first {scale_estimation_frames} frames: {estimated_scale:.3f}")



while True:
	ret, frame = cap.read()
	if not ret:
		break

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# Safe optical flow
	if prev_pts is not None and len(prev_pts) > 0:
		next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **lk_params)
	else:
		prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
		next_pts, status, _ = None, None, None

	gps_correction_applied = False
	gps_jump_meters = 0

	if next_pts is not None and status is not None:
		good_old = prev_pts[status.flatten() == 1]
		good_new = next_pts[status.flatten() == 1]

		if len(good_old) > 7:
			E, mask = cv2.findEssentialMat(good_new, good_old, focal=focal, pp=pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
			if E is not None:
				_, R_pose, t, mask_pose = cv2.recoverPose(E, good_new, good_old, focal=focal, pp=pp)

				dx = t[0][0]
				dz = t[2][0]

				# Kalman Predict
				x = F @ x
				P = F @ P @ F.T + Q

				# VO motion update
				x[0,0] += dx * estimated_scale
				x[1,0] += dz * estimated_scale


				# Accelerometer motion update
				if frame_idx < len(filtered_acl):
					acc_x = filtered_acl.iloc[frame_idx]['ax']
					acc_z = filtered_acl.iloc[frame_idx]['az']

					vx = acc_x * dt
					vz = acc_z * dt
					x[2,0] += vx
					x[3,0] += vz

					x[0,0] += x[2,0] * dt
					x[1,0] += x[3,0] * dt

				# GPS Correction with hopping detection
				if frame_idx < len(gps_interp_x):
					z_gps = np.array([[gps_interp_x[frame_idx]], [gps_interp_z[frame_idx]]])
					if frame_idx > 0:
						gps_jump_meters = np.linalg.norm([
							gps_interp_x[frame_idx] - gps_interp_x[frame_idx-1],
							gps_interp_z[frame_idx] - gps_interp_z[frame_idx-1]
						])
						if gps_jump_meters < 1.0:  # 1 meters threshold after each frame
							y = z_gps - (H @ x)
							S = H @ P @ H.T + R_kalman
							K = P @ H.T @ np.linalg.inv(S)
							x = x + K @ y
							P = (np.eye(4) - K @ H) @ P
							gps_correction_applied = True

				trajectory.append((x[0,0], x[1,0]))
				results.append([frame_idx, x[0,0], x[1,0], gps_interp_x[frame_idx], gps_interp_z[frame_idx]])

	prev_gray = gray.copy()
	prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
	frame_idx += 1

	# Draw
	traj_img = np.zeros((800, 800, 3), dtype=np.uint8)
	traj_arr = np.array(trajectory)

	if traj_arr.shape[0] > 1:
		for j in range(1, traj_arr.shape[0]):
			x1, z1 = traj_arr[j-1,0]*0.8 + 400, traj_arr[j-1,1]*0.8 + 400
			x2, z2 = traj_arr[j,0]*0.8 + 400, traj_arr[j,1]*0.8 + 400
			cv2.line(traj_img, (int(x1), int(z1)), (int(x2), int(z2)), (0,255,0), 2)
	if frame_idx < len(gps_interp_x):
		for j in range(1, frame_idx):
			x1, z1 = gps_interp_x[j-1]*0.8 + 400, gps_interp_z[j-1]*0.8 + 400
			x2, z2 = gps_interp_x[j]*0.8 + 400, gps_interp_z[j]*0.8 + 400
			cv2.line(traj_img, (int(x1), int(z1)), (int(x2), int(z2)), (255,0,0), 2)

	# Draw Current Positions
	current_time = time.time()
	fps_live = 1 / (current_time - last_time)
	last_time = current_time

	text1 = f'FPS: {fps_live:.2f}'
	text2 = f'VO X:{x[0,0]:.2f} Z:{x[1,0]:.2f}'
	if (frame_idx - 1) >= len(gps_interp_x):
		logging.warning(
			f"Frame {frame_idx - 1} out of GPS interpolation bounds! Max valid index: {len(gps_interp_x) - 1}")
		gps_x_val = np.nan
		gps_z_val = np.nan
	else:
		gps_x_val = gps_interp_x[frame_idx - 1]
		gps_z_val = gps_interp_z[frame_idx - 1]

	text3 = f'GPS X:{gps_x_val:.2f} Z:{gps_z_val:.2f}'
	text4 = f'GPS Jump: {gps_jump_meters:.2f}m'
	text5 = 'GPS Correction: ' + ('APPLIED YES' if gps_correction_applied else 'SKIPPED NO')

	cv2.putText(traj_img, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
	cv2.putText(traj_img, text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
	cv2.putText(traj_img, text3, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
	cv2.putText(traj_img, text4, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
	cv2.putText(traj_img, text5, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0) if gps_correction_applied else (0,0,255), 2)

	combined = np.hstack((cv2.resize(frame, (800, 800)), traj_img))

	cv2.imshow("Visual Odometry + GPS + Accelerometer Realtime", combined)
	out.write(combined)
	key = cv2.waitKey(1) & 0xFF
	if key == ord('q'):
		logging.info("Simulation interrupted by user.")
		break
	elif key == ord('p'):
			logging.info("Simulation paused. Press 'p' again to resume...")
			while True:
				pause_frame = combined.copy()
				cv2.putText(pause_frame, 'PAUSED', (300, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
				cv2.imshow("Visual Odometry + GPS + Accelerometer Realtime", pause_frame)
				key2 = cv2.waitKey(500) & 0xFF
				if key2 == ord('p'):
					logging.info("Resuming simulation...")
					break
				elif key2 == ord('q'):
					logging.info("Simulation interrupted during pause.")
					cap.release()
					cv2.destroyAllWindows()
					exit()

out.release()
# --- Save Results ---
logging.info("Saving results...")

os.makedirs('./output', exist_ok=True)

results_df = pd.DataFrame(results, columns=["frame", "vo_x", "vo_z", "gps_x", "gps_z"])
results_df['error_m'] = np.sqrt((results_df['vo_x'] - results_df['gps_x'])**2 + (results_df['vo_z'] - results_df['gps_z'])**2)
results_df.to_csv('./output/trajectory_comparison.csv', index=False)

plt.figure(figsize=(10,6))
plt.plot(results_df['vo_x'], results_df['vo_z'], label='VO+Kalman+Accel', color='g')
plt.plot(results_df['gps_x'], results_df['gps_z'], label='GPS', linestyle='--', color='b')
plt.legend()
plt.xlabel('X meters')
plt.ylabel('Z meters')
plt.title('Trajectory Comparison')
plt.grid()
plt.axis('equal')
plt.savefig('./output/trajectory_plot.png')
plt.show()

plt.figure(figsize=(10,5))
plt.plot(results_df['frame'], results_df['error_m'])
plt.title('Frame-wise Error (VO+Accel vs GPS)')
plt.xlabel('Frame')
plt.ylabel('Error (meters)')
plt.grid()
plt.savefig('./output/error_plot.png')
plt.show()

logging.info(f"Average Error: {results_df['error_m'].mean():.2f} meters")
logging.info(f"Max Error: {results_df['error_m'].max():.2f} meters")
logging.info(f"Min Error: {results_df['error_m'].min():.2f} meters")

logging.info("All outputs saved to './output/'. Done!")
