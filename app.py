from flask import Flask, request, send_file
from dotenv import load_dotenv
from flask_cors import CORS
import os 
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import json
import traceback
from dtaidistance import dtw
from datetime import datetime


load_dotenv()
app = Flask(__name__)
CORS(app)

app.config['DEBUG'] = os.environ.get('FLASK_DEBUG')

def engine():
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose


    # Initialize MediaPipe Pose model
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

    # Define text landmarks
    LANDMARKS = {
        'RIGHT_ANKLE': 28,
        'RIGHT_KNEE': 26,
        'RIGHT_HIP': 24,
        'LEFT_ANKLE': 27,
        'LEFT_KNEE': 25,
        'LEFT_HIP': 23
    }

    def calc_dist(p1, p2):
        a = np.array([p1.x, p1.y,p1.z])
        b = np.array([p2.x, p2.y, p2.z])

        squared_dist = np.sum((a-b)**2, axis=0)
        return np.sqrt(squared_dist)

    def calc_angle(p1, p2, p3):
        a = np.array([p1.x, p1.y,p1.z])
        b = np.array([p2.x, p2.y, p2.z])
        c = np.array([p3.x, p3.y,p3.z])

        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.rad2deg(np.arccos(cosine_angle))

        return angle


    def dumbell_thrust(landmarks):
        global leg_error_occurences
        global elbow_error_occurences
        
        errors = []

        left_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        right_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]

        left_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        right_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]

        left_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]

        left_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        right_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]

        left_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]

        left_knee_angle = calc_angle(left_ankle, left_knee, left_hip)
        right_knee_angle = calc_angle(right_ankle, right_knee, right_hip)


        # if left_knee_angle < 90 or right_knee_angle < 90:
        #     errors.append("Don't bend down too much")
        
        if abs(calc_dist(left_ankle, right_ankle) - calc_dist(left_hip, right_hip)) > 0.07:
            errors.append("Keep legs hip width apart")
        
        if abs(calc_dist(left_elbow, right_elbow) - calc_dist(left_shoulder, right_shoulder)) > 0.07:
            errors.append("Keep elbows parallel to shoulders")

        return errors


    # Function to smooth a list of values using a simple moving average filter
    def smooth_values(values, window_size=5):
        smoothed_values = []
        for i in range(len(values)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(values), i + window_size // 2 + 1)
            smoothed_values.append(np.mean(values[start_idx:end_idx]))
        return smoothed_values

    # Function to process a single video and extract angle values
    def process_video(video_path):
        cap = cv2.VideoCapture(video_path)
        angle_values_left = []
        angle_values_right = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect pose landmarks
            results = pose.process(frame_rgb)

            # Extract landmarks
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Ensure enough landmarks are detected
                if len(landmarks) >= 6:
                    # Landmark indices: 
                    # Right leg: RIGHT_ANKLE - RIGHT_KNEE - RIGHT_HIP
                    # Left leg: LEFT_ANKLE - LEFT_KNEE - LEFT_HIP

                    # Right leg
                    right_ankle = np.array([landmarks[LANDMARKS['RIGHT_ANKLE']].x, landmarks[LANDMARKS['RIGHT_ANKLE']].y, landmarks[LANDMARKS['RIGHT_ANKLE']].z])
                    right_knee = np.array([landmarks[LANDMARKS['RIGHT_KNEE']].x, landmarks[LANDMARKS['RIGHT_KNEE']].y, landmarks[LANDMARKS['RIGHT_KNEE']].z])
                    right_hip = np.array([landmarks[LANDMARKS['RIGHT_HIP']].x, landmarks[LANDMARKS['RIGHT_HIP']].y, landmarks[LANDMARKS['RIGHT_HIP']].z])

                    # Left leg
                    left_ankle = np.array([landmarks[LANDMARKS['LEFT_ANKLE']].x, landmarks[LANDMARKS['LEFT_ANKLE']].y, landmarks[LANDMARKS['LEFT_ANKLE']].z])
                    left_knee = np.array([landmarks[LANDMARKS['LEFT_KNEE']].x, landmarks[LANDMARKS['LEFT_KNEE']].y, landmarks[LANDMARKS['LEFT_KNEE']].z])
                    left_hip = np.array([landmarks[LANDMARKS['LEFT_HIP']].x, landmarks[LANDMARKS['LEFT_HIP']].y, landmarks[LANDMARKS['LEFT_HIP']].z])

                    # Calculate angle between ankle, knee, and hip (with respect to the knee)
                    angle_right = np.degrees(np.arctan2(np.linalg.norm(np.cross(right_hip - right_knee, right_ankle - right_knee)), np.dot(right_hip - right_knee, right_ankle - right_knee)))
                    angle_left = np.degrees(np.arctan2(np.linalg.norm(np.cross(left_hip - left_knee, left_ankle - left_knee)), np.dot(left_hip - left_knee, left_ankle - left_knee)))

                    # Append angle values to lists
                    angle_values_right.append(angle_right)
                    angle_values_left.append(angle_left)

        # Release video capture
        cap.release()

        return angle_values_left, angle_values_right

    # Paths to the two MP4 files
    video_path1 = 'arvin_good.mov'
    video_path2 = 'arvin_bad.mp4'

    # Process both videos
    angle_values_left1, angle_values_right1 = process_video(video_path1)
    angle_values_left2, angle_values_right2 = process_video(video_path2)

    # Calculate average angle values for each frame
    average_angle_values1 = [(left + right) / 2 for left, right in zip(angle_values_left1, angle_values_right1)]
    average_angle_values2 = [(left + right) / 2 for left, right in zip(angle_values_left2, angle_values_right2)]

    # Smooth the angle values using a moving average filter
    smooth_window_size = 5
    smoothed_values1 = smooth_values(average_angle_values1, window_size=smooth_window_size)
    smoothed_values2 = smooth_values(average_angle_values2, window_size=smooth_window_size)

    min_frame = np.argmin(smoothed_values2)

    s1 = np.array(smoothed_values1)
    s2 = np.array(smoothed_values2)
    path = dtw.warping_path(s1, s2)

    distance = dtw.distance(s1, s2) # print to see distance

    def save_json(errors):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data = {"timestamp": timestamp, "errors": errors}
        with open("error_log.json", "a") as f:
            json.dump(data, f)
            f.write("\n")

    curr_frame = 0

    # Initialize VideoCapture
    cap = cv2.VideoCapture('arvin_bad.mp4')

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object
    output_video_path = 'output_video_bad_form.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Define the codec (e.g., avc1 for H.264 format)
    _out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (frame_width, frame_height))

    ###

    # Setup mediapipe instance
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic


    # Main loop
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic, mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            # Recolor image to RGB
            frame = cv2.flip(frame, 1)
            frame_height, frame_width, _ = frame.shape
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks
            try:
                if curr_frame == min_frame and distance < 80:
                    #Todo 
                    cv2.putText(image, f'Ensure you bend down to a 90 degree knee angle!', (int(frame_width/2),int(frame_height/2)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 2)
                    #Add 20 frames (pause) to saved video
                    curr_frame += 1
                    print('reached')
                    for x in range(60):
                        _out.write(image)
                    continue

                landmarks = results.pose_landmarks.landmark
                errors = dumbell_thrust(landmarks)
                if errors:
                    save_json(errors)
                #print(errors)
                overlay = image.copy()
                alpha = 0.4
                elbow_clr = (0,255,0)
                leg_clr = (0,255,0)
                for error_string in errors:
                    if 'elbows' in error_string:
                        elbow_clr = (0, 0, 255)
                    if 'leg' in error_string:
                        leg_clr = (0, 0, 255)


                left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
                right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
                l_pixel_x1 = int(left_elbow.x * frame_width)
                l_pixel_y1 = int(left_elbow.y * frame_height)
                r_pixel_x1 = int(right_elbow.x * frame_width)
                r_pixel_y1 = int(right_elbow.y * frame_height)
                cv2.circle(overlay, (l_pixel_x1, l_pixel_y1), radius=20, color=elbow_clr, thickness=-1)
                cv2.circle(overlay, (r_pixel_x1, r_pixel_y1), radius=20, color=elbow_clr, thickness=-1)
                cv2.line(overlay, (l_pixel_x1, l_pixel_y1), (r_pixel_x1, r_pixel_y1), elbow_clr,15) 
                
                left_leg = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
                right_leg = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
                l_pixel_x2 = int(left_leg.x * frame_width)
                l_pixel_y2 = int(left_leg.y * frame_height)
                r_pixel_x2 = int(right_leg.x * frame_width)
                r_pixel_y2 = int(right_leg.y * frame_height)
                cv2.circle(overlay, (l_pixel_x2, l_pixel_y2), radius=20, color=leg_clr, thickness=-1)
                cv2.circle(overlay, (r_pixel_x2, r_pixel_y2), radius=20, color=leg_clr, thickness=-1)
                cv2.line(overlay, (l_pixel_x2, l_pixel_y2), (r_pixel_x2, r_pixel_y2), leg_clr,15) 
                

                image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

                if elbow_clr == (0, 0, 255):
                    cv2.putText(image, f'Keep elbows at shoulder width apart!', (r_pixel_x1+50,r_pixel_y1+50), cv2.FONT_HERSHEY_SIMPLEX, 0.85, elbow_clr, 2)
                else:
                    cv2.putText(image, f'Good Elbow Placement!', (r_pixel_x1+50,r_pixel_y1+50), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0,255,0), 2)
                if leg_clr == (0, 0, 255):
                    cv2.putText(image, f'Keep legs at hip width apart!', (r_pixel_x2+50,r_pixel_y2+50), cv2.FONT_HERSHEY_SIMPLEX, 0.85, leg_clr, 2)
                else:
                    cv2.putText(image, f'Good Feet Placement!', (r_pixel_x2+50,r_pixel_y2+50), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0,255,0), 2)
                for x in range(2):
                    _out.write(image)
                        
            except:
                print(traceback.format_exc())

            # Make detection
            results = pose.process(image)
            results_face = holistic.process(image)

            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                    )               
            
            curr_frame += 1
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    # Release VideoCapture and close all windows
    cap.release()
    _out.release()
    cv2.destroyAllWindows()

    return "good"


@app.route("/")
def hello():
    return "Hello World!"


@app.route("/test", methods=["POST"])
def test():

    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    file.save("arvin_bad.mp4")

    engine()

    return send_file("output_video_bad_form.mp4", as_attachment=True)


    

if __name__ == "__main__":
    app.run()   


