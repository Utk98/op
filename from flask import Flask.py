# # # from flask import Flask
# # # from flask_socketio import SocketIO
# # # from pyngrok import ngrok
# # # print("gvfd")
# # # app = Flask(__name__)
# # # # Ensure CORS is set up correctly to accept connections from your client's domain
# # # socketio = SocketIO(app, cors_allowed_origins="*")

# # # @app.route('/')
# # # def index():
# # #     return "Socket.IO server"

# # # @socketio.on('connect')
# # # def on_connect():
# # #     print('Client connected')
# # #     socketio.emit('message', 'Welcome, client!')

# # # @socketio.on('connect_error')
# # # def on_connect_error(data):
# # #     print('Connection failed:', data)

# # # @socketio.on('message')
# # # def handle_message(data):
# # #     print('Received message:', data)

# # # # ngrok.set_auth_token("2gDVBMbJ3zF6Fdccaicxr3QIzbu_7ho4uhZoAUaNg5MAQeAob")
# # # # public_url = ngrok.connect(5000)
# # # # print(" * ngrok tunnel \"{}\" -> \"http://127.0.0.1:5000/\"".format(public_url))

# # # if __name__ == '__main__':
# # #     # Run on 0.0.0.0 to accept connections on all public IPs
# # #     socketio.run(app, host='0.0.0.0', port=5000, debug=True)
    
# # # # # # # # # # # # # # # # # # # # # # # # # Import necessary libraries
# # # from flask import Flask, request, jsonify
# # # from flask_ngrok import run_with_ngrok
# # # import cv2
# # # import numpy as np
# # # import face_recognition
# # # from pyngrok import ngrok

# # # # Initialize Flask app
# # # app = Flask(__name__)
# # # run_with_ngrok(app)

# # # # Set up ngrok authentication token
# # # ngrok.set_auth_token("2gDk6gkcJSCa7YpuRI7xQPnzqQc_7cujt3BiEHW9HYTFZfMXH")

# # # KNOWN_IMAGE_PATH = "C:/Users/OMR-09/Pictures/img2.jpg"

# # # def detect_person_match(video_path):
# # #     # Load the known image
# # #     known_image = face_recognition.load_image_file(KNOWN_IMAGE_PATH)
# # #     known_encoding = face_recognition.face_encodings(known_image)[0]

# # #     # Open the video capture
# # #     cap = cv2.VideoCapture(video_path)

# # #     # Create a background subtractor object
# # #     bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=True)

# # #     # Initialize variables for face detection and background movement detection
# # #     face_locations = []

# # #     # Iterate over frames
# # #     while True:
# # #         ret, frame = cap.read()
# # #         if not ret:
# # #             break

# # #         # Apply background subtraction to detect movement in the background
# # #         fg_mask = bg_subtractor.apply(frame)

# # #         # Obtain the shadow value from the background subtractor
# # #         shadow_value = bg_subtractor.getShadowValue()

# # #         # Invert the shadow mask
# # #         fg_mask[fg_mask == shadow_value] = 0

# # #         # Find contours of moving objects
# # #         contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # #         # Check if any contours (movement) are detected
# # #         background_movement = False
# # #         for contour in contours:
# # #             area = cv2.contourArea(contour)
# # #             if area > 1000:  # Adjust threshold as needed
# # #                 background_movement = True
# # #                 break

# # #         if background_movement:
# # #             cap.release()
# # #             return "Movement"

# # #         # If no significant background movement is detected, check for face match
# # #         if not background_movement:
# # #             # Find face locations in the frame
# # #             face_locations = face_recognition.face_locations(frame)

# # #             # Check for more than one face detected or not match
# # #             if len(face_locations) > 1:
# # #                 cap.release()
# # #                 return "More than One Face Detected"

# # #             # # Compare face encoding in the frame with the known face encoding
# # #             # if len(face_locations) == 1:
# # #             #     face_encoding = face_recognition.face_encodings(frame, face_locations)[0]
# # #             #     match = face_recognition.compare_faces([known_encoding], face_encoding)
# # #             #     if match[0]:
# # #             #         cap.release()
# # #             #         return "Match"

# # #             # Detect neck bending
# # #             if len(face_locations) == 1:
# # #                 face_landmarks = face_recognition.face_landmarks(frame, face_locations)
# # #                 if face_landmarks:
# # #                     neck_angle = detect_neck_bending(face_landmarks[0])
# # #                     if neck_angle > 130 or neck_angle < 125:  # Adjust threshold as needed
# # #                         cap.release()
# # #                         return "Neck Bending Detected"

# # #     # If no significant background movement or face match is detected, return Not Match
# # #     cap.release()
# # #     return "Not Match"

# # # def detect_neck_bending(face_landmarks):
# # #     # Extract relevant landmarks for neck estimation
# # #     top_nose = face_landmarks['nose_bridge'][0]
# # #     bottom_nose = face_landmarks['nose_tip'][0]
# # #     top_chin = face_landmarks['chin'][8]
# # #     bottom_chin = face_landmarks['chin'][0]

# # #     # Calculate vectors for neck and face
# # #     neck_vector = np.array(bottom_chin) - np.array(top_chin)
# # #     face_vector = np.array(bottom_nose) - np.array(top_nose)

# # #     # Calculate angle between neck and face vectors
# # #     angle = np.degrees(np.arccos(np.dot(neck_vector, face_vector) /
# # #                                   (np.linalg.norm(neck_vector) * np.linalg.norm(face_vector))))
# # #     return angle

# # # @app.route('/match_person', methods=['POST'])
# # # def match_person():
# # #     if 'video' not in request.files:
# # #         return jsonify({'error': 'Missing video file'})

# # #     video_file = request.files['video']

# # #     if video_file.filename == '':
# # #         return jsonify({'error': 'No selected file'})

# # #     # Save the video file temporarily
# # #     video_path = 'temp_video.mp4'
# # #     video_file.save(video_path)

# # #     # Detect person match in the video with the known image
# # #     result = detect_person_match(video_path)

# # #     return jsonify({'result': result})

# # # if __name__ == '__main__':
# # #     public_url = ngrok.connect(5000).public_url
# # #     print(" * Running on", public_url)
# # #     try:
# # #         app.run()
# # #     except KeyboardInterrupt:
# # # # # #         print(" * Shutting down Flask app...")

# # # # # # # # # # # # # # # # # # # # # # # # # # # Import necessary libraries
# # # # # # # # # # # # # # # # # # # # # # # # # # from flask import Flask
# # # # # # # # # # # # # # # # # # # # # # # # # # from flask_socketio import SocketIO, emit
# # # # # # # # # # # # # # # # # # # # # # # # # # import cv2
# # # # # # # # # # # # # # # # # # # # # # # # # # import numpy as np
# # # # # # # # # # # # # # # # # # # # # # # # # # import face_recognition
# # # # # # # # # # # # # # # # # # # # # # # # # # from io import BytesIO
# # # # # # # # # # # # # # # # # # # # # # # # # # import subprocess
# # # # # # # # # # # # # # # # # # # # # # # # # # import base64

# # # # # # # # # # # # # # # # # # # # # # # # # # # Initialize Flask app
# # # # # # # # # # # # # # # # # # # # # # # # # # app = Flask(__name__)
# # # # # # # # # # # # # # # # # # # # # # # # # # app.config['SECRET_KEY'] = '2gDVBMbJ3zF6Fdccaicxr3QIzbu_7ho4uhZoAUaNg5MAQeAobe'
# # # # # # # # # # # # # # # # # # # # # # # # # # socketio = SocketIO(app, cors_allowed_origins="*")

# # # # # # # # # # # # # # # # # # # # # # # # # # KNOWN_IMAGE_PATH = "C:/Users/OMR-09/Pictures/img2.jpg"  # Change this to the path of your known person image

# # # # # # # # # # # # # # # # # # # # # # # # # # @socketio.on('connect')
# # # # # # # # # # # # # # # # # # # # # # # # # # def on_connect():
# # # # # # # # # # # # # # # # # # # # # # # # # #     print('Client connected')
# # # # # # # # # # # # # # # # # # # # # # # # # #     socketio.emit('message', 'Welcome, client!')

# # # # # # # # # # # # # # # # # # # # # # # # # # def detect_person_match(known_encoding, frame):
# # # # # # # # # # # # # # # # # # # # # # # # # #     # Load the known image
# # # # # # # # # # # # # # # # # # # # # # # # # #     known_image = face_recognition.load_image_file(KNOWN_IMAGE_PATH)
# # # # # # # # # # # # # # # # # # # # # # # # # #     known_encoding = face_recognition.face_encodings(known_image)[0]

# # # # # # # # # # # # # # # # # # # # # # # # # #     # Find face locations in the frame
# # # # # # # # # # # # # # # # # # # # # # # # # #     face_locations = face_recognition.face_locations(frame)

# # # # # # # # # # # # # # # # # # # # # # # # # #     # Check for more than one face detected
# # # # # # # # # # # # # # # # # # # # # # # # # #     if len(face_locations) > 1:
# # # # # # # # # # # # # # # # # # # # # # # # # #         return "More than One Face Detected"

# # # # # # # # # # # # # # # # # # # # # # # # # #     # Compare face encoding in the frame with the known face encoding
# # # # # # # # # # # # # # # # # # # # # # # # # #     if len(face_locations) == 1:
# # # # # # # # # # # # # # # # # # # # # # # # # #         face_encoding = face_recognition.face_encodings(frame, face_locations)[0]
# # # # # # # # # # # # # # # # # # # # # # # # # #         match = face_recognition.compare_faces([known_encoding], face_encoding)
# # # # # # # # # # # # # # # # # # # # # # # # # #         if match[0]:
# # # # # # # # # # # # # # # # # # # # # # # # # #             # Detect neck bending
# # # # # # # # # # # # # # # # # # # # # # # # # #             face_landmarks = face_recognition.face_landmarks(frame, face_locations)
# # # # # # # # # # # # # # # # # # # # # # # # # #             if face_landmarks:
# # # # # # # # # # # # # # # # # # # # # # # # # #                 neck_angle = detect_neck_bending(face_landmarks[0])
# # # # # # # # # # # # # # # # # # # # # # # # # #                 if neck_angle > 130 or neck_angle < 125:  # Adjust threshold as needed
# # # # # # # # # # # # # # # # # # # # # # # # # #                     return "Match with Neck Bending Detected"
# # # # # # # # # # # # # # # # # # # # # # # # # #             return "Match"
# # # # # # # # # # # # # # # # # # # # # # # # # #         else:
# # # # # # # # # # # # # # # # # # # # # # # # # #             return "Not Match"

# # # # # # # # # # # # # # # # # # # # # # # # # #     return "No Face Detected"

# # # # # # # # # # # # # # # # # # # # # # # # # # def detect_neck_bending(face_landmarks):
# # # # # # # # # # # # # # # # # # # # # # # # # #     # Extract relevant landmarks for neck estimation
# # # # # # # # # # # # # # # # # # # # # # # # # #     top_nose = face_landmarks['nose_bridge'][0]
# # # # # # # # # # # # # # # # # # # # # # # # # #     bottom_nose = face_landmarks['nose_tip'][0]
# # # # # # # # # # # # # # # # # # # # # # # # # #     top_chin = face_landmarks['chin'][8]
# # # # # # # # # # # # # # # # # # # # # # # # # #     bottom_chin = face_landmarks['chin'][0]

# # # # # # # # # # # # # # # # # # # # # # # # # #     # Calculate vectors for neck and face
# # # # # # # # # # # # # # # # # # # # # # # # # #     neck_vector = np.array(bottom_chin) - np.array(top_chin)
# # # # # # # # # # # # # # # # # # # # # # # # # #     face_vector = np.array(bottom_nose) - np.array(top_nose)

# # # # # # # # # # # # # # # # # # # # # # # # # #     # Calculate angle between neck and face vectors
# # # # # # # # # # # # # # # # # # # # # # # # # #     angle = np.degrees(np.arccos(np.dot(neck_vector, face_vector) /
# # # # # # # # # # # # # # # # # # # # # # # # # #                                   (np.linalg.norm(neck_vector) * np.linalg.norm(face_vector))))
# # # # # # # # # # # # # # # # # # # # # # # # # #     return angle

# # # # # # # # # # # # # # # # # # # # # # # # # # def detect_background_movement(previous_frame, current_frame):
# # # # # # # # # # # # # # # # # # # # # # # # # #     # Convert frames to grayscale
# # # # # # # # # # # # # # # # # # # # # # # # # #     prev_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
# # # # # # # # # # # # # # # # # # # # # # # # # #     curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

# # # # # # # # # # # # # # # # # # # # # # # # # #     # Compute absolute difference between frames
# # # # # # # # # # # # # # # # # # # # # # # # # #     frame_diff = cv2.absdiff(prev_gray, curr_gray)

# # # # # # # # # # # # # # # # # # # # # # # # # #     # Apply threshold to detect significant changes
# # # # # # # # # # # # # # # # # # # # # # # # # #     _, threshold = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)

# # # # # # # # # # # # # # # # # # # # # # # # # #     # Count the number of non-zero pixels
# # # # # # # # # # # # # # # # # # # # # # # # # #     non_zero_count = cv2.countNonZero(threshold)

# # # # # # # # # # # # # # # # # # # # # # # # # #     # Adjust threshold based on image size
# # # # # # # # # # # # # # # # # # # # # # # # # #     image_size = current_frame.shape[0] * current_frame.shape[1]
# # # # # # # # # # # # # # # # # # # # # # # # # #     threshold_ratio = non_zero_count / image_size

# # # # # # # # # # # # # # # # # # # # # # # # # #     # If more than 1% of the image has changed, consider it as background movement
# # # # # # # # # # # # # # # # # # # # # # # # # #     if threshold_ratio > 0.01:
# # # # # # # # # # # # # # # # # # # # # # # # # #         return True
# # # # # # # # # # # # # # # # # # # # # # # # # #     else:
# # # # # # # # # # # # # # # # # # # # # # # # # #         return False#
    
# # # # # # # # # # # # # # # # # # # # # # # # # # @socketio.on('video_data')
# # # # # # # # # # # # # # # # # # # # # # # # # # def handle_video_webm(webm_data):
# # # # # # # # # # # # # # # # # # # # # # # # # #     print("Video received")

# # # # # # # # # # # # # # # # # # # # # # # # # #     # Save the webm data to a file
# # # # # # # # # # # # # # # # # # # # # # # # # #     with open('input.webm', 'wb') as f:
# # # # # # # # # # # # # # # # # # # # # # # # # #         f.write(webm_data)

# # # # # # # # # # # # # # # # # # # # # # # # # #     # Use ffmpeg to decode the webm file to frames
# # # # # # # # # # # # # # # # # # # # # # # # # #     cmd = ['ffmpeg', '-i', 'input.webm', '-f', 'image2pipe', '-pix_fmt', 'rgb24', '-vcodec', 'rawvideo', '-']
# # # # # # # # # # # # # # # # # # # # # # # # # #     process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# # # # # # # # # # # # # # # # # # # # # # # # # #     # Read frames from ffmpeg output
# # # # # # # # # # # # # # # # # # # # # # # # # #     while True:
# # # # # # # # # # # # # # # # # # # # # # # # # #         frame = process.stdout.read(640 * 480 * 3)  # Adjust frame size as needed
# # # # # # # # # # # # # # # # # # # # # # # # # #         if not frame:
# # # # # # # # # # # # # # # # # # # # # # # # # #             break

# # # # # # # # # # # # # # # # # # # # # # # # # #         # Convert bytes to numpy array
# # # # # # # # # # # # # # # # # # # # # # # # # #         frame = np.frombuffer(frame, dtype=np.uint8).reshape(480, 640, 3)  # Adjust frame size as needed

# # # # # # # # # # # # # # # # # # # # # # # # # #         # Process the frame (detect person match)
# # # # # # # # # # # # # # # # # # # # # # # # # #         known_image = face_recognition.load_image_file(KNOWN_IMAGE_PATH)
# # # # # # # # # # # # # # # # # # # # # # # # # #         known_encoding = face_recognition.face_encodings(known_image)[0]
# # # # # # # # # # # # # # # # # # # # # # # # # #         result = detect_person_match(known_encoding, frame)

# # # # # # # # # # # # # # # # # # # # # # # # # #         # Send the result to the frontend
# # # # # # # # # # # # # # # # # # # # # # # # # #         emit('result', result)
# # # # # # # # # # # # # # # # # # # # # # # # # #         print("Result sent to frontend:", result)

# # # # # # # # # # # # # # # # # # # # # # # # # #     process.kill()
    
# # # # # # # # # # # # # # # # # # # # # # # # # # if __name__ == '__main__':
# # # # # # # # # # # # # # # # # # # # # # # # # #     detect_background_movement.previous_frame = None
# # # # # # # # # # # # # # # # # # # # # # # # # #     socketio.run(app, host='0.0.0.0', port=5000, debug=True)
    
    
# # # # # # # # # # # # # # # # # # # # # # # # # # # from flask import Flask
# # # # # # # # # # # # # # # # # # # # # # # # # # # from flask_socketio import SocketIO

# # # # # # # # # # # # # # # # # # # # # # # # # # # app = Flask(__name__)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # Ensure CORS is set up correctly to accept connections from your client's domain
# # # # # # # # # # # # # # # # # # # # # # # # # # # socketio = SocketIO(app, cors_allowed_origins="*")

# # # # # # # # # # # # # # # # # # # # # # # # # # # @app.route('/')
# # # # # # # # # # # # # # # # # # # # # # # # # # # def index():
# # # # # # # # # # # # # # # # # # # # # # # # # # #     return "Socket.IO server"

# # # # # # # # # # # # # # # # # # # # # # # # # # # @socketio.on('connect')
# # # # # # # # # # # # # # # # # # # # # # # # # # # def on_connect():
# # # # # # # # # # # # # # # # # # # # # # # # # # #     print('Client connected')
# # # # # # # # # # # # # # # # # # # # # # # # # # #     socketio.emit('message', 'Welcome, client!')

# # # # # # # # # # # # # # # # # # # # # # # # # # # @socketio.on('message')
# # # # # # # # # # # # # # # # # # # # # # # # # # # def handle_message(data):
# # # # # # # # # # # # # # # # # # # # # # # # # # #     print('Received message:', data)

# # # # # # # # # # # # # # # # # # # # # # # # # # # if __name__ == '__main__':
# # # # # # # # # # # # # # # # # # # # # # # # # # #     # Run on 0.0.0.0 to accept connections on all public IPs
# # # # # # # # # # # # # # # # # # # # # # # # # # #     socketio.run(app, host='0.0.0.0', port=5000, debug=True)




# # # # # # # # # # # # # # # # # # # # # # # # # # # from flask import Flask
# # # # # # # # # # # # # # # # # # # # # # # # # # # from flask_socketio import SocketIO, emit
# # # # # # # # # # # # # # # # # # # # # # # # # # # import os

# # # # # # # # # # # # # # # # # # # # # # # # # # # app = Flask(__name__)
# # # # # # # # # # # # # # # # # # # # # # # # # # # socketio = SocketIO(app,cors_allowed_origins="*")

# # # # # # # # # # # # # # # # # # # # # # # # # # # @socketio.on('connect')
# # # # # # # # # # # # # # # # # # # # # # # # # # # def handle_connect():
# # # # # # # # # # # # # # # # # # # # # # # # # # #     print('Client connected')

# # # # # # # # # # # # # # # # # # # # # # # # # # # @socketio.on('video_data')
# # # # # # # # # # # # # # # # # # # # # # # # # # # def handle_video_data(data):
# # # # # # # # # # # # # # # # # # # # # # # # # # #     print('Received video data')
# # # # # # # # # # # # # # # # # # # # # # # # # # #     # Process the video data here
# # # # # # # # # # # # # # # # # # # # # # # # # # #     process_result = "Processed Data" # Dummy result
# # # # # # # # # # # # # # # # # # # # # # # # # # #     emit('result', process_result) # Send result back to client
# # # # # # # # # # # # # # # # # # # # # # # # # # #     save_video_data(data)
    

# # # # # # # # # # # # # # # # # # # # # # # # # # # def save_video_data(data):
# # # # # # # # # # # # # # # # # # # # # # # # # # #     # Create a directory for saved videos if it doesn't exist
# # # # # # # # # # # # # # # # # # # # # # # # # # #     save_path = 'saved_videos'
# # # # # # # # # # # # # # # # # # # # # # # # # # #     if not os.path.exists(save_path):
# # # # # # # # # # # # # # # # # # # # # # # # # # #         os.makedirs(save_path)
    
# # # # # # # # # # # # # # # # # # # # # # # # # # #     # Define a file path
# # # # # # # # # # # # # # # # # # # # # # # # # # #     file_path = os.path.join(save_path, 'output.webm')
    
# # # # # # # # # # # # # # # # # # # # # # # # # # #     # Write the data to a file
# # # # # # # # # # # # # # # # # # # # # # # # # # #     with open(file_path, 'wb') as f:
# # # # # # # # # # # # # # # # # # # # # # # # # # #         f.write(data) # Write binary data to file
# # # # # # # # # # # # # # # # # # # # # # # # # # #     print(f'Video saved to {file_path}')

# # # # # # # # # # # # # # # # # # # # # # # # # # # if __name__ == '__main__':
# # # # # # # # # # # # # # # # # # # # # # # # # # #     socketio.run(app, debug=True,host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)

# # # # # # # # # # # # # # # # # # # # # # # # # import cv2
# # # # # # # # # # # # # # # # # # # # # # # # # import numpy as np
# # # # # # # # # # # # # # # # # # # # # # # # # from flask import Flask, render_template
# # # # # # # # # # # # # # # # # # # # # # # # # from flask_socketio import SocketIO, emit

# # # # # # # # # # # # # # # # # # # # # # # # # app = Flask(__name__)
# # # # # # # # # # # # # # # # # # # # # # # # # # app.config['SECRET_KEY'] = 'your_secret_key'
# # # # # # # # # # # # # # # # # # # # # # # # # socketio = SocketIO(app, cors_allowed_origins="*")

# # # # # # # # # # # # # # # # # # # # # # # # # # Function to detect faces in a frame
# # # # # # # # # # # # # # # # # # # # # # # # # def detect_faces(frame):
# # # # # # # # # # # # # # # # # # # # # # # # #     # Convert frame to grayscale
# # # # # # # # # # # # # # # # # # # # # # # # #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# # # # # # # # # # # # # # # # # # # # # # # # #     # Detect faces using Haar Cascade Classifier
# # # # # # # # # # # # # # # # # # # # # # # # #     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# # # # # # # # # # # # # # # # # # # # # # # # #     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# # # # # # # # # # # # # # # # # # # # # # # # #     return len(faces)

# # # # # # # # # # # # # # # # # # # # # # # # # # SocketIO event handler for receiving video frames
# # # # # # # # # # # # # # # # # # # # # # # # # @socketio.on('video_frame')
# # # # # # # # # # # # # # # # # # # # # # # # # def handle_video_frame(frame):
# # # # # # # # # # # # # # # # # # # # # # # # #     # Decode frame from base64 string
# # # # # # # # # # # # # # # # # # # # # # # # #     nparr = np.frombuffer(frame, np.uint8)
# # # # # # # # # # # # # # # # # # # # # # # # #     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

# # # # # # # # # # # # # # # # # # # # # # # # #     # Detect faces in the frame
# # # # # # # # # # # # # # # # # # # # # # # # #     faces_count = detect_faces(img)

# # # # # # # # # # # # # # # # # # # # # # # # #     # Send the count of detected faces back to the client
# # # # # # # # # # # # # # # # # # # # # # # # #     emit('faces_count', faces_count)

# # # # # # # # # # # # # # # # # # # # # # # # # # Render HTML page with SocketIO client
# # # # # # # # # # # # # # # # # # # # # # # # # @app.route('/')
# # # # # # # # # # # # # # # # # # # # # # # # # def index():
# # # # # # # # # # # # # # # # # # # # # # # # #     return render_template('index.html')

# # # # # # # # # # # # # # # # # # # # # # # # # if __name__ == '__main__':
# # # # # # # # # # # # # # # # # # # # # # # # #     socketio.run(app,debug=True,host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
# # # # # # # # # # # # # # # # # # # # # # # import cv2
# # # # # # # # # # # # # # # # # # # # # # # import numpy as np
# # # # # # # # # # # # # # # # # # # # # # # from flask import Flask, render_template
# # # # # # # # # # # # # # # # # # # # # # # from flask_socketio import SocketIO, emit
# # # # # # # # # # # # # # # # # # # # # # # import base64

# # # # # # # # # # # # # # # # # # # # # # # app = Flask(__name__)
# # # # # # # # # # # # # # # # # # # # # # # # app.config['SECRET_KEY'] = 'your_secret_key'
# # # # # # # # # # # # # # # # # # # # # # # socketio = SocketIO(app, cors_allowed_origins="*")

# # # # # # # # # # # # # # # # # # # # # # # # Function to detect faces in a frame
# # # # # # # # # # # # # # # # # # # # # # # def detect_faces(frame):
# # # # # # # # # # # # # # # # # # # # # # #     # Convert frame to grayscale
# # # # # # # # # # # # # # # # # # # # # # #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# # # # # # # # # # # # # # # # # # # # # # #     # Detect faces using Haar Cascade Classifier
# # # # # # # # # # # # # # # # # # # # # # #     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# # # # # # # # # # # # # # # # # # # # # # #     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# # # # # # # # # # # # # # # # # # # # # # #     return len(faces)

# # # # # # # # # # # # # # # # # # # # # # # # SocketIO event handler for client connection
# # # # # # # # # # # # # # # # # # # # # # # @socketio.on('connect')
# # # # # # # # # # # # # # # # # # # # # # # def handle_connect():
# # # # # # # # # # # # # # # # # # # # # # #     print('Client connected')

# # # # # # # # # # # # # # # # # # # # # # # # SocketIO event handler for receiving video frames
# # # # # # # # # # # # # # # # # # # # # # # # SocketIO event handler for receiving video frames

# # # # # # # # # # # # # # # # # # # # # # # # SocketIO event handler for receiving video frames
# # # # # # # # # # # # # # # # # # # # # # # @socketio.on('video_data')
# # # # # # # # # # # # # # # # # # # # # # # def handle_video_frame(frame_data):
# # # # # # # # # # # # # # # # # # # # # # #     print('Video received')

# # # # # # # # # # # # # # # # # # # # # # #     # Decode base64 encoded frame
# # # # # # # # # # # # # # # # # # # # # # #     frame_bytes = base64.b64decode(frame_data)
# # # # # # # # # # # # # # # # # # # # # # #     frame_arr = np.frombuffer(frame_bytes, dtype=np.uint8)
# # # # # # # # # # # # # # # # # # # # # # #     frame = cv2.imdecode(frame_arr, flags=cv2.IMREAD_COLOR)

# # # # # # # # # # # # # # # # # # # # # # #     # Check if the frame is empty
# # # # # # # # # # # # # # # # # # # # # # #     if frame is None:
# # # # # # # # # # # # # # # # # # # # # # #         emit('error', 'Empty frame')
# # # # # # # # # # # # # # # # # # # # # # #         return

# # # # # # # # # # # # # # # # # # # # # # #     # Detect faces in the frame
# # # # # # # # # # # # # # # # # # # # # # #     faces_count = detect_faces(frame)

# # # # # # # # # # # # # # # # # # # # # # #     # Send the count of detected faces back to the client in JSON format
# # # # # # # # # # # # # # # # # # # # # # #     result = {'faces_count': faces_count}
# # # # # # # # # # # # # # # # # # # # # # #     emit('result', result)

# # # # # # # # # # # # # # # # # # # # # # # # Render HTML page with SocketIO client
# # # # # # # # # # # # # # # # # # # # # # # @app.route('/')
# # # # # # # # # # # # # # # # # # # # # # # def index():
# # # # # # # # # # # # # # # # # # # # # # #     return render_template('index.html')

# # # # # # # # # # # # # # # # # # # # # # # if __name__ == '__main__':
# # # # # # # # # # # # # # # # # # # # # # #     socketio.run(app, debug=True,host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)

# # # # # # # # # # # # # # # # # # # # # # # from flask import Flask
# # # # # # # # # # # # # # # # # # # # # # # from flask_socketio import SocketIO
# # # # # # # # # # # # # # # # # # # # # # # import os
# # # # # # # # # # # # # # # # # # # # # # # import subprocess
# # # # # # # # # # # # # # # # # # # # # # # import datetime

# # # # # # # # # # # # # # # # # # # # # # # app = Flask(__name__)
# # # # # # # # # # # # # # # # # # # # # # # socketio = SocketIO(app, cors_allowed_origins="*", engineio_logger=True, logger=True)

# # # # # # # # # # # # # # # # # # # # # # # # Directory setup for saving chunks
# # # # # # # # # # # # # # # # # # # # # # # if not os.path.exists('video_chunks'):
# # # # # # # # # # # # # # # # # # # # # # #     os.makedirs('video_chunks')

# # # # # # # # # # # # # # # # # # # # # # # def merge_videos(directory):
# # # # # # # # # # # # # # # # # # # # # # #     # Generate a list of files sorted by creation time
# # # # # # # # # # # # # # # # # # # # # # #     files = sorted(
# # # # # # # # # # # # # # # # # # # # # # #         [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.mp4')],
# # # # # # # # # # # # # # # # # # # # # # #         key=os.path.getmtime
# # # # # # # # # # # # # # # # # # # # # # #     )
    
# # # # # # # # # # # # # # # # # # # # # # #     # Create a temporary file list for FFmpeg processing
# # # # # # # # # # # # # # # # # # # # # # #     list_path = os.path.join(directory, 'filelist.txt')
# # # # # # # # # # # # # # # # # # # # # # #     with open(list_path, 'w') as filelist:
# # # # # # # # # # # # # # # # # # # # # # #         for file in files:
# # # # # # # # # # # # # # # # # # # # # # #             filelist.write(f"file '{file}'\n")

# # # # # # # # # # # # # # # # # # # # # # #     # Output filename with timestamp to prevent overwriting
# # # # # # # # # # # # # # # # # # # # # # #     timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
# # # # # # # # # # # # # # # # # # # # # # #     output_filename = f'output_video_{timestamp}.mp4'
    
# # # # # # # # # # # # # # # # # # # # # # #     # FFmpeg command to concatenate all video files into MP4
# # # # # # # # # # # # # # # # # # # # # # #     subprocess.run([
# # # # # # # # # # # # # # # # # # # # # # #         'ffmpeg', '-f', 'concat', '-safe', '0', '-i', list_path,
# # # # # # # # # # # # # # # # # # # # # # #         '-c:v', 'libx264', '-crf', '23', '-preset', 'fast', os.path.join(directory, output_filename)
# # # # # # # # # # # # # # # # # # # # # # #     ], check=True)
    
# # # # # # # # # # # # # # # # # # # # # # #     print(f'Video merged and saved as {output_filename}')

# # # # # # # # # # # # # # # # # # # # # # # @socketio.on('connect')
# # # # # # # # # # # # # # # # # # # # # # # def handle_connect():
# # # # # # # # # # # # # # # # # # # # # # #     print('Client connected')

# # # # # # # # # # # # # # # # # # # # # # # @socketio.on('video_chunk')
# # # # # # # # # # # # # # # # # # # # # # # def handle_video_chunk(data):
# # # # # # # # # # # # # # # # # # # # # # #     print('Received video chunk')
# # # # # # # # # # # # # # # # # # # # # # #     # Each chunk gets a timestamped filename to maintain order without relying on accurate client-side chunk numbering
# # # # # # # # # # # # # # # # # # # # # # #     timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
# # # # # # # # # # # # # # # # # # # # # # #     chunk_path = f'video_chunks/chunk_{timestamp}.mp4'
# # # # # # # # # # # # # # # # # # # # # # #     with open(chunk_path, 'wb') as f:
# # # # # # # # # # # # # # # # # # # # # # #         f.write(data)
# # # # # # # # # # # # # # # # # # # # # # #     print(f'Chunk saved: {chunk_path}')

# # # # # # # # # # # # # # # # # # # # # # # @socketio.on('end_video')
# # # # # # # # # # # # # # # # # # # # # # # def handle_end_video():
# # # # # # # # # # # # # # # # # # # # # # #     print('End of video signal received')
# # # # # # # # # # # # # # # # # # # # # # #     merge_videos('video_chunks')

# # # # # # # # # # # # # # # # # # # # # # # if __name__ == '__main__':
# # # # # # # # # # # # # # # # # # # # # # #     socketio.run(app, debug=True, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
# # # # # # # # # # # # # # # # # # # # # import cv2
# # # # # # # # # # # # # # # # # # # # # import numpy as np
# # # # # # # # # # # # # # # # # # # # # from flask import Flask, render_template
# # # # # # # # # # # # # # # # # # # # # from flask_socketio import SocketIO, emit
# # # # # # # # # # # # # # # # # # # # # import base64

# # # # # # # # # # # # # # # # # # # # # app = Flask(__name__)
# # # # # # # # # # # # # # # # # # # # # socketio = SocketIO(app, cors_allowed_origins="*")

# # # # # # # # # # # # # # # # # # # # # # Function to detect faces in a frame
# # # # # # # # # # # # # # # # # # # # # def detect_faces(frame):
# # # # # # # # # # # # # # # # # # # # #     # Convert frame to grayscale
# # # # # # # # # # # # # # # # # # # # #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# # # # # # # # # # # # # # # # # # # # #     # Detect faces using Haar Cascade Classifier
# # # # # # # # # # # # # # # # # # # # #     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# # # # # # # # # # # # # # # # # # # # #     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# # # # # # # # # # # # # # # # # # # # #     return len(faces)

# # # # # # # # # # # # # # # # # # # # # # SocketIO event handler for client connection
# # # # # # # # # # # # # # # # # # # # # @socketio.on('connect')
# # # # # # # # # # # # # # # # # # # # # def handle_connect():
# # # # # # # # # # # # # # # # # # # # #     print('Client connected')

# # # # # # # # # # # # # # # # # # # # # # SocketIO event handler for receiving video frames
# # # # # # # # # # # # # # # # # # # # # @socketio.on('video_data')
# # # # # # # # # # # # # # # # # # # # # def handle_video_frame(frame_data):
# # # # # # # # # # # # # # # # # # # # #     print('Video received')

# # # # # # # # # # # # # # # # # # # # #     # Decode base64 encoded frame
# # # # # # # # # # # # # # # # # # # # #     frame_bytes = base64.b64decode(frame_data)
# # # # # # # # # # # # # # # # # # # # #     frame_arr = np.frombuffer(frame_bytes, dtype=np.uint8)
# # # # # # # # # # # # # # # # # # # # #     frame = cv2.imdecode(frame_arr, flags=cv2.IMREAD_COLOR)

# # # # # # # # # # # # # # # # # # # # #     # Check if the frame is empty
# # # # # # # # # # # # # # # # # # # # #     if frame is None:
# # # # # # # # # # # # # # # # # # # # #         emit('error', 'Empty frame')
# # # # # # # # # # # # # # # # # # # # #         return

# # # # # # # # # # # # # # # # # # # # #     # Write frame to memory as MP4 format
# # # # # # # # # # # # # # # # # # # # #     mp4_buf = cv2.imencode('.mp4', frame)[1].tostring()

# # # # # # # # # # # # # # # # # # # # #     # Decode MP4 buffer back to base64 string
# # # # # # # # # # # # # # # # # # # # #     mp4_base64 = base64.b64encode(mp4_buf).decode('utf-8')

# # # # # # # # # # # # # # # # # # # # #     # Detect faces in the frame
# # # # # # # # # # # # # # # # # # # # #     faces_count = detect_faces(frame)

# # # # # # # # # # # # # # # # # # # # #     # Send the count of detected faces and the MP4 video data back to the client
# # # # # # # # # # # # # # # # # # # # #     result = {'faces_count': faces_count, 'video_data': mp4_base64}
# # # # # # # # # # # # # # # # # # # # #     emit('result', result)

# # # # # # # # # # # # # # # # # # # # # # Render HTML page with SocketIO client
# # # # # # # # # # # # # # # # # # # # # @app.route('/')
# # # # # # # # # # # # # # # # # # # # # def index():
# # # # # # # # # # # # # # # # # # # # #     return render_template('index.html')

# # # # # # # # # # # # # # # # # # # # # if __name__ == '__main__':
# # # # # # # # # # # # # # # # # # # # #     socketio.run(app, debug=True, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
# # # # # # # # # # # # # # # # # # # # # from flask import Flask, render_template
# # # # # # # # # # # # # # # # # # # # # from flask_socketio import SocketIO, emit
# # # # # # # # # # # # # # # # # # # # # import cv2
# # # # # # # # # # # # # # # # # # # # # import numpy as np
# # # # # # # # # # # # # # # # # # # # # from imageio import get_reader
# # # # # # # # # # # # # # # # # # # # # from io import BytesIO
# # # # # # # # # # # # # # # # # # # # # import threading

# # # # # # # # # # # # # # # # # # # # # app = Flask(__name__)
# # # # # # # # # # # # # # # # # # # # # socketio = SocketIO(app, cors_allowed_origins="*")

# # # # # # # # # # # # # # # # # # # # # # Thread-safe buffer to store video stream data
# # # # # # # # # # # # # # # # # # # # # class VideoStreamBuffer:
# # # # # # # # # # # # # # # # # # # # #     def __init__(self):
# # # # # # # # # # # # # # # # # # # # #         self.buffer = BytesIO()
# # # # # # # # # # # # # # # # # # # # #         self.lock = threading.Lock()

# # # # # # # # # # # # # # # # # # # # #     def write(self, data):
# # # # # # # # # # # # # # # # # # # # #         with self.lock:
# # # # # # # # # # # # # # # # # # # # #             self.buffer.write(data)

# # # # # # # # # # # # # # # # # # # # #     def get_frames(self):
# # # # # # # # # # # # # # # # # # # # #         self.buffer.seek(0)
# # # # # # # # # # # # # # # # # # # # #         # Using imageio to read frames from binary stream
# # # # # # # # # # # # # # # # # # # # #         try:
# # # # # # # # # # # # # # # # # # # # #             for frame in get_reader(self.buffer, 'ffmpeg'):
# # # # # # # # # # # # # # # # # # # # #                 yield cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # Convert frame to BGR format used by OpenCV
# # # # # # # # # # # # # # # # # # # # #         finally:
# # # # # # # # # # # # # # # # # # # # #             self.reset()

# # # # # # # # # # # # # # # # # # # # #     def reset(self):
# # # # # # # # # # # # # # # # # # # # #         with self.lock:
# # # # # # # # # # # # # # # # # # # # #             self.buffer = BytesIO()

# # # # # # # # # # # # # # # # # # # # # video_stream_buffer = VideoStreamBuffer()

# # # # # # # # # # # # # # # # # # # # # @socketio.on('connect')
# # # # # # # # # # # # # # # # # # # # # def handle_connect():
# # # # # # # # # # # # # # # # # # # # #     print('Client connected')

# # # # # # # # # # # # # # # # # # # # # @socketio.on('video_chunk')
# # # # # # # # # # # # # # # # # # # # # def handle_video_chunk(data):
# # # # # # # # # # # # # # # # # # # # #     print('Received video chunk')
# # # # # # # # # # # # # # # # # # # # #     video_stream_buffer.write(data)

# # # # # # # # # # # # # # # # # # # # # @socketio.on('disconnect')
# # # # # # # # # # # # # # # # # # # # # def handle_disconnect():
# # # # # # # # # # # # # # # # # # # # #     print('Client disconnected')
# # # # # # # # # # # # # # # # # # # # #     process_video_stream() # Process video when the client disconnects

# # # # # # # # # # # # # # # # # # # # # def process_video_stream():
# # # # # # # # # # # # # # # # # # # # #     for frame in video_stream_buffer.get_frames():
# # # # # # # # # # # # # # # # # # # # #         processed_frame, faces_count = process_frame(frame)
# # # # # # # # # # # # # # # # # # # # #         # Process frame with OpenCV here
# # # # # # # # # # # # # # # # # # # # #         # Example: Detect edges in the frame
# # # # # # # # # # # # # # # # # # # # #         edges = cv2.Canny(processed_frame, 100, 200)
# # # # # # # # # # # # # # # # # # # # #         cv2.imshow('Edges', edges)
# # # # # # # # # # # # # # # # # # # # #         if cv2.waitKey(1) & 0xFF == ord('q'):
# # # # # # # # # # # # # # # # # # # # #             break
# # # # # # # # # # # # # # # # # # # # #     cv2.destroyAllWindows()

# # # # # # # # # # # # # # # # # # # # # def process_frame(frame):
# # # # # # # # # # # # # # # # # # # # #     # Convert to grayscale
# # # # # # # # # # # # # # # # # # # # #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
# # # # # # # # # # # # # # # # # # # # #     # Detect faces using Haar Cascade Classifier
# # # # # # # # # # # # # # # # # # # # #     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# # # # # # # # # # # # # # # # # # # # #     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
# # # # # # # # # # # # # # # # # # # # #     # Draw rectangles around faces
# # # # # # # # # # # # # # # # # # # # #     for (x, y, w, h) in faces:
# # # # # # # # # # # # # # # # # # # # #         cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
# # # # # # # # # # # # # # # # # # # # #     return frame, len(faces)

# # # # # # # # # # # # # # # # # # # # # if __name__ == '__main__':
# # # # # # # # # # # # # # # # # # # # #     threading.Thread(target=process_video_stream).start()
# # # # # # # # # # # # # # # # # # # # #     socketio.run(app, debug=True, host='0.0.0.0', port=5000)


# # # # # # # # # # # # # # # # # # # # # from flask import Flask
# # # # # # # # # # # # # # # # # # # # # from flask_socketio import SocketIO
# # # # # # # # # # # # # # # # # # # # # import cv2
# # # # # # # # # # # # # # # # # # # # # import numpy as np
# # # # # # # # # # # # # # # # # # # # # import subprocess
# # # # # # # # # # # # # # # # # # # # # import threading

# # # # # # # # # # # # # # # # # # # # # app = Flask(__name__)
# # # # # # # # # # # # # # # # # # # # # socketio = SocketIO(app, cors_allowed_origins="*")

# # # # # # # # # # # # # # # # # # # # # # Initialize OpenCV's Haar cascade for face detection
# # # # # # # # # # # # # # # # # # # # # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # # # # # # # # # # # # # # # # # # # # def start_ffmpeg_process():
# # # # # # # # # # # # # # # # # # # # #     command = [
# # # # # # # # # # # # # # # # # # # # #         'ffmpeg',
# # # # # # # # # # # # # # # # # # # # #         '-i', '-',
# # # # # # # # # # # # # # # # # # # # #         '-pix_fmt', 'bgr24',
# # # # # # # # # # # # # # # # # # # # #         '-vcodec', 'rawvideo',
# # # # # # # # # # # # # # # # # # # # #         '-an', '-sn',
# # # # # # # # # # # # # # # # # # # # #         '-f', 'image2pipe', '-'
# # # # # # # # # # # # # # # # # # # # #     ]
# # # # # # # # # # # # # # # # # # # # #     return subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# # # # # # # # # # # # # # # # # # # # # ffmpeg_process = start_ffmpeg_process()

# # # # # # # # # # # # # # # # # # # # # def video_processing_thread():
# # # # # # # # # # # # # # # # # # # # #     while True:
# # # # # # # # # # # # # # # # # # # # #         # Assuming 640x480 frame size
# # # # # # # # # # # # # # # # # # # # #         raw_frame = ffmpeg_process.stdout.read(640 * 480 * 3)
# # # # # # # # # # # # # # # # # # # # #         if not raw_frame:
# # # # # # # # # # # # # # # # # # # # #             break
# # # # # # # # # # # # # # # # # # # # #         frame = np.frombuffer(raw_frame, np.uint8).reshape((480, 640, 3))
# # # # # # # # # # # # # # # # # # # # #         face_count = count_faces(frame)
# # # # # # # # # # # # # # # # # # # # #         socketio.emit('result', {'count': face_count})
# # # # # # # # # # # # # # # # # # # # #         ffmpeg_process.stdout.flush()

# # # # # # # # # # # # # # # # # # # # # def count_faces(frame):
# # # # # # # # # # # # # # # # # # # # #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# # # # # # # # # # # # # # # # # # # # #     faces = face_cascade.detectMultiScale(gray, 1.1, 4)
# # # # # # # # # # # # # # # # # # # # #     return len(faces)

# # # # # # # # # # # # # # # # # # # # # @socketio.on('connect')
# # # # # # # # # # # # # # # # # # # # # def handle_connect():
# # # # # # # # # # # # # # # # # # # # #     print('Client connected')

# # # # # # # # # # # # # # # # # # # # # @socketio.on('video_chunk')
# # # # # # # # # # # # # # # # # # # # # def handle_video_chunk(data):
# # # # # # # # # # # # # # # # # # # # #     global ffmpeg_process
# # # # # # # # # # # # # # # # # # # # #     try:
        
# # # # # # # # # # # # # # # # # # # # #         ffmpeg_process.stdin.write(data)
# # # # # # # # # # # # # # # # # # # # #     except BrokenPipeError:
# # # # # # # # # # # # # # # # # # # # #         print("Restarting FFmpeg...")
        
# # # # # # # # # # # # # # # # # # # # #         ffmpeg_process = start_ffmpeg_process()
# # # # # # # # # # # # # # # # # # # # #         ffmpeg_process.stdin.write(data)

# # # # # # # # # # # # # # # # # # # # # @socketio.on('disconnect')
# # # # # # # # # # # # # # # # # # # # # def handle_disconnect():
# # # # # # # # # # # # # # # # # # # # #     print('Client disconnected')
# # # # # # # # # # # # # # # # # # # # #     ffmpeg_process.stdin.close()

# # # # # # # # # # # # # # # # # # # # # if __name__ == '__main__':
# # # # # # # # # # # # # # # # # # # # #     threading.Thread(target=video_processing_thread, daemon=True).start()
# # # # # # # # # # # # # # # # # # # # #     socketio.run(app, debug=True, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
# # # # # # # # # # # # # # # # # # # # import cv2
# # # # # # # # # # # # # # # # # # # # import numpy as np
# # # # # # # # # # # # # # # # # # # # from flask import Flask, render_template
# # # # # # # # # # # # # # # # # # # # from flask_socketio import SocketIO, emit
# # # # # # # # # # # # # # # # # # # # import base64

# # # # # # # # # # # # # # # # # # # # app = Flask(__name__)
# # # # # # # # # # # # # # # # # # # # socketio = SocketIO(app, cors_allowed_origins="*")

# # # # # # # # # # # # # # # # # # # # # Function to detect faces in a frame
# # # # # # # # # # # # # # # # # # # # def detect_faces(frame):
# # # # # # # # # # # # # # # # # # # #     # Convert frame to grayscale
# # # # # # # # # # # # # # # # # # # #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# # # # # # # # # # # # # # # # # # # #     # Detect faces using Haar Cascade Classifier
# # # # # # # # # # # # # # # # # # # #     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# # # # # # # # # # # # # # # # # # # #     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# # # # # # # # # # # # # # # # # # # #     return len(faces)

# # # # # # # # # # # # # # # # # # # # # SocketIO event handler for client connection
# # # # # # # # # # # # # # # # # # # # @socketio.on('connect')
# # # # # # # # # # # # # # # # # # # # def handle_connect():
# # # # # # # # # # # # # # # # # # # #     print('Client connected')

# # # # # # # # # # # # # # # # # # # # # SocketIO event handler for receiving video frames
# # # # # # # # # # # # # # # # # # # # @socketio.on('video_data')
# # # # # # # # # # # # # # # # # # # # def handle_video_frame(frame_data):
# # # # # # # # # # # # # # # # # # # #     print('Video received')

# # # # # # # # # # # # # # # # # # # #     # Decode base64 encoded frame
# # # # # # # # # # # # # # # # # # # #     try:
# # # # # # # # # # # # # # # # # # # #         frame_bytes = base64.b64decode(frame_data)
# # # # # # # # # # # # # # # # # # # #         frame_arr = np.frombuffer(frame_bytes, dtype=np.uint8)
# # # # # # # # # # # # # # # # # # # #         frame = cv2.imdecode(frame_arr, flags=cv2.IMREAD_COLOR)
# # # # # # # # # # # # # # # # # # # #     except Exception as e:
# # # # # # # # # # # # # # # # # # # #         print(f"Error decoding frame: {e}")
# # # # # # # # # # # # # # # # # # # #         emit('error', 'Error decoding frame')
# # # # # # # # # # # # # # # # # # # #         return

# # # # # # # # # # # # # # # # # # # #     # Check if the frame is empty
# # # # # # # # # # # # # # # # # # # #     if frame is None or frame.size == 0:
# # # # # # # # # # # # # # # # # # # #         print('Empty frame')
# # # # # # # # # # # # # # # # # # # #         emit('error', 'Empty frame')
# # # # # # # # # # # # # # # # # # # #         return

# # # # # # # # # # # # # # # # # # # #     # Detect faces in the frame
# # # # # # # # # # # # # # # # # # # #     faces_count = detect_faces(frame)
# # # # # # # # # # # # # # # # # # # #     print(f"Detected faces: {faces_count}")

# # # # # # # # # # # # # # # # # # # #     # Encode frame to MP4 format
# # # # # # # # # # # # # # # # # # # #     try:
# # # # # # # # # # # # # # # # # # # #         _, mp4_buf = cv2.imencode('.mp4', frame)
# # # # # # # # # # # # # # # # # # # #         mp4_base64 = base64.b64encode(mp4_buf).decode('utf-8')
# # # # # # # # # # # # # # # # # # # #     except Exception as e:
# # # # # # # # # # # # # # # # # # # #         print(f"Error encoding frame to MP4: {e}")
# # # # # # # # # # # # # # # # # # # #         emit('error', 'Error encoding frame to MP4')
# # # # # # # # # # # # # # # # # # # #         return

# # # # # # # # # # # # # # # # # # # #     # Send the count of detected faces and the MP4 video data back to the client
# # # # # # # # # # # # # # # # # # # #     result = {'faces_count': faces_count, 'video_data': mp4_base64}
# # # # # # # # # # # # # # # # # # # #     emit('result', result)

# # # # # # # # # # # # # # # # # # # # # Render HTML page with SocketIO client
# # # # # # # # # # # # # # # # # # # # @app.route('/')
# # # # # # # # # # # # # # # # # # # # def index():
# # # # # # # # # # # # # # # # # # # #     return render_template('index.html')

# # # # # # # # # # # # # # # # # # # # if __name__ == '__main__':
# # # # # # # # # # # # # # # # # # # #     socketio.run(app, debug=True, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
# # # # # # # # # # # # # # # # # # # import cv2
# # # # # # # # # # # # # # # # # # # import numpy as np
# # # # # # # # # # # # # # # # # # # from flask import Flask, render_template
# # # # # # # # # # # # # # # # # # # from flask_socketio import SocketIO, emit
# # # # # # # # # # # # # # # # # # # import base64

# # # # # # # # # # # # # # # # # # # app = Flask(__name__)
# # # # # # # # # # # # # # # # # # # socketio = SocketIO(app, cors_allowed_origins="*")

# # # # # # # # # # # # # # # # # # # # Function to detect faces in a frame
# # # # # # # # # # # # # # # # # # # def detect_faces(frame):
# # # # # # # # # # # # # # # # # # #     # Convert frame to grayscale
# # # # # # # # # # # # # # # # # # #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# # # # # # # # # # # # # # # # # # #     # Detect faces using Haar Cascade Classifier
# # # # # # # # # # # # # # # # # # #     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# # # # # # # # # # # # # # # # # # #     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# # # # # # # # # # # # # # # # # # #     return len(faces)

# # # # # # # # # # # # # # # # # # # # SocketIO event handler for client connection
# # # # # # # # # # # # # # # # # # # @socketio.on('connect')
# # # # # # # # # # # # # # # # # # # def handle_connect():
# # # # # # # # # # # # # # # # # # #     print('Client connected')

# # # # # # # # # # # # # # # # # # # # SocketIO event handler for receiving video frames
# # # # # # # # # # # # # # # # # # # @socketio.on('video_data')
# # # # # # # # # # # # # # # # # # # def handle_video_frame(frame_data):
# # # # # # # # # # # # # # # # # # #     print('Video received')

# # # # # # # # # # # # # # # # # # #     # Decode base64 encoded frame
# # # # # # # # # # # # # # # # # # #     frame_bytes = base64.b64decode(frame_data)
# # # # # # # # # # # # # # # # # # #     frame_arr = np.frombuffer(frame_bytes, dtype=np.uint8)
# # # # # # # # # # # # # # # # # # #     frame = cv2.imdecode(frame_arr, flags=cv2.IMREAD_COLOR)

# # # # # # # # # # # # # # # # # # #     # Check if the frame is empty
# # # # # # # # # # # # # # # # # # #     if frame is None:
# # # # # # # # # # # # # # # # # # #         emit('error', 'Empty frame')
# # # # # # # # # # # # # # # # # # #         return

# # # # # # # # # # # # # # # # # # #     # Write frame to memory as MP4 format
# # # # # # # # # # # # # # # # # # #     mp4_buf = cv2.imencode('.mp4', frame)[1].tostring()

# # # # # # # # # # # # # # # # # # #     # Decode MP4 buffer back to base64 string
# # # # # # # # # # # # # # # # # # #     mp4_base64 = base64.b64encode(mp4_buf).decode('utf-8')

# # # # # # # # # # # # # # # # # # #     # Detect faces in the frame
# # # # # # # # # # # # # # # # # # #     faces_count = detect_faces(frame)

# # # # # # # # # # # # # # # # # # #     # Send the count of detected faces and the MP4 video data back to the client
# # # # # # # # # # # # # # # # # # #     result = {'faces_count': faces_count, 'video_data': mp4_base64}
# # # # # # # # # # # # # # # # # # #     emit('result', result)

# # # # # # # # # # # # # # # # # # # # Render HTML page with SocketIO client
# # # # # # # # # # # # # # # # # # # @app.route('/')
# # # # # # # # # # # # # # # # # # # def index():
# # # # # # # # # # # # # # # # # # #     return render_template('index.html')

# # # # # # # # # # # # # # # # # # # if __name__ == '__main__':
# # # # # # # # # # # # # # # # # # #     socketio.run(app, debug=True, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
# # # # # # # # # # # # # # # # # # # import cv2
# # # # # # # # # # # # # # # # # # # import numpy as np
# # # # # # # # # # # # # # # # # # # from flask import Flask, render_template
# # # # # # # # # # # # # # # # # # # from flask_socketio import SocketIO, emit
# # # # # # # # # # # # # # # # # # # import face_recognition
# # # # # # # # # # # # # # # # # # # import tempfile
# # # # # # # # # # # # # # # # # # # import os

# # # # # # # # # # # # # # # # # # # app = Flask(__name__)
# # # # # # # # # # # # # # # # # # # socketio = SocketIO(app, cors_allowed_origins="*")

# # # # # # # # # # # # # # # # # # # KNOWN_IMAGE_PATH = "C:/Users/OMR-09/Pictures/img2.jpg"

# # # # # # # # # # # # # # # # # # # # Load the known image
# # # # # # # # # # # # # # # # # # # known_image = face_recognition.load_image_file(KNOWN_IMAGE_PATH)
# # # # # # # # # # # # # # # # # # # known_encoding = face_recognition.face_encodings(known_image)[0]

# # # # # # # # # # # # # # # # # # # # Function to detect faces and neck bending in a frame
# # # # # # # # # # # # # # # # # # # def detect_person_match(frame):
# # # # # # # # # # # # # # # # # # #     # Convert frame to RGB (face_recognition expects RGB)
# # # # # # # # # # # # # # # # # # #     rgb_frame = frame[:, :, ::-1]

# # # # # # # # # # # # # # # # # # #     # Find face locations in the frame
# # # # # # # # # # # # # # # # # # #     face_locations = face_recognition.face_locations(rgb_frame)

# # # # # # # # # # # # # # # # # # #     # Check for more than one face detected
# # # # # # # # # # # # # # # # # # #     if len(face_locations) > 1:
# # # # # # # # # # # # # # # # # # #         return "More than One Face Detected"

# # # # # # # # # # # # # # # # # # #     # Check for face match and neck bending
# # # # # # # # # # # # # # # # # # #     if len(face_locations) == 1:
# # # # # # # # # # # # # # # # # # #         face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
# # # # # # # # # # # # # # # # # # #         match = face_recognition.compare_faces([known_encoding], face_encoding)

# # # # # # # # # # # # # # # # # # #         if match[0]:
# # # # # # # # # # # # # # # # # # #             face_landmarks = face_recognition.face_landmarks(rgb_frame, face_locations)
# # # # # # # # # # # # # # # # # # #             if face_landmarks:
# # # # # # # # # # # # # # # # # # #                 neck_angle = detect_neck_bending(face_landmarks[0])
# # # # # # # # # # # # # # # # # # #                 if neck_angle > 130 or neck_angle < 125:  # Adjust threshold as needed
# # # # # # # # # # # # # # # # # # #                     return "Neck Bending Detected"
# # # # # # # # # # # # # # # # # # #                 return "Match"

# # # # # # # # # # # # # # # # # # #     return "Not Match"

# # # # # # # # # # # # # # # # # # # # Function to detect neck bending based on facial landmarks
# # # # # # # # # # # # # # # # # # # def detect_neck_bending(face_landmarks):
# # # # # # # # # # # # # # # # # # #     # Extract relevant landmarks for neck estimation
# # # # # # # # # # # # # # # # # # #     top_nose = face_landmarks['nose_bridge'][0]
# # # # # # # # # # # # # # # # # # #     bottom_nose = face_landmarks['nose_tip'][0]
# # # # # # # # # # # # # # # # # # #     top_chin = face_landmarks['chin'][8]
# # # # # # # # # # # # # # # # # # #     bottom_chin = face_landmarks['chin'][0]

# # # # # # # # # # # # # # # # # # #     # Calculate vectors for neck and face
# # # # # # # # # # # # # # # # # # #     neck_vector = np.array(bottom_chin) - np.array(top_chin)
# # # # # # # # # # # # # # # # # # #     face_vector = np.array(bottom_nose) - np.array(top_nose)

# # # # # # # # # # # # # # # # # # #     # Calculate angle between neck and face vectors
# # # # # # # # # # # # # # # # # # #     angle = np.degrees(np.arccos(np.dot(neck_vector, face_vector) /
# # # # # # # # # # # # # # # # # # #                                 (np.linalg.norm(neck_vector) * np.linalg.norm(face_vector))))
# # # # # # # # # # # # # # # # # # #     return angle

# # # # # # # # # # # # # # # # # # # # SocketIO event handler for client connection
# # # # # # # # # # # # # # # # # # # @socketio.on('connect')
# # # # # # # # # # # # # # # # # # # def handle_connect():
# # # # # # # # # # # # # # # # # # #     print('Client connected')

# # # # # # # # # # # # # # # # # # # # SocketIO event handler for receiving video frames
# # # # # # # # # # # # # # # # # # # @socketio.on('video_data')
# # # # # # # # # # # # # # # # # # # def handle_video_frame(frame_data):
# # # # # # # # # # # # # # # # # # #     print('Video data received')

# # # # # # # # # # # # # # # # # # #     # Convert Uint8Array to numpy array
# # # # # # # # # # # # # # # # # # #     frame_arr = np.frombuffer(frame_data, dtype=np.uint8)

# # # # # # # # # # # # # # # # # # #     # Write frame data to a temporary file
# # # # # # # # # # # # # # # # # # #     with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as tmpfile:
# # # # # # # # # # # # # # # # # # #         tmpfile.write(frame_arr)
# # # # # # # # # # # # # # # # # # #         tmpfile_path = tmpfile.name

# # # # # # # # # # # # # # # # # # #     # Debug: Check if the temporary file exists and its size
# # # # # # # # # # # # # # # # # # #     if not os.path.exists(tmpfile_path):
# # # # # # # # # # # # # # # # # # #         print('Temporary file does not exist')
# # # # # # # # # # # # # # # # # # #         emit('error', 'Temporary file does not exist')
# # # # # # # # # # # # # # # # # # #         return

# # # # # # # # # # # # # # # # # # #     print(f'Temporary file created at: {tmpfile_path}')
# # # # # # # # # # # # # # # # # # #     print(f'Temporary file size: {os.path.getsize(tmpfile_path)} bytes')

# # # # # # # # # # # # # # # # # # #     # Read the video file using OpenCV
# # # # # # # # # # # # # # # # # # #     cap = cv2.VideoCapture(tmpfile_path)

# # # # # # # # # # # # # # # # # # #     # Check if the video capture opened successfully
# # # # # # # # # # # # # # # # # # #     if not cap.isOpened():
# # # # # # # # # # # # # # # # # # #         print('Error opening video file')
# # # # # # # # # # # # # # # # # # #         emit('error', 'Error opening video file')
# # # # # # # # # # # # # # # # # # #         os.remove(tmpfile_path)
# # # # # # # # # # # # # # # # # # #         return

# # # # # # # # # # # # # # # # # # #     result = "Not Match"
# # # # # # # # # # # # # # # # # # #     while True:
# # # # # # # # # # # # # # # # # # #         ret, frame = cap.read()
# # # # # # # # # # # # # # # # # # #         if not ret:
# # # # # # # # # # # # # # # # # # #             break

# # # # # # # # # # # # # # # # # # #         # Detect person match in each frame
# # # # # # # # # # # # # # # # # # #         result = detect_person_match(frame)
# # # # # # # # # # # # # # # # # # #         if result != "Not Match":
# # # # # # # # # # # # # # # # # # #             break

# # # # # # # # # # # # # # # # # # #     cap.release()
# # # # # # # # # # # # # # # # # # #     os.remove(tmpfile_path)

# # # # # # # # # # # # # # # # # # #     print(f"Result: {result}")

# # # # # # # # # # # # # # # # # # #     # Send the result back to the client
# # # # # # # # # # # # # # # # # # #     emit('result', {'result': result})

# # # # # # # # # # # # # # # # # # # # Render HTML page with SocketIO client
# # # # # # # # # # # # # # # # # # # @app.route('/')
# # # # # # # # # # # # # # # # # # # def index():
# # # # # # # # # # # # # # # # # # #     return render_template('index.html')

# # # # # # # # # # # # # # # # # # # if __name__ == '__main__':
# # # # # # # # # # # # # # # # # # #     socketio.run(app, debug=True, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
# # # # # # # # # # # # # # # # # # import cv2
# # # # # # # # # # # # # # # # # # import numpy as np
# # # # # # # # # # # # # # # # # # from flask import Flask, render_template
# # # # # # # # # # # # # # # # # # from flask_socketio import SocketIO, emit
# # # # # # # # # # # # # # # # # # import face_recognition
# # # # # # # # # # # # # # # # # # import tempfile
# # # # # # # # # # # # # # # # # # import os

# # # # # # # # # # # # # # # # # # app = Flask(__name__)
# # # # # # # # # # # # # # # # # # socketio = SocketIO(app, cors_allowed_origins="*")

# # # # # # # # # # # # # # # # # # KNOWN_IMAGE_PATH = "C:/Users/OMR-09/Pictures/img2.jpg"

# # # # # # # # # # # # # # # # # # # Load the known image
# # # # # # # # # # # # # # # # # # known_image = face_recognition.load_image_file(KNOWN_IMAGE_PATH)
# # # # # # # # # # # # # # # # # # known_encoding = face_recognition.face_encodings(known_image)[0]

# # # # # # # # # # # # # # # # # # # Function to detect faces and neck bending in a frame
# # # # # # # # # # # # # # # # # # def detect_person_match(frame):
# # # # # # # # # # # # # # # # # #     # Convert frame to RGB (face_recognition expects RGB)
# # # # # # # # # # # # # # # # # #     rgb_frame = frame[:, :, ::-1]

# # # # # # # # # # # # # # # # # #     # Find face locations in the frame
# # # # # # # # # # # # # # # # # #     face_locations = face_recognition.face_locations(rgb_frame)

# # # # # # # # # # # # # # # # # #     # Check for more than one face detected
# # # # # # # # # # # # # # # # # #     if len(face_locations) > 1:
# # # # # # # # # # # # # # # # # #         return "More than One Face Detected"

# # # # # # # # # # # # # # # # # #     # Check for face match and neck bending
# # # # # # # # # # # # # # # # # #     if len(face_locations) == 1:
# # # # # # # # # # # # # # # # # #         face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
# # # # # # # # # # # # # # # # # #         match = face_recognition.compare_faces([known_encoding], face_encoding)

# # # # # # # # # # # # # # # # # #         if match[0]:
# # # # # # # # # # # # # # # # # #             face_landmarks = face_recognition.face_landmarks(rgb_frame, face_locations)
# # # # # # # # # # # # # # # # # #             if face_landmarks:
# # # # # # # # # # # # # # # # # #                 neck_angle = detect_neck_bending(face_landmarks[0])
# # # # # # # # # # # # # # # # # #                 if neck_angle > 130 or neck_angle < 125:  # Adjust threshold as needed
# # # # # # # # # # # # # # # # # #                     return "Neck Bending Detected"
# # # # # # # # # # # # # # # # # #                 return "Match"

# # # # # # # # # # # # # # # # # #     return "Not Match"

# # # # # # # # # # # # # # # # # # # Function to detect neck bending based on facial landmarks
# # # # # # # # # # # # # # # # # # def detect_neck_bending(face_landmarks):
# # # # # # # # # # # # # # # # # #     # Extract relevant landmarks for neck estimation
# # # # # # # # # # # # # # # # # #     top_nose = face_landmarks['nose_bridge'][0]
# # # # # # # # # # # # # # # # # #     bottom_nose = face_landmarks['nose_tip'][0]
# # # # # # # # # # # # # # # # # #     top_chin = face_landmarks['chin'][8]
# # # # # # # # # # # # # # # # # #     bottom_chin = face_landmarks['chin'][0]

# # # # # # # # # # # # # # # # # #     # Calculate vectors for neck and face
# # # # # # # # # # # # # # # # # #     neck_vector = np.array(bottom_chin) - np.array(top_chin)
# # # # # # # # # # # # # # # # # #     face_vector = np.array(bottom_nose) - np.array(top_nose)

# # # # # # # # # # # # # # # # # #     # Calculate angle between neck and face vectors
# # # # # # # # # # # # # # # # # #     angle = np.degrees(np.arccos(np.dot(neck_vector, face_vector) /
# # # # # # # # # # # # # # # # # #                                 (np.linalg.norm(neck_vector) * np.linalg.norm(face_vector))))
# # # # # # # # # # # # # # # # # #     return angle

# # # # # # # # # # # # # # # # # # # SocketIO event handler for client connection
# # # # # # # # # # # # # # # # # # @socketio.on('connect')
# # # # # # # # # # # # # # # # # # def handle_connect():
# # # # # # # # # # # # # # # # # #     print('Client connected')

# # # # # # # # # # # # # # # # # # # SocketIO event handler for receiving video frames
# # # # # # # # # # # # # # # # # # @socketio.on('video_data')
# # # # # # # # # # # # # # # # # # def handle_video_frame(frame_data):
# # # # # # # # # # # # # # # # # #     print('Video data received')

# # # # # # # # # # # # # # # # # #     # Convert Uint8Array to numpy array
# # # # # # # # # # # # # # # # # #     frame_arr = np.frombuffer(frame_data, dtype=np.uint8)

# # # # # # # # # # # # # # # # # #     # Write frame data to a temporary file
# # # # # # # # # # # # # # # # # #     with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as tmpfile:
# # # # # # # # # # # # # # # # # #         tmpfile.write(frame_arr)
# # # # # # # # # # # # # # # # # #         tmpfile_path = tmpfile.name

# # # # # # # # # # # # # # # # # #     # Debug: Check if the temporary file exists and its size
# # # # # # # # # # # # # # # # # #     if not os.path.exists(tmpfile_path):
# # # # # # # # # # # # # # # # # #         print('Temporary file does not exist')
# # # # # # # # # # # # # # # # # #         emit('error', 'Temporary file does not exist')
# # # # # # # # # # # # # # # # # #         return

# # # # # # # # # # # # # # # # # #     print(f'Temporary file created at: {tmpfile_path}')
# # # # # # # # # # # # # # # # # #     print(f'Temporary file size: {os.path.getsize(tmpfile_path)} bytes')

# # # # # # # # # # # # # # # # # #     # Read the video file using OpenCV
# # # # # # # # # # # # # # # # # #     cap = cv2.VideoCapture(tmpfile_path)

# # # # # # # # # # # # # # # # # #     # Check if the video capture opened successfully
# # # # # # # # # # # # # # # # # #     if not cap.isOpened():
# # # # # # # # # # # # # # # # # #         print('Error opening video file')
# # # # # # # # # # # # # # # # # #         emit('error', 'Error opening video file')
# # # # # # # # # # # # # # # # # #         os.remove(tmpfile_path)
# # # # # # # # # # # # # # # # # #         return

# # # # # # # # # # # # # # # # # #     result = "Not Match"
# # # # # # # # # # # # # # # # # #     while True:
# # # # # # # # # # # # # # # # # #         ret, frame = cap.read()
# # # # # # # # # # # # # # # # # #         if not ret:
# # # # # # # # # # # # # # # # # #             break

# # # # # # # # # # # # # # # # # #         # Detect person match in each frame
# # # # # # # # # # # # # # # # # #         result = detect_person_match(frame)
# # # # # # # # # # # # # # # # # #         if result != "Not Match":
# # # # # # # # # # # # # # # # # #             break

# # # # # # # # # # # # # # # # # #     cap.release()
# # # # # # # # # # # # # # # # # #     os.remove(tmpfile_path)

# # # # # # # # # # # # # # # # # #     print(f"Result: {result}")

# # # # # # # # # # # # # # # # # #     # Send the result back to the client
# # # # # # # # # # # # # # # # # #     emit('result', {'result': result})

# # # # # # # # # # # # # # # # # # # Render HTML page with SocketIO client
# # # # # # # # # # # # # # # # # # @app.route('/')
# # # # # # # # # # # # # # # # # # def index():
# # # # # # # # # # # # # # # # # #     return render_template('index.html')

# # # # # # # # # # # # # # # # # # if __name__ == '__main__':
# # # # # # # # # # # # # # # # # #     socketio.run(app, debug=True, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
# # # # # # # # # # # # # # # # import cv2
# # # # # # # # # # # # # # # # import numpy as np
# # # # # # # # # # # # # # # # from flask import Flask, render_template
# # # # # # # # # # # # # # # # from flask_socketio import SocketIO, emit
# # # # # # # # # # # # # # # # import face_recognition
# # # # # # # # # # # # # # # # import tempfile
# # # # # # # # # # # # # # # # import os
# # # # # # # # # # # # # # # # import ffmpeg

# # # # # # # # # # # # # # # # app = Flask(__name__)
# # # # # # # # # # # # # # # # socketio = SocketIO(app, cors_allowed_origins="*")

# # # # # # # # # # # # # # # # KNOWN_IMAGE_PATH = "C:/Users/OMR-09/Pictures/img2.jpg"

# # # # # # # # # # # # # # # # # Load the known image
# # # # # # # # # # # # # # # # known_image = face_recognition.load_image_file(KNOWN_IMAGE_PATH)
# # # # # # # # # # # # # # # # known_encoding = face_recognition.face_encodings(known_image)[0]

# # # # # # # # # # # # # # # # # Function to detect faces and neck bending in a frame
# # # # # # # # # # # # # # # # def detect_person_match(frame):
# # # # # # # # # # # # # # # #     # Convert frame to RGB (face_recognition expects RGB)
# # # # # # # # # # # # # # # #     rgb_frame = frame[:, :, ::-1]

# # # # # # # # # # # # # # # #     # Find face locations in the frame
# # # # # # # # # # # # # # # #     face_locations = face_recognition.face_locations(rgb_frame)

# # # # # # # # # # # # # # # #     # Check for more than one face detected
# # # # # # # # # # # # # # # #     if len(face_locations) > 1:
# # # # # # # # # # # # # # # #         return "More than One Face Detected"

# # # # # # # # # # # # # # # #     # Check for face match and neck bending
# # # # # # # # # # # # # # # #     if len(face_locations) == 1:
# # # # # # # # # # # # # # # #         face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
# # # # # # # # # # # # # # # #         match = face_recognition.compare_faces([known_encoding], face_encoding)

# # # # # # # # # # # # # # # #         if match[0]:
# # # # # # # # # # # # # # # #             face_landmarks = face_recognition.face_landmarks(rgb_frame, face_locations)
# # # # # # # # # # # # # # # #             if face_landmarks:
# # # # # # # # # # # # # # # #                 neck_angle = detect_neck_bending(face_landmarks[0])
# # # # # # # # # # # # # # # #                 if neck_angle > 130 or neck_angle < 125:  # Adjust threshold as needed
# # # # # # # # # # # # # # # #                     return "Neck Bending Detected"
# # # # # # # # # # # # # # # #                 return "Match"

# # # # # # # # # # # # # # # #     return "Not Match"

# # # # # # # # # # # # # # # # # Function to detect neck bending based on facial landmarks
# # # # # # # # # # # # # # # # def detect_neck_bending(face_landmarks):
# # # # # # # # # # # # # # # #     # Extract relevant landmarks for neck estimation
# # # # # # # # # # # # # # # #     top_nose = face_landmarks['nose_bridge'][0]
# # # # # # # # # # # # # # # #     bottom_nose = face_landmarks['nose_tip'][0]
# # # # # # # # # # # # # # # #     top_chin = face_landmarks['chin'][8]
# # # # # # # # # # # # # # # #     bottom_chin = face_landmarks['chin'][0]

# # # # # # # # # # # # # # # #     # Calculate vectors for neck and face
# # # # # # # # # # # # # # # #     neck_vector = np.array(bottom_chin) - np.array(top_chin)
# # # # # # # # # # # # # # # #     face_vector = np.array(bottom_nose) - np.array(top_nose)

# # # # # # # # # # # # # # # #     # Calculate angle between neck and face vectors
# # # # # # # # # # # # # # # #     angle = np.degrees(np.arccos(np.dot(neck_vector, face_vector) /
# # # # # # # # # # # # # # # #                                 (np.linalg.norm(neck_vector) * np.linalg.norm(face_vector))))
# # # # # # # # # # # # # # # #     return angle

# # # # # # # # # # # # # # # # # SocketIO event handler for client connection
# # # # # # # # # # # # # # # # @socketio.on('connect')
# # # # # # # # # # # # # # # # def handle_connect():
# # # # # # # # # # # # # # # #     print('Client connected')

# # # # # # # # # # # # # # # # # SocketIO event handler for receiving video frames
# # # # # # # # # # # # # # # # @socketio.on('video_data')
# # # # # # # # # # # # # # # # def handle_video_frame(frame_data):
# # # # # # # # # # # # # # # #     print('Video data received')

# # # # # # # # # # # # # # # #     # Convert Uint8Array to numpy array
# # # # # # # # # # # # # # # #     frame_arr = np.frombuffer(frame_data, dtype=np.uint8)

# # # # # # # # # # # # # # # #     # Write frame data to a temporary file
# # # # # # # # # # # # # # # #     with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as tmpfile:
# # # # # # # # # # # # # # # #         tmpfile.write(frame_arr)
# # # # # # # # # # # # # # # #         tmpfile_path = tmpfile.name

# # # # # # # # # # # # # # # #     # Debug: Check if the temporary file exists and its size
# # # # # # # # # # # # # # # #     if not os.path.exists(tmpfile_path):
# # # # # # # # # # # # # # # #         print('Temporary file does not exist')
# # # # # # # # # # # # # # # #         emit('error', 'Temporary file does not exist')
# # # # # # # # # # # # # # # #         return

# # # # # # # # # # # # # # # #     print(f'Temporary file created at: {tmpfile_path}')
# # # # # # # # # # # # # # # #     print(f'Temporary file size: {os.path.getsize(tmpfile_path)} bytes')

# # # # # # # # # # # # # # # #     # Convert .webm file to .mp4 using ffmpeg-python
# # # # # # # # # # # # # # # #     converted_file_path = tmpfile_path.replace('.webm', '.mp4')
# # # # # # # # # # # # # # # #     try:
# # # # # # # # # # # # # # # #         ffmpeg.input(tmpfile_path).output(converted_file_path).run()
# # # # # # # # # # # # # # # #     except ffmpeg.Error as e:
# # # # # # # # # # # # # # # #         print(f'Error converting video: {e}')
# # # # # # # # # # # # # # # #         emit('error', 'Error converting video')
# # # # # # # # # # # # # # # #         os.remove(tmpfile_path)
# # # # # # # # # # # # # # # #         return

# # # # # # # # # # # # # # # #     # Read the video file using OpenCV
# # # # # # # # # # # # # # # #     cap = cv2.VideoCapture(converted_file_path)

# # # # # # # # # # # # # # # #     # Check if the video capture opened successfully
# # # # # # # # # # # # # # # #     if not cap.isOpened():
# # # # # # # # # # # # # # # #         print('Error opening video file')
# # # # # # # # # # # # # # # #         emit('error', 'Error opening video file')
# # # # # # # # # # # # # # # #         os.remove(tmpfile_path)
# # # # # # # # # # # # # # # #         os.remove(converted_file_path)
# # # # # # # # # # # # # # # #         return

# # # # # # # # # # # # # # # #     result = "Not Match"
# # # # # # # # # # # # # # # #     while True:
# # # # # # # # # # # # # # # #         ret, frame = cap.read()
# # # # # # # # # # # # # # # #         if not ret:
# # # # # # # # # # # # # # # #             break

# # # # # # # # # # # # # # # #         # Detect person match in each frame
# # # # # # # # # # # # # # # #         result = detect_person_match(frame)
# # # # # # # # # # # # # # # #         if result != "Not Match":
# # # # # # # # # # # # # # # #             break

# # # # # # # # # # # # # # # #     cap.release()
# # # # # # # # # # # # # # # #     os.remove(tmpfile_path)
# # # # # # # # # # # # # # # #     os.remove(converted_file_path)

# # # # # # # # # # # # # # # #     print(f"Result: {result}")

# # # # # # # # # # # # # # # #     # Send the result back to the client
# # # # # # # # # # # # # # # #     emit('result', {'result': result})

# # # # # # # # # # # # # # # # # Render HTML page with SocketIO client
# # # # # # # # # # # # # # # # @app.route('/')
# # # # # # # # # # # # # # # # def index():
# # # # # # # # # # # # # # # #     return render_template('index.html')

# # # # # # # # # # # # # # # # if __name__ == '__main__':
# # # # # # # # # # # # # # # #     socketio.run(app, debug=True, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
# # # # # # # # # # # # # # # # # from flask import Flask
# # # # # # # # # # # # # # # # # from flask_socketio import SocketIO
# # # # # # # # # # # # # # # # # import cv2
# # # # # # # # # # # # # # # # # import numpy as np
# # # # # # # # # # # # # # # # # import subprocess
# # # # # # # # # # # # # # # # # import threading

# # # # # # # # # # # # # # # # # app = Flask(__name__)
# # # # # # # # # # # # # # # # # socketio = SocketIO(app, cors_allowed_origins="*")

# # # # # # # # # # # # # # # # # # Initialize OpenCV's Haar cascade for face detection
# # # # # # # # # # # # # # # # # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # # # # # # # # # # # # # # # # def start_ffmpeg_process():
# # # # # # # # # # # # # # # # #     command = [
# # # # # # # # # # # # # # # # #         'ffmpeg',
# # # # # # # # # # # # # # # # #         '-i', '-',
# # # # # # # # # # # # # # # # #         '-pix_fmt', 'bgr24',
# # # # # # # # # # # # # # # # #         '-vcodec', 'rawvideo',
# # # # # # # # # # # # # # # # #         '-an', '-sn',
# # # # # # # # # # # # # # # # #         '-f', 'image2pipe', '-'
# # # # # # # # # # # # # # # # #     ]
# # # # # # # # # # # # # # # # #     return subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# # # # # # # # # # # # # # # # # ffmpeg_process = start_ffmpeg_process()

# # # # # # # # # # # # # # # # # def video_processing_thread():
# # # # # # # # # # # # # # # # #     while True:
# # # # # # # # # # # # # # # # #         # Assuming 640x480 frame size
# # # # # # # # # # # # # # # # #         raw_frame = ffmpeg_process.stdout.read(640 * 480 * 3)
# # # # # # # # # # # # # # # # #         if not raw_frame:
# # # # # # # # # # # # # # # # #             break
# # # # # # # # # # # # # # # # #         frame = np.frombuffer(raw_frame, np.uint8).reshape((480, 640, 3))
# # # # # # # # # # # # # # # # #         face_count = count_faces(frame)
# # # # # # # # # # # # # # # # #         socketio.emit('result', {'count': face_count})
# # # # # # # # # # # # # # # # #         ffmpeg_process.stdout.flush()

# # # # # # # # # # # # # # # # # def count_faces(frame):
# # # # # # # # # # # # # # # # #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# # # # # # # # # # # # # # # # #     faces = face_cascade.detectMultiScale(gray, 1.1, 4)
# # # # # # # # # # # # # # # # #     return len(faces)

# # # # # # # # # # # # # # # # # @socketio.on('connect')
# # # # # # # # # # # # # # # # # def handle_connect():
# # # # # # # # # # # # # # # # #     print('Client connected')

# # # # # # # # # # # # # # # # # @socketio.on('video_chunk')
# # # # # # # # # # # # # # # # # def handle_video_chunk(data):
# # # # # # # # # # # # # # # # #     global ffmpeg_process
# # # # # # # # # # # # # # # # #     try:
# # # # # # # # # # # # # # # # #         ffmpeg_process.stdin.write(data)
# # # # # # # # # # # # # # # # #     except BrokenPipeError:
# # # # # # # # # # # # # # # # #         print("Restarting FFmpeg...")
# # # # # # # # # # # # # # # # #         ffmpeg_process = start_ffmpeg_process()
# # # # # # # # # # # # # # # # #         ffmpeg_process.stdin.write(data)

# # # # # # # # # # # # # # # # # @socketio.on('disconnect')
# # # # # # # # # # # # # # # # # def handle_disconnect():
# # # # # # # # # # # # # # # # #     print('Client disconnected')
# # # # # # # # # # # # # # # # #     ffmpeg_process.stdin.close()

# # # # # # # # # # # # # # # # # if __name__ == '__main__':
# # # # # # # # # # # # # # # # #     threading.Thread(target=video_processing_thread, daemon=True).start()
# # # # # # # # # # # # # # # # #     socketio.run(app, debug=True, host='0.0.0.0', port=5000)
# # # # # # # # # # # # # # # import cv2
# # # # # # # # # # # # # # # import numpy as np
# # # # # # # # # # # # # # # from flask import Flask, render_template
# # # # # # # # # # # # # # # from flask_socketio import SocketIO, emit
# # # # # # # # # # # # # # # import face_recognition
# # # # # # # # # # # # # # # import tempfile
# # # # # # # # # # # # # # # import os
# # # # # # # # # # # # # # # import base64
# # # # # # # # # # # # # # # import ffmpeg

# # # # # # # # # # # # # # # app = Flask(__name__)
# # # # # # # # # # # # # # # socketio = SocketIO(app, cors_allowed_origins="*")

# # # # # # # # # # # # # # # KNOWN_IMAGE_PATH = "C:/Users/OMR-09/Pictures/img2.jpg"

# # # # # # # # # # # # # # # # Load the known image
# # # # # # # # # # # # # # # known_image = face_recognition.load_image_file(KNOWN_IMAGE_PATH)
# # # # # # # # # # # # # # # known_encoding = face_recognition.face_encodings(known_image)[0]

# # # # # # # # # # # # # # # # Function to detect faces and neck bending in a frame
# # # # # # # # # # # # # # # def detect_person_match(frame):
# # # # # # # # # # # # # # #     # Convert frame to RGB (face_recognition expects RGB)
# # # # # # # # # # # # # # #     rgb_frame = frame[:, :, ::-1]

# # # # # # # # # # # # # # #     # Find face locations in the frame
# # # # # # # # # # # # # # #     face_locations = face_recognition.face_locations(rgb_frame)

# # # # # # # # # # # # # # #     # Check for more than one face detected
# # # # # # # # # # # # # # #     if len(face_locations) > 1:
# # # # # # # # # # # # # # #         return "More than One Face Detected"

# # # # # # # # # # # # # # #     # Check for face match and neck bending
# # # # # # # # # # # # # # #     if len(face_locations) == 1:
# # # # # # # # # # # # # # #         face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
# # # # # # # # # # # # # # #         match = face_recognition.compare_faces([known_encoding], face_encoding)

# # # # # # # # # # # # # # #         if match[0]:
# # # # # # # # # # # # # # #             face_landmarks = face_recognition.face_landmarks(rgb_frame, face_locations)
# # # # # # # # # # # # # # #             if face_landmarks:
# # # # # # # # # # # # # # #                 neck_angle = detect_neck_bending(face_landmarks[0])
# # # # # # # # # # # # # # #                 if neck_angle > 130 or neck_angle < 125:  # Adjust threshold as needed
# # # # # # # # # # # # # # #                     return "Neck Bending Detected"
# # # # # # # # # # # # # # #                 return "Match"

# # # # # # # # # # # # # # #     return "Not Match"

# # # # # # # # # # # # # # # # Function to detect neck bending based on facial landmarks
# # # # # # # # # # # # # # # def detect_neck_bending(face_landmarks):
# # # # # # # # # # # # # # #     # Extract relevant landmarks for neck estimation
# # # # # # # # # # # # # # #     top_nose = face_landmarks['nose_bridge'][0]
# # # # # # # # # # # # # # #     bottom_nose = face_landmarks['nose_tip'][0]
# # # # # # # # # # # # # # #     top_chin = face_landmarks['chin'][8]
# # # # # # # # # # # # # # #     bottom_chin = face_landmarks['chin'][0]

# # # # # # # # # # # # # # #     # Calculate vectors for neck and face
# # # # # # # # # # # # # # #     neck_vector = np.array(bottom_chin) - np.array(top_chin)
# # # # # # # # # # # # # # #     face_vector = np.array(bottom_nose) - np.array(top_nose)

# # # # # # # # # # # # # # #     # Calculate angle between neck and face vectors
# # # # # # # # # # # # # # #     angle = np.degrees(np.arccos(np.dot(neck_vector, face_vector) /
# # # # # # # # # # # # # # #                                 (np.linalg.norm(neck_vector) * np.linalg.norm(face_vector))))
# # # # # # # # # # # # # # #     return angle

# # # # # # # # # # # # # # # # SocketIO event handler for client connection
# # # # # # # # # # # # # # # @socketio.on('connect')
# # # # # # # # # # # # # # # def handle_connect():
# # # # # # # # # # # # # # #     print('Client connected')

# # # # # # # # # # # # # # # # SocketIO event handler for receiving video frames
# # # # # # # # # # # # # # # @socketio.on('video_data')
# # # # # # # # # # # # # # # def handle_video_frame(frame_data):
# # # # # # # # # # # # # # #     print('Video data received')

# # # # # # # # # # # # # # #     # Convert base64 string to numpy array
# # # # # # # # # # # # # # #     frame_arr = np.frombuffer(base64.b64decode(frame_data), dtype=np.uint8)

# # # # # # # # # # # # # # #     # Write frame data to a temporary file
# # # # # # # # # # # # # # #     with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as tmpfile:
# # # # # # # # # # # # # # #         tmpfile.write(frame_arr)
# # # # # # # # # # # # # # #         tmpfile_path = tmpfile.name

# # # # # # # # # # # # # # #     # Debug: Check if the temporary file exists and its size
# # # # # # # # # # # # # # #     if not os.path.exists(tmpfile_path):
# # # # # # # # # # # # # # #         print('Temporary file does not exist')
# # # # # # # # # # # # # # #         emit('error', 'Temporary file does not exist')
# # # # # # # # # # # # # # #         return

# # # # # # # # # # # # # # #     print(f'Temporary file created at: {tmpfile_path}')
# # # # # # # # # # # # # # #     print(f'Temporary file size: {os.path.getsize(tmpfile_path)} bytes')

# # # # # # # # # # # # # # #     # Convert .webm file to .mp4 using ffmpeg-python
# # # # # # # # # # # # # # #     converted_file_path = tmpfile_path.replace('.webm', '.mp4')
# # # # # # # # # # # # # # #     try:
# # # # # # # # # # # # # # #         ffmpeg.input(tmpfile_path).output(converted_file_path).run()
# # # # # # # # # # # # # # #     except ffmpeg.Error as e:
# # # # # # # # # # # # # # #         print(f'Error converting video: {e}')
# # # # # # # # # # # # # # #         emit('error', 'Error converting video')
# # # # # # # # # # # # # # #         os.remove(tmpfile_path)
# # # # # # # # # # # # # # #         return

# # # # # # # # # # # # # # #     # Read the video file using OpenCV
# # # # # # # # # # # # # # #     cap = cv2.VideoCapture(converted_file_path)

# # # # # # # # # # # # # # #     # Check if the video capture opened successfully
# # # # # # # # # # # # # # #     if not cap.isOpened():
# # # # # # # # # # # # # # #         print('Error opening video file')
# # # # # # # # # # # # # # #         emit('error', 'Error opening video file')
# # # # # # # # # # # # # # #         os.remove(tmpfile_path)
# # # # # # # # # # # # # # #         os.remove(converted_file_path)
# # # # # # # # # # # # # # #         return

# # # # # # # # # # # # # # #     result = "Not Match"
# # # # # # # # # # # # # # #     while True:
# # # # # # # # # # # # # # #         ret, frame = cap.read()
# # # # # # # # # # # # # # #         if not ret:
# # # # # # # # # # # # # # #             break

# # # # # # # # # # # # # # #         # Detect person match in each frame
# # # # # # # # # # # # # # #         result = detect_person_match(frame)
# # # # # # # # # # # # # # #         if result != "Not Match":
# # # # # # # # # # # # # # #             break

# # # # # # # # # # # # # # #     cap.release()
# # # # # # # # # # # # # # #     os.remove(tmpfile_path)
# # # # # # # # # # # # # # #     os.remove(converted_file_path)

# # # # # # # # # # # # # # #     print(f"Result: {result}")

# # # # # # # # # # # # # # #     # Send the result back to the client
# # # # # # # # # # # # # # #     emit('result', {'result': result})

# # # # # # # # # # # # # # # # Render HTML page with SocketIO client
# # # # # # # # # # # # # # # @app.route('/')
# # # # # # # # # # # # # # # def index():
# # # # # # # # # # # # # # #     return render_template('index.html')

# # # # # # # # # # # # # # # if __name__ == '__main__':
# # # # # # # # # # # # # # #     socketio.run(app, debug=True, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
# # # # # # # # # # # # # # import io
# # # # # # # # # # # # # # import cv2
# # # # # # # # # # # # # # import numpy as np
# # # # # # # # # # # # # # from flask import Flask, render_template
# # # # # # # # # # # # # # from flask_socketio import SocketIO, emit
# # # # # # # # # # # # # # import face_recognition
# # # # # # # # # # # # # # import tempfile
# # # # # # # # # # # # # # import os
# # # # # # # # # # # # # # import ffmpeg

# # # # # # # # # # # # # # app = Flask(__name__)
# # # # # # # # # # # # # # socketio = SocketIO(app, cors_allowed_origins="*")

# # # # # # # # # # # # # # KNOWN_IMAGE_PATH = "C:/Users/OMR-09/Pictures/img2.jpg"

# # # # # # # # # # # # # # # Load the known image
# # # # # # # # # # # # # # known_image = face_recognition.load_image_file(KNOWN_IMAGE_PATH)
# # # # # # # # # # # # # # known_encoding = face_recognition.face_encodings(known_image)[0]

# # # # # # # # # # # # # # # Function to detect faces and neck bending in a frame
# # # # # # # # # # # # # # def detect_person_match(frame):
# # # # # # # # # # # # # #     # Convert frame to RGB (face_recognition expects RGB)
# # # # # # # # # # # # # #     rgb_frame = frame[:, :, ::-1]

# # # # # # # # # # # # # #     # Find face locations in the frame
# # # # # # # # # # # # # #     face_locations = face_recognition.face_locations(rgb_frame)

# # # # # # # # # # # # # #     # Check for more than one face detected
# # # # # # # # # # # # # #     if len(face_locations) > 1:
# # # # # # # # # # # # # #         return "More than One Face Detected"

# # # # # # # # # # # # # #     # Check for face match and neck bending
# # # # # # # # # # # # # #     if len(face_locations) == 1:
# # # # # # # # # # # # # #         face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
# # # # # # # # # # # # # #         match = face_recognition.compare_faces([known_encoding], face_encoding)

# # # # # # # # # # # # # #         if match[0]:
# # # # # # # # # # # # # #             face_landmarks = face_recognition.face_landmarks(rgb_frame, face_locations)
# # # # # # # # # # # # # #             if face_landmarks:
# # # # # # # # # # # # # #                 neck_angle = detect_neck_bending(face_landmarks[0])
# # # # # # # # # # # # # #                 if neck_angle > 130 or neck_angle < 125:  # Adjust threshold as needed
# # # # # # # # # # # # # #                     return "Neck Bending Detected"
# # # # # # # # # # # # # #                 return "Match"

# # # # # # # # # # # # # #     return "Not Match"

# # # # # # # # # # # # # # # Function to detect neck bending based on facial landmarks
# # # # # # # # # # # # # # def detect_neck_bending(face_landmarks):
# # # # # # # # # # # # # #     # Extract relevant landmarks for neck estimation
# # # # # # # # # # # # # #     top_nose = face_landmarks['nose_bridge'][0]
# # # # # # # # # # # # # #     bottom_nose = face_landmarks['nose_tip'][0]
# # # # # # # # # # # # # #     top_chin = face_landmarks['chin'][8]
# # # # # # # # # # # # # #     bottom_chin = face_landmarks['chin'][0]

# # # # # # # # # # # # # #     # Calculate vectors for neck and face
# # # # # # # # # # # # # #     neck_vector = np.array(bottom_chin) - np.array(top_chin)
# # # # # # # # # # # # # #     face_vector = np.array(bottom_nose) - np.array(top_nose)

# # # # # # # # # # # # # #     # Calculate angle between neck and face vectors
# # # # # # # # # # # # # #     angle = np.degrees(np.arccos(np.dot(neck_vector, face_vector) /
# # # # # # # # # # # # # #                                 (np.linalg.norm(neck_vector) * np.linalg.norm(face_vector))))
# # # # # # # # # # # # # #     return angle

# # # # # # # # # # # # # # # SocketIO event handler for client connection
# # # # # # # # # # # # # # @socketio.on('connect')
# # # # # # # # # # # # # # def handle_connect():
# # # # # # # # # # # # # #     print('Client connected')

# # # # # # # # # # # # # # # SocketIO event handler for receiving video frames
# # # # # # # # # # # # # # @socketio.on('video_data')
# # # # # # # # # # # # # # def handle_video_frame(frame_data):
# # # # # # # # # # # # # #     print('Video data received')

# # # # # # # # # # # # # #     # Convert Uint8Array to numpy array
# # # # # # # # # # # # # #     frame_arr = np.frombuffer(frame_data, dtype=np.uint8)

# # # # # # # # # # # # # #     # Convert frame data to OpenCV-compatible format
# # # # # # # # # # # # # #     frame = cv2.imdecode(frame_arr, cv2.IMREAD_COLOR)

# # # # # # # # # # # # # #     # Create an in-memory file-like object
# # # # # # # # # # # # # #     with io.BytesIO() as memfile:
# # # # # # # # # # # # # #         # Write frame data to the in-memory file
# # # # # # # # # # # # # #         cv2.imwrite(memfile, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

# # # # # # # # # # # # # #         # Reset file pointer to beginning
# # # # # # # # # # # # # #         memfile.seek(0)

# # # # # # # # # # # # # #         # Convert the in-memory file to MP4 using ffmpeg-python
# # # # # # # # # # # # # #         try:
# # # # # # # # # # # # # #             ffmpeg.input('pipe:', format='image2pipe', pix_fmt='bgr24', vcodec='mjpeg', r=25).output(memfile, format='mp4', vcodec='h264_nvenc').run(input=memfile.read())
# # # # # # # # # # # # # #         except ffmpeg.Error as e:
# # # # # # # # # # # # # #             print(f'Error converting video: {e}')
# # # # # # # # # # # # # #             emit('error', 'Error converting video')
# # # # # # # # # # # # # #             return

# # # # # # # # # # # # # #     # Read the converted MP4 file from the in-memory buffer
# # # # # # # # # # # # # #     memfile.seek(0)
# # # # # # # # # # # # # #     converted_data = memfile.read()

# # # # # # # # # # # # # #     # Process the converted data (you can pass it to other functions like detect_person_match)

# # # # # # # # # # # # # #     print("Conversion completed")

# # # # # # # # # # # # # # # Render HTML page with SocketIO client
# # # # # # # # # # # # # # @app.route('/')
# # # # # # # # # # # # # # def index():
# # # # # # # # # # # # # #     return render_template('index.html')

# # # # # # # # # # # # # # if __name__ == '__main__':
# # # # # # # # # # # # # #     socketio.run(app, debug=True, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
# # # # # # # # # import io
# # # # # # # # # import cv2
# # # # # # # # # import numpy as np
# # # # # # # # # from flask import Flask, request, jsonify
# # # # # # # # # from flask_socketio import SocketIO, emit
# # # # # # # # # import face_recognition
# # # # # # # # # import tempfile
# # # # # # # # # import os
# # # # # # # # # from moviepy.editor import ImageSequenceClip

# # # # # # # # # app = Flask(__name__)
# # # # # # # # # socketio = SocketIO(app, cors_allowed_origins="*")

# # # # # # # # # KNOWN_IMAGE_PATH = "C:/Users/OMR-09/Pictures/img2.jpg"

# # # # # # # # # # Load the known image
# # # # # # # # # known_image = face_recognition.load_image_file(KNOWN_IMAGE_PATH)
# # # # # # # # # known_encoding = face_recognition.face_encodings(known_image)[0]

# # # # # # # # # # Function to detect faces and neck bending in a frame
# # # # # # # # # def detect_person_match(frame):
# # # # # # # # #     # Convert frame to RGB (face_recognition expects RGB)
# # # # # # # # #     rgb_frame = frame[:, :, ::-1]

# # # # # # # # #     # Find face locations in the frame
# # # # # # # # #     face_locations = face_recognition.face_locations(rgb_frame)

# # # # # # # # #     # Check for more than one face detected
# # # # # # # # #     if len(face_locations) > 1:
# # # # # # # # #         return "More than One Face Detected"

# # # # # # # # #     # Check for face match and neck bending
# # # # # # # # #     if len(face_locations) == 1:
# # # # # # # # #         face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
# # # # # # # # #         match = face_recognition.compare_faces([known_encoding], face_encoding)

# # # # # # # # #         if match[0]:
# # # # # # # # #             face_landmarks = face_recognition.face_landmarks(rgb_frame, face_locations)
# # # # # # # # #             if face_landmarks:
# # # # # # # # #                 neck_angle = detect_neck_bending(face_landmarks[0])
# # # # # # # # #                 if neck_angle > 130 or neck_angle < 125:  # Adjust threshold as needed
# # # # # # # # #                     return "Neck Bending Detected"
# # # # # # # # #                 return "Match"

# # # # # # # # #     return "Not Match"

# # # # # # # # # # Function to detect neck bending based on facial landmarks
# # # # # # # # # def detect_neck_bending(face_landmarks):
# # # # # # # # #     # Extract relevant landmarks for neck estimation
# # # # # # # # #     top_nose = face_landmarks['nose_bridge'][0]
# # # # # # # # #     bottom_nose = face_landmarks['nose_tip'][0]
# # # # # # # # #     top_chin = face_landmarks['chin'][8]
# # # # # # # # #     bottom_chin = face_landmarks['chin'][0]

# # # # # # # # #     # Calculate vectors for neck and face
# # # # # # # # #     neck_vector = np.array(bottom_chin) - np.array(top_chin)
# # # # # # # # #     face_vector = np.array(bottom_nose) - np.array(top_nose)

# # # # # # # # #     # Calculate angle between neck and face vectors
# # # # # # # # #     angle = np.degrees(np.arccos(np.dot(neck_vector, face_vector) /
# # # # # # # # #                                 (np.linalg.norm(neck_vector) * np.linalg.norm(face_vector))))
# # # # # # # # #     return angle

# # # # # # # # # # Route to handle video data
# # # # # # # # # @app.route('/video_data', methods=['POST'])
# # # # # # # # # def handle_video_data():
# # # # # # # # #     print('Video data received')

# # # # # # # # #     # Check if the request contains video data
# # # # # # # # #     if 'video' not in request.files:
# # # # # # # # #         return jsonify({"error": "No video part in the request"}), 400

# # # # # # # # #     video_file = request.files['video']

# # # # # # # # #     # Create a temporary directory to store frames
# # # # # # # # #     with tempfile.TemporaryDirectory() as temp_dir:
# # # # # # # # #         # Save the video file to the temporary directory
# # # # # # # # #         video_path = os.path.join(temp_dir, 'input_video.mp4')
# # # # # # # # #         video_file.save(video_path)

# # # # # # # # #         # Read the video file frame by frame
# # # # # # # # #         cap = cv2.VideoCapture(video_path)
# # # # # # # # #         frame_filenames = []
# # # # # # # # #         frame_count = 0

# # # # # # # # #         while cap.isOpened():
# # # # # # # # #             ret, frame = cap.read()
# # # # # # # # #             if not ret:
# # # # # # # # #                 break

# # # # # # # # #             # Save each frame as an image file
# # # # # # # # #             frame_path = os.path.join(temp_dir, f'frame_{frame_count}.jpg')
# # # # # # # # #             cv2.imwrite(frame_path, frame)
# # # # # # # # #             frame_filenames.append(frame_path)
# # # # # # # # #             frame_count += 1

# # # # # # # # #         cap.release()

# # # # # # # # #         # Convert frames to MP4 using moviepy
# # # # # # # # #         try:
# # # # # # # # #             clip = ImageSequenceClip(frame_filenames, fps=24)
# # # # # # # # #             output_file_path = os.path.join(temp_dir, 'output_video.mp4')
# # # # # # # # #             clip.write_videofile(output_file_path, codec='libx264')
# # # # # # # # #         except Exception as e:
# # # # # # # # #             print(f'Error converting video: {e}')
# # # # # # # # #             return jsonify({"error": "Error converting video"}), 500

# # # # # # # # #         # Read the converted MP4 file
# # # # # # # # #         with open(output_file_path, 'rb') as mp4_file:
# # # # # # # # #             converted_data = mp4_file.read()

# # # # # # # # #         # Process the converted data (you can pass it to other functions like detect_person_match)
# # # # # # # # #         # For demonstration, let's use the first frame
# # # # # # # # #         frame = cv2.imread(frame_filenames[0])
# # # # # # # # #         result = detect_person_match(frame)
# # # # # # # # #         print(result)

# # # # # # # # #         return jsonify({"message": "Conversion completed", "result": result}), 200

# # # # # # # # # # SocketIO event handler for client connection
# # # # # # # # # @socketio.on('connect')
# # # # # # # # # def handle_connect():
# # # # # # # # #     print('Client connected')
# # # # # # # # #     emit('result', {'message': 'Server connected'})

# # # # # # # # # # Render HTML page with SocketIO client
# # # # # # # # # # @app.route('/')
# # # # # # # # # # def index():
# # # # # # # # # #     return render_template('index.html')

# # # # # # # # # if __name__ == '__main__':
# # # # # # # # #     socketio.run(app, debug=True, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)

# # # # # # # # # # import cv2
# # # # # # # # # # import numpy as np
# # # # # # # # # # from flask import Flask, render_template
# # # # # # # # # # from flask_socketio import SocketIO, emit

# # # # # # # # # # app = Flask(__name__)
# # # # # # # # # # socketio = SocketIO(app,cors_allowed_origins="*")

# # # # # # # # # # # Define video writer object
# # # # # # # # # # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# # # # # # # # # # out = None

# # # # # # # # # # def start_video_writer():
# # # # # # # # # #     global out
# # # # # # # # # #     if out is None:
# # # # # # # # # #         out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))

# # # # # # # # # # def stop_video_writer():
# # # # # # # # # #     global out
# # # # # # # # # #     if out is not None:
# # # # # # # # # #         out.release()
# # # # # # # # # #         out = None

# # # # # # # # # # def process_frame(frame):
# # # # # # # # # #     # Your frame processing code using OpenCV goes here
# # # # # # # # # #     # For example, you can convert the frame to grayscale
# # # # # # # # # #     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# # # # # # # # # #     return gray_frame

# # # # # # # # # # @socketio.on('stream')
# # # # # # # # # # def handle_stream(frame):
# # # # # # # # # #     start_video_writer()  # Start video writer if not already started

# # # # # # # # # #     frame = np.frombuffer(frame, dtype=np.uint8)
# # # # # # # # # #     frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
# # # # # # # # # #     processed_frame = process_frame(frame)
# # # # # # # # # #     out.write(processed_frame)  # Write the processed frame to the video file

# # # # # # # # # # @app.route('/')
# # # # # # # # # # def index():
# # # # # # # # # #     return render_template('index.html')

# # # # # # # # # # if __name__ == '__main__':
# # # # # # # # # #     socketio.run(app, debug=True,host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
# # # # # # # # # # # # from flask import Flask, request, jsonify
# # # # # # # # # # import cv2
# # # # # # # # # # import numpy as np
# # # # # # # # # # import base64
# # # # # # # # # # import tempfile
# # # # # # # # # # import os

# # # # # # # # # # app = Flask(__name__)

# # # # # # # # # # @app.route('/')
# # # # # # # # # # def index():
# # # # # # # # # #     print("Server is connected.")
# # # # # # # # # #     return "Flask server is running"

# # # # # # # # # # @app.route('/video_data', methods=['POST'])
# # # # # # # # # # def process_video():
# # # # # # # # # #     print("Video received.")
# # # # # # # # # #     file = request.files['video']
# # # # # # # # # #     if not file:
# # # # # # # # # #         print("Error: No file received")
# # # # # # # # # #         return jsonify({"error": "No file received"}), 400

# # # # # # # # # #     # Save the received file to a temporary location
# # # # # # # # # #     temp_file = tempfile.NamedTemporaryFile(delete=False)
# # # # # # # # # #     temp_file.write(file.read())
# # # # # # # # # #     temp_file.close()

# # # # # # # # # #     # Read the video from the temporary file
# # # # # # # # # #     cap = cv2.VideoCapture(temp_file.name)
# # # # # # # # # #     if not cap.isOpened():
# # # # # # # # # #         print("Error: Could not open video")
# # # # # # # # # #         return jsonify({"error": "Could not open video"}), 400

# # # # # # # # # #     # Process the video frame by frame
# # # # # # # # # #     frames = []
# # # # # # # # # #     frame_count = 0
# # # # # # # # # #     while True:
# # # # # # # # # #         ret, frame = cap.read()
# # # # # # # # # #         if not ret:
# # # # # # # # # #             print(f"Finished reading frames. Total frames read: {frame_count}")
# # # # # # # # # #             break
# # # # # # # # # #         if frame is None or frame.size == 0:
# # # # # # # # # #             print("Error: Received empty frame")
# # # # # # # # # #             return jsonify({"error": "Received empty frame"}), 400
        
# # # # # # # # # #         # Perform OpenCV operations (example: converting to grayscale)
# # # # # # # # # #         gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# # # # # # # # # #         frames.append(gray_frame)
# # # # # # # # # #         frame_count += 1
    
# # # # # # # # # #     cap.release()
# # # # # # # # # #     os.remove(temp_file.name)  # Clean up the temporary file

# # # # # # # # # #     if not frames:
# # # # # # # # # #         print("Error: No frames were processed")
# # # # # # # # # #         return jsonify({"error": "No frames were processed"}), 400

# # # # # # # # # #     # Write the processed frames to a new video file
# # # # # # # # # #     height, width = frames[0].shape
# # # # # # # # # #     out_temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
# # # # # # # # # #     out = cv2.VideoWriter(out_temp_file.name, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (width, height), isColor=False)

# # # # # # # # # #     for frame in frames:
# # # # # # # # # #         out.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    
# # # # # # # # # #     out.release()

# # # # # # # # # #     # Read the processed video file and encode it as base64
# # # # # # # # # #     with open(out_temp_file.name, 'rb') as f:
# # # # # # # # # #         processed_video = f.read()
    
# # # # # # # # # #     encoded_video = base64.b64encode(processed_video).decode('utf-8')
# # # # # # # # # #     os.remove(out_temp_file.name)  # Clean up the temporary file

# # # # # # # # # #     return jsonify({"processed_video": encoded_video})

# # # # # # # # # # if __name__ == '__main__':
# # # # # # # # # #     print("Starting Flask server...")
# # # # # # # # # #     app.run()
    
    
    
# # # # # # # # # # # # from flask import Flask
# # # # # # # # # # # # from flask_socketio import SocketIO
# # # # # # # # # # # # import os
# # # # # # # # # # # # import subprocess

# # # # # # # # # # # # app = Flask(__name__)
# # # # # # # # # # # # socketio = SocketIO(app,cors_allowed_origins="*",engineio_logger=True, logger=True,)
# # # # # # # # # # # # # from flask import Flask
# # # # # # # # # # # # # from flask_socketio import SocketIO
# # # # # # # # # # # # # import os
# # # # # # # # # # # # # import subprocess

# # # # # # # # # # # # # app = Flask(__name__)
# # # # # # # # # # # # # socketio = SocketIO(app)

# # # # # # # # # # # # # Directory setup for saving chunks
# # # # # # # # # # # # if not os.path.exists('video_chunks'):
# # # # # # # # # # # #     os.makedirs('video_chunks')

# # # # # # # # # # # # def merge_videos(directory):
# # # # # # # # # # # #     # Generate a list of files sorted by creation time
# # # # # # # # # # # #     files = sorted(
# # # # # # # # # # # #         [os.path.join(directory, f) for f in os.listdir(directory)],
# # # # # # # # # # # #         key=os.path.getmtime
# # # # # # # # # # # #     )
# # # # # # # # # # # #     with open('filelist.txt', 'w') as filelist:
# # # # # # # # # # # #         for file in files:
# # # # # # # # # # # #             filelist.write(f"file '{file}'\n")
    
# # # # # # # # # # # #     # FFmpeg command to concatenate all video files into MP4
# # # # # # # # # # # #     subprocess.run([
# # # # # # # # # # # #         'ffmpeg', '-f', 'concat', '-safe', '0', '-i', 'filelist.txt',
# # # # # # # # # # # #         '-c', 'copy', '-strict', '-2', 'output_video.mp4' # '-strict -2' may be necessary for some FFmpeg versions to handle MP4
# # # # # # # # # # # #     ], check=True)
# # # # # # # # # # # #     # Optional cleanup: Remove files after merging to save space
# # # # # # # # # # # #     # for file in files:
# # # # # # # # # # # #     #     os.remove(file)
# # # # # # # # # # # #     # os.remove('filelist.txt')

# # # # # # # # # # # # @socketio.on('connect')
# # # # # # # # # # # # def handle_connect():
# # # # # # # # # # # #     print('Client connected')

# # # # # # # # # # # # @socketio.on('video_data')
# # # # # # # # # # # # def handle_video_chunk(data):
# # # # # # # # # # # #     print('Received video chunk')
# # # # # # # # # # # #     chunk_number = len(os.listdir('video_chunks')) + 1
# # # # # # # # # # # #     chunk_path = f'video_chunks/chunk_{chunk_number}.mp4'
# # # # # # # # # # # #     with open(chunk_path, 'wb') as f:
# # # # # # # # # # # #         f.write(data)
# # # # # # # # # # # #     print(f'Chunk saved: {chunk_path}')
# # # # # # # # # # # #     # Optional: merge after each chunk
# # # # # # # # # # # #     merge_videos('video_chunks')

# # # # # # # # # # # # if __name__ == '__main__':
# # # # # # # # # # # #     socketio.run(app, debug=True,host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True,)

# # # # # # # # # # # from flask import Flask, request, jsonify
# # # # # # # # # # # from moviepy.editor import ImageSequenceClip
# # # # # # # # # # # import os

# # # # # # # # # # # app = Flask(__name__)

# # # # # # # # # # # # Directory to save uploaded frames
# # # # # # # # # # # UPLOAD_FOLDER = 'frames'
# # # # # # # # # # # os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# # # # # # # # # # # @app.route('/video_data', methods=['POST'])
# # # # # # # # # # # def upload_frames():
# # # # # # # # # # #     if 'frames' not in request.files:
# # # # # # # # # # #         return jsonify({"error": "No frames part in the request"}), 400

# # # # # # # # # # #     frames = request.files.getlist('frames')

# # # # # # # # # # #     if not frames:
# # # # # # # # # # #         return jsonify({"error": "No frames uploaded"}), 400

# # # # # # # # # # #     frame_filenames = []
# # # # # # # # # # #     for i, frame in enumerate(frames):
# # # # # # # # # # #         frame_path = os.path.join(UPLOAD_FOLDER, f'frame_{i}.png')
# # # # # # # # # # #         frame.save(frame_path)
# # # # # # # # # # #         frame_filenames.append(frame_path)

# # # # # # # # # # #     try:
# # # # # # # # # # #         # Create a video clip from the frame filenames
# # # # # # # # # # #         clip = ImageSequenceClip(frame_filenames, fps=24)
# # # # # # # # # # #         # Write the video clip to a file
# # # # # # # # # # #         clip.write_videofile("output_video.mp4", codec='libx264')

# # # # # # # # # # #         # Clean up the saved frame files
# # # # # # # # # # #         for frame_file in frame_filenames:
# # # # # # # # # # #             os.remove(frame_file)

# # # # # # # # # # #         return jsonify({"message": "Video created successfully"}), 200

# # # # # # # # # # #     except Exception as e:
# # # # # # # # # # #         return jsonify({"error": str(e)}), 500

# # # # # # # # # # # def start_server():
# # # # # # # # # # #     app.run(debug=True)

# # # # # # # # # # # if __name__ == '__main__':
# # # # # # # # # # #     start_server()
# # # # # # # # # # from flask import Flask
# # # # # # # # # # from flask_socketio import SocketIO
# # # # # # # # # # import cv2
# # # # # # # # # # import numpy as np
# # # # # # # # # # from imageio import get_reader
# # # # # # # # # # from io import BytesIO
# # # # # # # # # # import threading

# # # # # # # # # # app = Flask(__name__)
# # # # # # # # # # socketio = SocketIO(app, cors_allowed_origins="*")

# # # # # # # # # # # Thread-safe buffer to store video stream data
# # # # # # # # # # class VideoStreamBuffer:
# # # # # # # # # #     def __init__(self):
# # # # # # # # # #         self.buffer = BytesIO()
# # # # # # # # # #         self.lock = threading.Lock()

# # # # # # # # # #     def write(self, data):
# # # # # # # # # #         with self.lock:
# # # # # # # # # #             self.buffer.write(data)

# # # # # # # # # #     def get_frames(self):
# # # # # # # # # #         self.buffer.seek(0)
# # # # # # # # # #         # Using imageio to read frames from binary stream
# # # # # # # # # #         try:
# # # # # # # # # #             for frame in get_reader(self.buffer, 'ffmpeg'):
# # # # # # # # # #                 yield cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # Convert frame to BGR format used by OpenCV
# # # # # # # # # #         finally:
# # # # # # # # # #             self.reset()

# # # # # # # # # #     def reset(self):
# # # # # # # # # #         with self.lock:
# # # # # # # # # #             self.buffer = BytesIO()

# # # # # # # # # # video_stream_buffer = VideoStreamBuffer()

# # # # # # # # # # @socketio.on('connect')
# # # # # # # # # # def handle_connect():
# # # # # # # # # #     print('Client connected')

# # # # # # # # # # @socketio.on('video_chunk')
# # # # # # # # # # def handle_video_chunk(data):
# # # # # # # # # #     print('Received video chunk')
# # # # # # # # # #     video_stream_buffer.write(data)

# # # # # # # # # # @socketio.on('disconnect')
# # # # # # # # # # def handle_disconnect():
# # # # # # # # # #     print('Client disconnected')
# # # # # # # # # #     process_video_stream() # Process video when the client disconnects

# # # # # # # # # # def process_video_stream():
# # # # # # # # # #     for frame in video_stream_buffer.get_frames():
# # # # # # # # # #         processed_frame = process_frame(frame)
# # # # # # # # # #         # Process frame with OpenCV here
# # # # # # # # # #         # Example: Detect edges in the frame
# # # # # # # # # #         edges = cv2.Canny(processed_frame, 100, 200)
# # # # # # # # # #         cv2.imshow('Edges', edges)
# # # # # # # # # #         if cv2.waitKey(1) & 0xFF == ord('q'):
# # # # # # # # # #             break
# # # # # # # # # #     cv2.destroyAllWindows()

# # # # # # # # # # def process_frame(frame):
# # # # # # # # # #     # Example processing: Convert to grayscale
# # # # # # # # # #     return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# # # # # # # # # # if __name__ == '__main__':
# # # # # # # # # #     socketio.run(app, debug=True, host='0.0.0.0', port=5000)
# # # # # # # # # #     threading.Thread(target=process_video_stream).start()

# # # # # # # # # import io
# # # # # # # # # import cv2
# # # # # # # # # import numpy as np
# # # # # # # # # from flask import Flask, render_template
# # # # # # # # # from flask_socketio import SocketIO, emit
# # # # # # # # # import face_recognition
# # # # # # # # # import tempfile
# # # # # # # # # import os
# # # # # # # # # from moviepy.editor import ImageSequenceClip

# # # # # # # # # app = Flask(__name__)
# # # # # # # # # socketio = SocketIO(app, cors_allowed_origins="*")

# # # # # # # # # KNOWN_IMAGE_PATH = "C:/Users/OMR-09/Pictures/img2.jpg"

# # # # # # # # # # Load the known image
# # # # # # # # # known_image = face_recognition.load_image_file(KNOWN_IMAGE_PATH)
# # # # # # # # # known_encoding = face_recognition.face_encodings(known_image)[0]

# # # # # # # # # # Function to detect faces and neck bending in a frame
# # # # # # # # # def detect_person_match(frame):
# # # # # # # # #     # Convert frame to RGB (face_recognition expects RGB)
# # # # # # # # #     rgb_frame = frame[:, :, ::-1]

# # # # # # # # #     # Find face locations in the frame
# # # # # # # # #     face_locations = face_recognition.face_locations(rgb_frame)

# # # # # # # # #     # Check for more than one face detected
# # # # # # # # #     if len(face_locations) > 1:
# # # # # # # # #         return "More than One Face Detected"

# # # # # # # # #     # Check for face match and neck bending
# # # # # # # # #     if len(face_locations) == 1:
# # # # # # # # #         face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
# # # # # # # # #         match = face_recognition.compare_faces([known_encoding], face_encoding)

# # # # # # # # #         if match[0]:
# # # # # # # # #             face_landmarks = face_recognition.face_landmarks(rgb_frame, face_locations)
# # # # # # # # #             if face_landmarks:
# # # # # # # # #                 neck_angle = detect_neck_bending(face_landmarks[0])
# # # # # # # # #                 if neck_angle > 130 or neck_angle < 125:  # Adjust threshold as needed
# # # # # # # # #                     return "Neck Bending Detected"
# # # # # # # # #                 return "Match"

# # # # # # # # #     return "Not Match"

# # # # # # # # # # Function to detect neck bending based on facial landmarks
# # # # # # # # # def detect_neck_bending(face_landmarks):
# # # # # # # # #     # Extract relevant landmarks for neck estimation
# # # # # # # # #     top_nose = face_landmarks['nose_bridge'][0]
# # # # # # # # #     bottom_nose = face_landmarks['nose_tip'][0]
# # # # # # # # #     top_chin = face_landmarks['chin'][8]
# # # # # # # # #     bottom_chin = face_landmarks['chin'][0]

# # # # # # # # #     # Calculate vectors for neck and face
# # # # # # # # #     neck_vector = np.array(bottom_chin) - np.array(top_chin)
# # # # # # # # #     face_vector = np.array(bottom_nose) - np.array(top_nose)

# # # # # # # # #     # Calculate angle between neck and face vectors
# # # # # # # # #     angle = np.degrees(np.arccos(np.dot(neck_vector, face_vector) /
# # # # # # # # #                                 (np.linalg.norm(neck_vector) * np.linalg.norm(face_vector))))
# # # # # # # # #     return angle

# # # # # # # # # # SocketIO event handler for client connection
# # # # # # # # # @socketio.on('video_data')
# # # # # # # # # def handle_video_frame(frame_data):
# # # # # # # # #     print('Video data received')

# # # # # # # # #     # Convert Uint8Array to numpy array
# # # # # # # # #     frame_arr = np.frombuffer(frame_data, dtype=np.uint8)

# # # # # # # # #     # Convert frame data to OpenCV-compatible format
# # # # # # # # #     frame = cv2.imdecode(frame_arr, cv2.IMREAD_COLOR)

# # # # # # # # #     if frame is None:
# # # # # # # # #         print("Error: Received empty frame")
# # # # # # # # #         emit('error', 'Received empty frame')
# # # # # # # # #         return

# # # # # # # # #     if frame.size == 0:
# # # # # # # # #         print("Error: Received frame with size 0")
# # # # # # # # #         emit('error', 'Received frame with size 0')
# # # # # # # # #         return

# # # # # # # # #     print("Frame shape:", frame.shape)

# # # # # # # # #     # Store the frame temporarily for conversion to video
# # # # # # # # #     frame_filenames = []
# # # # # # # # #     with tempfile.TemporaryDirectory() as temp_dir:
# # # # # # # # #         temp_frame_path = os.path.join(temp_dir, 'frame.jpg')
# # # # # # # # #         if not cv2.imwrite(temp_frame_path, frame):
# # # # # # # # #             print("Error writing frame to temporary file")
# # # # # # # # #             emit('error', 'Error writing frame to temporary file')
# # # # # # # # #             return

# # # # # # # # #         frame_filenames.append(temp_frame_path)

# # # # # # # # #         # Convert frames to MP4 using moviepy
# # # # # # # # #         try:
# # # # # # # # #             clip = ImageSequenceClip(frame_filenames, fps=24)
# # # # # # # # #             output_file_path = os.path.join(temp_dir, 'output_video.mp4')
# # # # # # # # #             clip.write_videofile(output_file_path, codec='libx264')
# # # # # # # # #         except Exception as e:
# # # # # # # # #             print(f'Error converting video: {e}')
# # # # # # # # #             emit('error', 'Error converting video')
# # # # # # # # #             return

# # # # # # # # #         # Read the converted MP4 file
# # # # # # # # #         with open(output_file_path, 'rb') as mp4_file:
# # # # # # # # #             converted_data = mp4_file.read()

# # # # # # # # #     # Process the converted data (you can pass it to other functions like detect_person_match)
# # # # # # # # #     result = detect_person_match(frame)
# # # # # # # # #     print(result)

# # # # # # # # #     emit('conversion_complete', {'message': 'Conversion completed', 'result': result})

# # # # # # # # # # Render HTML page with SocketIO client
# # # # # # # # # @app.route('/')
# # # # # # # # # def index():
# # # # # # # # #     return render_template('index.html')

# # # # # # # # # if __name__ == '__main__':
# # # # # # # # #     socketio.run(app, debug=True, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
# # # # # # # # from flask import Flask
# # # # # # # # from flask_socketio import SocketIO
# # # # # # # # import os
# # # # # # # # import subprocess

# # # # # # # # app = Flask(__name__)
# # # # # # # # socketio = SocketIO(app)

# # # # # # # # # Directory setup for saving chunks
# # # # # # # # if not os.path.exists('video_chunks'):
# # # # # # # #     os.makedirs('video_chunks')

# # # # # # # # def merge_videos(directory):
# # # # # # # #     # Generate a list of files sorted by creation time
# # # # # # # #     files = sorted(
# # # # # # # #         [os.path.join(directory, f) for f in os.listdir(directory)],
# # # # # # # #         key=os.path.getmtime
# # # # # # # #     )
# # # # # # # #     with open('filelist.txt', 'w') as filelist:
# # # # # # # #         for file in files:
# # # # # # # #             filelist.write(f"file '{file}'\n")
    
# # # # # # # #     # FFmpeg command to concatenate all video files into MP4
# # # # # # # #     subprocess.run([
# # # # # # # #         'ffmpeg', '-f', 'concat', '-safe', '0', '-i', 'filelist.txt',
# # # # # # # #         '-c', 'copy', '-strict', '-2', 'output_video.mp4' # '-strict -2' may be necessary for some FFmpeg versions to handle MP4
# # # # # # # #     ], check=True)
# # # # # # # #     # Optional cleanup: Remove files after merging to save space
# # # # # # # #     # for file in files:
# # # # # # # #     #     os.remove(file)
# # # # # # # #     # os.remove('filelist.txt')

# # # # # # # # @socketio.on('connect')
# # # # # # # # def handle_connect():
# # # # # # # #     print('Client connected')

# # # # # # # # @socketio.on('video_chunk')
# # # # # # # # def handle_video_chunk(data):
# # # # # # # #     print('Received video chunk')
# # # # # # # #     chunk_number = len(os.listdir('video_chunks')) + 1
# # # # # # # #     chunk_path = f'video_chunks/chunk_{chunk_number}.mp4'
# # # # # # # #     with open(chunk_path, 'wb') as f:
# # # # # # # #         f.write(data)
# # # # # # # #     print(f'Chunk saved: {chunk_path}')
# # # # # # # #     # Optional: merge after each chunk
# # # # # # # #     merge_videos('video_chunks')

# # # # # # # # if __name__ == '__main__':
# # # # # # # #     socketio.run(app, debug=True)
# # # # # # # from flask import Flask, jsonify
# # # # # # # from flask_socketio import SocketIO
# # # # # # # import cv2
# # # # # # # import numpy as np
# # # # # # # from base64 import b64encode
# # # # # # # import imageio
# # # # # # # import io
# # # # # # # import tempfile
# # # # # # # import os 

# # # # # # # app = Flask(__name__)
# # # # # # # socketio = SocketIO(app, cors_allowed_origins="*")


# # # # # # # def process_video_stream(video_data):
# # # # # # #     try:
# # # # # # #         # Create a temporary file to write the video data
# # # # # # #         with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as tmpfile:
# # # # # # #             tmpfile.write(video_data)
# # # # # # #             tmpfile.flush()  # Ensure all data is written
# # # # # # #             tmpfile_path = tmpfile.name  # Save the path to access later

# # # # # # #         # Now read the video file using imageio with ffmpeg
# # # # # # #         with imageio.get_reader(tmpfile_path, 'ffmpeg') as reader:
# # # # # # #             frames = []
# # # # # # #             for image in reader:
# # # # # # #                 gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# # # # # # #                 _, buffer = cv2.imencode('.jpg', gray_image)
# # # # # # #                 frames.append(b64encode(buffer).decode('utf-8'))
# # # # # # #             return frames
# # # # # # #     except Exception as e:
# # # # # # #         print("Error processing video from file:", e)
# # # # # # #         return []
# # # # # # #     finally:
# # # # # # #         # Cleanup: Ensure to delete the temp file
# # # # # # #         if os.path.exists(tmpfile_path):
# # # # # # #             os.remove(tmpfile_path)
            
            
# # # # # # # @socketio.on('video_data')
# # # # # # # def handle_video_data(data):
# # # # # # #     print("Received video data type:", type(data))
# # # # # # #     print("Data size:", len(data) if isinstance(data, bytes) else "N/A")
# # # # # # #     try:
# # # # # # #         frames = process_video_stream(data)
# # # # # # #         # Emit the processed frames back to the client
# # # # # # #         for frame in frames:
# # # # # # #             socketio.emit('processed_frame', {'data': frame})
# # # # # # #     except Exception as e:
# # # # # # #         print("Error processing video stream:", e)
# # # # # # #         raise e

# # # # # # # @socketio.on('connect')
# # # # # # # def test_connect():
# # # # # # #     print('Client connected')

# # # # # # # @socketio.on('disconnect')
# # # # # # # def test_disconnect():
# # # # # # #     print('Client disconnected')

# # # # # # # if __name__ == '__main__':
# # # # # # #     socketio.run(app, debug=True,host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)


# # # # # # # app.py


# # # # # # from flask import Flask, request, jsonify
# # # # # # from flask_socketio import SocketIO, emit
# # # # # # import cv2
# # # # # # import numpy as np
# # # # # # import face_recognition
# # # # # # from pyngrok import ngrok
# # # # # # import base64

# # # # # # # Initialize Flask app
# # # # # # app = Flask(__name__)
# # # # # # socketio = SocketIO(app, cors_allowed_origins="*")

# # # # # # # Set up ngrok authentication token
# # # # # # ngrok.set_auth_token("2gDVBMbJ3zF6Fdccaicxr3QIzbu_7ho4uhZoAUaNg5MAQeAob")

# # # # # # KNOWN_IMAGE_PATH = "C:/Users/OMR-09/Pictures/img2.jpg"

# # # # # # # Load the known image
# # # # # # known_image = face_recognition.load_image_file(KNOWN_IMAGE_PATH)
# # # # # # known_encoding = face_recognition.face_encodings(known_image)[0]

# # # # # # # Create a background subtractor object
# # # # # # bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=True)

# # # # # # def detect_neck_bending(face_landmarks):
# # # # # #     # Extract relevant landmarks for neck estimation
# # # # # #     top_nose = face_landmarks['nose_bridge'][0]
# # # # # #     bottom_nose = face_landmarks['nose_tip'][0]
# # # # # #     top_chin = face_landmarks['chin'][8]
# # # # # #     bottom_chin = face_landmarks['chin'][0]

# # # # # #     # Calculate vectors for neck and face
# # # # # #     neck_vector = np.array(bottom_chin) - np.array(top_chin)
# # # # # #     face_vector = np.array(bottom_nose) - np.array(top_nose)

# # # # # #     # Calculate angle between neck and face vectors
# # # # # #     angle = np.degrees(np.arccos(np.dot(neck_vector, face_vector) /
# # # # # #                                   (np.linalg.norm(neck_vector) * np.linalg.norm(face_vector))))
# # # # # #     return angle

# # # # # # @socketio.on('connect')
# # # # # # def handle_connect():
# # # # # #     print('Client connected')

# # # # # # @socketio.on('disconnect')
# # # # # # def handle_disconnect():
# # # # # #     print('Client disconnected')

# # # # # # @socketio.on('video_data')
# # # # # # def handle_video_frame(data):
# # # # # #     # Decode the base64 frame
# # # # # #     frame = base64.b64decode(data['frame'])
# # # # # #     np_frame = np.frombuffer(frame, dtype=np.uint8)
# # # # # #     img = cv2.imdecode(np_frame, cv2.IMREAD_COLOR)

# # # # # #     # Apply background subtraction to detect movement in the background
# # # # # #     fg_mask = bg_subtractor.apply(img)

# # # # # #     # Obtain the shadow value from the background subtractor
# # # # # #     shadow_value = bg_subtractor.getShadowValue()

# # # # # #     # Invert the shadow mask
# # # # # #     fg_mask[fg_mask == shadow_value] = 0

# # # # # #     # Find contours of moving objects
# # # # # #     contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # # # # #     # Check if any contours (movement) are detected
# # # # # #     background_movement = False
# # # # # #     for contour in contours:
# # # # # #         area = cv2.contourArea(contour)
# # # # # #         if area > 1000:  # Adjust threshold as needed
# # # # # #             background_movement = True
# # # # # #             break

# # # # # #     if background_movement:
# # # # # #         emit('movement_detected', {'result': 'Movement'})
# # # # # #         return

# # # # # #     # If no significant background movement is detected, check for face match
# # # # # #     face_locations = face_recognition.face_locations(img)

# # # # # #     # Check for more than one face detected
# # # # # #     if len(face_locations) > 1:
# # # # # #         emit('movement_detected', {'result': 'More than One Face Detected'})
# # # # # #         return

# # # # # #     # Detect neck bending
# # # # # #     if len(face_locations) == 1:
# # # # # #         face_landmarks = face_recognition.face_landmarks(img, face_locations)
# # # # # #         if face_landmarks:
# # # # # #             neck_angle = detect_neck_bending(face_landmarks[0])
# # # # # #             if neck_angle > 130 or neck_angle < 125:  # Adjust threshold as needed
# # # # # #                 emit('movement_detected', {'result': 'Neck Bending Detected'})
# # # # # #                 return

# # # # # #     # If no significant background movement or face match is detected, return Not Match
# # # # # #     emit('movement_detected', {'result': 'Not Match'})

# # # # # # if __name__ == '__main__':
# # # # # #     public_url = ngrok.connect(5000).public_url
# # # # # #     print(" * Running on", public_url)
# # # # # #     try:
# # # # # #         socketio.run(app, host='0.0.0.0', port=5000)
# # # # # #     except KeyboardInterrupt:
# # # # # #         print(" * Shutting down Flask app...")

# # # # from flask import Flask, jsonify
# # # # from flask_socketio import SocketIO, emit
# # # # import cv2
# # # # import numpy as np
# # # # import face_recognition
# # # # from pyngrok import ngrok
# # # # import os

# # # # # Initialize Flask app and SocketIO
# # # # app = Flask(__name__)
# # # # socketio = SocketIO(app, cors_allowed_origins="*")

# # # # # Set up ngrok authentication token
# # # # ngrok.set_auth_token("2gDVBMbJ3zF6Fdccaicxr3QIzbu_7ho4uhZoAUaNg5MAQeAob")

# # # # KNOWN_IMAGE_PATH = "C:/Users/OMR-09/Pictures/img2.jpg"

# # # # def detect_person_match(video_path):
# # # #     # Load the known image
# # # #     known_image = face_recognition.load_image_file(KNOWN_IMAGE_PATH)
# # # #     known_encoding = face_recognition.face_encodings(known_image)[0]

# # # #     # Open the video capture
# # # #     cap = cv2.VideoCapture(video_path)

# # # #     # Create a background subtractor object
# # # #     bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=True)

# # # #     # Initialize variables for face detection and background movement detection
# # # #     face_locations = []

# # # #     # Iterate over frames
# # # #     while True:
# # # #         ret, frame = cap.read()
# # # #         if not ret:
# # # #             break

# # # #         # Apply background subtraction to detect movement in the background
# # # #         fg_mask = bg_subtractor.apply(frame)

# # # #         # Obtain the shadow value from the background subtractor
# # # #         shadow_value = bg_subtractor.getShadowValue()

# # # #         # Invert the shadow mask
# # # #         fg_mask[fg_mask == shadow_value] = 0

# # # #         # Find contours of moving objects
# # # #         contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # # #         # Check if any contours (movement) are detected
# # # #         background_movement = False
# # # #         for contour in contours:
# # # #             area = cv2.contourArea(contour)
# # # #             if area > 1000:  # Adjust threshold as needed
# # # #                 background_movement = True
# # # #                 break
  
# # # #         if background_movement:
# # # #             cap.release()
# # # #             return "Movement"

# # # #         # If no significant background movement is detected, check for face match
# # # #         if not background_movement:
# # # #             # Find face locations in the frame
# # # #             face_locations = face_recognition.face_locations(frame)
            
# # # #             if len(face_locations) == 0:
# # # #                 cap.release()
# # # #                 return "No Face Detected"
            
# # # #             # Check for more than one face detected or not match
# # # #             if len(face_locations) > 1:
# # # #                 cap.release()
# # # #                 return "More than One Face Detected"

# # # #             # Detect neck bending
# # # #             if len(face_locations) == 1:
# # # #                 face_landmarks = face_recognition.face_landmarks(frame, face_locations)
# # # #                 if face_landmarks:
# # # #                     neck_angle = detect_neck_bending(face_landmarks[0])
# # # #                     if neck_angle > 130 or neck_angle < 125:  # Adjust threshold as needed
# # # #                         cap.release()
# # # #                         return "Neck Bending Detected"

# # # #     # If no significant background movement or face match is detected, return Not Match
# # # #     cap.release()
# # # #     return "Not Match"

# # # # def detect_neck_bending(face_landmarks):
# # # #     # Extract relevant landmarks for neck estimation
# # # #     top_nose = face_landmarks['nose_bridge'][0]
# # # #     bottom_nose = face_landmarks['nose_tip'][0]
# # # #     top_chin = face_landmarks['chin'][8]
# # # #     bottom_chin = face_landmarks['chin'][0]

# # # #     # Calculate vectors for neck and face
# # # #     neck_vector = np.array(bottom_chin) - np.array(top_chin)
# # # #     face_vector = np.array(bottom_nose) - np.array(top_nose)

# # # #     # Calculate angle between neck and face vectors
# # # #     angle = np.degrees(np.arccos(np.dot(neck_vector, face_vector) /
# # # #                                   (np.linalg.norm(neck_vector) * np.linalg.norm(face_vector))))
# # # #     return angle

# # # # @socketio.on('connect')
# # # # def handle_connect():
# # # #     print('Server connected')

# # # # @socketio.on('video_data')
# # # # def handle_video_frame(data):
# # # #     print('Video received')

    
# # # #     # Save the received video file
# # # #     video_path = 'temp_video.mp4'
# # # #     with open(video_path, 'wb') as f:
# # # #         f.write(data)

# # # #     # Process the video file
# # # #     result = detect_person_match(video_path)

# # # #     # Remove the temporary video file
# # # #     os.remove(video_path)

# # # #     # Send back the result
# # # #     emit('frame_result', {'result': result})

# # # # if __name__ == '__main__':
# # # #     public_url = ngrok.connect(5000).public_url
# # # #     print(" * Running on", public_url)
# # # #     socketio.run(app, debug=True, host='0.0.0.0', port=5000)

# # # # # from flask import Flask
# # # # # from flask_socketio import SocketIO, emit
# # # # # import os

# # # # # app = Flask(__name__)
# # # # # socketio = SocketIO(app)

# # # # # @socketio.on('connect')
# # # # # def handle_connect():
# # # # #     print('Client connected')

# # # # # @socketio.on('video_data')
# # # # # def handle_video_data(data):
# # # # #     print('Received video data')
# # # # #     # Assuming 'data' is the binary blob of video data
# # # # #     save_video_data(data)

# # # # # def save_video_data(data):
# # # # #     # Create a directory for saved videos if it doesn't exist
# # # # #     save_path = 'saved_videos'
# # # # #     if not os.path.exists(save_path):
# # # # #         os.makedirs(save_path)
    
# # # # #     # Define a file path
# # # # #     file_path = os.path.join(save_path, 'output.webm')
    
# # # # #     # Write the data to a file
# # # # #     with open(file_path, 'wb') as f:
# # # # #         f.write(data) # Write binary data to file
# # # # #     print(f'Video saved to {file_path}')

# # # # # if __name__ == '__main__':
# # # # #     socketio.run(app, debug=True)
# # # from flask import Flask
# # # from flask_socketio import SocketIO
# # # import os
# # # import subprocess

# # # app = Flask(__name__)
# # # socketio = SocketIO(app,cors_allowed_origins="*")

# # # # Directory setup for saving chunks
# # # if not os.path.exists('video_chunks'):
# # #     os.makedirs('video_chunks')

# # # def merge_videos(directory):
# # #     # Generate a list of files sorted by creation time
# # #     files = sorted(
# # #         [os.path.join(directory, f) for f in os.listdir(directory)],
# # #         key=os.path.getmtime
# # #     )
# # #     with open('filelist.txt', 'w') as filelist:
# # #         for file in files:
# # #             filelist.write(f"file '{file}'\n")
    
# # #     # FFmpeg command to concatenate all video files into MP4
# # #     subprocess.run([
# # #         'ffmpeg', '-f', 'concat', '-safe', '0', '-i', 'filelist.txt',
# # #         '-c', 'copy', '-strict', '-2', 'output_video.mp4' # '-strict -2' may be necessary for some FFmpeg versions to handle MP4
# # #     ], check=True)
# # #     # Optional cleanup: Remove files after merging to save space
# # #     # for file in files:
# # #     #     os.remove(file)
# # #     # os.remove('filelist.txt')

# # # @socketio.on('connect')
# # # def handle_connect():
# # #     print('Client connected')

# # # @socketio.on('video_data')
# # # def handle_video_chunk(data):
# # #     print('Received video chunk')
# # #     chunk_number = len(os.listdir('video_chunks')) + 1
# # #     chunk_path = f'video_chunks/chunk_{chunk_number}.mp4'
# # #     with open(chunk_path, 'wb') as f:
# # #         f.write(data)
# # #     print(f'Chunk saved: {chunk_path}')
# # #     # Optional: merge after each chunk
# # #     merge_videos('video_chunks')

# # # if __name__ == '__main__':
# # #     socketio.run(app, debug=True,host='0.0.0.0', port=5000)

# # import numpy as np
# # import dlib
# # import cv2
# # import matplotlib.pyplot as plt
# # from PIL import Image

# # detector = dlib.get_frontal_face_detector()
# # predictor = dlib.shape_predictor("C:/Users/OMR-09/Downloads/shape_predictor_68_face_landmarks.dat")


# # # Start video capture from webcam
# # cap = cv2.VideoCapture(0)

# # while True:
# #     ret, frame = cap.read()
# #     if not ret:
# #         break
    
# #     # Convert the frame to RGB (dlib uses RGB)
# #     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
# #     # Detect face
# #     rects = detector(frame_rgb)
# #     if len(rects) == 0:
# #         continue
    
# #     rect = rects[0]
# #     sp = predictor(frame_rgb, rect)
# #     landmarks = np.array([[p.x, p.y] for p in sp.parts()])
    
# #     # Extracting nose bridge coordinates
# #     nose_bridge_x = []
# #     nose_bridge_y = []
# #     for i in [28, 29, 30, 31, 33, 34, 35]:
# #         nose_bridge_x.append(landmarks[i][0])
# #         nose_bridge_y.append(landmarks[i][1])

# #     # Extracting glasses lens coordinates (points 37-42)
# #     lens_x = [landmarks[i][0] for i in range(36, 42)]
# #     lens_y = [landmarks[i][1] for i in range(36, 42)]

# #     # x_min and x_max for nose bridge
# #     x_min = min(nose_bridge_x)
# #     x_max = max(nose_bridge_x)
# #     # ymin (from top eyebrow coordinate), ymax
# #     y_min = landmarks[20][1]
# #     y_max = landmarks[31][1]

# #     # Crop the region for nose bridge
# #     img_nose = Image.fromarray(frame_rgb)
# #     img_nose = img_nose.crop((x_min, y_min, x_max, y_max))
    
# #     # Crop the region for glasses lens
# #     x_min_lens = min(lens_x)
# #     x_max_lens = max(lens_x)
# #     y_min_lens = min(lens_y)
# #     y_max_lens = max(lens_y)
# #     img_lens = Image.fromarray(frame_rgb)
# #     img_lens = img_lens.crop((x_min_lens, y_min_lens, x_max_lens, y_max_lens))
    
# #     # Convert to numpy array and blur the images
# #     img_nose_blur = cv2.GaussianBlur(np.array(img_nose), (3, 3), sigmaX=0, sigmaY=0)
# #     img_lens_blur = cv2.GaussianBlur(np.array(img_lens), (3, 3), sigmaX=0, sigmaY=0)

# #     # Detect edges
# #     edges_nose = cv2.Canny(image=img_nose_blur, threshold1=100, threshold2=200)
# #     edges_lens = cv2.Canny(image=img_lens_blur, threshold1=100, threshold2=200)

# #     # Center strip for nose bridge
# #     edges_center_nose = edges_nose.T[(int(len(edges_nose.T) / 2))]
# #     # Center strip for glasses lens
# #     edges_center_lens = edges_lens.T[(int(len(edges_lens.T) / 2))]

# #     # Check if glasses are present
# #     if 255 in edges_center_nose:
# #         print("Glasses are present")
# #     else:
# #         print("Glasses are absent")
# from flask import Flask, request
# from flask_cors import CORS
# from flask_socketio import SocketIO
# import os
# import time

# app = Flask(__name__)
# CORS(app) # Allow all routes in CORS

# socketio = SocketIO(app, cors_allowed_origins="*", ping_timeout=20000, max_http_buffer_size=10 * 1024 * 1024, ping_interval=25000)

# # Directory to store video files
# video_dir = os.path.join(os.path.dirname(__file__), 'videos')
# if not os.path.exists(video_dir):
#     os.makedirs(video_dir)

# def create_file_write_stream(file_name):
#     file_path = os.path.join(video_dir, f"{file_name}.webm")
#     file_stream = open(file_path, 'wb')
#     return file_stream, file_path

# @socketio.on('connect')
# def handle_connect():
#     print('Client connected')

# @socketio.on('video_data')
# def handle_video_data(data):
#     file_name = f"video_{int(time.time() * 1000)}"
#     file_stream, file_path = create_file_write_stream(file_name)

#     file_stream.write(data)
#     file_stream.close()
#     print(f"File {file_path} has been saved.")

# @socketio.on('disconnect')
# def handle_disconnect():
#     print('Client disconnected')

# @socketio.on_error() # Handles the default namespace
# def error_handler(e):
#     print(f"Socket error: {e}")

# if __name__ == '__main__':
#     socketio.run(app, port=5000, host='0.0.0.0', debug=True)



from flask import Flask
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.exceptions import NotFound

# create a server instance
app = Flask(__name__)

@app.route('/')
def index():
    return "Hello World!!!"

@app.route('/help')
def helpPage():
    return "This is the Help Page"

hostedApp = Flask(__name__)
hostedApp.wsgi_app = DispatcherMiddleware(NotFound(), {
    "/myApp": app
})
# run the server
hostedApp.run(host="0.0.0.0", port=50100, debug=True)