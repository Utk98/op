# # # # # # from flask import Flask, request
# # # # # # from flask_cors import CORS
# # # # # # from flask_socketio import SocketIO
# # # # # # import os
# # # # # # import time

# # # # # # app = Flask(__name__)
# # # # # # CORS(app) # Allow all routes in CORS

# # # # # # socketio = SocketIO(app, cors_allowed_origins="*")

# # # # # # # Directory to store video files
# # # # # # video_dir = os.path.join(os.path.dirname(__file__), 'videos')
# # # # # # if not os.path.exists(video_dir):
# # # # # #     os.makedirs(video_dir)

# # # # # # def create_file_write_stream(file_name):
# # # # # #     file_path = os.path.join(video_dir, f"{file_name}.webm")
# # # # # #     file_stream = open(file_path, 'wb')
# # # # # #     return file_stream, file_path

# # # # # # @socketio.on('connect')
# # # # # # def handle_connect():
# # # # # #     print('Client connected')

# # # # # # @socketio.on('video_data')
# # # # # # def handle_video_data(data):
# # # # # #     file_name = f"video_{int(time.time() * 1000)}"
# # # # # #     file_stream, file_path = create_file_write_stream(file_name)

# # # # # #     file_stream.write(data)
# # # # # #     file_stream.close()
# # # # # #     print(f"File {file_path} has been saved.")

# # # # # # @socketio.on('disconnect')
# # # # # # def handle_disconnect():
# # # # # #     print('Client disconnected')

# # # # # # @socketio.on_error() # Handles the default namespace
# # # # # # def error_handler(e):
# # # # # #     print(f"Socket error: {e}")

# # # # # # if __name__ == '__main__':
# # # # # #     socketio.run(app, port=5000, host='0.0.0.0', debug=True)
    
# # # # #     # -------------------------------------------------------------------------------------------------
# # # # # # from flask import Flask, request
# # # # # # from flask_cors import CORS
# # # # # # from flask_socketio import SocketIO
# # # # # # import os
# # # # # # import time

# # # # # # app = Flask(__name__)
# # # # # # CORS(app) # Allow all routes in CORS

# # # # # # socketio = SocketIO(app, cors_allowed_origins="*")

# # # # # # # Directory to store video files
# # # # # # video_dir = os.path.join(os.path.dirname(__file__), 'videos')
# # # # # # if not os.path.exists(video_dir):
# # # # # #     os.makedirs(video_dir)

# # # # # # def create_file_write_stream(file_name):
# # # # # #     file_path = os.path.join(video_dir, f"{file_name}.webm")
# # # # # #     file_stream = open(file_path, 'wb')
# # # # # #     return file_stream, file_path

# # # # # # current_video_file = None

# # # # # # @socketio.on('connect')
# # # # # # def handle_connect():
# # # # # #     print('Client connected')

# # # # # # @socketio.on('video_data')
# # # # # # def handle_video_data(data):
# # # # # #     global current_video_file
    
# # # # # #     # Delete previous video file if exists
# # # # # #     if current_video_file:
# # # # # #         os.remove(current_video_file)
    
# # # # # #     # Create a new file for the video
# # # # # #     file_name = "video"
# # # # # #     file_stream, file_path = create_file_write_stream(file_name)

# # # # # #     file_stream.write(data)
# # # # # #     file_stream.close()
# # # # # #     print(f"File {file_path} has been saved.")
    
# # # # # #     current_video_file = file_path

# # # # # # @socketio.on('disconnect')
# # # # # # def handle_disconnect():
# # # # # #     print('Client disconnected')

# # # # # # @socketio.on_error() # Handles the default namespace
# # # # # # def error_handler(e):
# # # # # #     print(f"Socket error: {e}")

# # # # # # if __name__ == '__main__':
# # # # # #     socketio.run(app, port=5000, host='0.0.0.0', debug=True)

# # # # # # -----------------------------------------------------------------------------------------------------
# # # # # # from flask import Flask, request
# # # # # # from flask_cors import CORS
# # # # # # from flask_socketio import SocketIO
# # # # # # import os
# # # # # # import time
# # # # # # import threading
# # # # # # import shutil

# # # # # # app = Flask(__name__)
# # # # # # CORS(app)  # Allow all routes in CORS
# # # # # # socketio = SocketIO(app, cors_allowed_origins="*",ping_timeout=20000,max_http_buffer_size=10 * 1024 * 1024, ping_interval=25000)

# # # # # # # Directory to store video files
# # # # # # video_dir = os.path.join(os.path.dirname(__file__), 'videos')
# # # # # # if not os.path.exists(video_dir):
# # # # # #     os.makedirs(video_dir)

# # # # # # def clear_video_directory():
# # # # # #     """Clears the contents of the video directory."""
# # # # # #     for filename in os.listdir(video_dir):
# # # # # #         file_path = os.path.join(video_dir, filename)
# # # # # #         try:
# # # # # #             if os.path.isfile(file_path) or os.path.islink(file_path):
# # # # # #                 os.unlink(file_path)
# # # # # #             elif os.path.isdir(file_path):
# # # # # #                 shutil.rmtree(file_path)
# # # # # #         except Exception as e:
# # # # # #             print(f"Failed to delete {file_path}. Reason: {e}")

# # # # # # # Clear the video directory initially
# # # # # # clear_video_directory()

# # # # # # # A dictionary to hold the file streams and locks for each client
# # # # # # client_files = {}
# # # # # # client_locks = {}

# # # # # # @socketio.on('connect')
# # # # # # def handle_connect():
# # # # # #     print("hjnhjbc")
# # # # # #     client_id = request.sid
# # # # # #     clear_video_directory()  # Clear the video directory before creating a new file
# # # # # #     file_name = f"video_{client_id}_{int(time.time() * 1000)}.webm"
# # # # # #     file_path = os.path.join(video_dir, file_name)
# # # # # #     file_stream = open(file_path, 'wb')
# # # # # #     client_files[client_id] = file_stream
# # # # # #     client_locks[client_id] = threading.Lock()
# # # # # #     print(f"Client {client_id} connected, previous files deleted, and new file {file_path} created.")

# # # # # # @socketio.on('video_data')
# # # # # # def handle_video_data(data):
# # # # # #     client_id = request.sid
# # # # # #     if client_id in client_files:
# # # # # #         with client_locks[client_id]:
# # # # # #             client_files[client_id].write(data)
# # # # # #             print(f"Received data from client {client_id}.")

# # # # # # @socketio.on('disconnect')
# # # # # # def handle_disconnect():
# # # # # #     client_id = request.sid
# # # # # #     if client_id in client_files:
# # # # # #         with client_locks[client_id]:
# # # # # #             client_files[client_id].close()
# # # # # #         del client_files[client_id]
# # # # # #         del client_locks[client_id]
# # # # # #         print(f"Client {client_id} disconnected and file closed.")

# # # # # # @socketio.on_error()  # Handles the default namespace
# # # # # # def error_handler(e):
# # # # # #     print(f"Socket error: {e}")

# # # # # # if __name__ == '__main__':
# # # # # #     socketio.run(app, port=5000, host='0.0.0.0', debug=True)

# # # # # # --------------------------------------------------------------------------------------
# # # # # # from flask import Flask, request, jsonify
# # # # # # from flask_cors import CORS
# # # # # # from flask_socketio import SocketIO
# # # # # # import os
# # # # # # import threading
# # # # # # import cv2
# # # # # # import face_recognition

# # # # # # app = Flask(__name__)
# # # # # # CORS(app)  # Allow all routes in CORS
# # # # # # socketio = SocketIO(app, cors_allowed_origins="*", ping_timeout=6000)

# # # # # # # Directory to store video files
# # # # # # video_dir = os.path.join(os.path.dirname(__file__), 'videos')
# # # # # # if not os.path.exists(video_dir):
# # # # # #     os.makedirs(video_dir)

# # # # # # # A dictionary to hold the file paths for each client
# # # # # # client_files = {}

# # # # # # def create_file_write_stream(client_id):
# # # # # #     file_path = os.path.join(video_dir, f"video_{client_id}.webm")
# # # # # #     return file_path

# # # # # # def detect_face_match(video_path):
# # # # # #     known_face = face_recognition.load_image_file("C:/Users/OMR-09/Pictures/img2.jpg")
# # # # # #     known_face_encoding = face_recognition.face_encodings(known_face)[0]

# # # # # #     cap = cv2.VideoCapture(video_path)
# # # # # #     face_locations = []
# # # # # #     face_encodings = []
# # # # # #     matches = []

# # # # # #     while True:
# # # # # #         ret, frame = cap.read()

# # # # # #         if not ret:
# # # # # #             break

# # # # # #         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# # # # # #         face_locations = face_recognition.face_locations(rgb_frame)
# # # # # #         face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

# # # # # #         for face_encoding in face_encodings:
# # # # # #             match = face_recognition.compare_faces([known_face_encoding], face_encoding)
# # # # # #             matches.append(match[0])

# # # # # #     cap.release()

# # # # # #     if True in matches:
# # # # # #         return True
# # # # # #     else:
# # # # # #         return False

# # # # # # @socketio.on('connect')
# # # # # # def handle_connect():
# # # # # #     client_id = request.sid
# # # # # #     client_files[client_id] = create_file_write_stream(client_id)
# # # # # #     print(f"Client {client_id} connected.")

# # # # # # @socketio.on('video_data')
# # # # # # def handle_video_data(data):
# # # # # #     client_id = request.sid
# # # # # #     file_path = client_files.get(client_id)
# # # # # #     if file_path:
# # # # # #         with open(file_path, 'ab') as f:
# # # # # #             f.write(data)
# # # # # #             print(f"Received data from client {client_id}.")

# # # # # # @socketio.on('disconnect')
# # # # # # def handle_disconnect():
# # # # # #     client_id = request.sid
# # # # # #     file_path = client_files.pop(client_id, None)
# # # # # #     if file_path and os.path.exists(file_path):
# # # # # #         result = detect_face_match(file_path)
# # # # # #         os.remove(file_path)  # Delete the video file after processing
# # # # # #         print(f"Client {client_id} disconnected and file removed.")
# # # # # #         # socketio.emit('face_match_result', {'client_id': client_id, 'match': result})
# # # # # #         socketio.emit('result', {'client_id': client_id, 'match': result})

# # # # # # @socketio.on_error()  # Handles the default namespace
# # # # # # def error_handler(e):
# # # # # #     print(f"Socket error: {e}")

# # # # # # if __name__ == '__main__':
# # # # # #     socketio.run(app, port=5000, host='0.0.0.0', debug=True)


# # # # # # -----------------------------------------------------------------------------------------------------
# # # # # # # # # from flask import Flask, request
# # # # # # # # # from flask_cors import CORS
# # # # # # # # # from flask_socketio import SocketIO


# # # # # # # # # import os
# # # # # # # # # import time
# # # # # # # # # import threading
# # # # # # # # # import shutil
# # # # # # # # # import cv2
# # # # # # # # # import numpy as np
# # # # # # # # # import face_recognition

# # # # # # # # # app = Flask(__name__)
# # # # # # # # # CORS(app)  # Allow all routes in CORS
# # # # # # # # # socketio = SocketIO(app, cors_allowed_origins="*", ping_timeout=20000, max_http_buffer_size=10 * 1024 * 1024, ping_interval=25000)

# # # # # # # # # # Directory to store video files
# # # # # # # # # video_dir = "videos/video_4e35PO1iowKqtr74AAAB_1715947036449.webm"
# # # # # # # # # if not os.path.exists(video_dir):
# # # # # # # # #     os.makedirs(video_dir)

# # # # # # # # # KNOWN_IMAGE_PATH = "C:/Users/OMR-09/Pictures/img2.jpg"

# # # # # # # # # def create_file_write_stream(file_name):
# # # # # # # # #     file_path = os.path.join(video_dir, f"{file_name}.webm")
# # # # # # # # #     file_stream = open(file_path, 'wb')
# # # # # # # # #     return file_stream, file_path

# # # # # # # # # def clear_video_directory():
# # # # # # # # #     """Clears the contents of the video directory."""
# # # # # # # # #     for filename in os.listdir(video_dir):
# # # # # # # # #         file_path = os.path.join(video_dir, filename)
# # # # # # # # #         try:
# # # # # # # # #             if os.path.isfile(file_path) or os.path.islink(file_path):
# # # # # # # # #                 os.unlink(file_path)
# # # # # # # # #             elif os.path.isdir(file_path):
# # # # # # # # #                 shutil.rmtree(file_path)
# # # # # # # # #         except Exception as e:
# # # # # # # # #             print(f"Failed to delete {file_path}. Reason: {e}")

# # # # # # # # # # Clear the video directory initially
# # # # # # # # # clear_video_directory()

# # # # # # # # # # A dictionary to hold the file streams and locks for each client
# # # # # # # # # client_files = {}
# # # # # # # # # client_locks = {}

# # # # # # # # # def detect_person_match(video_path):
# # # # # # # # #     # Load the known image
# # # # # # # # #     known_image = face_recognition.load_image_file(KNOWN_IMAGE_PATH)
# # # # # # # # #     known_encoding = face_recognition.face_encodings(known_image)[0]

# # # # # # # # #     # Open the video capture
# # # # # # # # #     cap = cv2.VideoCapture(video_path)

# # # # # # # # #     # Create a background subtractor object
# # # # # # # # #     bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=True)

# # # # # # # # #     # Initialize variables for face detection and background movement detection
# # # # # # # # #     face_locations = []

# # # # # # # # #     # Iterate over frames
# # # # # # # # #     while True:
# # # # # # # # #         ret, frame = cap.read()
# # # # # # # # #         if not ret:
# # # # # # # # #             break

# # # # # # # # #         # Apply background subtraction to detect movement in the background
# # # # # # # # #         fg_mask = bg_subtractor.apply(frame)

# # # # # # # # #         # Obtain the shadow value from the background subtractor
# # # # # # # # #         shadow_value = bg_subtractor.getShadowValue()

# # # # # # # # #         # Invert the shadow mask
# # # # # # # # #         fg_mask[fg_mask == shadow_value] = 0

# # # # # # # # #         # Find contours of moving objects
# # # # # # # # #         contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # # # # # # # #         # Check if any contours (movement) are detected
# # # # # # # # #         background_movement = False
# # # # # # # # #         for contour in contours:
# # # # # # # # #             area = cv2.contourArea(contour)
# # # # # # # # #             if area > 1000:  # Adjust threshold as needed
# # # # # # # # #                 background_movement = True
# # # # # # # # #                 break

# # # # # # # # #         if background_movement:
# # # # # # # # #             cap.release()
# # # # # # # # #             return "Movement"

# # # # # # # # #         # If no significant background movement is detected, check for face match
# # # # # # # # #         if not background_movement:
# # # # # # # # #             # Find face locations in the frame
# # # # # # # # #             face_locations = face_recognition.face_locations(frame)

# # # # # # # # #             # Check for more than one face detected or not match
# # # # # # # # #             if len(face_locations) > 1:
# # # # # # # # #                 cap.release()
# # # # # # # # #                 return "More than One Face Detected"

# # # # # # # # #             # Compare face encoding in the frame with the known face encoding
# # # # # # # # #             if len(face_locations) == 1:
# # # # # # # # #                 face_encoding = face_recognition.face_encodings(frame, face_locations)[0]
# # # # # # # # #                 match = face_recognition.compare_faces([known_encoding], face_encoding)
# # # # # # # # #                 if match[0]:
# # # # # # # # #                     cap.release()
# # # # # # # # #                     return "Match"

# # # # # # # # #             # Detect neck bending
# # # # # # # # #             if len(face_locations) == 1:
# # # # # # # # #                 face_landmarks = face_recognition.face_landmarks(frame, face_locations)
# # # # # # # # #                 if face_landmarks:
# # # # # # # # #                     neck_angle = detect_neck_bending(face_landmarks[0])
# # # # # # # # #                     if neck_angle > 130 or neck_angle < 125:  # Adjust threshold as needed
# # # # # # # # #                         cap.release()
# # # # # # # # #                         return "Neck Bending Detected"

# # # # # # # # #     # If no significant background movement or face match is detected, return Not Match
# # # # # # # # #     cap.release()
# # # # # # # # #     return "Not Match"

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
# # # # # # # # #                                   (np.linalg.norm(neck_vector) * np.linalg.norm(face_vector))))
# # # # # # # # #     return angle

# # # # # # # # # @socketio.on('connect')
# # # # # # # # # def handle_connect():
# # # # # # # # #     print("Client connected")
# # # # # # # # #     client_id = request.sid
# # # # # # # # #     clear_video_directory()  # Clear the video directory before creating a new file
# # # # # # # # #     file_name = f"video_{client_id}_{int(time.time() * 1000)}"
# # # # # # # # #     file_stream, file_path = create_file_write_stream(file_name)
# # # # # # # # #     client_files[client_id] = file_stream
# # # # # # # # #     client_locks[client_id] = threading.Lock()
# # # # # # # # #     print(f"Client {client_id} connected, previous files deleted, and new file {file_path} created.")

# # # # # # # # # @socketio.on('video_data')
# # # # # # # # # def handle_video_data(data):
# # # # # # # # #     client_id = request.sid
# # # # # # # # #     if client_id in client_files:
# # # # # # # # #         with client_locks[client_id]:
# # # # # # # # #             client_files[client_id].write(data)
# # # # # # # # #             print(f"Received data from client {client_id}.")

# # # # # # # # # @socketio.on('disconnect')
# # # # # # # # # def handle_disconnect():
# # # # # # # # #     client_id = request.sid
# # # # # # # # #     if client_id in client_files:
# # # # # # # # #         with client_locks[client_id]:
# # # # # # # # #             client_files[client_id].close()
# # # # # # # # #         del client_files[client_id]
# # # # # # # # #         del client_locks[client_id]
# # # # # # # # #         print(f"Client {client_id} disconnected and file closed.")
# # # # # # # # #         # Call detect_person_match function when streaming is complete
# # # # # # # # #         video_path = os.path.join(video_dir, f"video_{client_id}_*.webm")
# # # # # # # # #         result = detect_person_match(video_path)
# # # # # # # # #         print("Result:", result)
# # # # # # # # #         # Emit the result back to the client
# # # # # # # # #         socketio.emit('result', result, room=client_id)

# # # # # # # # # @socketio.on_error()  # Handles the default namespace
# # # # # # # # # def error_handler(e):
# # # # # # # # #     print(f"Socket error: {e}")

# # # # # # # # # if __name__ == '__main__':
# # # # # # # # #     socketio.run(app, port=5000, host='0.0.0.0', debug=True)

# # # # # # # ---------------------------------------------------------------------------------------------------------------------------

# # # # # # # from keras.models import load_model
# # # # # # # from keras.preprocessing.image import img_to_array
# # # # # # # import cv2
# # # # # # # import numpy as np
# # # # # # # from time import sleep
# # # # # # # from collections import Counter

# # # # # # # # Load the face detection and emotion classifier models
# # # # # # # face_classifier = cv2.CascadeClassifier(r'C:\Users\OMR-09\Downloads\Emotion_Detection_CNN-main-20240518T033125Z-001\Emotion_Detection_CNN-main\Emotion_Detection_CNN-main\haarcascade_frontalface_default.xml')
# # # # # # # classifier = load_model(r'C:\Users\OMR-09\Downloads\Emotion_Detection_CNN-main-20240518T033125Z-001\Emotion_Detection_CNN-main\Emotion_Detection_CNN-main\model.h5')

# # # # # # # # Emotion labels
# # # # # # # emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# # # # # # # # Path to the video file
# # # # # # # video_path = r'C:\Users\OMR-09\Desktop\new\videos\video_chunk_1716007537948.webm'

# # # # # # # # Start video capture from the file
# # # # # # # cap = cv2.VideoCapture(video_path)

# # # # # # # # Dictionary to keep count of each emotion
# # # # # # # emotion_counter = Counter()

# # # # # # # while cap.isOpened():
# # # # # # #     ret, frame = cap.read()
# # # # # # #     if not ret:
# # # # # # #         break  # Exit the loop if no frame is captured

# # # # # # #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# # # # # # #     faces = face_classifier.detectMultiScale(gray)

# # # # # # #     for (x, y, w, h) in faces:
# # # # # # #         roi_gray = gray[y:y+h, x:x+w]
# # # # # # #         roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

# # # # # # #         if np.sum([roi_gray]) != 0:
# # # # # # #             roi = roi_gray.astype('float') / 255.0
# # # # # # #             roi = img_to_array(roi)
# # # # # # #             roi = np.expand_dims(roi, axis=0)

# # # # # # #             prediction = classifier.predict(roi)[0]
# # # # # # #             label = emotion_labels[prediction.argmax()]
# # # # # # #             emotion_counter[label] += 1  # Increment the count for the detected emotion

# # # # # # #     # sleep(5)  # Wait for 5 seconds before the next detection

# # # # # # #     if cv2.waitKey(1) & 0xFF == ord('q'):
# # # # # # #         break

# # # # # # # cap.release()
# # # # # # # cv2.destroyAllWindows()

# # # # # # # # Find the most common emotion
# # # # # # # most_common_emotion = emotion_counter.most_common(1)
# # # # # # # if most_common_emotion:
# # # # # # #     print("Most frequently detected emotion:", most_common_emotion[0][0])
# # # # # # # else:
# # # # # # #     print("No emotions detected.")



# # # # # # # # --------------------------------------------------------------------------------------------------------------------------?.....camera 
# # # # # # # # from keras.models import load_model
# # # # # # # # from time import sleep
# # # # # # # # from keras.preprocessing.image import img_to_array
# # # # # # # # from keras.preprocessing import image
# # # # # # # # import cv2
# # # # # # # # import numpy as np

# # # # # # # # face_classifier = cv2.CascadeClassifier(r'C:\Users\OMR-09\Downloads\Emotion_Detection_CNN-main-20240518T033125Z-001\Emotion_Detection_CNN-main\Emotion_Detection_CNN-main\haarcascade_frontalface_default.xml')
# # # # # # # # classifier =load_model(r'C:\Users\OMR-09\Downloads\Emotion_Detection_CNN-main-20240518T033125Z-001\Emotion_Detection_CNN-main\Emotion_Detection_CNN-main\model.h5')

# # # # # # # # emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

# # # # # # # # cap = cv2.VideoCapture(0)


# # # # # # # # while True:
# # # # # # # #     _, frame = cap.read()
# # # # # # # #     labels = []
# # # # # # # #     gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
# # # # # # # #     faces = face_classifier.detectMultiScale(gray)

# # # # # # # #     for (x,y,w,h) in faces:
# # # # # # # #         cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
# # # # # # # #         roi_gray = gray[y:y+h,x:x+w]
# # # # # # # #         roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)



# # # # # # # #         if np.sum([roi_gray])!=0:
# # # # # # # #             roi = roi_gray.astype('float')/255.0
# # # # # # # #             roi = img_to_array(roi)
# # # # # # # #             roi = np.expand_dims(roi,axis=0)

# # # # # # # #             prediction = classifier.predict(roi)[0]
# # # # # # # #             label=emotion_labels[prediction.argmax()]
# # # # # # # #             label_position = (x,y)
# # # # # # # #             cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
# # # # # # # #         else:
# # # # # # # #             cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
# # # # # # # #     cv2.imshow('Emotion Detector',frame)
# # # # # # # #     if cv2.waitKey(1) & 0xFF == ord('q'):
# # # # # # # #         break

# # # # # # # # cap.release()
# # # # # # # # cv2.destroyAllWindows()
# # # # # # # ---------------------------------------------------------------------------------------------------------------

# # # # # # # # Import necessary libraries
# # # # # # # from flask import Flask, request, jsonify
# # # # # # # from flask_ngrok import run_with_ngrok
# # # # # # # import cv2
# # # # # # # from flask_cors import CORS
# # # # # # # from pyngrok import ngrok
# # # # # # # import numpy as np
# # # # # # # import face_recognition
# # # # # # # from keras.models import load_model
# # # # # # # from keras.preprocessing.image import img_to_array
# # # # # # # import cv2
# # # # # # # import numpy as np
# # # # # # # from time import sleep
# # # # # # # from collections import Counter

# # # # # # # # Initialize Flask app
# # # # # # # port_no = 5000
# # # # # # # app = Flask(__name__)
# # # # # # # ngrok.set_auth_token("2gDVBMbJ3zF6Fdccaicxr3QIzbu_7ho4uhZoAUaNg5MAQeAob")
# # # # # # # public_url = ngrok.connect(port_no).public_url
# # # # # # # CORS(app)
# # # # # # # run_with_ngrok(app)

# # # # # # # KNOWN_IMAGE_PATH = "C:/Users/OMR-09/Downloads/IMG20240502144421.jpg"

# # # # # # # face_classifier = cv2.CascadeClassifier("C:/Users/OMR-09/Downloads/Emotion_Detection_CNN-main-20240518T033125Z-001/Emotion_Detection_CNN-main\Emotion_Detection_CNN-main/haarcascade_frontalface_default.xml")
# # # # # # # classifier = load_model("C:/Users/OMR-09/Downloads/Emotion_Detection_CNN-main-20240518T033125Z-001/Emotion_Detection_CNN-main\Emotion_Detection_CNN-main/model.h5")

# # # # # # # def emotion_fdetect(video_path):


# # # # # # #     # Emotion labels
# # # # # # #     emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# # # # # # #     # Start video capture from the file
# # # # # # #     cap = cv2.VideoCapture(video_path)

# # # # # # #     # Dictionary to keep count of each emotion
# # # # # # #     emotion_counter = Counter()

# # # # # # #     while cap.isOpened():
# # # # # # #         ret, frame = cap.read()
# # # # # # #         if not ret:
# # # # # # #             break  # Exit the loop if no frame is captured

# # # # # # #         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# # # # # # #         faces = face_classifier.detectMultiScale(gray)

# # # # # # #         for (x, y, w, h) in faces:
# # # # # # #             roi_gray = gray[y:y+h, x:x+w]
# # # # # # #             roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

# # # # # # #             if np.sum([roi_gray]) != 0:
# # # # # # #                 roi = roi_gray.astype('float') / 255.0
# # # # # # #                 roi = img_to_array(roi)
# # # # # # #                 roi = np.expand_dims(roi, axis=0)

# # # # # # #                 prediction = classifier.predict(roi)[0]
# # # # # # #                 label = emotion_labels[prediction.argmax()]
# # # # # # #                 emotion_counter[label] += 1  # Increment the count for the detected emotion

# # # # # # #         if cv2.waitKey(1) & 0xFF == ord('q'):
# # # # # # #             break

# # # # # # #     cap.release()
# # # # # # #     cv2.destroyAllWindows()

# # # # # # #     # Find the most common emotion
# # # # # # #     most_common_emotion = emotion_counter.most_common(1)
# # # # # # #     if most_common_emotion:
# # # # # # #         return most_common_emotion[0][0]
# # # # # # #     else:
# # # # # # #         return "No emotions detected."

# # # # # # # def detect_person_match(video_path):
# # # # # # #     # Load the known image
# # # # # # #     known_image = face_recognition.load_image_file(KNOWN_IMAGE_PATH)
# # # # # # #     known_encoding = face_recognition.face_encodings(known_image)[0]

# # # # # # #     # Open the video capture
# # # # # # #     cap = cv2.VideoCapture(video_path)

# # # # # # #     # Find face locations and encodings in the first frame
# # # # # # #     ret, frame = cap.read()
# # # # # # #     if not ret:
# # # # # # #         cap.release()
# # # # # # #         return "Video Capture Error"

# # # # # # #     face_locations = face_recognition.face_locations(frame)
# # # # # # #     face_count = len(face_locations)

# # # # # # #     # Check for no face detected
# # # # # # #     if face_count == 0:
# # # # # # #         cap.release()
# # # # # # #         return "No Face Detected"

# # # # # # #     # Check for more than one face detected or not match
# # # # # # #     if face_count > 1:
# # # # # # #         cap.release()
# # # # # # #         return "More than One Face Detected"

# # # # # # #     # Compare face encodings in the frame with the known face encoding
# # # # # # #     face_encodings = face_recognition.face_encodings(frame, face_locations)
# # # # # # #     match = face_recognition.compare_faces([known_encoding], face_encodings[0])
# # # # # # #     if match[0]:
# # # # # # #         # Detect neck bending
# # # # # # #         neck_bending = detect_neck_bending(frame, face_locations[0])
# # # # # # #         if neck_bending:
# # # # # # #             # If both face match and neck bending are detected, release the video capture and return Neck Movement
# # # # # # #             cap.release()
# # # # # # #             return "Neck Movement"
# # # # # # #         else:
# # # # # # #             # If only face match is detected, release the video capture and return Match
# # # # # # #             cap.release()
# # # # # # #             return "Match"
# # # # # # #     else:
# # # # # # #         # If face does not match, release the video capture and return Not Match
# # # # # # #         cap.release()
# # # # # # #         return "Not Match"

# # # # # # # def detect_neck_bending(frame, face_location):
# # # # # # #     # Get face landmarks
# # # # # # #     face_landmarks = face_recognition.face_landmarks(frame, [face_location])[0]

# # # # # # #     # Extract required landmarks
# # # # # # #     top_nose = face_landmarks['nose_bridge'][0]
# # # # # # #     bottom_nose = face_landmarks['nose_tip'][0]
# # # # # # #     top_chin = face_landmarks['chin'][8]
# # # # # # #     bottom_chin = face_landmarks['chin'][0]

# # # # # # #     # Calculate vectors for neck and face
# # # # # # #     neck_vector = np.array(bottom_chin) - np.array(top_chin)
# # # # # # #     face_vector = np.array(bottom_nose) - np.array(top_nose)

# # # # # # #     # Calculate angle between neck and face vectors
# # # # # # #     angle = np.degrees(np.arccos(np.dot(neck_vector, face_vector) /
# # # # # # #                                   (np.linalg.norm(neck_vector) * np.linalg.norm(face_vector))))

# # # # # # #     # Adjust threshold as needed
# # # # # # #     return angle > 130 or angle < 125

# # # # # # # @app.route('/match_person', methods=['POST'])
# # # # # # # def match_person():
# # # # # # #     if 'video' not in request.files:
# # # # # # #         return jsonify({'error': 'Missing video file'})

# # # # # # #     video_file = request.files['video']

# # # # # # #     if video_file.filename == '':
# # # # # # #         return jsonify({'error': 'No selected file'})

# # # # # # #     # Save the video file temporarily
# # # # # # #     video_path = 'temp_video.mp4'
# # # # # # #     video_file.save(video_path)

# # # # # # #     # Detect person match in the video with the known image
# # # # # # #     result = detect_person_match(video_path)
# # # # # # #     emotion = emotion_fdetect(video_path)
# # # # # # #     print(emotion)
# # # # # # #     return jsonify({'result': result, 'emotion': emotion})
# # # # # # #     # return jsonify({'emotion':emotion})

# # # # # # # if __name__ == '__main__':
# # # # # # #     app.run()

# # # # # # # -------------------------------------------------------------------------------------------------------------------------------
# # # # # # from flask import Flask, request, jsonify
# # # # # # from flask_cors import CORS
# # # # # # from flask_socketio import SocketIO
# # # # # # import os
# # # # # # import time
# # # # # # import shutil
# # # # # # import cv2
# # # # # # import numpy as np
# # # # # # import face_recognition
# # # # # # from keras.models import load_model
# # # # # # from keras.preprocessing.image import img_to_array
# # # # # # from collections import Counter

# # # # # # app = Flask(__name__)
# # # # # # CORS(app)  # Allow all routes in CORS
# # # # # # socketio = SocketIO(app, cors_allowed_origins="*", ping_timeout=20000, max_http_buffer_size=10 * 1024 * 1024, ping_interval=25000)

# # # # # # # Define constants
# # # # # # KNOWN_IMAGE_PATH = "C:/Users/OMR-09/Pictures/IMG20240502144421.jpg"
# # # # # # HAARCASCADE_XML_PATH = "C:/Users/OMR-09/Downloads/Emotion_Detection_CNN-main-20240518T033125Z-001/Emotion_Detection_CNN-main/Emotion_Detection_CNN-main/haarcascade_frontalface_default.xml"
# # # # # # MODEL_H5_PATH = "C:/Users/OMR-09/Downloads/Emotion_Detection_CNN-main-20240518T033125Z-001/Emotion_Detection_CNN-main/Emotion_Detection_CNN-main/model.h5"
# # # # # # EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
# # # # # # VIDEO_DIR = os.path.join(os.path.dirname(__file__), 'videos')

# # # # # # # Clear the video directory initially
# # # # # # def clear_video_directory():
# # # # # #     """Clears the contents of the video directory."""
# # # # # #     for filename in os.listdir(VIDEO_DIR):
# # # # # #         file_path = os.path.join(VIDEO_DIR, filename)
# # # # # #         try:
# # # # # #             if os.path.isfile(file_path) or os.path.islink(file_path):
# # # # # #                 os.unlink(file_path)
# # # # # #             elif os.path.isdir(file_path):
# # # # # #                 shutil.rmtree(file_path)
# # # # # #         except Exception as e:
# # # # # #             print(f"Failed to delete {file_path}. Reason: {e}")

# # # # # # clear_video_directory()

# # # # # # # Load the ML models
# # # # # # face_classifier = cv2.CascadeClassifier(HAARCASCADE_XML_PATH)
# # # # # # classifier = load_model(MODEL_H5_PATH)

# # # # # # def emotion_fdetect(video_path):
# # # # # #     """Detects emotions in a video file."""
# # # # # #     emotion_counter = Counter()
# # # # # #     cap = cv2.VideoCapture(video_path)
# # # # # #     while cap.isOpened():
# # # # # #         ret, frame = cap.read()
# # # # # #         if not ret:
# # # # # #             break
# # # # # #         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# # # # # #         faces = face_classifier.detectMultiScale(gray)
# # # # # #         for (x, y, w, h) in faces:
# # # # # #             roi_gray = gray[y:y+h, x:x+w]
# # # # # #             roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
# # # # # #             if np.sum([roi_gray]) != 0:
# # # # # #                 roi = roi_gray.astype('float') / 255.0
# # # # # #                 roi = img_to_array(roi)
# # # # # #                 roi = np.expand_dims(roi, axis=0)
# # # # # #                 prediction = classifier.predict(roi)[0]
# # # # # #                 label = EMOTION_LABELS[prediction.argmax()]
# # # # # #                 emotion_counter[label] += 1
# # # # # #     cap.release()
# # # # # #     cv2.destroyAllWindows()
# # # # # #     most_common_emotion = emotion_counter.most_common(1)
# # # # # #     if most_common_emotion:
# # # # # #         return most_common_emotion[0][0]
# # # # # #     else:
# # # # # #         return "No emotions detected."

# # # # # # def detect_person_match(video_path):
# # # # # #     """Compares faces in a video file with a known face."""
# # # # # #     known_image = face_recognition.load_image_file(KNOWN_IMAGE_PATH)
# # # # # #     known_encoding = face_recognition.face_encodings(known_image)[0]
# # # # # #     cap = cv2.VideoCapture(video_path)
# # # # # #     ret, frame = cap.read()
# # # # # #     if not ret:
# # # # # #         cap.release()
# # # # # #         return "Video Capture Error"
# # # # # #     face_locations = face_recognition.face_locations(frame)
# # # # # #     face_count = len(face_locations)
# # # # # #     if face_count == 0:
# # # # # #         cap.release()
# # # # # #         return "No Face Detected"
# # # # # #     if face_count > 1:
# # # # # #         cap.release()
# # # # # #         return "More than One Face Detected"
# # # # # #     face_encodings = face_recognition.face_encodings(frame, face_locations)
# # # # # #     match = face_recognition.compare_faces([known_encoding], face_encodings[0])
# # # # # #     if match[0]:
# # # # # #         neck_bending = detect_neck_bending(frame, face_locations[0])
# # # # # #         if neck_bending:
# # # # # #             cap.release()
# # # # # #             return "Neck Movement"
# # # # # #         else:
# # # # # #             cap.release()
# # # # # #             return "Match"
# # # # # #     else:
# # # # # #         cap.release()
# # # # # #         return "Not Match"

# # # # # # def detect_neck_bending(frame, face_location):
# # # # # #     """Detects neck bending based on face landmarks."""
# # # # # #     face_landmarks = face_recognition.face_landmarks(frame, [face_location])[0]
# # # # # #     top_nose = face_landmarks['nose_bridge'][0]
# # # # # #     bottom_nose = face_landmarks['nose_tip'][0]
# # # # # #     top_chin = face_landmarks['chin'][8]
# # # # # #     bottom_chin = face_landmarks['chin'][0]
# # # # # #     neck_vector = np.array(bottom_chin) - np.array(top_chin)
# # # # # #     face_vector = np.array(bottom_nose) - np.array(top_nose)
# # # # # #     angle = np.degrees(np.arccos(np.dot(neck_vector, face_vector) /
# # # # # #                                   (np.linalg.norm(neck_vector) * np.linalg.norm(face_vector))))
# # # # # #     return angle > 130 or angle < 125

# # # # # # @socketio.on('connect')
# # # # # # def handle_connect():
# # # # # #     """Handles client connection."""
# # # # # #     print("Client connected")
# # # # # #     clear_video_directory()
# # # # # #     print("Client connected and previous files deleted.")

# # # # # # @socketio.on('video_data')
# # # # # # def handle_video_data(data):
# # # # # #     """Handles received video data."""
# # # # # #     file_name = f"video_chunk_{int(time.time() * 1000)}.webm"
# # # # # #     file_path = os.path.join(VIDEO_DIR, file_name)
# # # # # #     print(file_path)
# # # # # #     with open(file_path, 'wb') as file:
# # # # # #         file.write(data)
# # # # # #         print("Received video data.")
# # # # # #     result = detect_person_match(file_path)
# # # # # #     emotion = emotion_fdetect(file_path)
# # # # # #     print(f"Result: {result}, Emotion: {emotion}")
# # # # # #     socketio.emit('result', {'result': result, 'emotion': emotion})

# # # # # # @socketio.on('disconnect')
# # # # # # def handle_disconnect():
# # # # # #     """Handles client disconnection."""
# # # # # #     print("Client disconnected")

# # # # # # @socketio.on_error() 
# # # # # # def error_handler(e):
# # # # # #     """Handles socket errors."""
# # # # # #     print(f"Socket error: {e}")

# # # # # # if __name__ == '__main__':
# # # # # #     socketio.run(app, port=5000, host='0.0.0.0', debug=True)
# # # # # # ----------------------------------------------------------------------------------------------------------------------------------------------------
# # # # # Import necessary libraries ------- chl rha hai par video aman sir se leni hai
# # # # # from flask import Flask, request, jsonify
# # # # # from flask_cors import CORS
# # # # # from flask_socketio import SocketIO
# # # # # import os
# # # # # import time
# # # # # import shutil
# # # # # import cv2
# # # # # import numpy as np
# # # # # import face_recognition
# # # # # from keras.models import load_model
# # # # # from keras.preprocessing.image import img_to_array
# # # # # from collections import Counter

# # # # # app = Flask(__name__)
# # # # # CORS(app)  # Allow all routes in CORS
# # # # # socketio = SocketIO(app, cors_allowed_origins="*", ping_timeout=20000, max_http_buffer_size=10 * 1024 * 1024, ping_interval=25000)

# # # # # video_dir = os.path.join(os.path.dirname(__file__), 'videos')
# # # # # if not os.path.exists(video_dir):
# # # # #     os.makedirs(video_dir)


# # # # # def create_file_write_stream(file_name):
# # # # #     file_path = os.path.join(video_dir, f"{file_name}.webm")
# # # # #     file_stream = open(file_path, 'wb')
# # # # #     return file_stream, file_path

# # # # # current_video_file = None


# # # # # file_counter = 1

# # # # # @socketio.on('video_data')
# # # # # def handle_video_data(data):
# # # # #     global current_video_file
    
# # # # #     # Delete previous video file if exists
# # # # #     if current_video_file:
# # # # #         os.remove(current_video_file)
    
# # # # #     # Create a new file for the video
# # # # #     file_name = "video"
# # # # #     file_stream, file_path = create_file_write_stream(file_name)

# # # # #     file_stream.write(data)
# # # # #     file_stream.close()
# # # # #     print(f"File {file_path} has been saved.")
    
# # # # #     current_video_file = file_path
# # # # #     # Trigger emotion detection and face recognition processes
# # # # #     # emotion = emotion_fdetect(file_path)
# # # # #     result = detect_person_match(file_path)
    
# # # # #     # print(emotion)
# # # # #     print(result)
# # # # #     time.sleep(30)
# # # # #     # socketio.emit('emotion', emotion)
# # # # #     socketio.emit('result', result)
    
    
# # # # #     time.sleep(30)
    
# # # # # KNOWN_IMAGE_PATH = "C:/Users/OMR-09/Pictures/IMG20240502144421.jpg" # Change this to the path of your known person image
# # # # # face_classifier = cv2.CascadeClassifier("C:/Users/OMR-09/Downloads/Emotion_Detection_CNN-main-20240518T033125Z-001/Emotion_Detection_CNN-main/Emotion_Detection_CNN-main/haarcascade_frontalface_default.xml")
# # # # # classifier = load_model("C:/Users/OMR-09/Downloads/Emotion_Detection_CNN-main-20240518T033125Z-001/Emotion_Detection_CNN-main/Emotion_Detection_CNN-main/model.h5")

# # # # # def emotion_fdetect(video_path):
# # # # #     # Emotion labels
# # # # #     emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# # # # #     # Start video capture from the file
# # # # #     cap = cv2.VideoCapture(video_path)

# # # # #     # Dictionary to keep count of each emotion
# # # # #     emotion_counter = Counter()

# # # # #     while cap.isOpened():
# # # # #         ret, frame = cap.read()
# # # # #         if not ret:
# # # # #             break  # Exit the loop if no frame is captured

# # # # #         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# # # # #         faces = face_classifier.detectMultiScale(gray)

# # # # #         for (x, y, w, h) in faces:
# # # # #             roi_gray = gray[y:y+h, x:x+w]
# # # # #             roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

# # # # #             if np.sum([roi_gray]) != 0:
# # # # #                 roi = roi_gray.astype('float') / 255.0
# # # # #                 roi = img_to_array(roi)
# # # # #                 roi = np.expand_dims(roi, axis=0)

# # # # #                 prediction = classifier.predict(roi)[0]
# # # # #                 label = emotion_labels[prediction.argmax()]
# # # # #                 emotion_counter[label] += 1  # Increment the count for the detected emotion

# # # # #         if cv2.waitKey(1) & 0xFF == ord('q'):
# # # # #             break

# # # # #     cap.release()
# # # # #     cv2.destroyAllWindows()

# # # # #     # Find the most common emotion
# # # # #     most_common_emotion = emotion_counter.most_common(1)
# # # # #     if most_common_emotion:
# # # # #         return most_common_emotion[0][0]
# # # # #     else:
# # # # #         return "No emotions detected."


# # # # # def detect_person_match(video_path):
# # # # #     # Load the known image
# # # # #     known_image = face_recognition.load_image_file(KNOWN_IMAGE_PATH)
# # # # #     known_encoding = face_recognition.face_encodings(known_image)[0]

# # # # #     # Open the video capture
# # # # #     cap = cv2.VideoCapture(video_path)

# # # # #     # Find face locations and encodings in the first frame
# # # # #     ret, frame = cap.read()
# # # # #     if not ret:
# # # # #         cap.release()
# # # # #         return "Video Capture Error"

# # # # #     face_locations = face_recognition.face_locations(frame)
# # # # #     face_count = len(face_locations)

# # # # #     # Check for no face detected
# # # # #     if face_count == 0:
# # # # #         cap.release()
# # # # #         return "No Face Detected"

# # # # #     # Check for more than one face detected or not match
# # # # #     if face_count > 1:
# # # # #         cap.release()
# # # # #         return "More than One Face Detected"

# # # # #     # Compare face encodings in the frame with the known face encoding
# # # # #     face_encodings = face_recognition.face_encodings(frame, face_locations)
# # # # #     match = face_recognition.compare_faces([known_encoding], face_encodings[0])
# # # # #     if match[0]:
# # # # #         # Detect neck bending
# # # # #         neck_bending = detect_neck_bending(frame, face_locations[0])
# # # # #         if neck_bending:
# # # # #             # If both face match and neck bending are detected, release the video capture and return Neck Movement
# # # # #             cap.release()
# # # # #             return "Neck Movement"
# # # # #         else:
# # # # #             # If only face match is detected, release the video capture and return Match
# # # # #             cap.release()
# # # # #             return "Match"
# # # # #     else:
# # # # #         # If face does not match, release the video capture and return Not Match
# # # # #         cap.release()
# # # # #         return "Not Match"

# # # # # def detect_neck_bending(frame, face_location):
# # # # #     # Get face landmarks
# # # # #     face_landmarks = face_recognition.face_landmarks(frame, [face_location])[0]

# # # # #     # Extract required landmarks
# # # # #     top_nose = face_landmarks['nose_bridge'][0]
# # # # #     bottom_nose = face_landmarks['nose_tip'][0]
# # # # #     top_chin = face_landmarks['chin'][8]
# # # # #     bottom_chin = face_landmarks['chin'][0]

# # # # #     # Calculate vectors for neck and face
# # # # #     neck_vector = np.array(bottom_chin) - np.array(top_chin)
# # # # #     face_vector = np.array(bottom_nose) - np.array(top_nose)

# # # # #     # Calculate angle between neck and face vectors
# # # # #     angle = np.degrees(np.arccos(np.dot(neck_vector, face_vector) /
# # # # #                                   (np.linalg.norm(neck_vector) * np.linalg.norm(face_vector))))

# # # # #     # Adjust threshold as needed
# # # # #     return angle > 130 or angle < 125



    
# # # # # @socketio.on('connect')
# # # # # def handle_connect():
# # # # #     """Handles client connection."""
# # # # #     print("Client connected")
# # # # #     # video_path = "C:/Users/OMR-09/Desktop/new/NEW1/video_chunk_1716007537924.webm"
# # # # #     video_path = "C:/Users/OMR-09/Desktop/new/videos/video.webm"
# # # # #     # emotion = emotion_fdetect(video_path)
# # # # #     # result = (video_path)
# # # # #     result = detect_person_match(video_path)
# # # # #     # print(emotion)
# # # # #     print(result)
# # # # #     time.sleep(30)
# # # # #     # socketio.emit('emotion', emotion)
    
# # # # #     socketio.emit('result',result)
# # # # #     time.sleep(30)

    
# # # # # @socketio.on('disconnect')
# # # # # def handle_disconnect():
# # # # #     """Handles client disconnection."""
# # # # #     print("Client disconnected")

# # # # # @socketio.on_error() 
# # # # # def error_handler(e):
# # # # #     """Handles socket errors."""
# # # # #     print(f"Socket error: {e}")

# # # # # if __name__ == '__main__':
# # # # #     socketio.run(app, port=5000, host='0.0.0.0', debug=True)
# # # # #     # --------------------------------------------------------------------------------------------------------------------yaha tak
# # # # # # @app.route('/match_person', methods=['POST'])
# # # # # # def match_person():
# # # # # #     if 'video' not in request.files:
# # # # # #         return jsonify({'error': 'Missing video file'})

# # # # # #     video_file = request.files['video']

# # # # # #     if video_file.filename == '':
# # # # # #         return jsonify({'error': 'No selected file'})

# # # # # #     # Save the video file temporarily
# # # # # #     video_path = 'temp_video.mp4'
# # # # # #     video_file.save(video_path)

# # # # # #     # Detect person match in the video with the known image
# # # # # #     result = detect_person_match(video_path)
# # # # # #     emotion = emotion_fdetect(video_path)
# # # # # #     print(emotion)
# # # # # #     return jsonify({'result': result, 'emotion': emotion})
# # # # # #     # return jsonify({'emotion':emotion})

# # # # # if __name__ == '__main__':
# # # # #     app.run()

# # # # # # # -----------------------------------------------------------------------------------------------------------------------------------------------------
# # # # from flask import Flask, request
# # # # from flask_cors import CORS
# # # # from flask_socketio import SocketIO
# # # # import os
# # # # import time

# # # # app = Flask(__name__)
# # # # CORS(app)  # Allow all routes in CORS

# # # # socketio = SocketIO(app, cors_allowed_origins="*")

# # # # # Directory to store video files
# # # # video_dir = os.path.join(os.path.dirname(__file__), 'videos')
# # # # if not os.path.exists(video_dir):
# # # #     os.makedirs(video_dir)

# # # # def create_file_write_stream(file_name):
# # # #     file_path = os.path.join(video_dir, f"{file_name}.webm")
# # # #     file_stream = open(file_path, 'wb')
# # # #     return file_stream, file_path

# # # # @socketio.on('connect')
# # # # def handle_connect():
# # # #     print('Client connected')

# # # # @socketio.on('video_data')
# # # # def handle_video_data(data):
# # # #     file_name = f"video_{int(time.time() * 1000)}"
# # # #     file_stream, file_path = create_file_write_stream(file_name)

# # # #     file_stream.write(data)
# # # #     file_stream.close()
# # # #     print(f"File {file_path} has been saved.")

# # # # @socketio.on('disconnect')
# # # # def handle_disconnect():
# # # #     print('Client disconnected')

# # # # @socketio.on_error()  # Handles the default namespace
# # # # def error_handler(e):
# # # #     print(f"Socket error: {e}")

# # # # if __name__ == '__main__':
# # # #     socketio.run(app, port=5000)


# # # # --------------------------------------------------------------------------
# # # # Import necessary libraries
# # # from flask import Flask, request, jsonify
# # # from flask_ngrok import run_with_ngrok
# # # import cv2
# # # from flask_cors import CORS
# # # from pyngrok import ngrok
# # # import numpy as np
# # # import face_recognition
# # # from keras.models import load_model
# # # from keras.preprocessing.image import img_to_array
# # # import cv2
# # # import numpy as np
# # # from time import sleep
# # # from collections import Counter

# # # # Initialize Flask app
# # # port_no = 5000
# # # app = Flask(__name__)
# # # ngrok.set_auth_token("2gDVBMbJ3zF6Fdccaicxr3QIzbu_7ho4uhZoAUaNg5MAQeAob")
# # # public_url = ngrok.connect(port_no).public_url
# # # CORS(app)
# # # run_with_ngrok(app)

# # # KNOWN_IMAGE_PATH = "/content/WIN_20230528_23_11_07_Pro.jpg" # Change this to the path of your known person image
# # # face_classifier = cv2.CascadeClassifier("/content/haarcascade_frontalface_default.xml")
# # # classifier = load_model("/content/model.h5")

# # # def emotion_fdetect(video_path):


# # #     # Emotion labels
# # #     emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# # #     # Start video capture from the file
# # #     cap = cv2.VideoCapture(video_path)

# # #     # Dictionary to keep count of each emotion
# # #     emotion_counter = Counter()

# # #     while cap.isOpened():
# # #         ret, frame = cap.read()
# # #         if not ret:
# # #             break  # Exit the loop if no frame is captured

# # #         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# # #         faces = face_classifier.detectMultiScale(gray)

# # #         for (x, y, w, h) in faces:
# # #             roi_gray = gray[y:y+h, x:x+w]
# # #             roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

# # #             if np.sum([roi_gray]) != 0:
# # #                 roi = roi_gray.astype('float') / 255.0
# # #                 roi = img_to_array(roi)
# # #                 roi = np.expand_dims(roi, axis=0)

# # #                 prediction = classifier.predict(roi)[0]
# # #                 label = emotion_labels[prediction.argmax()]
# # #                 emotion_counter[label] += 1  # Increment the count for the detected emotion

# # #         if cv2.waitKey(1) & 0xFF == ord('q'):
# # #             break

# # #     cap.release()
# # #     cv2.destroyAllWindows()

# # #     # Find the most common emotion
# # #     most_common_emotion = emotion_counter.most_common(1)
# # #     if most_common_emotion:
# # #         return most_common_emotion[0][0]
# # #     else:
# # #         return "No emotions detected."

# # # def detect_person_match(video_path):
# # #     # Load the known image
# # #     known_image = face_recognition.load_image_file(KNOWN_IMAGE_PATH)
# # #     known_encoding = face_recognition.face_encodings(known_image)[0]

# # #     # Open the video capture
# # #     cap = cv2.VideoCapture(video_path)

# # #     # Find face locations and encodings in the first frame
# # #     ret, frame = cap.read()
# # #     if not ret:
# # #         cap.release()
# # #         return "Video Capture Error"

# # #     face_locations = face_recognition.face_locations(frame)
# # #     face_count = len(face_locations)

# # #     # Check for no face detected
# # #     if face_count == 0:
# # #         cap.release()
# # #         return "No Face Detected"

# # #     # Check for more than one face detected or not match
# # #     if face_count > 1:
# # #         cap.release()
# # #         return "More than One Face Detected"

# # #     # Compare face encodings in the frame with the known face encoding
# # #     face_encodings = face_recognition.face_encodings(frame, face_locations)
# # #     match = face_recognition.compare_faces([known_encoding], face_encodings[0])
# # #     if match[0]:
# # #         # Detect neck bending
# # #         neck_bending = detect_neck_bending(frame, face_locations[0])
# # #         if neck_bending:
# # #             # If both face match and neck bending are detected, release the video capture and return Neck Movement
# # #             cap.release()
# # #             return "Neck Movement"
# # #         else:
# # #             # If only face match is detected, release the video capture and return Match
# # #             cap.release()
# # #             return "Match"
# # #     else:
# # #         # If face does not match, release the video capture and return Not Match
# # #         cap.release()
# # #         return "Not Match"

# # # def detect_neck_bending(frame, face_location):
# # #     # Get face landmarks
# # #     face_landmarks = face_recognition.face_landmarks(frame, [face_location])[0]

# # #     # Extract required landmarks
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

# # #     # Adjust threshold as needed
# # #     return angle > 130 or angle < 125

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
# # #     emotion = emotion_fdetect(video_path)
# # #     print(emotion)
# # #     return jsonify({'result': result, 'emotion': emotion})
# # #     # return jsonify({'emotion':emotion})

# # # if __name__ == '__main__':
# # #     app.run()
# # #--------------------------------------------------------------------------------------------------------------------------------------------------------

# # from flask import Flask, request
# # from flask_cors import CORS
# # from flask_socketio import SocketIO
# # import os
# # import time
# # import ssl
# # import cv2
# # from OpenSSL import SSL
# # import numpy as np
# # from keras.models import load_model
# # from keras.preprocessing.image import img_to_array
# # import face_recognition
# # from collections import Counter
# # from threading import Thread

# # app = Flask(__name__)
# # CORS(app)
# # socketio = SocketIO(app, cors_allowed_origins="https://66545c64620e7f82814d91c7--velvety-meerkat-dbc2c1.netlify.app",ssl_context='adhoc',ping_timeout=20000, max_http_buffer_size=10 * 1024 * 1024, ping_interval=25000)

# # # Directory to store video files
# # video_dir = os.path.join(os.path.dirname(__file__), 'videos')
# # if not os.path.exists(video_dir):
# #     os.makedirs(video_dir)

# # # Load ML models with error handling
# # try:
# #     face_classifier = cv2.CascadeClassifier( "C:/Users/OMR-09/Downloads/Emotion_Detection_CNN-main-20240518T033125Z-001/Emotion_Detection_CNN-main/Emotion_Detection_CNN-main/haarcascade_frontalface_default.xml")
# #     emotion_classifier = load_model("C:/Users/OMR-09/Downloads/Emotion_Detection_CNN-main-20240518T033125Z-001/Emotion_Detection_CNN-main/Emotion_Detection_CNN-main/model.h5")
# #     emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
# #     KNOWN_IMAGE_PATH = "C:/Users/OMR-09/Pictures/img2.jpg"  # Change this to the path of your known person image
# # except Exception as e:
# #     print(f"Error loading models: {e}")
# #     exit(1)

# # def create_file_write_stream(file_name):
# #     try:
# #         file_path = os.path.join(video_dir, f"{file_name}.webm")
# #         file_stream = open(file_path, 'wb')
# #         return file_stream, file_path
# #     except Exception as e:
# #         print(f"Error creating file write stream: {e}")
# #         return None, None

# # def emotion_fdetect(video_path):
# #     emotion_counter = Counter()
# #     cap = cv2.VideoCapture(video_path)

# #     if not cap.isOpened():
# #         return "Video Capture Error"

# #     while cap.isOpened():
# #         ret, frame = cap.read()
# #         if not ret:
# #             break

# #         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# #         faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# #         for (x, y, w, h) in faces:
# #             roi_gray = gray[y:y+h, x:x+w]
# #             roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

# #             if np.sum([roi_gray]) != 0:
# #                 roi = roi_gray.astype('float') / 255.0
# #                 roi = img_to_array(roi)
# #                 roi = np.expand_dims(roi, axis=0)

# #                 prediction = emotion_classifier.predict(roi)[0]
# #                 label = emotion_labels[prediction.argmax()]
# #                 emotion_counter[label] += 1

# #     cap.release()
# #     most_common_emotion = emotion_counter.most_common(1)
# #     if most_common_emotion:
# #         return most_common_emotion[0][0]
# #     else:
# #         return "No emotions detected."

# # def detect_person_match(video_path):
# #     try:
# #         known_image = face_recognition.load_image_file(KNOWN_IMAGE_PATH)
# #         known_encoding = face_recognition.face_encodings(known_image)[0]
# #     except Exception as e:
# #         print(f"Error loading known image: {e}")
# #         return "Error loading known image"

# #     cap = cv2.VideoCapture(video_path)
# #     if not cap.isOpened():
# #         return "Video Capture Error"

# #     ret, frame = cap.read()
# #     if not ret:
# #         cap.release()
# #         return "Video Capture Error"

# #     face_locations = face_recognition.face_locations(frame)
# #     face_count = len(face_locations)

# #     if face_count == 0:
# #         cap.release()
# #         return "No Face Detected"

# #     if face_count > 1:
# #         cap.release()
# #         return "More than One Face Detected"

# #     face_encodings = face_recognition.face_encodings(frame, face_locations)
# #     match = face_recognition.compare_faces([known_encoding], face_encodings[0])
# #     if match[0]:
# #         neck_bending = detect_neck_bending(frame, face_locations[0])
# #         cap.release()
# #         if neck_bending:
# #             return "Neck Movement"
# #         else:
# #             return "Match"
# #     else:
# #         cap.release()
# #         return "Not Match"

# # def detect_neck_bending(frame, face_location):
# #     face_landmarks = face_recognition.face_landmarks(frame, [face_location])
# #     if not face_landmarks:
# #         return False

# #     face_landmarks = face_landmarks[0]
# #     top_nose = face_landmarks['nose_bridge'][0]
# #     bottom_nose = face_landmarks['nose_tip'][-1]
# #     top_chin = face_landmarks['chin'][8]
# #     bottom_chin = face_landmarks['chin'][0]

# #     neck_vector = np.array(bottom_chin) - np.array(top_chin)
# #     face_vector = np.array(bottom_nose) - np.array(top_nose)

# #     angle = np.degrees(np.arccos(np.dot(neck_vector, face_vector) /
# #                                   (np.linalg.norm(neck_vector) * np.linalg.norm(face_vector))))

# #     return angle > 130 or angle < 125

# # def process_videos():
# #     processed_files = set()
# #     while True:
# #         for file_name in os.listdir(video_dir):
# #             if file_name.endswith('.webm') and file_name not in processed_files:
# #                 file_path = os.path.join(video_dir, file_name)
# #                 print(f"Processing file: {file_path}")

# #                 result = detect_person_match(file_path)
# #                 emotion = emotion_fdetect(file_path)
# #                 socketio.emit('result', {'result': result, 'emotion': emotion})
# #                 print(result)
# #                 print(emotion)
                
# #                 processed_files.add(file_name)

# #         time.sleep(5)  # Adjust the sleep duration as needed

# # @socketio.on('connect')
# # def handle_connect():
# #     print('Client connected')

# # @socketio.on('video_data')
# # def handle_video_data(data):
# #     file_name = f"video_{int(time.time() * 1000)}"
# #     file_stream, file_path = create_file_write_stream(file_name)

# #     if file_stream is None:
# #         socketio.emit('result', {'result': 'File Write Error', 'emotion': 'N/A'})
# #         return

# #     try:
# #         file_stream.write(data)
# #     except Exception as e:
# #         print(f"Error writing to file: {e}")
# #         socketio.emit('result', {'result': 'File Write Error', 'emotion': 'N/A'})
# #     finally:
# #         file_stream.close()

# #     print(f"File {file_path} has been saved.")

# # @socketio.on('disconnect')
# # def handle_disconnect():
# #     print('Client disconnected')

# # @socketio.on_error()
# # def error_handler(e):
# #     print(f"Socket error: {e}")
    
# # context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
# # context.load_cert_chain('cert.pem', 'key.pem')

# # if __name__ == '__main__':
# #     # Start the video processing thread
# #     video_processing_thread = Thread(target=process_videos)
# #     video_processing_thread.daemon = True
# #     video_processing_thread.start()
# #     ssl_context = SSL.Context(SSL.PROTOCOL_TLSv1_2)
# #     ssl_context.use_privatekey_file('key.pem')
# #     ssl_context.use_certificate_file('cert.pem')

# #     socketio.run(app, port=5000, host='0.0.0.0',ssl_context=context, debug=True,certfile='cert.pem',keyfile='key.pem')

# #--------------------------------------------------------------------------------------------------------------------------
# # from flask import Flask, request
# # from flask_socketio import SocketIO, emit
# # import os
# # import base64
# # from flask_cors import CORS
# # import time

# # app = Flask(__name__)
# # CORS(app, origins="https://66545eb19c181f80368e15dd--velvety-meerkat-dbc2c1.netlify.app")
# # socketio = SocketIO(app, cors_allowed_origins="https://66545eb19c181f80368e15dd--velvety-meerkat-dbc2c1.netlify.app")
# # print("fgv")

# # # Directory to store video files
# # video_dir = os.path.join(os.path.dirname(__file__), 'videos')
# # if not os.path.exists(video_dir):
# #     os.makedirs(video_dir)

# # def create_file_write_stream(file_name):
# #     file_path = os.path.join(video_dir, f"{file_name}.webm")    
# #     file_stream = open(file_path, 'wb')
# #     return file_stream, file_path

# # print("lllvk")
# # @socketio.on('video_data')
# # def handle_video_data(data):
# #     file_name = f"video_{int(time.time())}"
# #     file_stream, file_path = create_file_write_stream(file_name)

# #     file_stream.write(base64.b64decode(data))
# #     file_stream.close()
# #     print(f"File {file_path} has been saved.")

# # @socketio.on('disconnect')
# # def handle_disconnect():
# #     print('Client disconnected')

# # @socketio.on_error_default # handles all namespaces without an explicit error handler
# # def error_handler(e):
# #     print('An error occurred:', e)
# # print("op")
# # if __name__ == '__main__':
# #     # ssl_context = ('C:/Users/OMR-09/cert.pem', 'C:/Users/OMR-09/key.pem')
# #     print("l")
# #     socketio.run(app, host='0.0.0.0', port=5000, debug=True)
# # ----------------------------------------------------------------------------------------------------------------------------------------------

# # from flask import Flask, request
# # from flask_socketio import SocketIO, emit
# # import os
# # import base64
# # import time

# # app = Flask(__name__)
# # socketio = SocketIO(app, cors_allowed_origins="https://66545eb19c181f80368e15dd--velvety-meerkat-dbc2c1.netlify.app")

# # # Directory to store video files
# # video_dir = os.path.join(os.path.dirname(__file__), 'videos')
# # if not os.path.exists(video_dir):
# #     os.makedirs(video_dir)

# # def create_file_write_stream(file_name):
# #     file_path = os.path.join(video_dir, f"{file_name}.webm")
# #     file_stream = open(file_path, 'wb')
# #     return file_stream, file_path

# # @socketio.on('video_data')
# # def handle_video_data(data):
# #     file_name = f"video_{int(time.time())}"
# #     file_stream, file_path = create_file_write_stream(file_name)

# #     file_stream.write(base64.b64decode(data))
# #     file_stream.close()
# #     print(f"File {file_path} has been saved.")

# # @socketio.on('disconnect')
# # def handle_disconnect():
# #     print('Client disconnected')

# # @socketio.on_error_default  # handles all namespaces without an explicit error handler
# # def error_handler(e):
# #     print('An error occurred:', e)

# # if __name__ == '__main__':
# #     socketio.run(app, port=5000,host='0.0.0.0',debug=True,ssl_context=("cert.pem", "key.pem"))

# #-----------------------------------------------------------------------------------------------------------------------------------------------------------

# # from flask import Flask, request
# # from flask_cors import CORS
# # from flask_socketio import SocketIO
# # import os
# # import time
# # import cv2
# # import numpy as np
# # from keras.models import load_model
# # from keras.preprocessing.image import img_to_array
# # import face_recognition
# # from collections import Counter
# # from threading import Thread

# # app = Flask(__name__)
# # CORS(app)
# # socketio = SocketIO(app, cors_allowed_origins="*",ping_timeout=20000, max_http_buffer_size=10 * 1024 * 1024, ping_interval=25000)

# # # Directory to store video files
# # video_dir = os.path.join(os.path.dirname(__file__), 'videos')
# # if not os.path.exists(video_dir):
# #     os.makedirs(video_dir)

# # # Load ML models with error handling
# # try:
# #     face_classifier = cv2.CascadeClassifier( "C:/Users/OMR-09/Downloads/Emotion_Detection_CNN-main-20240518T033125Z-001/Emotion_Detection_CNN-main/Emotion_Detection_CNN-main/haarcascade_frontalface_default.xml")
# #     emotion_classifier = load_model("C:/Users/OMR-09/Downloads/Emotion_Detection_CNN-main-20240518T033125Z-001/Emotion_Detection_CNN-main/Emotion_Detection_CNN-main/model.h5")
# #     emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
# #     KNOWN_IMAGE_PATH = "C:/Users/OMR-09/Pictures/img2.jpg"  # Change this to the path of your known person image
# # except Exception as e:
# #     print(f"Error loading models: {e}")
# #     exit(1)

# # def create_file_write_stream(file_name):
# #     try:
# #         file_path = os.path.join(video_dir, f"{file_name}.webm")
# #         file_stream = open(file_path, 'wb')
# #         return file_stream, file_path
# #     except Exception as e:
# #         print(f"Error creating file write stream: {e}")
# #         return None, None

# # def emotion_fdetect(video_path):
# #     emotion_counter = Counter()
# #     cap = cv2.VideoCapture(video_path)

# #     if not cap.isOpened():
# #         return "Video Capture Error"

# #     while cap.isOpened():
# #         ret, frame = cap.read()
# #         if not ret:
# #             break

# #         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# #         faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# #         for (x, y, w, h) in faces:
# #             roi_gray = gray[y:y+h, x:x+w]
# #             roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

# #             if np.sum([roi_gray]) != 0:
# #                 roi = roi_gray.astype('float') / 255.0
# #                 roi = img_to_array(roi)
# #                 roi = np.expand_dims(roi, axis=0)

# #                 prediction = emotion_classifier.predict(roi)[0]
# #                 label = emotion_labels[prediction.argmax()]
# #                 emotion_counter[label] += 1

# #     cap.release()
# #     most_common_emotion = emotion_counter.most_common(1)
# #     if most_common_emotion:
# #         return most_common_emotion[0][0]
# #     else:
# #         return "No emotions detected."

# # def detect_person_match(video_path):
# #     try:
# #         known_image = face_recognition.load_image_file(KNOWN_IMAGE_PATH)
# #         known_encoding = face_recognition.face_encodings(known_image)[0]
# #     except Exception as e:
# #         print(f"Error loading known image: {e}")
# #         return "Error loading known image"

# #     cap = cv2.VideoCapture(video_path)
# #     if not cap.isOpened():
# #         return "Video Capture Error"

# #     ret, frame = cap.read()
# #     if not ret:
# #         cap.release()
# #         return "Video Capture Error"

# #     face_locations = face_recognition.face_locations(frame)
# #     face_count = len(face_locations)

# #     if face_count == 0:
# #         cap.release()
# #         return "No Face Detected"

# #     if face_count > 1:
# #         cap.release()
# #         return "More than One Face Detected"

# #     face_encodings = face_recognition.face_encodings(frame, face_locations)
# #     match = face_recognition.compare_faces([known_encoding], face_encodings[0])
# #     if match[0]:
# #         neck_bending = detect_neck_bending(frame, face_locations[0])
# #         cap.release()
# #         if neck_bending:
# #             return "Neck Movement"
# #         else:
# #             return "Match"
# #     else:
# #         cap.release()
# #         return "Not Match"

# # def detect_neck_bending(frame, face_location):
# #     face_landmarks = face_recognition.face_landmarks(frame, [face_location])
# #     if not face_landmarks:
# #         return False

# #     face_landmarks = face_landmarks[0]
# #     top_nose = face_landmarks['nose_bridge'][0]
# #     bottom_nose = face_landmarks['nose_tip'][-1]
# #     top_chin = face_landmarks['chin'][8]
# #     bottom_chin = face_landmarks['chin'][0]

# #     neck_vector = np.array(bottom_chin) - np.array(top_chin)
# #     face_vector = np.array(bottom_nose) - np.array(top_nose)

# #     angle = np.degrees(np.arccos(np.dot(neck_vector, face_vector) /
# #                                   (np.linalg.norm(neck_vector) * np.linalg.norm(face_vector))))

# #     return angle > 130 or angle < 125

# # def process_videos():
# #     processed_files = set()
# #     while True:
# #         for file_name in os.listdir(video_dir):
# #             if file_name.endswith('.webm') and file_name not in processed_files:
# #                 file_path = os.path.join(video_dir, file_name)
# #                 print(f"Processing file: {file_path}")
        
# #                 result = detect_person_match(file_path)
# #                 emotion = emotion_fdetect(file_path)
# #                 socketio.emit( {'result': result, 'emotion': emotion})
# #                 print(result)
# #                 print(emotion)

# #                 processed_files.add(file_name)

# #         time.sleep(5)  # Adjust the sleep duration as needed

# # @socketio.on('connect')
# # def handle_connect():
# #     print('Client connected')
# #     socketio.emit("connect")

# # @socketio.on('video_data')
# # def handle_video_data(data):
# #     file_name = f"video_{int(time.time() * 1000)}"
# #     file_stream, file_path = create_file_write_stream(file_name)

# #     if file_stream is None:
# #         socketio.emit('result', {'result': 'File Write Error', 'emotion': 'N/A'})
# #         return

# #     try:
# #         file_stream.write(data)
# #     except Exception as e:
# #         print(f"Error writing to file: {e}")
# #         socketio.emit('result', {'result': 'File Write Error', 'emotion': 'N/A'})
# #     finally:
# #         file_stream.close()

# #     print(f"File {file_path} has been saved.")


# # @socketio.on('disconnect')
# # def handle_disconnect():
# #     print('Client disconnected')

# # @socketio.on_error()
# # def error_handler(e):
# #     print(f"Socket error: {e}")

# # if __name__ == '__main__':
# #     # Start the video processing thread
# #     video_processing_thread = Thread(target=process_videos)
# #     video_processing_thread.daemon = True
# #     video_processing_thread.start()

# #     socketio.run(app, port=5000, host='0.0.0.0', debug=True)

# #=-----------------------------------------------------------------------------------------------------------------------------------------------------------
# # from flask import Flask, request
# # from flask_cors import CORS
# # from flask_socketio import SocketIO
# # import os
# # import time
# # import cv2
# # import numpy as np
# # from keras.models import load_model
# # from keras.preprocessing.image import img_to_array
# # import face_recognition
# # from collections import Counter
# # from threading import Thread

# # app = Flask(__name__)
# # CORS(app)
# # socketio = SocketIO(app, cors_allowed_origins="*",ping_timeout=20000, max_http_buffer_size=10 * 1024 * 1024, ping_interval=25000)

# # # Directory to store video files
# # video_dir = os.path.join(os.path.dirname(__file__), 'videos')
# # if not os.path.exists(video_dir):
# #     os.makedirs(video_dir)

# # # Load ML models with error handling
# # try:
# #     face_classifier = cv2.CascadeClassifier( "C:/Users/OMR-09/Downloads/Emotion_Detection_CNN-main-20240518T033125Z-001/Emotion_Detection_CNN-main/Emotion_Detection_CNN-main/haarcascade_frontalface_default.xml")
# #     emotion_classifier = load_model("C:/Users/OMR-09/Downloads/Emotion_Detection_CNN-main-20240518T033125Z-001/Emotion_Detection_CNN-main/Emotion_Detection_CNN-main/model.h5")
# #     emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
# #     KNOWN_IMAGE_PATH = "C:/Users/OMR-09/Pictures/img2.jpg"  # Change this to the path of your known person image
# # except Exception as e:
# #     print(f"Error loading models: {e}")
# #     exit(1)

# # def create_file_write_stream(file_name):
# #     try:
# #         file_path = os.path.join(video_dir, f"{file_name}.webm")
# #         file_stream = open(file_path, 'wb')
# #         return file_stream, file_path
# #     except Exception as e:
# #         print(f"Error creating file write stream: {e}")
# #         return None, None

# # def emotion_fdetect(video_path):
# #     emotion_counter = Counter()
# #     cap = cv2.VideoCapture(video_path)

# #     if not cap.isOpened():
# #         return "Video Capture Error"

# #     while cap.isOpened():
# #         ret, frame = cap.read()
# #         if not ret:
# #             break

# #         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# #         faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# #         for (x, y, w, h) in faces:
# #             roi_gray = gray[y:y+h, x:x+w]
# #             roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

# #             if np.sum([roi_gray]) != 0:
# #                 roi = roi_gray.astype('float') / 255.0
# #                 roi = img_to_array(roi)
# #                 roi = np.expand_dims(roi, axis=0)

# #                 prediction = emotion_classifier.predict(roi)[0]
# #                 label = emotion_labels[prediction.argmax()]
# #                 emotion_counter[label] += 1

# #     cap.release()
# #     most_common_emotion = emotion_counter.most_common(1)
# #     if most_common_emotion:
# #         return most_common_emotion[0][0]
# #     else:
# #         return "No emotions detected."

# # def detect_person_match(video_path):
# #     try:
# #         known_image = face_recognition.load_image_file(KNOWN_IMAGE_PATH)
# #         known_encoding = face_recognition.face_encodings(known_image)[0]
# #     except Exception as e:
# #         print(f"Error loading known image: {e}")
# #         return "Error loading known image"

# #     cap = cv2.VideoCapture(video_path)
# #     if not cap.isOpened():
# #         return "Video Capture Error"

# #     ret, frame = cap.read()
# #     if not ret:
# #         cap.release()
# #         return "Video Capture Error"

# #     face_locations = face_recognition.face_locations(frame)
# #     face_count = len(face_locations)

# #     if face_count == 0:
# #         cap.release()
# #         return "No Face Detected"

# #     if face_count > 1:
# #         cap.release()
# #         return "More than One Face Detected"

# #     face_encodings = face_recognition.face_encodings(frame, face_locations)
# #     match = face_recognition.compare_faces([known_encoding], face_encodings[0])
# #     if match[0]:
# #         neck_bending = detect_neck_bending(frame, face_locations[0])
# #         cap.release()
# #         if neck_bending:
# #             return "Neck Movement"
# #         else:
# #             return "Match"
# #     else:
# #         cap.release()
# #         return "Not Match"

# # def detect_neck_bending(frame, face_location):
# #     face_landmarks = face_recognition.face_landmarks(frame, [face_location])
# #     if not face_landmarks:
# #         return False

# #     face_landmarks = face_landmarks[0]
# #     top_nose = face_landmarks['nose_bridge'][0]
# #     bottom_nose = face_landmarks['nose_tip'][-1]
# #     top_chin = face_landmarks['chin'][8]
# #     bottom_chin = face_landmarks['chin'][0]

# #     neck_vector = np.array(bottom_chin) - np.array(top_chin)
# #     face_vector = np.array(bottom_nose) - np.array(top_nose)

# #     angle = np.degrees(np.arccos(np.dot(neck_vector, face_vector) /
# #                                   (np.linalg.norm(neck_vector) * np.linalg.norm(face_vector))))

# #     return angle > 130 or angle < 125

# # def process_videos():
# #     processed_files = set()
# #     while True:
# #         for file_name in os.listdir(video_dir):
# #             if file_name.endswith('.webm') and file_name not in processed_files:
# #                 file_path = os.path.join(video_dir, file_name)
# #                 print(f"Processing file: {file_path}")

# #                 result = detect_person_match(file_path)
# #                 emotion = emotion_fdetect(file_path)
# #                 socketio.emit('result', {'result': result, 'emotion': emotion})
# #                 print(result)
# #                 print(emotion)

# #                 processed_files.add(file_name)

# #         time.sleep(5)  # Adjust the sleep duration as needed

# # @socketio.on('connect')
# # def handle_connect():
# #     print('Client connected')
# #     socketio.emit("connect")
    

# # @socketio.on('video_data')
# # def handle_video_data(data):
# #     file_name = f"video_{int(time.time() * 1000)}"
# #     file_stream, file_path = create_file_write_stream(file_name)

# #     if file_stream is None:
# #         socketio.emit('result', {'result': 'File Write Error', 'emotion': 'N/A'})
# #         return

# #     try:
# #         file_stream.write(data)
# #         result = detect_person_match(file_path)
# #         emotion = emotion_fdetect(file_path)
# #         socketio.emit('result', {'result': result, 'emotion': emotion})
# #     except Exception as e:
# #         print(f"Error writing to file: {e}")
# #         socketio.emit('result', {'result': 'File Write Error', 'emotion': 'N/A'})
# #     finally:
# #         result = detect_person_match(file_path)
# #         emotion = emotion_fdetect(file_path)
# #         socketio.emit('result', {'result': result, 'emotion': emotion})
# #         file_stream.close()

# #     print(f"File {file_path} has been saved.")
# #     result = detect_person_match(file_path)
# #     emotion = emotion_fdetect(file_path)
# #     socketio.emit('result', {'result': result, 'emotion': emotion})

# # @socketio.on('disconnect')
# # def handle_disconnect():
# #     print('Client disconnected')

# # @socketio.on_error()
# # def error_handler(e):
# #     print(f"Socket error: {e}")

# # if __name__ == '__main__':
# #     # Start the video processing thread
# #     video_processing_thread = Thread(target=process_videos)
# #     video_processing_thread.daemon = True
# #     video_processing_thread.start()

# #     socketio.run(app, port=5000, host='0.0.0.0', debug=True)

# from flask import Flask, request
# from flask_cors import CORS
# from flask_socketio import SocketIO
# import os
# import time
# import cv2
# import numpy as np
# from keras.models import load_model
# from keras.preprocessing.image import img_to_array
# import face_recognition
# from collections import Counter
# from threading import Thread

# app = Flask(__name__)
# CORS(app)
# socketio = SocketIO(app, cors_allowed_origins="*",ping_timeout=20000, max_http_buffer_size=10 * 1024 * 1024, ping_interval=25000)

# # Directory to store video files
# video_dir = os.path.join(os.path.dirname(__file__), 'videos')
# if not os.path.exists(video_dir):
#     os.makedirs(video_dir)

# # Load ML models with error handling
# try:
#     face_classifier = cv2.CascadeClassifier( "C:/Users/OMR-09/Downloads/Emotion_Detection_CNN-main-20240518T033125Z-001/Emotion_Detection_CNN-main/Emotion_Detection_CNN-main/haarcascade_frontalface_default.xml")
#     emotion_classifier = load_model("C:/Users/OMR-09/Downloads/Emotion_Detection_CNN-main-20240518T033125Z-001/Emotion_Detection_CNN-main/Emotion_Detection_CNN-main/model.h5")
#     emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
#     KNOWN_IMAGE_PATH = "C:/Users/OMR-09/Pictures/img2.jpg"  # Change this to the path of your known person image
# except Exception as e:
#     print(f"Error loading models: {e}")
#     exit(1)

# def create_file_write_stream(file_name):
#     try:
#         file_path = os.path.join(video_dir, f"{file_name}.webm")
#         file_stream = open(file_path, 'wb')
#         return file_stream, file_path
#     except Exception as e:
#         print(f"Error creating file write stream: {e}")
#         return None, None

# def emotion_fdetect(video_path):
#     emotion_counter = Counter()
#     cap = cv2.VideoCapture(video_path)

#     if not cap.isOpened():
#         return "Video Capture Error"

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#         for (x, y, w, h) in faces:
#             roi_gray = gray[y:y+h, x:x+w]
#             roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

#             if np.sum([roi_gray]) != 0:
#                 roi = roi_gray.astype('float') / 255.0
#                 roi = img_to_array(roi)
#                 roi = np.expand_dims(roi, axis=0)

#                 prediction = emotion_classifier.predict(roi)[0]
#                 label = emotion_labels[prediction.argmax()]
#                 emotion_counter[label] += 1

#     cap.release()
#     most_common_emotion = emotion_counter.most_common(1)
#     if most_common_emotion:
#         return most_common_emotion[0][0]
#     else:
#         return "No emotions detected."

# def detect_person_match(video_path):
#     try:
#         known_image = face_recognition.load_image_file(KNOWN_IMAGE_PATH)
#         known_encoding = face_recognition.face_encodings(known_image)[0]
#     except Exception as e:
#         print(f"Error loading known image: {e}")
#         return "Error loading known image"

#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         return "Video Capture Error"

#     ret, frame = cap.read()
#     if not ret:
#         cap.release()
#         return "Video Capture Error"

#     face_locations = face_recognition.face_locations(frame)
#     face_count = len(face_locations)

#     if face_count == 0:
#         cap.release()
#         return "No Face Detected"

#     if face_count > 1:
#         cap.release()
#         return "More than One Face Detected"

#     face_encodings = face_recognition.face_encodings(frame, face_locations)
#     match = face_recognition.compare_faces([known_encoding], face_encodings[0])
#     if match[0]:
#         neck_bending = detect_neck_bending(frame, face_locations[0])
#         cap.release()
#         if neck_bending:
#             return "Neck Movement"
#         else:
#             return "Match"
#     else:
#         cap.release()
#         return "Not Match"

# def detect_neck_bending(frame, face_location):
#     face_landmarks = face_recognition.face_landmarks(frame, [face_location])
#     if not face_landmarks:
#         return False

#     face_landmarks = face_landmarks[0]
#     top_nose = face_landmarks['nose_bridge'][0]
#     bottom_nose = face_landmarks['nose_tip'][-1]
#     top_chin = face_landmarks['chin'][8]
#     bottom_chin = face_landmarks['chin'][0]

#     neck_vector = np.array(bottom_chin) - np.array(top_chin)
#     face_vector = np.array(bottom_nose) - np.array(top_nose)

#     angle = np.degrees(np.arccos(np.dot(neck_vector, face_vector) /
#                                   (np.linalg.norm(neck_vector) * np.linalg.norm(face_vector))))

#     return angle > 130 or angle < 125

# def process_videos():
#     processed_files = set()
#     while True:
#         for file_name in os.listdir(video_dir):
#             if file_name.endswith('.webm') and file_name not in processed_files:
#                 file_path = os.path.join(video_dir, file_name)
#                 print(f"Processing file: {file_path}")

#                 result = detect_person_match(file_path)
#                 emotion = emotion_fdetect(file_path)
#                 socketio.emit('result', {'result': result, 'emotion': emotion})
#                 print(result)
#                 print(emotion)
#                 result = detect_person_match(file_path)
#                 emotion = emotion_fdetect(file_path)
#                 socketio.emit('result', {'result': result, 'emotion': emotion})
#                 processed_files.add(file_name)

#         time.sleep(5)  # Adjust the sleep duration as needed

# @socketio.on('connect')
# def handle_connect():
#     print('Client connected')

# @socketio.on('video_data')
# def handle_video_data(data):
#     file_name = f"video_{int(time.time() * 1000)}"
#     file_stream, file_path = create_file_write_stream(file_name)

#     if file_stream is None:
#         socketio.emit('result', {'result': 'File Write Error', 'emotion': 'N/A'})
#         return

#     try:
#         file_stream.write(data)
#     except Exception as e:
#         print(f"Error writing to file: {e}")
#         socketio.emit('result', {'result': 'File Write Error', 'emotion': 'N/A'})
#     finally:
#         result = detect_person_match(file_path)
#         emotion = emotion_fdetect(file_path)
#         socketio.emit('result', {'result': result, 'emotion': emotion})
#         file_stream.close()

#     print(f"File {file_path} has been saved.")
#     result = detect_person_match(file_path)
#     emotion = emotion_fdetect(file_path)
#     socketio.emit('result', {'result': result, 'emotion': emotion})

# @socketio.on('disconnect')
# def handle_disconnect():
#     print('Client disconnected')

# @socketio.on_error()
# def error_handler(e):
#     print(f"Socket error: {e}")

# if __name__ == '__main__':
#     # Start the video processing thread
#     video_processing_thread = Thread(target=process_videos)
#     video_processing_thread.daemon = True
#     video_processing_thread.start()

#     socketio.run(app, port=5000, host='0.0.0.0', debug=True)
# -------------------------------------------------------------------------------------------------------------------------
# from flask import Flask, request
# from flask_cors import CORS
# from flask_socketio import SocketIO
# import os
# import time
# import cv2
# import numpy as np
# from keras.models import load_model
# from keras.preprocessing.image import img_to_array
# import face_recognition
# from collections import Counter
# from threading import Thread

# app = Flask(__name__)
# CORS(app)
# socketio = SocketIO(app, cors_allowed_origins="*", ping_timeout=20000, max_http_buffer_size=10 * 1024 * 1024, ping_interval=25000)

# # Directory to store video files
# video_dir = os.path.join(os.path.dirname(__file__), 'videos')
# if not os.path.exists(video_dir):
#     os.makedirs(video_dir)

# # Load ML models with error handling
# try:
    # face_classifier = cv2.CascadeClassifier(
    #     "C:/Users/OMR-09/Downloads/Emotion_Detection_CNN-main-20240518T033125Z-001/Emotion_Detection_CNN-main/Emotion_Detection_CNN-main/haarcascade_frontalface_default.xml")
    # emotion_classifier = load_model(
    #     "C:/Users/OMR-09/Downloads/Emotion_Detection_CNN-main-20240518T033125Z-001/Emotion_Detection_CNN-main/Emotion_Detection_CNN-main/model.h5")
    # emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    # KNOWN_IMAGE_PATH = "C:/Users/OMR-09/Pictures/Camera Roll/WIN_20240529_14_15_13_Pro.jpg"  # Change this to the path of your known person image
# except Exception as e:
#     print(f"Error loading models: {e}")
#     exit(1)

# def create_file_write_stream(file_name):
#     try:
#         file_path = os.path.join(video_dir, f"{file_name}.webm")
#         file_stream = open(file_path, 'wb')
#         return file_stream, file_path
#     except Exception as e:
#         print(f"Error creating file write stream: {e}")
#         return None, None

# def emotion_fdetect(video_path):
#     emotion_counter = Counter()
#     cap = cv2.VideoCapture(video_path)

#     if not cap.isOpened():
#         return "Video Capture Error"

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#         for (x, y, w, h) in faces:
#             roi_gray = gray[y:y+h, x:x+w]
#             roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

#             if np.sum([roi_gray]) != 0:
#                 roi = roi_gray.astype('float') / 255.0
#                 roi = img_to_array(roi)
#                 roi = np.expand_dims(roi, axis=0)

#                 prediction = emotion_classifier.predict(roi)[0]
#                 label = emotion_labels[prediction.argmax()]
#                 emotion_counter[label] += 1

#     cap.release()
#     most_common_emotion = emotion_counter.most_common(1)
#     if most_common_emotion:
#         return most_common_emotion[0][0]
#     else:
#         return "No emotions detected."

# def detect_person_match(video_path):
#     try:
#         known_image = face_recognition.load_image_file(KNOWN_IMAGE_PATH)
#         known_encoding = face_recognition.face_encodings(known_image)[0]
#     except Exception as e:
#         print(f"Error loading known image: {e}")
#         return "Error loading known image"

#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         return "Video Capture Error"

#     ret, frame = cap.read()
#     if not ret:
#         cap.release()
#         return "Video Capture Error"

#     face_locations = face_recognition.face_locations(frame)
#     face_count = len(face_locations)

#     if face_count == 0:
#         cap.release()
#         return "No Face Detected"

#     if face_count > 1:
#         cap.release()
#         return "More than One Face Detected"

#     face_encodings = face_recognition.face_encodings(frame, face_locations)
#     match = face_recognition.compare_faces([known_encoding], face_encodings[0])
#     if match[0]:
#         neck_bending = detect_neck_bending(frame, face_locations[0])
#         cap.release()
#         if neck_bending:
#             return "Neck Movement"
#         else:
#             return "Match"
#     else:
#         cap.release()
#         return "Not Match"

# def detect_neck_bending(frame, face_location):
#     face_landmarks = face_recognition.face_landmarks(frame, [face_location])
#     if not face_landmarks:
#         return False

#     face_landmarks = face_landmarks[0]
#     top_nose = face_landmarks['nose_bridge'][0]
#     bottom_nose = face_landmarks['nose_tip'][-1]
#     top_chin = face_landmarks['chin'][8]
#     bottom_chin = face_landmarks['chin'][0]

#     neck_vector = np.array(bottom_chin) - np.array(top_chin)
#     face_vector = np.array(bottom_nose) - np.array(top_nose)

#     angle = np.degrees(np.arccos(np.dot(neck_vector, face_vector) /
#                                   (np.linalg.norm(neck_vector) * np.linalg.norm(face_vector))))

#     return angle > 130 or angle < 125

# def process_videos():
#     while True:
#         for file_name in os.listdir(video_dir):
#             if file_name.endswith('.webm'):
#                 file_path = os.path.join(video_dir, file_name)
#                 print(f"Processing file: {file_path}")

#                 result = detect_person_match(file_path)
#                 emotion = emotion_fdetect(file_path)
#                 socketio.emit('result', {'result': result, 'emotion': emotion})
#                 print(f"Result: {result}, Emotion: {emotion}")

#         time.sleep(5)  # Adjust the sleep duration as needed

# @socketio.on('connect')
# def handle_connect():
#     print('Client connected')

# @socketio.on('video_data')
# def handle_video_data(data):
#     file_name = f"video_{int(time.time() * 1000)}"
#     file_stream, file_path = create_file_write_stream(file_name)

#     if file_stream is None:
#         socketio.emit('result', {'result': 'File Write Error', 'emotion': 'N/A'})
#         return

#     try:
#         file_stream.write(data)
#     except Exception as e:
#         print(f"Error writing to file: {e}")
#         socketio.emit('result', {'result': 'File Write Error', 'emotion': 'N/A'})
#     finally:
#         result = detect_person_match(file_path)
#         emotion = emotion_fdetect(file_path)
#         socketio.emit('result', {'result': result, 'emotion': emotion})
#         file_stream.close()

#     print(f"File {file_path} has been saved.")
#     result = detect_person_match(file_path)
#     emotion = emotion_fdetect(file_path)
#     socketio.emit('result', {'result': result, 'emotion': emotion})

# @socketio.on('disconnect')
# def handle_disconnect():
#     print('Client disconnected')

# @socketio.on_error()
# def error_handler(e):
#     print(f"Socket error: {e}")

# if __name__ == '__main__':
#     # Start the video processing thread
#     video_processing_thread = Thread(target=process_videos)
#     video_processing_thread.daemon = True
#     video_processing_thread.start()

#     socketio.run(app, port=5000, host='0.0.0.0', debug=True)
# ---------------------------------------------------------------------------------------------------------------------------------------
# from flask import Flask, request
# from flask_cors import CORS
# from flask_socketio import SocketIO
# import os
# import time
# import cv2
# import numpy as np
# from keras.models import load_model
# from keras.preprocessing.image import img_to_array
# import face_recognition
# from collections import Counter
# from threading import Thread
# import queue

# app = Flask(__name__)
# CORS(app)
# socketio = SocketIO(app, cors_allowed_origins="*", ping_timeout=20000, max_http_buffer_size=10 * 1024 * 1024, ping_interval=25000)

# # Directory to store video files
# video_dir = os.path.join(os.path.dirname(__file__), 'videos')
# if not os.path.exists(video_dir):
#     os.makedirs(video_dir)

# # Load ML models with error handling
# try:
#     face_classifier = cv2.CascadeClassifier(
#         "C:/Users/OMR-09/Downloads/Emotion_Detection_CNN-main-20240518T033125Z-001/Emotion_Detection_CNN-main/Emotion_Detection_CNN-main/haarcascade_frontalface_default.xml")
#     emotion_classifier = load_model(
#         "C:/Users/OMR-09/Downloads/Emotion_Detection_CNN-main-20240518T033125Z-001/Emotion_Detection_CNN-main/Emotion_Detection_CNN-main/model.h5")
#     emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
#     KNOWN_IMAGE_PATH = "C:/Users/OMR-09/Pictures/Camera Roll/WIN_20240529_14_15_13_Pro.jpg"  # Change this to the path of your known person image
# except Exception as e:
#     print(f"Error loading models: {e}")
#     exit(1)

# # Create a queue to hold video data
# video_queue = queue.Queue()

# def create_file_write_stream(file_name):
#     try:
#         file_path = os.path.join(video_dir, f"{file_name}.webm")
#         file_stream = open(file_path, 'wb')
#         return file_stream, file_path
#     except Exception as e:
#         print(f"Error creating file write stream: {e}")
#         return None, None

# def emotion_fdetect(video_path):
#     emotion_counter = Counter()
#     cap = cv2.VideoCapture(video_path)

#     if not cap.isOpened():
#         return "Video Capture Error"

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#         for (x, y, w, h) in faces:
#             roi_gray = gray[y:y+h, x:x+w]
#             roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

#             if np.sum([roi_gray]) != 0:
#                 roi = roi_gray.astype('float') / 255.0
#                 roi = img_to_array(roi)
#                 roi = np.expand_dims(roi, axis=0)

#                 prediction = emotion_classifier.predict(roi)[0]
#                 label = emotion_labels[prediction.argmax()]
#                 emotion_counter[label] += 1

#     cap.release()
#     most_common_emotion = emotion_counter.most_common(1)
#     if most_common_emotion:
#         return most_common_emotion[0][0]
#     else:
#         return "No emotions detected."

# def detect_person_match(video_path):
#     try:
#         known_image = face_recognition.load_image_file(KNOWN_IMAGE_PATH)
#         known_encoding = face_recognition.face_encodings(known_image)[0]
#     except Exception as e:
#         print(f"Error loading known image: {e}")
#         return "Error loading known image"

#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         return "Video Capture Error"

#     ret, frame = cap.read()
#     if not ret:
#         cap.release()
#         return "Video Capture Error"

#     face_locations = face_recognition.face_locations(frame)
#     face_count = len(face_locations)

#     if face_count == 0:
#         cap.release()
#         return "No Face Detected"

#     if face_count > 1:
#         cap.release()
#         return "More than One Face Detected"

#     face_encodings = face_recognition.face_encodings(frame, face_locations)
#     match = face_recognition.compare_faces([known_encoding], face_encodings[0])
#     if match[0]:
#         neck_bending = detect_neck_bending(frame, face_locations[0])
#         cap.release()
#         if neck_bending:
#             return "Neck Movement"
#         else:
#             return "Match"
#     else:
#         cap.release()
#         return "Not Match"

# def detect_neck_bending(frame, face_location):
#     face_landmarks = face_recognition.face_landmarks(frame, [face_location])
#     if not face_landmarks:
#         return False

#     face_landmarks = face_landmarks[0]
#     top_nose = face_landmarks['nose_bridge'][0]
#     bottom_nose = face_landmarks['nose_tip'][0]
#     top_chin = face_landmarks['chin'][8]
#     bottom_chin = face_landmarks['chin'][0]

#     neck_vector = np.array(bottom_chin) - np.array(top_chin)
#     face_vector = np.array(bottom_nose) - np.array(top_nose)

#     angle = np.degrees(np.arccos(np.dot(neck_vector, face_vector) /
#                                   (np.linalg.norm(neck_vector) * np.linalg.norm(face_vector))))
#     print(angle)
#     return angle > 131 or angle < 120

# @socketio.on('connect')
# def handle_connect():
#     print('Client connected')

# @socketio.on('video_data')
# def handle_video_data(data):
#     file_name = f"video_{int(time.time() * 1000)}"
#     file_stream, file_path = create_file_write_stream(file_name)

#     if file_stream is None:
#         socketio.emit('result', {'result': 'File Write Error', 'emotion': 'N/A'})
#         return

#     try:
#         file_stream.write(data)
#     except Exception as e:
#         print(f"Error writing to file: {e}")
#         socketio.emit('result', {'result': 'File Write Error', 'emotion': 'N/A'})
#     finally:
#         file_stream.close()
#         video_queue.put(file_path)  # Put the file path into the queue

#     # If the queue has only one video, process it
#     if video_queue.qsize() == 1:
#         process_next_video()

# def process_next_video():
#     file_path = video_queue.get()  # Get the next video file path from the queue
#     result = detect_person_match(file_path)
#     emotion = emotion_fdetect(file_path)
#     socketio.emit('result', {'result': result, 'emotion': emotion})

#     # If there are more videos in the queue, process the next one
#     if not video_queue.empty():
#         process_next_video()

# @socketio.on('disconnect')
# def handle_disconnect():
#     print('Client disconnected')


# @socketio.on_error()
# def error_handler(e):
#     print('Client disconnected')
#     cleanup_videos()

# def cleanup_videos():
#     try:
#         import shutil
#         shutil.rmtree(video_dir)
#         print("Videos directory deleted.")
#     except Exception as e:
#         print(f"Error deleting videos directory: {e}")
        
# if __name__ == '__main__':
#     socketio.run(app, port=5000, host='0.0.0.0', debug=True)
# ---------------------------------------------------------------------------------------------------------------------
from flask import Flask, request
from flask_cors import CORS
from flask_socketio import SocketIO
import os
import time
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import face_recognition
from collections import Counter
from threading import Thread
import queue

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", ping_timeout=20000, max_http_buffer_size=10 * 1024 * 1024, ping_interval=25000)

# Directory to store video files
video_dir = os.path.join(os.path.dirname(__file__), 'videos')
if not os.path.exists(video_dir):
    os.makedirs(video_dir)

# Load ML models with error handling
try:
    face_classifier = cv2.CascadeClassifier(
        "C:/Users/OMR-09/Downloads/Emotion_Detection_CNN-main-20240518T033125Z-001/Emotion_Detection_CNN-main/Emotion_Detection_CNN-main/haarcascade_frontalface_default.xml")
    emotion_classifier = load_model(
        "C:/Users/OMR-09/Downloads/Emotion_Detection_CNN-main-20240518T033125Z-001/Emotion_Detection_CNN-main/Emotion_Detection_CNN-main/model.h5")
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    KNOWN_IMAGE_PATH = "C:/Users/OMR-09/Pictures/Camera Roll/WIN_20240529_14_15_13_Pro.jpg"  # Change this to the path of your known person image
except Exception as e:
    print(f"Error loading models: {e}")
    exit(1)

# Create a queue to hold video data
video_queue = queue.Queue()

def create_file_write_stream(file_name):
    try:
        file_path = os.path.join(video_dir, f"{file_name}.webm")
        file_stream = open(file_path, 'wb')
        return file_stream, file_path
    except Exception as e:
        print(f"Error creating file write stream: {e}")
        return None, None

def emotion_fdetect(video_path):
    emotion_counter = Counter()
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return "Video Capture Error"

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                prediction = emotion_classifier.predict(roi)[0]
                label = emotion_labels[prediction.argmax()]
                emotion_counter[label] += 1

    cap.release()
    most_common_emotion = emotion_counter.most_common(1)
    if most_common_emotion:
        return most_common_emotion[0][0]
    else:
        return "No emotions detected."

def detect_person_match(video_path):
    try:
        known_image = face_recognition.load_image_file(KNOWN_IMAGE_PATH)
        known_encoding = face_recognition.face_encodings(known_image)[0]
    except Exception as e:
        print(f"Error loading known image: {e}")
        return "Error loading known image"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "Video Capture Error"

    ret, frame = cap.read()
    if not ret:
        cap.release()
        return "Video Capture Error"

    face_locations = face_recognition.face_locations(frame)
    face_count = len(face_locations)

    if face_count == 0:
        cap.release()
        return "No Face Detected"

    if face_count > 1:
        cap.release()
        return "More than One Face Detected"

    face_encodings = face_recognition.face_encodings(frame, face_locations)
    match = face_recognition.compare_faces([known_encoding], face_encodings[0])
    if match[0]:
        neck_bending = detect_neck_bending(frame, face_locations[0])
        cap.release()
        if neck_bending:
            return "Neck Movement"
        else:
            return "Match"
    else:
        cap.release()
        return "Not Match"

def detect_neck_bending(frame, face_location):
    face_landmarks = face_recognition.face_landmarks(frame, [face_location])
    if not face_landmarks:
        return False

    face_landmarks = face_landmarks[0]
    top_nose = face_landmarks['nose_bridge'][0]
    bottom_nose = face_landmarks['nose_tip'][-1]
    top_chin = face_landmarks['chin'][8]
    bottom_chin = face_landmarks['chin'][0]

    neck_vector = np.array(bottom_chin) - np.array(top_chin)
    face_vector = np.array(bottom_nose) - np.array(top_nose)

    angle = np.degrees(np.arccos(np.dot(neck_vector, face_vector) /
                                  (np.linalg.norm(neck_vector) * np.linalg.norm(face_vector))))

    return angle > 130 or angle < 125

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('video_data')
def handle_video_data(data):
    file_name = f"video_{int(time.time() * 1000)}"
    file_stream, file_path = create_file_write_stream(file_name)

    if file_stream is None:
        socketio.emit('result', {'result': 'File Write Error', 'emotion': 'N/A'})
        return

    try:
        file_stream.write(data)
    except Exception as e:
        print(f"Error writing to file: {e}")
        socketio.emit('result', {'result': 'File Write Error', 'emotion': 'N/A'})
    finally:
        file_stream.close()
        video_queue.put(file_path)  # Put the file path into the queue

    # If the queue has only one video, process it
    if video_queue.qsize() == 1:
        process_next_video()

def process_next_video():
    file_path = video_queue.get()  # Get the next video file path from the queue
    result = detect_person_match(file_path)
    emotion = emotion_fdetect(file_path)
    socketio.emit('result', {'result': result, 'emotion': emotion})

    # If there are more videos in the queue, process the next one
    if not video_queue.empty():
        process_next_video()

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on_error()
def error_handler(e):
    print(f"Socket error: {e}")

if __name__ == '__main__':
    socketio.run(app, port=5000, host='0.0.0.0', debug=True,keyfile='C:/Users/OMR-09/Desktop/new/create-cert-key.pem', certfile='C:/Users/OMR-09/Desktop/new/create-cert.pem')

    #------ek ke bad ek vala hai bilkul