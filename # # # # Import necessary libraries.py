# # # # Import necessary libraries
# # # from flask import Flask, request, jsonify
# # # import cv2
# # # from flask_cors import CORS
# # # import numpy as np
# # # import face_recognition

# # # # Initialize Flask app
# # # port_no = 5000
# # # app = Flask(__name__)
# # # # ngrok.set_auth_token("2gDVBMbJ3zF6Fdccaicxr3QIzbu_7ho4uhZoAUaNg5MAQeAob")
# # # # public_url = ngrok.connect(port_no).public_url
# # # CORS(app)
# # # # run_with_ngrok(app)

# # # KNOWN_IMAGE_PATH = "/content/WIN_20230528_23_10_59_Pro.jpg" # Change this to the path of your known person image

# # # def emotion_fdetect(video_path):
# # # face_classifier = cv2.CascadeClassifier(r'c:\Users\UTKARSH SINGH\Downloads\Emotion_Detection_CNN-main\Emotion_Detection_CNN-main\haarcascade_frontalface_default.xml')
# # # classifier = load_model(r'C:\Users\UTKARSH SINGH\Downloads\Emotion_Detection_CNN-main\Emotion_Detection_CNN-main\model.h5')

# # # # Emotion labels
# # # emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# # # # Path to the video file
# # # video_path = r'C:\Users\UTKARSH SINGH\Pictures\Camera Roll\WIN_20240409_14_50_50_Pro.mp4'

# # # # Start video capture from the file
# # # cap = cv2.VideoCapture(video_path)

# # # # Dictionary to keep count of each emotion
# # # emotion_counter = Counter()

# # # while cap.isOpened():
# # #     ret, frame = cap.read()
# # #     if not ret:
# # #         break  # Exit the loop if no frame is captured

# # #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# # #     faces = face_classifier.detectMultiScale(gray)

# # #     for (x, y, w, h) in faces:
# # #         roi_gray = gray[y:y+h, x:x+w]
# # #         roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

# # #         if np.sum([roi_gray]) != 0:
# # #             roi = roi_gray.astype('float') / 255.0
# # #             roi = img_to_array(roi)
# # #             roi = np.expand_dims(roi, axis=0)

# # #             prediction = classifier.predict(roi)[0]
# # #             label = emotion_labels[prediction.argmax()]
# # #             emotion_counter[label] += 1  # Increment the count for the detected emotion

# # #     # sleep(5)  # Wait for 5 seconds before the next detection

# # #     if cv2.waitKey(1) & 0xFF == ord('q'):
# # #         break

# # # cap.release()
# # # cv2.destroyAllWindows()

# # # # Find the most common emotion
# # # most_common_emotion = emotion_counter.most_common(1)
# # # if most_common_emotion:
# # #     print("Most frequently detected emotion:", most_common_emotion[0][0])
# # # else:
# # #     print("No emotions detected.")

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

# # #     return jsonify({'result': result})

# # # if __name__ == '__main__':
# # #     app.run()
# # import numpy as np
# # import dlib
# # import cv2
# # import matplotlib.pyplot as plt
# # from PIL import Image

# # detector = dlib.get_frontal_face_detector()
# # predictor = dlib.shape_predictor("C:/Users/UTKARSH SINGH/Downloads/facial-landmarks-recognition-master/facial-landmarks-recognition-master/shape_predictor_68_face_landmarks.dat")

# # # Path to the image you want to analyze
# # path = "C:/Users/UTKARSH SINGH/Pictures/Camera Roll/uyt.jpg"

# # img = dlib.load_rgb_image(path)
# # plt.imshow(img)
# # plt.show()

# # rect = detector(img)[0]
# # sp = predictor(img, rect)
# # landmarks = np.array([[p.x, p.y] for p in sp.parts()])

# # nose_bridge_x = []
# # nose_bridge_y = []
# # for i in [28, 29, 30, 31, 33, 34, 35]:
# #     nose_bridge_x.append(landmarks[i][0])
# #     nose_bridge_y.append(landmarks[i][1])

# # # x_min and x_max
# # x_min = min(nose_bridge_x)
# # x_max = max(nose_bridge_x)
# # # ymin (from top eyebrow coordinate), ymax
# # y_min = landmarks[20][1]
# # y_max = landmarks[31][1]

# # img2 = Image.open(path)
# # img2 = img2.crop((x_min, y_min, x_max, y_max))
# # plt.imshow(img2)
# # # plt.show()

# # img_blur = cv2.GaussianBlur(np.array(img2), (3, 3), sigmaX=0, sigmaY=0)
# # edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)
# # plt.imshow(edges, cmap=plt.get_cmap('gray'))
# # plt.show()

# # # Center strip
# # edges_center = edges.T[(int(len(edges.T) / 2))]
# # if 255 in edges_center:
# #     print("Glasses are present")
# # else:
# #     print("Glasses are absent")
# import numpy as np
# import dlib
# import cv2
# import matplotlib.pyplot as plt
# from PIL import Image

# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("C:/Users/UTKARSH SINGH/Downloads/facial-landmarks-recognition-master/facial-landmarks-recognition-master/shape_predictor_68_face_landmarks.dat")

# # Path to the image you want to analyze
# path = "C:/Users/UTKARSH SINGH/Pictures/Camera Roll/uyt.jpg"
# # path = "c:/Users/UTKARSH SINGH/Downloads/360_F_502151561_oZKT2pDgQokfhU7del9rJcMQMiT22eGJ.jpg"
# # path ="C:/Users/UTKARSH SINGH/Downloads/Fashion-man-wearing-sunglasses-Stock-Photo-02.jpg"

# img = dlib.load_rgb_image(path)
# plt.imshow(img)
# plt.show()

# rect = detector(img)[0]
# sp = predictor(img, rect)
# landmarks = np.array([[p.x, p.y] for p in sp.parts()])

# # Extracting nose bridge coordinates
# nose_bridge_x = []
# nose_bridge_y = []
# for i in [28, 29, 30, 31, 33, 34, 35]:
#     nose_bridge_x.append(landmarks[i][0])
#     nose_bridge_y.append(landmarks[i][1])

# # Extracting glasses lens coordinates (points 37-48)
# lens_x = [landmarks[i][0] for i in range(36, 48)]
# lens_y = [landmarks[i][1] for i in range(36, 48)]

# # x_min and x_max for nose bridge
# x_min = min(nose_bridge_x)
# x_max = max(nose_bridge_x)
# # ymin (from top eyebrow coordinate), ymax
# y_min = landmarks[20][1]
# y_max = landmarks[31][1]

# # Crop the region for nose bridge
# img_nose = Image.open(path)
# img_nose = img_nose.crop((x_min, y_min, x_max, y_max))
# plt.imshow(img_nose)
# plt.show()

# # Crop the region for glasses lens
# img_lens = Image.open(path)
# x_min_lens = min(lens_x)
# x_max_lens = max(lens_x)
# y_min_lens = min(lens_y)
# y_max_lens = max(lens_y)
# img_lens = img_lens.crop((x_min_lens, y_min_lens, x_max_lens, y_max_lens))
# plt.imshow(img_lens)
# plt.show()

# # Convert to numpy array and blur the images
# img_nose_blur = cv2.GaussianBlur(np.array(img_nose), (3, 3), sigmaX=0, sigmaY=0)
# img_lens_blur = cv2.GaussianBlur(np.array(img_lens), (3, 3), sigmaX=0, sigmaY=0)

# # Detect edges
# edges_nose = cv2.Canny(image=img_nose_blur, threshold1=100, threshold2=200)
# edges_lens = cv2.Canny(image=img_lens_blur, threshold1=100, threshold2=200)

# # Center strip for nose bridge
# edges_center_nose = edges_nose.T[(int(len(edges_nose.T) / 2))]
# # Center strip for glasses lens
# edges_center_lens = edges_lens.T[(int(len(edges_lens.T) / 2))]

# # Check if glasses are present
# if 255 in edges_center_nose:
#     print("Glasses are present")
# else:
#     print("Glasses are absent")

# # Get the color histogram of the lens region
# hist_lens = cv2.calcHist([np.array(img_lens)], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])

# # Find the most present color
# max_color = np.unravel_index(np.argmax(hist_lens), hist_lens.shape)
# print("Most present color in glasses lens (BGR):", max_color)

# # Convert BGR to RGB
# max_color_rgb = max_color[::-1]
# print("Most present color in glasses lens (RGB):", max_color_rgb)

#----------------------------------------------------------------------------------------
# import numpy as np
# import dlib
# import cv2
# import matplotlib.pyplot as plt
# from PIL import Image

# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("C:/Users/UTKARSH SINGH/Downloads/facial-landmarks-recognition-master/facial-landmarks-recognition-master/shape_predictor_68_face_landmarks.dat")

# # Path to the image you want to analyze
# path = "C:/Users/UTKARSH SINGH/Downloads/Fashion-man-wearing-sunglasses-Stock-Photo-02.jpg"

# img = dlib.load_rgb_image(path)
# plt.imshow(img)
# plt.show()

# rect = detector(img)[0]
# sp = predictor(img, rect)
# landmarks = np.array([[p.x, p.y] for p in sp.parts()])

# # Extracting nose bridge coordinates
# nose_bridge_x = []
# nose_bridge_y = []
# for i in [28, 29, 30, 31, 33, 34, 35]:
#     nose_bridge_x.append(landmarks[i][0])
#     nose_bridge_y.append(landmarks[i][1])

# # Extracting glasses lens coordinates (points 37-42)
# lens_x = [landmarks[i][0] for i in range(36, 42)]
# lens_y = [landmarks[i][1] for i in range(36, 42)]

# # x_min and x_max for nose bridge
# x_min = min(nose_bridge_x)
# x_max = max(nose_bridge_x)
# # ymin (from top eyebrow coordinate), ymax
# y_min = landmarks[20][1]
# y_max = landmarks[31][1]

# # Crop the region for nose bridge
# img_nose = Image.open(path)
# img_nose = img_nose.crop((x_min, y_min, x_max, y_max))
# plt.imshow(img_nose)
# plt.show()

# # Crop the region for glasses lens
# img_lens = Image.open(path)
# x_min_lens = min(lens_x)
# x_max_lens = max(lens_x)
# y_min_lens = min(lens_y)
# y_max_lens = max(lens_y)
# img_lens = img_lens.crop((x_min_lens, y_min_lens, x_max_lens, y_max_lens))
# plt.imshow(img_lens)
# plt.show()

# # Convert to numpy array and blur the images
# img_nose_blur = cv2.GaussianBlur(np.array(img_nose), (3, 3), sigmaX=0, sigmaY=0)
# img_lens_blur = cv2.GaussianBlur(np.array(img_lens), (3, 3), sigmaX=0, sigmaY=0)

# # Detect edges
# edges_nose = cv2.Canny(image=img_nose_blur, threshold1=100, threshold2=200)
# edges_lens = cv2.Canny(image=img_lens_blur, threshold1=100, threshold2=200)

# # Center strip for nose bridge
# edges_center_nose = edges_nose.T[(int(len(edges_nose.T) / 2))]
# # Center strip for glasses lens
# edges_center_lens = edges_lens.T[(int(len(edges_lens.T) / 2))]

# # Check if glasses are present
# if 255 in edges_center_nose:
#     print("Glasses are present")
# else:
#     print("Glasses are absent")

# # Get the color histogram of the lens region
# hist_lens = cv2.calcHist([np.array(img_lens)], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])

# # Find the most present color
# max_color = np.unravel_index(np.argmax(hist_lens), hist_lens.shape)
# print("Most present color in glasses lens (BGR):", max_color)

# # Convert BGR to RGB
# max_color_rgb = max_color[::-1]
# print("Most present color in glasses lens (RGB):", max_color_rgb)

# # Check if lens is transparent or not
# if 255 in edges_center_lens:
#     print("Glasses lens is not transparent")
# else:
#     print("Glasses lens is transparent")
#------------------------------------------------------------------------------------
import numpy as np
import dlib
import cv2
import matplotlib.pyplot as plt
from PIL import Image

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:/Users/UTKARSH SINGH/Downloads/facial-landmarks-recognition-master/facial-landmarks-recognition-master/shape_predictor_68_face_landmarks.dat")

# Path to the image you want to analyze
path = "C:/Users/UTKARSH SINGH/Downloads/Fashion-man-wearing-sunglasses-Stock-Photo-02.jpg"

img = dlib.load_rgb_image(path)
plt.imshow(img)
plt.show()

rect = detector(img)[0]
sp = predictor(img, rect)
landmarks = np.array([[p.x, p.y] for p in sp.parts()])

# Extracting nose bridge coordinates
nose_bridge_x = []
nose_bridge_y = []
for i in [28, 29, 30, 31, 33, 34, 35]:
    nose_bridge_x.append(landmarks[i][0])
    nose_bridge_y.append(landmarks[i][1])

# Extracting glasses lens coordinates (points 37-42)
lens_x = [landmarks[i][0] for i in range(36, 42)]
lens_y = [landmarks[i][1] for i in range(36, 42)]

# x_min and x_max for nose bridge
x_min = min(nose_bridge_x)
x_max = max(nose_bridge_x)
# ymin (from top eyebrow coordinate), ymax
y_min = landmarks[20][1]
y_max = landmarks[31][1]

# Crop the region for nose bridge
img_nose = Image.open(path)
img_nose = img_nose.crop((x_min, y_min, x_max, y_max))
plt.imshow(img_nose)
plt.show()

# Crop the region for glasses lens
img_lens = Image.open(path)
x_min_lens = min(lens_x)
x_max_lens = max(lens_x)
y_min_lens = min(lens_y)
y_max_lens = max(lens_y)
img_lens = img_lens.crop((x_min_lens, y_min_lens, x_max_lens, y_max_lens))
plt.imshow(img_lens)
plt.show()

# Convert to numpy array and blur the images
img_nose_blur = cv2.GaussianBlur(np.array(img_nose), (3, 3), sigmaX=0, sigmaY=0)
img_lens_blur = cv2.GaussianBlur(np.array(img_lens), (3, 3), sigmaX=0, sigmaY=0)

# Detect edges
edges_nose = cv2.Canny(image=img_nose_blur, threshold1=100, threshold2=200)
edges_lens = cv2.Canny(image=img_lens_blur, threshold1=100, threshold2=200)

# Center strip for nose bridge
edges_center_nose = edges_nose.T[(int(len(edges_nose.T) / 2))]
# Center strip for glasses lens
edges_center_lens = edges_lens.T[(int(len(edges_lens.T) / 2))]

# Check if glasses are present
if 255 in edges_center_nose:
    print("Glasses are present")
else:
    print("Glasses are absent")

# Get the color histogram of the lens region
hist_lens = cv2.calcHist([np.array(img_lens)], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])

# Find the most present color
max_color = np.unravel_index(np.argmax(hist_lens), hist_lens.shape)
print("Most present color in glasses lens (BGR):", max_color)

# Convert BGR to RGB
max_color_rgb = max_color[::-1]
print("Most present color in glasses lens (RGB):", max_color_rgb)

# Calculate the average pixel intensity within the lens region
average_intensity = np.mean(np.array(img_lens))

# Define a threshold to differentiate between transparent and opaque
threshold = 100  # Adjust this threshold as needed

# Check if the average intensity is above the threshold
if average_intensity > threshold:
    print("Glasses lens is opaque")
else:
    print("Glasses lens is transparent")

# import numpy as np
# import dlib
# import cv2
# import matplotlib.pyplot as plt
# from PIL import Image

# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("C:/Users/UTKARSH SINGH/Downloads/facial-landmarks-recognition-master/facial-landmarks-recognition-master/shape_predictor_68_face_landmarks.dat")

# # Path to the image you want to analyze
# path = "C:/Users/UTKARSH SINGH/Downloads/Fashion-man-wearing-sunglasses-Stock-Photo-02.jpg"

# img = dlib.load_rgb_image(path)
# plt.imshow(img)
# plt.show()

# rect = detector(img)[0]
# sp = predictor(img, rect)
# landmarks = np.array([[p.x, p.y] for p in sp.parts()])

# # Extracting nose bridge coordinates
# nose_bridge_x = []
# nose_bridge_y = []
# for i in [28, 29, 30, 31, 33, 34, 35]:
#     nose_bridge_x.append(landmarks[i][0])
#     nose_bridge_y.append(landmarks[i][1])

# # Extracting glasses lens coordinates (points 37-42)
# lens_x = [landmarks[i][0] for i in range(36, 42)]
# lens_y = [landmarks[i][1] for i in range(36, 42)]

# # x_min and x_max for nose bridge
# x_min = min(nose_bridge_x)
# x_max = max(nose_bridge_x)
# # ymin (from top eyebrow coordinate), ymax
# y_min = landmarks[20][1]
# y_max = landmarks[31][1]

# # Crop the region for nose bridge
# img_nose = Image.open(path)
# img_nose = img_nose.crop((x_min, y_min, x_max, y_max))
# plt.imshow(img_nose)
# plt.show()

# # Crop the region for glasses lens
# img_lens = Image.open(path)
# x_min_lens = min(lens_x)
# x_max_lens = max(lens_x)
# y_min_lens = min(lens_y)
# y_max_lens = max(lens_y)
# img_lens = img_lens.crop((x_min_lens, y_min_lens, x_max_lens, y_max_lens))
# plt.imshow(img_lens)
# plt.show()

# # Convert to numpy array and blur the images
# img_nose_blur = cv2.GaussianBlur(np.array(img_nose), (3, 3), sigmaX=0, sigmaY=0)
# img_lens_blur = cv2.GaussianBlur(np.array(img_lens), (3, 3), sigmaX=0, sigmaY=0)

# # Check if glasses are present
# if np.mean(img_nose_blur) < 200:  # Threshold for average pixel intensity
#     print("Glasses are present")
# else:
#     print("Glasses are absent")

# # Get the color histogram of the lens region
# hist_lens = cv2.calcHist([np.array(img_lens)], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])

# # Find the most present color
# max_color = np.unravel_index(np.argmax(hist_lens), hist_lens.shape)
# print("Most present color in glasses lens (BGR):", max_color)

# # Convert BGR to RGB
# max_color_rgb = max_color[::-1]
# print("Most present color in glasses lens (RGB):", max_color_rgb)
