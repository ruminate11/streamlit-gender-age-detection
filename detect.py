import cv2
import gdown
import os

# File IDs from the Google Drive links
file_id_1 = '1xEe2isH9MATXxf8tQJkxPNtgXKBG4-20'  # Age model
file_id_2 = '1R7YOFev-OIqJyPodGyaPFAgQgNoVrrwb'  # Gender model

# Google Drive URLs
url_1 = f'https://drive.google.com/uc?id={file_id_1}'
url_2 = f'https://drive.google.com/uc?id={file_id_2}'

# Paths to save the models
age_model_path = 'age_net.caffemodel'
gender_model_path = 'gender_net.caffemodel'
age_proto_path = 'age_deploy.prototxt'
gender_proto_path = 'gender_deploy.prototxt'

# Check if the models already exist, if not, download them
if not os.path.exists(age_model_path):
    print("Downloading age model...")
    try:
        gdown.download(url_1, age_model_path, quiet=False)
    except Exception as e:
        print(f"Error downloading age model: {e}")

if not os.path.exists(gender_model_path):
    print("Downloading gender model...")
    try:
        gdown.download(url_2, gender_model_path, quiet=False)
    except Exception as e:
        print(f"Error downloading gender model: {e}")

# Check if proto files exist, if not, download them
if not os.path.exists(age_proto_path):
    print("Downloading age proto file...")
    try:
        gdown.download('https://drive.google.com/uc?id=1gscbHGvMwUlo8uLVD68vA64Ej_xiSxj9', age_proto_path, quiet=False)
    except Exception as e:
        print(f"Error downloading age proto: {e}")

if not os.path.exists(gender_proto_path):
    print("Downloading gender proto file...")
    try:
        gdown.download('https://drive.google.com/uc?id=1X2Q5NvwItZodqGlOa_y7tU71dTAswb6q', gender_proto_path, quiet=False)
    except Exception as e:
        print(f"Error downloading gender proto: {e}")

# Load pre-trained models for face, age, and gender detection
try:
    faceProto = "opencv_face_detector.pbtxt"
    faceModel = "opencv_face_detector_uint8.pb"
    ageNet = cv2.dnn.readNet(age_model_path, age_proto_path)
    genderNet = cv2.dnn.readNet(gender_model_path, gender_proto_path)
    faceNet = cv2.dnn.readNet(faceModel, faceProto)
except Exception as e:
    print(f"Error loading models: {e}")
    exit()

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Function to detect face
def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    return frameOpencvDnn, faceBoxes

# Function to detect age and gender
def detect_gender_age(frame):
    resultImg, faceBoxes = highlightFace(faceNet, frame)
    if not faceBoxes:
        return resultImg, "No face detected"
    
    results = []
    for faceBox in faceBoxes:
        face = frame[max(0, faceBox[1] - 20):min(faceBox[3] + 20, frame.shape[0] - 1),
                     max(0, faceBox[0] - 20):min(faceBox[2] + 20, frame.shape[1] - 1)]

        # Gender detection
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]

        # Age detection
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]
        
        results.append(f"{gender}, {age}")

        cv2.putText(resultImg, f"{gender}, {age}", (faceBox[0], faceBox[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    return resultImg, results

# Example of how to run the code (add your webcam feed or image loading here)
if __name__ == "__main__":
    # Example webcam setup
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Detect gender and age
        resultImg, output = detect_gender_age(frame)

        # Show the output image
        cv2.imshow('Gender and Age Detection', resultImg)

        # Print the results (you can adjust this to your needs)
        print(output)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
