import cv2
import face_recognition
import numpy as np
import os
import onnxruntime as ort
import matplotlib.pyplot as plt

# Function to load ONNX model
def load_onnx_model(model_path):
    try:
        session = ort.InferenceSession(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        session = None
    return session

# Path to the custom ONNX model
model_path = 'D:/A_C_GPT/MODEL12/weights/best.onnx'

session = load_onnx_model(model_path)

def preprocess(image):
    image = cv2.resize(image, (640, 640))
    image = image.transpose(2, 0, 1)  # HWC to CHW
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)
    image /= 255.0  # Normalize to [0, 1]
    return image

def non_max_suppression(boxes, scores, iou_threshold):
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.25, nms_threshold=iou_threshold)
    return indices.flatten() if len(indices) > 0 else []

def postprocess(outputs, image_shape, input_shape, iou_threshold=0.4):
    boxes, scores, class_ids = [], [], []
    for output in outputs:
        for det in output:
            if det[4] > 0.25:  # Confidence threshold
                x_center, y_center, width, height = det[0], det[1], det[2], det[3]
                left = int((x_center - width / 2) * (image_shape[1] / input_shape[3]))
                top = int((y_center - height / 2) * (image_shape[0] / input_shape[2]))
                right = int((x_center + width / 2) * (image_shape[1] / input_shape[3]))
                bottom = int((y_center + height / 2) * (image_shape[0] / input_shape[2]))
                boxes.append([left, top, right, bottom])
                scores.append(det[4])
                class_ids.append(int(det[5]))

    indices = non_max_suppression(boxes, scores, iou_threshold)
    filtered_boxes = [boxes[i] for i in indices]
    filtered_scores = [scores[i] for i in indices]
    filtered_class_ids = [class_ids[i] for i in indices]

    return filtered_boxes, filtered_scores, filtered_class_ids

def detect_and_recognize_faces(image_path):
    if session is None:
        print("Model not loaded correctly. Exiting...")
        return [], [], None
    
    # Load the image
    image = cv2.imread(image_path)
    input_image = preprocess(image)
    
    # Run inference
    outputs = session.run(None, {session.get_inputs()[0].name: input_image})
    
    # Postprocess the outputs
    boxes, scores, class_ids = postprocess(outputs[0], image.shape, input_image.shape)
    
    # Filter only face class (assuming 'face' class index is 0)
    face_locations = []
    for box, score, class_id in zip(boxes, scores, class_ids):
        if class_id == 0:
            face_locations.append((box[1], box[2], box[3], box[0]))
    
    # Convert the image to RGB (opencv loads images in BGR format)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Encode the faces
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    
    return face_locations, face_encodings, image

def display_image_with_boxes(image, face_locations):
    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    
    # Convert BGR image to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Display the image using matplotlib
    plt.imshow(rgb_image)
    plt.axis('off')  # Hide axes
    plt.show()

def test_photos(directory):
    if session is None:
        print("Model not loaded correctly. Exiting...")
        return
    
    image_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    for image_path in image_paths:
        face_locations, face_encodings, image = detect_and_recognize_faces(image_path)
        
        if face_locations:
            display_image_with_boxes(image, face_locations)

# Example usage
test_photos('Test_album')
