import cv2
import face_recognition
import numpy as np
import os
from shutil import copyfile
from openpyxl import Workbook, load_workbook
import onnxruntime as ort
import streamlit as st
from PIL import Image
from io import BytesIO

# Path to the custom ONNX model
model_path = 'MODEL12/weights/best.onnx'
session = ort.InferenceSession(model_path)

# Excel file path
excel_path = 'face_detection_log.xlsx'

# Global variable for face encodings dictionary
face_encodings_dict = {}

def create_excel_sheet():
    # Create a new Excel workbook and sheet if it doesn't exist
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = 'Face Detection Log'
    sheet.append(['Face Name', 'Image Path'])
    workbook.save(excel_path)

def update_excel_sheet(face_name, image_path):
    # Load the existing workbook
    workbook = load_workbook(excel_path)
    sheet = workbook.active
    # Append the new face name and image path
    sheet.append([face_name, image_path])
    workbook.save(excel_path)

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
                bottom = int((x_center + height / 2) * (image_shape[0] / input_shape[2]))
                boxes.append([left, top, right, bottom])
                scores.append(det[4])
                class_ids.append(int(det[5]))

    indices = non_max_suppression(boxes, scores, iou_threshold)
    filtered_boxes = [boxes[i] for i in indices]
    filtered_scores = [scores[i] for i in indices]
    filtered_class_ids = [class_ids[i] for i in indices]

    return filtered_boxes, filtered_scores, filtered_class_ids

def detect_and_recognize_faces(image):
    input_image = preprocess(image)
    outputs = session.run(None, {session.get_inputs()[0].name: input_image})
    boxes, scores, class_ids = postprocess(outputs[0], image.shape, input_image.shape)
    
    face_locations = []
    for box, score, class_id in zip(boxes, scores, class_ids):
        if class_id == 0:
            face_locations.append((box[1], box[2], box[3], box[0]))

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    
    return face_locations, face_encodings

def display_faces_with_boxes(image, face_locations):
    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    return image

def main():
    st.title("Real-Time Face Recognition")
    
    # Sidebar options
    st.sidebar.title("Options")
    option = st.sidebar.radio("Choose Mode", ("Upload Image", "Use Webcam"))
    
    if option == "Upload Image":
        uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            face_locations, face_encodings = detect_and_recognize_faces(image)

            if face_locations:
                annotated_image = display_faces_with_boxes(image.copy(), face_locations)
                st.image(annotated_image, channels="BGR", caption="Detected Faces")
            else:
                st.warning("No faces detected!")
    
    elif option == "Use Webcam":
        run_webcam = st.button("Start Webcam")
        if run_webcam:
            capture = cv2.VideoCapture(0)
            stframe = st.empty()
            while True:
                ret, frame = capture.read()
                if not ret:
                    break

                face_locations, face_encodings = detect_and_recognize_faces(frame)

                if face_locations:
                    frame = display_faces_with_boxes(frame, face_locations)

                stframe.image(frame, channels="BGR")

                if st.button("Stop Webcam"):
                    capture.release()
                    stframe.empty()
                    break

if __name__ == "__main__":
    main()
