import cv2
import numpy as np

# Load the pre-trained MobileNet SSD model
# Download the model files from https://github.com/chuanqi305/MobileNet-SSD
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')

# Define the classes in MobileNet SSD (using the COCO dataset)
classNames = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

# Open a USB webcam (device 0 is the default)
cap = cv2.VideoCapture(0)

# Set the video capture resolution (optional, for better performance)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

while True:
    # Read the frame from the webcam
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture image")
        break

    # Resize the frame to 300x300 as required by the MobileNet SSD model
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5, True, False)

    # Set the input to the model
    net.setInput(blob)

    # Perform the forward pass and get the detections
    detections = net.forward()

    # Initialize person count
    person_count = 0

    # Iterate over all detections and draw bounding boxes around detected persons
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        # Only consider detections with a high confidence and that belong to the "person" class (class ID = 15)
        if confidence > 0.2:  # Confidence threshold
            class_id = int(detections[0, 0, i, 1])
            if class_id == 15:  # Person class ID
                # Get the bounding box coordinates
                box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (startX, startY, endX, endY) = box.astype("int")

                # Draw a rectangle around the person
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                person_count += 1

    # Display the number of detected people
    cv2.putText(frame, f"People Count: {person_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show the resulting frame
    cv2.imshow("Person Counting", frame)

    # Break on 'ESC' key press
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
