import cv2
import numpy as np
from jetson_inference import detectNet
from jetson_utils import videoSource, videoOutput, cudaToNumpy, cudaFromNumpy
import time

# Initialize the object detection network with a threshold of 0.5 for detection confidence
net = detectNet("ssd-mobilenet-v2", threshold=0.5)

# Set up the video source (camera) and video output (display)
camera = videoSource("/dev/video1")  # Video source from camera (V4L2)
display = videoOutput("display://0")  # Display output on the screen

# Main loop
while display.IsStreaming():
    start_time = time.time()  # Record start time for FPS calculation

    img = camera.Capture()

    if img is None:  # Capture timeout
        continue

    # Perform object detection on the captured frame
    detections = net.Detect(img)
    
    # Convert CUDA image to NumPy array for OpenCV processing
    img_cv = cudaToNumpy(img)  # Convert to NumPy array

    # Loop over the detections and classify all objects
    for detection in detections:
        # Get the class label for the detected object
        class_label = net.GetClassDesc(detection.ClassID)
        
        # Get the bounding box coordinates using Left, Top, Right, Bottom
        top_left = (int(detection.Left), int(detection.Top))
        bot_right = (int(detection.Right), int(detection.Bottom))
        
        # Draw the bounding box around the detected object using OpenCV
        color = (0, 255, 0)  # Green color for the bounding box
        cv2.rectangle(img_cv, top_left, bot_right, color, 4)  # Draw bounding box

        # Put the class label and confidence on the image
        label = f"{class_label}: {detection.Confidence*100:.2f}%"
        cv2.putText(img_cv, label, (top_left[0], top_left[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Convert the image back from NumPy array to CUDA image
    img = cudaFromNumpy(img_cv)

    # Render the image with bounding boxes and class labels
    display.Render(img)

    # Calculate FPS (Frames Per Second) and update the status
    elapsed_time = time.time() - start_time
    fps = 1 / elapsed_time
    display.SetStatus(f"Real-Time Object Detection | Network FPS: {fps:.0f}")


