import cv2
import numpy as np
from jetson_inference import detectNet
from jetson_utils import videoSource, videoOutput, cudaToNumpy, cudaFromNumpy

# Initialize the object detection network with a threshold of 0.5 for detection confidence
net = detectNet("ssd-mobilenet-v2", threshold=0.5)

# Set up the video source (camera) and video output (display)
camera = videoSource("/dev/video0")  # Video source from camera (V4L2)
display = videoOutput("display://0")  # Display output on the screen

# Main loop
while display.IsStreaming():
    img = camera.Capture()

    if img is None:  # Capture timeout
        continue

    # Perform object detection on the captured frame
    detections = net.Detect(img)
    
    # Convert CUDA image to NumPy array for OpenCV processing
    img_cv = cudaToNumpy(img)  # Convert to NumPy array

    # Loop over the detections
    for detection in detections:
        # Debug: Print ClassID and Class Label for all detections
        print(f"Detected ClassID: {detection.ClassID}, Label: {net.GetClassDesc(detection.ClassID)}")
        
        # Filter detections to only include 'person' (ClassID 1)
        if detection.ClassID == 1:  # Class ID 1 corresponds to 'person' in SSD-MobileNet-V2
            # Get the bounding box coordinates using Left, Top, Right, Bottom
            top_left = (int(detection.Left), int(detection.Top))
            bot_right = (int(detection.Right), int(detection.Bottom))
            
            # Draw the bounding box around the detected person using OpenCV
            cv2.rectangle(img_cv, top_left, bot_right, (0, 255, 0), 4)  # Green box

    # Convert the image back from NumPy array to CUDA image
    img = cudaFromNumpy(img_cv)

    # Render the image with bounding boxes for detected persons
    display.Render(img)
    
    # Set the status message showing the FPS of the network
    display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
