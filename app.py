from flask import Flask, render_template, request, redirect, url_for, make_response
import cv2
import numpy as np
import os

app = Flask(__name__)

cap = None

@app.route("/", methods=["GET", "POST"])
def homepage():
    return render_template('homepage.html')

@app.route("/exit", methods=["GET", "POST"])
def exit():
    
    global cap
       
    if cap is not None:
        cap.release()
        cv2.destroyAllWindows()
        # cap = None
    
    return render_template('upload.html')

@app.route("/upload", methods=["GET", "POST"])
def upload():
    
    global cap
    
    if request.method == "POST":
        
        file = request.files['video']

        # Save file to static folder
        file.save(os.path.join('static', file.filename))

        # Initialize video capture
        cap = cv2.VideoCapture(os.path.join('static', file.filename))

        # Load YOLO weights and configurations
        net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL_FP16)
        # Define classes for detection
        classes = []

        with open("coco.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]

        # Set minimum probability for detection
        confidence_threshold = 0.5

        # Set non-maximum suppression threshold
        nms_threshold = 0.4

        # Set video size
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Loop through frames
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                break
            height, width, channels = frame.shape

            # Convert frame to blob for input to neural network
            blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)

            # Set input for neural network
            net.setInput(blob)

            # Get output layer names
            layer_names = net.getLayerNames()
            output_layers = [layer_names[layer_id - 1] for layer_id in net.getUnconnectedOutLayers()]

            # Run forward pass through neural network
            outputs = net.forward(output_layers)

            # Initialize variables for object detection
            boxes = []
            confidences = []
            class_ids = []

            # Loop through outputs and detect objects
            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > confidence_threshold:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            # Apply non-maximum suppression to eliminate overlapping boxes
            indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
            
            near_color = (0, 0, 255) # Red
            far_color = (0, 255, 0) # Green
            text_color = (255, 255, 255)

            # Loop through indices and draw boxes and labels around objects
            if len(indices) > 0:
                for i in indices.flatten():
                    box = boxes[i]
                    x, y, w, h = box
                
                    # Determine center of box
                    center_x = int(x + w / 2)
                    center_y = int(y + h / 2)

                    # Determine if object is near or far based on y position
                    if center_y < 400:
                        color = far_color
                        text = "FAR"
                    else:
                        color = near_color
                        text = "NEAR"

                    # Draw box and label on frame
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness=2)
                    cv2.putText(frame, f"{classes[class_ids[i]]} {confidences[i]:.2f} {text}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, thickness=2)

            # Display resulting frame
            cv2.imshow("Frame", frame)

            # Press 'q' to quit
            if cv2.waitKey(1) == ord('q'):
                break

        # Release video capture and destroy all windows
        cap.release()
        cv2.destroyAllWindows()


        return render_template('upload.html')

    
    return render_template('upload.html')

if __name__ == "__main__":
    app.run(debug=True)
