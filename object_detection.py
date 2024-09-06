import numpy as np
import argparse
import os

try:
    import cv2 as cv
except ImportError:
    raise ImportError("Can't find OpenCV Python module. If you've built it from sources without installing it, configure the environment variable PYTHONPATH to 'opencv_build_dir/lib' directory.")

inWidth = 300
inHeight = 300
WHRatio = inWidth / float(inHeight)
inScaleFactor = 0.007843
meanVal = 127.5

# Initialize obj.txt file
f = open("/home/pi/ObjectDetection/obj.txt", "w+")
f.write("")
f.close()

# Start text-to-speech in background
os.system("sudo python /home/pi/ObjectDetection/texttospeech.py &")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script for object detection using deep learning TensorFlow frameworks.')
    parser.add_argument("--video", help="path to video file. If empty, camera's stream will be used")
    parser.add_argument("--prototxt", default="ssd_mobilenet_v1_coco.pbtxt")
    parser.add_argument("--weights", default="frozen_inference_graph.pb")
    parser.add_argument("--num_classes", default=90, type=int)
    parser.add_argument("--thr", default=0.2, type=float, help="confidence threshold to filter out weak detections")
    args = parser.parse_args()

    # Load the network
    net = cv.dnn.readNetFromTensorflow(args.weights, args.prototxt)
    swapRB = True

    classNames = {0: 'background', 1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus',
                  7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign',
                  18: 'dog', 19: 'horse', 24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 23: 'bear',
                  37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard',
                  42: 'surfboard', 46: 'wine glass', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich',
                  55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake',
                  62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv',
                  73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave',
                  86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}

    # Capture video stream
    if args.video:
        cap = cv.VideoCapture(args.video)
    else:
        cap = cv.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        blob = cv.dnn.blobFromImage(frame, inScaleFactor, (inWidth, inHeight), (meanVal, meanVal, meanVal), swapRB)
        net.setInput(blob)
        detections = net.forward()

        cols = frame.shape[1]
        rows = frame.shape[0]

        if cols / float(rows) > WHRatio:
            cropSize = (int(rows * WHRatio), rows)
        else:
            cropSize = (cols, int(cols / WHRatio))

        y1 = int((rows - cropSize[1]) / 2)
        y2 = y1 + cropSize[1]
        x1 = int((cols - cropSize[0]) / 2)
        x2 = x1 + cropSize[0]
        frame = frame[y1:y2, x1:x2]
        cols = frame.shape[1]
        rows = frame.shape[0]

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > args.thr:
                class_id = int(detections[0, 0, i, 1])
                xLeftBottom = int(detections[0, 0, i, 3] * cols)
                yLeftBottom = int(detections[0, 0, i, 4] * rows)
                xRightTop = int(detections[0, 0, i, 5] * cols)
                yRightTop = int(detections[0, 0, i, 6] * rows)

                cv.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop), (0, 255, 0))

                if class_id in classNames:
                    label = classNames[class_id]
                    print("--- OBJECT NAME :", label)

                    # Write detected object label to file
                    f = open("/home/pi/ObjectDetection/obj.txt", "w")
                    f.write(label)
                    f.close()

                    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    yLeftBottom = max(yLeftBottom, labelSize[1])
                    cv.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]),
                                 (xLeftBottom + labelSize[0], yLeftBottom + baseLine), (255, 255, 255), cv.FILLED)
                    cv.putText(frame, label, (xLeftBottom, yLeftBottom), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        cv.imshow("Object Detection", frame)

        if cv.waitKey(1) == ord('q'):
            f = open("/home/pi/ObjectDetection/obj.txt", "w+")
            f.write("")
            f.close()
            os.system("sudo pkill -f texttospeech.py")
            break

    cap.release()
    cv.destroyAllWindows()
