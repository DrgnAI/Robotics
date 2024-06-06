import sys
import cv2
import numpy as np
import os
import socket
import time
import ikomia
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Constants
TARGET_BOX_AREA = 20000
TEST = 30000
THRESHOLD = 0.8


def calculate_histogram(frame, bbox):
    """
    Calculate the histogram for a region of interest (ROI) in the given frame.

    Args:
        frame (numpy.ndarray): The input frame from which the histogram is calculated.
        bbox (tuple): The bounding box coordinates (x1, y1, width, height) of the ROI.

    Returns:
        numpy.ndarray: The normalized histogram of the ROI.
    """
    x1, y1, width, height = bbox
    x1 = int(x1)
    y1 = int(y1)
    width = int(width)
    height = int(height)
    roi = frame[y1:y1+height, x1:x1+width]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)),
                       np.array((180., 255., 255.)))
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    return roi_hist


def compare_histograms(hist1, hist2):
    """
    Compare two histograms using correlation.

    Args:
        hist1 (numpy.ndarray): The first histogram.
        hist2 (numpy.ndarray): The second histogram.

    Returns:
        float: The correlation score between the two histograms.
    """
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)


def calculate_speed_from_box_size(current_area, target_area):
    """
    Calculate the speed based on the difference between the current and target areas.

    Args:
        current_area (float): The area of the current bounding box.
        target_area (float): The desired area of the bounding box.

    Returns:
        float: The calculated speed for the robot.
    """
    error = current_area - target_area
    Kp = 0.00002  # Proportional gain, tune this parameter based on testing
    test_error = TEST - error
    speed = Kp * test_error
    print(f'CURRENT AREA: {current_area}')
    print(f'ERROR: {error}')
    print(f'Speed: {speed}')
    if current_area > 25000:
        return 0
    else:
        return max(min(speed, 0.35), 0)


def main():
    """
    Main function to initialize socket connection, set up the workflow, process video frames, and control the robot.
    """
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    listener.bind(('192.168.1.223', 8888))
    listener.listen(5)
    print('Waiting for connect...')

    # Init your workflow
    wf = Workflow()

    # Add object detection algorithm
    detector = wf.add_task(name="infer_yolo_v7", auto_connect=True)

    # Add ByteTrack tracking algorithm
    tracking = wf.add_task(name="infer_deepsort", auto_connect=True)
    tracking.set_parameters({
        "categories": "person",  # Set to track only humans
        "conf_thres": "0.5"
    })

    # Get video properties for the output
    frame_width = 320
    frame_height = 480
    frame_rate = 30
    person_id = None

    while True:
        client_executor, addr = listener.accept()
        print('Accept new connection from %s:%s...' % addr)

        head_string = client_executor.recv(1024)
        if not head_string:
            break

        head_array = np.fromstring(head_string, dtype='int', sep=',')
        cols = head_array[0]
        rows = head_array[1]

        print(cols, rows)
        send_message = "working"
        client_executor.send(bytes(send_message.encode('utf-8')))

        t1 = cv2.getTickCount()  # Time

        received_length_bytes = client_executor.recv(4)
        received_length = int.from_bytes(
            received_length_bytes, byteorder='big')
        received_data = b''
        while len(received_data) < received_length:
            received_data += client_executor.recv(
                received_length - len(received_data))

        nparr = np.frombuffer(received_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Run the workflow on current frame
        wf.run_on(array=frame)

        # Get results
        image_out = tracking.get_output(0)
        obj_detect_out = tracking.get_output(1)

        result = ""
        client_executor.send(bytes(result.encode('utf-8')))

        # Convert the result to BGR color space for displaying
        img_out = image_out.get_image_with_graphics(obj_detect_out)
        img_res = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)

        cv2.imshow('Robot Camera Feed', img_res)
        cv2.waitKey(1)

        print(addr, "data transfer finished")

        bboxes = obj_detect_out.get_objects()
        if bboxes:
            if person_id is None:
                person_id = bboxes[0].id
                bbox = bboxes[0].box
                reference_histogram = calculate_histogram(frame, bbox)
                print(f"Locked onto person with ID: {person_id}")

            bbox = next((b for b in bboxes if b.id == person_id), None)
            if not bbox:
                max_score = 0
                for b in bboxes:
                    current_histogram = calculate_histogram(frame, b.box)
                    score = compare_histograms(
                        reference_histogram, current_histogram)
                    if score > max_score:
                        max_score = score
                        bbox = b
                        if score > THRESHOLD:
                            person_id = b.id
                            reference_histogram = current_histogram

            if bbox:
                x1, y1, width, height = bbox.box
                current_area = width * height
                center_x = x1 + (width / 2)
                frame_center = frame_width / 2

                if center_x < frame_center - 60:  # Person is on the left
                    result = f"robot.left(0.4)\n"
                elif center_x > frame_center + 60:  # Person is on the right
                    result = f"robot.right(0.4)\n"
                else:
                    speed = calculate_speed_from_box_size(
                        current_area, TARGET_BOX_AREA)
                    result = f"robot.forward({speed})\n"
            else:
                result = "robot.stop()\n"
                person_id = None
        else:
            result = "robot.stop()"

        cv2.imshow('frame', frame)
        cv2.waitKey(1)

        client_executor.send(bytes(result.encode('utf-8')))
        print(f"Result sent: {result.strip()}")

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
