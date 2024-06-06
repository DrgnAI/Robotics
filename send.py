from RobotClass import Robot
import datetime
import sys
from IPython.display import display, HTML
import ipywidgets.widgets as widgets
import traitlets
from traitlets.config.configurable import SingletonConfigurable

import cv2
import numpy as np

import pyrealsense2 as rs

import threading
import time
import random
import socket
from time import gmtime, strftime
from random import randint
import os


class Camera(SingletonConfigurable):
    """
    A class representing a camera interface using RealSense technology, capable of capturing color and depth images.

    Attributes:
        color_value (traitlets.Any): The color image captured by the camera.
        depth_value (numpy.ndarray): The depth image captured by the camera.
        pipeline (rs.pipeline): The RealSense pipeline for capturing images.
        configuration (rs.config): The configuration for the RealSense pipeline.
        color_width (int): The width of the color image.
        color_height (int): The height of the color image.
        color_fps (int): The frames per second for the color image.
        depth_width (int): The width of the depth image.
        depth_height (int): The height of the depth image.
        depth_fps (int): The frames per second for the depth image.
        thread_runnning_flag (bool): A flag to control the image capturing thread.
        pipeline_started (bool): A flag indicating if the pipeline has started.
    """

    color_value = traitlets.Any()

    def __init__(self):
        """
        Initialize the Camera with color and depth streams, and start the pipeline.
        """
        super(Camera, self).__init__()

        self.pipeline = rs.pipeline()
        self.configuration = rs.config()

        # Set resolution for the color camera
        self.color_width = 640
        self.color_height = 480
        self.color_fps = 30
        self.configuration.enable_stream(
            rs.stream.color, self.color_width, self.color_height, rs.format.bgr8, self.color_fps)

        # Set resolution for the depth camera
        self.depth_width = 640
        self.depth_height = 480
        self.depth_fps = 30
        self.configuration.enable_stream(
            rs.stream.depth, self.depth_width, self.depth_height, rs.format.z16, self.depth_fps)

        # Flag to control the thread
        self.thread_runnning_flag = False

        # Start the RGBD sensor
        self.pipeline.start(self.configuration)
        self.pipeline_started = True
        frames = self.pipeline.wait_for_frames()

        # Capture the first color image
        color_frame = frames.get_color_frame()
        image = np.asanyarray(color_frame.get_data())
        self.color_value = image

        # Capture the first depth image
        depth_frame = frames.get_depth_frame()
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(
            depth_image, alpha=0.03), cv2.COLORMAP_JET)
        self.depth_value = depth_colormap

    def _capture_frames(self):
        """
        Capture frames from the RealSense camera in a separate thread.
        """
        while self.thread_runnning_flag:  # Continue until the thread_runnning_flag is set to False
            frames = self.pipeline.wait_for_frames()  # Receive data from RGBD sensor

            color_frame = frames.get_color_frame()  # Get the color image
            # Convert color image to numpy array
            image = np.asanyarray(color_frame.get_data())
            self.color_value = image  # Assign the numpy array image to the color_value variable

            depth_frame = frames.get_depth_frame()  # Get the depth image
            # Convert depth data to numpy array
            depth_image = np.asanyarray(depth_frame.get_data())

            # We only consider the central area of the vision sensor
            depth_image[:190, :] = 0
            depth_image[290:, :] = 0
            depth_image[:, :160] = 0
            depth_image[:, 480:] = 0

            # For object avoidance, we don't consider the distance that is lower than 100mm or bigger than 1000mm
            depth_image[depth_image < 100] = 0
            depth_image[depth_image > 1000] = 0

            # If all values in the depth image are 0, the depth[depth != 0] command will fail
            # We set a specific value here to prevent this failure
            depth_image[0, 0] = 2000

            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(
                depth_image, alpha=0.03), cv2.COLORMAP_JET)
            if depth_image[depth_image != 0].min() < 400:
                cv2.putText(depth_colormap, 'warning!!!', (320, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                self.warning_flag = 1
            else:
                self.warning_flag = 0
            self.depth_value = depth_colormap

    def start(self):
        """
        Start the data capture thread.
        """
        if not self.thread_runnning_flag:  # Only process if no thread is running yet
            # Flag to control the operation of the _capture_frames function
            self.thread_runnning_flag = True
            # Link thread with the function
            self.thread = threading.Thread(target=self._capture_frames)
            self.thread.start()  # Start the thread

    def stop(self):
        """
        Stop the data capture thread.
        """
        if self.thread_runnning_flag:
            self.thread_runnning_flag = False  # Exit the while loop in _capture_frames
            self.thread.join()  # Wait for the thread to exit


def bgr8_to_jpeg(value):
    """
    Convert a numpy array to JPEG encoded data for displaying.

    Args:
        value (numpy.ndarray): The image data to be converted.

    Returns:
        bytes: The JPEG encoded image data.
    """
    return bytes(cv2.imencode('.jpg', value)[1])


# Create a camera object
camera = Camera.instance()
camera.start()  # Start capturing the data


# Initialize the Robot class
robot = Robot()

# Create widgets for displaying the images
# Determine the width of the color image
display_color = widgets.Image(format='jpeg', width='45%')
# Determine the width of the depth image
display_depth = widgets.Image(format='jpeg', width='45%')
layout = widgets.Layout(width='100%')

sidebyside = widgets.HBox([display_color, display_depth],
                          layout=layout)  # Horizontal layout
display(sidebyside)  # Display the widget


def process(change):
    """
    Callback function invoked when traitlets detects a change in the color image.

    Args:
        change (dict): The dictionary containing information about the change.
    """
    image = change['new']  # Retrieve data from the input dict

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(('192.168.1.223', 8888))
    frame = image
    # Resize to a small size; no need for high resolution
    img = cv2.resize(frame, (320, 240))
    head_string = str(img.shape[1]) + ',' + str(img.shape[0])
    sock.sendall(head_string.encode())
    answer = sock.recv(1024)
    print(answer.decode('utf-8'))

    _, img_encoded = cv2.imencode('.jpg', img)
    sock.sendall(len(img_encoded).to_bytes(4, byteorder='big'))
    sock.sendall(img_encoded)
    result = sock.recv(1024)
    eval(result)
    print(result.decode('utf-8'))
    sock.close()

    display_color.value = bgr8_to_jpeg(cv2.resize(image, (160, 120)))
    display_depth.value = bgr8_to_jpeg(
        cv2.resize(camera.depth_value, (160, 120)))


# The camera.observe function will monitor the color_value variable.
# If this value changes, the process function will be executed.
camera.observe(process, names='color_value')
