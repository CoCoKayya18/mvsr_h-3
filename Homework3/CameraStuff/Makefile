# Makefile for setting up and running the ROS stream publisher, camera info service, and camera calibration

# Define variables
SHELL := /bin/bash
ROS_SETUP = source /opt/ros/noetic/setup.bash
SCRIPT = streamToVideo.py
SERVICE_SCRIPT = set_camera_info_service.py
RTMP_STREAM = rtmp://192.168.0.4/live/WSL_Stream
DISPLAY_ENV = $(shell echo $$(hostname -I | awk '{print $$1}'):0)

# Default target
all: setup run

# Setup environment
setup:
	# @echo "Setting up the ROS environment..."
	# @$(SHELL) -c '$(ROS_SETUP)'
	# @echo "Installing necessary packages..."
	# @sudo apt-get update
	# @sudo apt-get install -y ffmpeg python3-opencv ros-noetic-rospy ros-noetic-roscpp ros-noetic-std-msgs ros-noetic-sensor-msgs ros-noetic-camera-info-manager ros-noetic-cv-bridge ros-noetic-camera-calibration x11-apps qt5-default qt5-qmake qtbase5-dev-tools gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly

# Run the stream publisher script and the camera info service
run: setup
	@echo "Killing any existing roscore instances..."
	@pkill roscore || true
	@echo "Starting roscore..."
	@$(SHELL) -c '$(ROS_SETUP) && roscore &' # Start roscore in the background
	@sleep 5 # Wait a bit for roscore to start
	@echo "Running the stream publisher script..."
	@$(SHELL) -c 'DISPLAY=$(DISPLAY_ENV) $(ROS_SETUP) && python3 $(SCRIPT) &' # Run the stream publisher in the background
	@sleep 5 # Wait a bit for the stream to start
	@echo "Running the camera info service..."
	@$(SHELL) -c 'DISPLAY=$(DISPLAY_ENV) $(ROS_SETUP) && python3 $(SERVICE_SCRIPT) &' # Run the camera info service in the background

# Run ROS camera calibration
calibrate: setup
	@echo "Killing any existing roscore instances..."
	@pkill roscore || true
	@echo "Starting roscore..."
	@$(SHELL) -c '$(ROS_SETUP) && roscore &' # Start roscore in the background
	@sleep 5 # Wait a bit for roscore to start
	@echo "Running the stream publisher script..."
	@$(SHELL) -c 'DISPLAY=$(DISPLAY_ENV) $(ROS_SETUP) && python3 $(SCRIPT) &' # Run the stream publisher in the background
	@sleep 5 # Wait a bit for the stream to start
	@echo "Running the camera info service..."
	@$(SHELL) -c 'DISPLAY=$(DISPLAY_ENV) $(ROS_SETUP) && python3 $(SERVICE_SCRIPT) &' # Run the camera info service in the background
	@sleep 5 # Wait a bit for the services to be ready
	@echo "Running ROS camera calibration..."
	@$(SHELL) -c 'DISPLAY=$(DISPLAY_ENV) $(ROS_SETUP) && rosrun camera_calibration cameracalibrator.py --size 8x6 --square 0.025 image:=/camera/image_raw camera:=/camera'

# Clean up environment (if needed)
clean:
	@echo "Cleaning up..."
	@sudo apt-get remove -y ffmpeg python3-opencv ros-noetic-rospy ros-noetic-roscpp ros-noetic-std-msgs ros-noetic-sensor-msgs ros-noetic-camera-info-manager ros-noetic-cv-bridge ros-noetic-camera-calibration x11-apps qt5-default qt5-qmake qtbase5-dev-tools gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly
	@sudo apt-get autoremove -y
