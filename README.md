# terrapin-ros

This project is a Mapping and Deep Learning Object Recognition Project for use with the Turtlebot. (June - 8th September 2017)

## Description:

-	Produce a 2D occupancy grid map and 3D point cloud using RTAB-Map
-	Autonomously navigate an unknown environment
-	Detect and identify objects in a room
-	Calibrate camera to mask textures

## Hardware Requirements:
-	Turtlebot 2
-	A USB cable (that works, ensure it does) to connect the Kobuki Base and Laptop
-	A Laptop running Ubuntu 16.04 and ROS Kinetic 
-	A camera: 

    - Xbox Kinect v2 connected to the 12v 5 amps socket on the Turtlebot. (Please note the existing cable for this is poor and will need a permanent solution with proper parts.)
 
    - Zed Camera connected to the laptop

## Installation (assumes knowledge of catkin workspaces):
0)	Install Turtlebot packages (replace for kinetic, some will not work) following instructions found here:

       http://wiki.ros.org/turtlebot/Tutorials/indigo/Turtlebot%20Installation
       
1)	Install RTABMAP-ros following instructions found here:

       https://github.com/introlab/rtabmap_ros

2)	Install the Exploration package: 
```
cd ~/catkin_ws/src
git clone https://github.com/bnurbekov/Turtlebot_Navigation
cd ..
catkin_make
```
3)	Install the Google Cloud SDK following instructions found here:

    https://cloud.google.com/sdk/downloads

4)	Enable the SDK for the Google Vision API and install the client library following instructions found here:

    https://cloud.google.com/vision/docs/reference/libraries

5)	Clone and catkin_make in a catkin workspace: 

```
cd ~/catkin_ws/src
git clone https://github.com/mcgeorgiev/terrapin-ros
cd ..
catkin_make
```

6) (OPTIONAL) Tensorflow will need to be trained and placed in `~/terrapin-ros/src/tensorflow/tf_files` following instructions found here:

    https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#0


### Camera Installation (Follow the installation instructions for each piece of software exactly!):
a)	Kinect v2:
  1) https://github.com/OpenKinect/libfreenect2
  2) https://github.com/code-iai/iai_kinect2
  
b)	Zed Camera:
  1) https://www.stereolabs.com/blog/index.php/2015/09/07/use-your-zed-camera-with-ros/ (Including the SDK instruction)

### Install Python dependencies:

```
cd ~/catkin_ws/src/terrapin-ros 
pip install –r requirements.txt
```

## How to run the package (Run each command in a new terminal):

0)	Ensure that the catkin workspace directory is sourced for all terminals used. Usually:

`source ~/catkin_ws/devel/setup.bash`

1)	Run the launch file specific to your camera, either:
```
roslaunch terrapin-ros kinect.launch
OR
roslaunch terrapin-ros zed.launch
```
This will launch the `turtlebot_bringup`, `rtabmap_ros`, `rtabmap visualisation`, specific camera node and `depthimage_to_laserscan` nodes.

2)	Run the object detection programme:

`roslaunch terrapin-ros stream.py`

3)	Run RViz:

...

4)	Run the frontier exploration nodes:
```
rosrun final_project control.py
rosrun final_project mapping.py
```

 - Turtlebot should start mapping! However autonomous navigation can be replaced with tele-operation. 
Replace step 4) with: `roslaunch turtlebot-telep keyboard.launch `

- A calibration tool can be ran which will create a text file with calibration details. This will mask out the selected area. (E.g. flooring)
Run: `python calibration.py` and point at an area and press ‘q’ or ‘p’.

<img src="/readme_images/ex3pc.png" alt="" style="width: 200px;"/>

![Alt text](/readme_images/ex1pc.png)
![Alt text](/readme_images/hsv.png)


