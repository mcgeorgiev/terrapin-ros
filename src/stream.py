#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
import cv2
import time, sys
from cv_bridge import CvBridge
import numpy as np
from numpy import inf
from skimage.filters import threshold_otsu
from skimage import morphology
from sklearn.cluster import DBSCAN
import names
import matplotlib.pyplot as plt
from Marker import Mark_Maker
from google_query import GoogleVision
from tensor_flow.tf_files.label_image import TensorFlow
from sklearn.preprocessing import StandardScaler
from sklearn import mixture


class Region:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.roi = None
        self.corners = 0
        self.mean = None

    def crop(self, frame):
        self.roi = frame[self.y1:self.y2, self.x1:self.x2]
        self.fast()

    def fast(self):
        # Initiate FAST object with default values
        fast = cv2.FastFeatureDetector_create()
        kp = fast.detect(self.roi,None)
        self.corners = len(kp)
        points = np.asarray([point.pt for point in kp])
        self.mean = points.mean(axis=0)
        # img3 = None
        # img3 = cv2.drawKeypoints(self.roi, kp, img3)
        # cv2.imwrite(str(self.x1)+"kp3.png",img3)


class Camera:
    def __init__(self, sensor):
        ###
        self.bridge = CvBridge()
        self.color_image = None
        self.depth_image = None
        self.gray_image = None
        self.sensor = sensor
        rospy.init_node('stream', anonymous=True)
        if self.sensor == "hack_kinect2":
            color_topic = "ir"
            depth_topic = "small_depth"
        elif self.sensor == "kinect2":
            color_topic = "/kinect2/qhd/image_color_rect"
            depth_topic = "/kinect2/qhd/image_depth_rect"
        elif self.sensor == "gazebo":
            color_topic = "/camera/rgb/image_raw"
            depth_topic = "/camera/depth/image_raw"
        elif self.sensor == "zed":
            color_topic = "/rgb/image_rect_color"
            depth_topic = "/depth/depth_registered"

        rospy.Subscriber(color_topic, Image, self.callback)
        rospy.Subscriber(depth_topic, Image, self.depth_callback)


        self.depth_mask = None
        self.current_depth_threshold = 0
        self.dbscan_scale = 0.12
        # 1500
        self.depth_max_threshold = 2000
        self.mask = None
        self.labels = None
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        self.marker_object = Mark_Maker(self.sensor)
        self.frame = None
        self.google_vision = GoogleVision()
        self.tensorflow = TensorFlow()

        self.gmm = mixture.GMM(n_components=1, covariance_type='full')
        self.average_color = self.set_floor_hsv()


    def callback(self, data):
        if self.sensor == "hack_kinect2":
            image = self.bridge.imgmsg_to_cv2(data, "16UC1")
        elif self.sensor == "kinect2" or self.sensor == "gazebo" or self.sensor == "zed":
            image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        self.color_image = np.copy(image)



    def depth_callback(self, data):
        if self.sensor == "hack_kinect2":
            image = self.bridge.imgmsg_to_cv2(data, "8UC1")
        elif self.sensor == "kinect2" or self.sensor == "gazebo" or self.sensor == "zed":
            image = self.bridge.imgmsg_to_cv2(data, "32FC1")
        depth_image = np.copy(image)
        if self.sensor == "gazebo" or self.sensor == "zed":
            where_are_NaNs = np.isnan(depth_image)
            depth_image[where_are_NaNs] = 0
        if self.sensor == "zed":
            depth_image *= 1000 #zed cam is in meters while processing happens in mm
            depth_image[depth_image == -inf] = 0
            depth_image[depth_image >= 1E308] = 0
        # # # depth_image = depth_image[:depth_image.shape[0]-2] # shave off
        depth_threshold = threshold_otsu(depth_image)
        if depth_threshold > self.depth_max_threshold:
            depth_threshold = self.depth_max_threshold
        depth_image[depth_image > depth_threshold] = 0
        depth_image[depth_image > 0] = 255
        self.depth_image = depth_image
        self.current_depth_thresh = depth_threshold

    def create_mask(self):
        try:
            if self.depth_mask.all() == None:
                return
        except AttributeError as e:
            pass

        if self.sensor == "hack_kinect2":
            self.gray_image = self.color_image
        elif self.sensor == "kinect2" or self.sensor == "gazebo" or self.sensor == "zed":
            self.gray_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2GRAY)

        self.hsv = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2HSV) # assuming converted is your original stream...

        mask1 = self.gray_image < threshold_otsu(self.gray_image)
        mask2 = self.gray_image > threshold_otsu(self.gray_image)
        # mask = mask1 if np.sum(mask1) < np.sum(mask2) else mask2
        mask = mask2 # not the floor

        mask = mask * self.depth_image
        mask = np.asarray(mask, dtype=np.uint8)
        mask = morphology.remove_small_objects(mask, 50)

        mask[mask > 0] = 255
        self.mask = mask

        blurred_hsv = cv2.medianBlur(self.hsv,15)
        hsv_mask = cv2.inRange(blurred_hsv, self.average_color*0.1, self.average_color*1.9)
        self.mask = self.mask - hsv_mask

    def segment(self):

        X, x_size, y_size = self.create_features(self.mask, self.gray_image)

        regions = {}
        if X.shape[0] > 100:
            X_prime = np.copy(X)
            X_prime = StandardScaler().fit_transform(X_prime)
            #0.7,20
            db = DBSCAN(eps=1, min_samples=20).fit(X_prime)
            labels = db.labels_
            unique_labels = set(labels)

            # Create the regions of interest
            for key in unique_labels:

                # this is the sklean label for noise
                if key == -1:
                    continue

                class_member_mask = (labels == key)

                class_feats = X[class_member_mask, :]
                # probably a noisey cluster detection
                if class_feats.shape[0] < 20:
                    continue


                self.gmm.fit(class_feats)
                covars = np.sqrt(np.asarray(self.gmm._get_covars()))
                alpha = 1
                x1 = int((self.gmm.means_[0, 0] - alpha * covars[0, 0, 0]) * x_size)
                x2 = int((self.gmm.means_[0, 0] + alpha * covars[0, 0, 0]) * x_size)
                y1 = int((self.gmm.means_[0, 1] - alpha * covars[0, 1, 1]) * y_size)
                y2 = int((self.gmm.means_[0, 1] + alpha * covars[0, 1, 1]) * y_size)

                mean_depth = self.gmm.means_[0, 3] * self.current_depth_thresh
                regions[key] = Region(x1, y1, x2, y2)

            for key, region in regions.items():

                regions[key].crop(self.color_image)
                if regions[key].corners > 400:

                    # query the region
                    existing_marker = self.marker_object.add_marker(regions[key].x1+int(regions[key].mean[1]), regions[key].y1+int(regions[key].mean[0]))
                    tensor = False
                    if not existing_marker:
                        name, score = self.tensorflow.query(regions[key].roi)
                        tensor = True


                        if score < 0.5:
                            name, score = self.google_vision.query(regions[key].roi)
                            if score < 0.8: # if the google vision api gives a bad reading
                                continue
                        try:
                            self.marker_object.markerArray.markers[-1].ns = name
                            self.marker_object.markerArray.markers[-1].text = name
                        except:
                            pass
                    else:
                        name = existing_marker.ns

                    # display the region
                    try:
                        radius = int(((regions[key].x2 - regions[key].x1) + (regions[key].y2 - regions[key].y1))/4)
                        cv2.circle(self.frame, (regions[key].x1+int(regions[key].mean[1]), regions[key].y1+int(regions[key].mean[0])), 10, (0,0,255), -1)
                        cv2.circle(self.frame, (regions[key].x1+int(regions[key].mean[1]), regions[key].y1+int(regions[key].mean[0])), radius, (0,255,255), 1)
                        cv2.putText(self.frame, str(regions[key].corners),(regions[key].x1+int(regions[key].mean[1]),regions[key].y1+int(regions[key].mean[0])), self.font, 1,(0,200,200),2 ,cv2.LINE_AA)
                        cv2.putText(self.frame, str(name),(regions[key].x1+int(regions[key].mean[1]),regions[key].y1+int(regions[key].mean[0])-50), self.font, 1,(255,0,255),3 ,cv2.LINE_AA)

                        # if name == "guitar" or name == "coke can":
                        #     cv2.putText(self.frame, "*",(regions[key].x1+int(regions[key].mean[1])-50,regions[key].y1+int(regions[key].mean[0])-50), self.font, 5,(255,0,255),5 ,cv2.LINE_AA)
                        #     try:
                        #         self.marker_object.markerArray.markers[-1].color.r = 1.0
                        #         self.marker_object.markerArray.markers[-1].color.g = 0.0
                        #         self.marker_object.markerArray.markers[-1].color.b = 0.0
                        #     except:
                        #         pass

                    except Exception as e:
                        print e


    def create_features(self, mask, gray_image):
        x_size, y_size = mask.shape[0], mask.shape[1]
        gX, gY = np.meshgrid(range(mask.shape[0]), range(mask.shape[1]))
        X = np.zeros((mask.size, 5))
        X[:, 0] = gX.transpose().ravel() * 1.0 / x_size
        X[:, 1] = gY.transpose().ravel() * 1.0 / y_size
        X[:, 2] = mask.ravel()
        X[:, 3] = self.mask.ravel() * 1.0 / self.current_depth_thresh
        X[:, 4] = gray_image.ravel() * 1.0 / gray_image.max()
        num_samples = X.shape[0]
        step_size = int(num_samples / 3000)

        if step_size == 0:
            step_size = 1

        X = X[0::step_size, :]
        return X, x_size, y_size

    def set_floor_hsv(self):
        try:
            with open("calib_hsv.txt", "rb") as f:
                data = f.readlines()
            all_readings = [eval(item.replace('\n','')) for item in data]
            # return the latest reading
            return np.asarray(all_readings[-1])
        except:
            return np.asarray([  17.17480556,   51.27622222,  105.36063889])

    def listen(self):
        while not rospy.is_shutdown():
            self.frame = self.color_image

            self.create_mask()

            self.segment()


            channeled_mask = cv2.cvtColor(self.mask, cv2.COLOR_GRAY2BGR)
            frame = cv2.resize(self.frame, None, fx=0.8, fy=0.8, interpolation=cv2.INTER_AREA)
            channeled_mask = cv2.resize(channeled_mask, None, fx=0.8, fy=0.8, interpolation=cv2.INTER_NEAREST)
            blank_image = np.zeros((frame.shape[0]+channeled_mask.shape[0],frame.shape[1],3), np.uint8)
            blank_image[blank_image==0] = 125
            blank_image[0:frame.shape[0], 0:frame.shape[1]] = frame
            blank_image[frame.shape[0]:frame.shape[0] + channeled_mask.shape[0], 0:frame.shape[1]] = channeled_mask
            cv2.imshow("Split Window", blank_image)

            print "LENGTH OF MARKERS:",len(self.marker_object.markerArray.markers)
            for marker in self.marker_object.markerArray.markers:
                print marker.ns, marker.pose.position.x, marker.pose.position.y, marker.pose.position.z
            key = cv2.waitKey(delay=1)
            if key == ord('q'):
                break


if __name__ == '__main__':
    if len(sys.argv) == 0:
        print "Please enter a camera as an argument"
        sys.exit()
    if sys.argv[1] not in ["zed", "kinect2", "gazebo"]:
        print "Please enter a valid camera type: zed, kinect2, gazebo"
        sys.exit()
    camera = sys.argv[1]
    c = Camera(camera)
    time.sleep(5)
    c.listen()
    cv2.destroyAllWindows()
