#!/usr/bin/env python
import rospy
import tf2_ros
from sensor_msgs.msg import PointCloud2
import PyKDL
from PyKDL import Vector
import PyKDL
import sensor_msgs.point_cloud2  as pc2
from geometry_msgs.msg import PointStamped, Vector3Stamped
import numpy as np
import tf2_geometry_msgs
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
import sys
class Mark_Maker:
    def __init__(self, camera):
        if camera == "gazebo":
            topic = '/camera/depth/points'
        elif camera == "kinect2":
            topic = '/kinect2/qhd/points'
        elif camera == "zed":
            topic = "/point_cloud/cloud_registered"
        self.camera = camera
        rospy.Subscriber(topic, PointCloud2, self.point_cloud_callback)
        self.pc_frame_id = ""
        self.tf_buffer = tf2_ros.Buffer()
        self.tf2_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.translation = None
        self.rotation = None
        self.transform = None
        self.markerArray = MarkerArray()
        self.count = 0
        self.MARKERS_MAX = 1
        self.point_3d_array = None
        self.radius = 0.8 # radius of the sphere used to localised objects from the center
        self.publisher = rospy.Publisher('visualization_marker_array', MarkerArray, queue_size=10)

    def get_xyz(self, x, y):
        return self.point_3d_array[y][x]

    def to_msg_vector(self, vector):
        msg = PointStamped()
        msg.header.frame_id = self.pc_frame_id
        msg.header.stamp = rospy.Time(0)
        msg.point.x = vector[0]
        msg.point.y = vector[1]
        msg.point.z = vector[2]
        return msg

    def point_cloud_callback(self, msg):
        point_cloud = msg

        print point_cloud.height, point_cloud.width
        self.pc_frame_id = point_cloud.header.frame_id
        if self.pc_frame_id[0] == "/":
            self.pc_frame_id = self.pc_frame_id[1:]
        point_list = []
        for p in pc2.read_points(point_cloud, field_names = ("x", "y", "z")):
            point_list.append((p[0],p[1],p[2]))
        point_array = np.array(point_list)
        self.point_3d_array = np.reshape(point_array, (point_cloud.height,point_cloud.width,3))
        self.transform = self.tf_buffer.lookup_transform("map",
                                                        self.pc_frame_id,
                                                        rospy.Time(0),
                                                        rospy.Duration(10))

        self.translation = self.transform.transform.translation
        self.rotation = self.transform.transform.rotation
        print "recieved point cloud data"
        # Publish the MarkerArray
        self.publisher.publish(self.markerArray)

    def mark(self, x, y, z):
        print "MARKER at", x,y,z
        marker = Marker()
        marker.header.frame_id = "/map"
        marker.type = marker.TEXT_VIEW_FACING 
        marker.action = marker.ADD
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z
        marker.id = self.count
        marker.ns = "-"


        # We add the new marker to the MarkerArray, removing the oldest
        # marker from it when necessary
        # if(self.count > self.MARKERS_MAX):
        #     self.markerArray.markers.pop(0)

        self.markerArray.markers.append(marker)

        # # Renumber the marker IDs
        # id = 0
        # for m in self.markerArray.markers:
        #     m.id = id
        #     id += 1

        # Publish the MarkerArray
        self.publisher.publish(self.markerArray)

        self.count += 1

    def transform_to_kdl(self, t):
        return PyKDL.Frame(PyKDL.Rotation.Quaternion(t.transform.rotation.x, t.transform.rotation.y,
                                                  t.transform.rotation.z, t.transform.rotation.w),
                        PyKDL.Vector(t.transform.translation.x,
                                     t.transform.translation.y,
                                     t.transform.translation.z))

    def do_transform_vector3(self,vector3, transform):
        p = self.transform_to_kdl(transform) * PyKDL.Vector(vector3.point.x, vector3.point.y, vector3.point.z)
        res = Vector3Stamped()
        res.vector.x = p[0]
        res.vector.y = p[1]
        res.vector.z = p[2]
        res.header = transform.header
        return res

    def add_marker(self, _x, _y):
        # if self.camera in ["zed", "kinect2"]:
        #     tmp = _x
        #     _x = _y
        #     _y = tmp
        try:
            x,y,z = self.get_xyz(_x,_y)
        except:
            return False
        vec = self.to_msg_vector(Vector(x,y,z))
        if self.transform:
            transformed_vec = self.do_transform_vector3(vec, self.transform)
            dx,dy,dz = transformed_vec.vector.x, transformed_vec.vector.y, transformed_vec.vector.z
            existing_marker = self.inside_spheres(dx,dy,dz)
            if np.isnan(dx) or np.isnan(dy) or np.isnan(dz):
                return False
            if not self.inside_spheres(dx,dy,dz):
                self.mark(dx,dy,dz)
                print "published marker at", _y, _x
                return False
            else:
                print "already at location"
                return existing_marker
                # self.xyz_to_uv(dx, dy, dz)


    def isclose(self, a, b, rel_tol=1e-09, abs_tol=0.0):
        return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

    def xyz_to_uv(self, x,y,z):
        vec = self.to_msg_vector(Vector(x,y,z))
        transform = self.tf_buffer.lookup_transform("camera_depth_optical_frame","map",rospy.Time(0),rospy.Duration(10))
        transformed_vec = self.do_transform_vector3(vec, transform)
        dx,dy,dz = transformed_vec.vector.x, transformed_vec.vector.y, transformed_vec.vector.z
        point_index = np.where(np.isclose(self.point_3d_array, [dx,dy,dz], 0.000001))
        print x, y, z, "is ", str(point_index)
        try:
            y_count = np.bincount(point_index[0])
            x_count = np.bincount(point_index[1])
            return np.argmax(y_count), np.argmax(x_count)
        except:
            return -1,-1

    def inside_spheres(self, x, y, z):
        for marker in self.markerArray.markers:
            cx = marker.pose.position.x
            cy = marker.pose.position.y
            cz = marker.pose.position.z
            if ((x - cx)**2 + (y - cy)**2 + (z - cz)**2) < (self.radius**2):
                print "*&*&*&*"
                return marker
        return False

    def listen(self):
        while not rospy.is_shutdown():
            print "existing"
            # try:
            #     x,y,z = self.get_xyz(320,240)
            #     vec = self.to_msg_vector(Vector(x,y,z))
            #     if self.transform:
            #         transformed_vec = self.do_transform_vector3(vec, self.transform)
            #         x,y,z = transformed_vec.vector.x, transformed_vec.vector.y, transformed_vec.vector.z
            #         print x,y,z
            #         self.mark(x,y,z)
            #         print "published marker"
            # except Exception as e:
            #     print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

if __name__ == "__main__":
    m = Mark_Maker('gazebo')
    m.listen()
