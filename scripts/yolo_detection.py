#!/usr/bin/env python3
import rospy
from ultralytics_ros.msg import YoloResult
from vision_msgs.msg import Detection2D
from mavros_msgs.srv import (
    CommandLong,
    CommandLongRequest,
    CommandLongResponse,
    SetMode,
    WaypointSetCurrent,
    WaypointSetCurrentRequest,
    WaypointPull,
)
from mavros_msgs.msg import WaypointReached, VFR_HUD, StatusText
from geometry_msgs.msg import Pose2D
import time, cv2, math, sys
from gps_mavros.srv import GetGPSData, GetGPSDataResponse
from waypoint_mavros.srv import AddWaypointResponse, AddWaypoint, AddWaypointRequest
from waypoint_mavros.srv import DelWaypointResponse, DelWaypoint, DelWaypointRequest
from sensor_msgs.msg import NavSatFix
from waypoint_mavros.srv import (
    UpdateMissionResponse,
    UpdateMission,
    UpdateMissionRequest,
)
from collections import deque
import os, subprocess, rospkg
from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from payload import ServoController

# from  camera_frame import WaypointManager
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

ALT = 20  # in meters (this is ~55 ft)

# both in degrees
HFOV = 68.75
VFOV = 53.13


class Detected_Object_Waypoints:
    def __init__(self):
        self.detected_objects = (
            deque()
        )  # Queue to store detected objects and their GPS waypoints

    def add_object(self, object_name, lat, long, alt, index):
        """
        Adds a detected object with its name and calculated GPS coordinates to the queue.
        """
        if len(self.detected_objects) >= 4:
            return

        detected_object = {
            "name": object_name,
            "latitude": lat,
            "longitude": long,
            "alt": alt,
            "index": index,
        }
        self.detected_objects.append(detected_object)
        rospy.loginfo(
            f"Object '{object_name}' added at LAT: {lat}, LONG: {long}, ALT: {alt}"
        )

    def get_detected_objects(self):
        """
        Returns the queue of detected objects with their GPS waypoints.
        """
        return self.detected_objects

    def rotate_waypoints(self, rotate=-1):
        if self.detected_objects:
            self.detected_objects.rotate(rotate)

    def clear_queue(self):
        return self.detected_objects.clear()


class YoloResultSubscriber:

    def __init__(self):
        self.subscriber = rospy.Subscriber(
            "yolo_result", YoloResult, self.callback, queue_size=1
        )

        self.detected_object_waypoints = Detected_Object_Waypoints()
        self.last_before_rtl = 0
        self.next_after_takeoff = 0
        self.takeoff_index = 0
        self.rtl_index = 0
        self.lap = 0

        self.servo_controller = ServoController()

        rospy.Subscriber(
            "/mavros/mission/reached", WaypointReached, self.update_waypoint_reached
        )
        rospy.Subscriber("/mavros/vfr_hud", VFR_HUD, self.speed_cb)
        self.status_pub = rospy.Publisher(
            "/mavros/statustext/send", StatusText, queue_size=10
        )
        self.detected_photo_pub = rospy.Publisher(
            "/camera/object_detected", Bool, queue_size=10
        )

        rospy.Subscriber("/mavros/statustext/recv", StatusText, self.restart_callback)

        rospy.wait_for_service("/mavros/mission/set_current")
        self.set_wp_srv = rospy.ServiceProxy(
            "/mavros/mission/set_current", WaypointSetCurrent
        )
        self.waypoint_pull_client = rospy.ServiceProxy(
            "/mavros/mission/pull", WaypointPull
        )

        self.last_status_time = 0
        self.status_interval = 8  # seconds between GCS messages

        
        rospy.wait_for_service("/mavros/set_mode")
        self.set_mode = rospy.ServiceProxy("/mavros/set_mode", SetMode)

        self.lasttime = time.time()
        self.run_detection_once = False
        self.waypoint_reached = 0
        
        class_names = rospy.get_param("/yolo_class_names", None)
        while class_names is None:
            rospy.logwarn_throttle_identical(
                5, "Waiting for /yolo_class_names to be set..."
            )
            class_names = rospy.get_param("/yolo_class_names", None)
        self.class_names = eval(class_names)

        self.fetch_mission_indices()

        self.latest_yolo_image_msg = None
        self.bridge = CvBridge()
        self.bridgeObject = CvBridge()
        rp = rospkg.RosPack()
        rospy.Subscriber("/yolo_image", Image, self.yolo_image_callback)
        self.detected_object_path = os.path.join(rp.get_path("video_cam"), "detected")

        if not os.path.exists(self.detected_object_path):
            os.makedirs(self.detected_object_path)

        self.latest_image_msg = None

        self.within_geofence = False

        #Simulation Geofence
        # min_lat1 = min(-35.3633813, -35.3629898, -35.3629416, -35.3633222)
        # max_lat1 = max(-35.3633813, -35.3629898, -35.3629416, -35.3633222)

        # min_lon1 = min(149.1648799, 149.1648397, 149.1654968, 149.1655505)
        # max_lon1 = max(149.1648799, 149.1648397, 149.1654968, 149.1655505)

        # min_lat2 = min(-35.3622067, -35.3615461, -35.3614936, -35.3621629)
        # max_lat2 = max(-35.3622067, -35.3615461, -35.3614936, -35.3621629)

        # min_lon2 = min(149.1647565, 149.1646492, 149.1653734, 149.1654003)
        # max_lon2 = max(149.1647565, 149.1646492, 149.1653734, 149.1654003)

        #Farm
        # min_lat = min(34.0432765, 34.0429942, 34.0426230, 34.0429009)
        # max_lat = max(34.0432765, 34.0429942, 34.0426230, 34.0429009)

        # min_lon = min(-117.8124315, -117.8126916, -117.8120291, -117.8117099)  
        # max_lon = max(-117.8124315, -117.8126916, -117.8120291, -117.8117099)   

        #UAV_Lab
        # min_lat = min(34.059302, 34.058841, 34.058697, 34.059199)
        # max_lat = max(34.059302, 34.058841, 34.058697, 34.059199)

        # min_lon = min(-117.820718, -117.821009, -117.820617, -117.620389)
        # max_lon = max(-117.820718, -117.821009, -117.820617, -117.620389)

        #Soccer Field
        # min_lat = min(34.0527216, 34.0524172, 34.0519594, 34.0523305)
        # max_lat = max(34.0527216, 34.0524172, 34.0519594, 34.0523305)

        # min_lon = min(-117.8191423, -117.8181526, -117.8183833, -117.8193676)
        # max_lon = max(-117.8191423, -117.8181526, -117.8183833, -117.8193676)

        # Runway 1 Geofence (Maryland)
        min_lat1 = min(38.3153622, 38.3156463, 38.3159388, 38.3156653)
        max_lat1 = max(38.3153622, 38.3156463, 38.3159388, 38.3156653)

        min_lon1 = min(-76.5508904, -76.5525976, -76.5525077, -76.5507992)
        max_lon1 = max(-76.5508904, -76.5525976, -76.5525077, -76.5507992)  

        #Runway 2 Geofence (Maryland)
        min_lat2 = min(38.3145083, 38.3147814, 38.3144952, 38.3141858)
        max_lat2 = max(38.3145083, 38.3147814, 38.3144952, 38.3141858)

        min_lon2 = min(-76.5458706, -76.5457834, -76.5440708, -76.5441687)
        max_lon2 = max(-76.5458706, -76.5457834, -76.5440708, -76.5441687)

        self.GEOFENCE1 = {
        "min_lat1": min_lat1,
        "max_lat1": max_lat1,
        "min_lon1": min_lon1,
        "max_lon1": max_lon1
        } 

        self.GEOFENCE2 = {
        "min_lat2": min_lat2,
        "max_lat2": max_lat2,
        "min_lon2": min_lon2,
        "max_lon2": max_lon2
        } 

        rospy.Subscriber("/mavros/global_position/global", NavSatFix, self.geofence_check)
       
    def geofence_check(self,msg):
        lat = msg.latitude
        long = msg.longitude
        self.within_geofence = (
            (self.GEOFENCE1["min_lat1"] <= lat <= self.GEOFENCE1["max_lat1"] and
            self.GEOFENCE1["min_lon1"] <= long <= self.GEOFENCE1["max_lon1"]) or 

            (self.GEOFENCE2["min_lat2"] <= lat <= self.GEOFENCE2["max_lat2"] and
            self.GEOFENCE2["min_lon2"] <= long <= self.GEOFENCE2["max_lon2"])
        )

        if self.within_geofence:
            rospy.loginfo_throttle(10, f"{GREEN}Geofence status: Inside{RESET}")
            message = f"within geofence"
            self.send_status(message, True)
        else:
            rospy.loginfo_throttle(10, f"{YELLOW}Geofence status: Outside{RESET}")
            message = f"NOT within geofence"
            self.send_status(message, True)

    def fetch_mission_indices(self):
        num_waypoints = rospy.get_param("/num_waypoints", None)
        while num_waypoints is None:
            rospy.logwarn_throttle_identical(
                5, "Waiting for /num_waypoints to be set..."
            )
            num_waypoints = rospy.get_param("/num_waypoints", None)
        self.num_waypoints = int(num_waypoints)
        rospy.loginfo(self.num_waypoints)

        takeoff_index = rospy.get_param("/takeoff_index", None)
        while takeoff_index is None:
            rospy.logwarn_throttle_identical(
                5, "Waiting for /takeoff_index to be set..."
            )
            takeoff_index = rospy.get_param("/takeoff_index", None)
        self.takeoff_index = int(takeoff_index)

        rtl_index = rospy.get_param("/rtl_index", None)
        while rtl_index is None:
            rospy.logwarn_throttle_identical(5, "Waiting for /rtl_index to be set...")
            rtl_index = rospy.get_param("/rtl_index", None)
        self.rtl_index = int(rtl_index)

        next_after_takeoff = rospy.get_param("/next_after_takeoff", None)
        while next_after_takeoff is None:
            rospy.logwarn_throttle_identical(
                5, "Waiting for /next_after_takeoff to be set..."
            )
            next_after_takeoff = rospy.get_param("/next_after_takeoff", None)
        self.next_after_takeoff = int(next_after_takeoff)

        last_before_rtl = rospy.get_param("/last_before_rtl", None)
        while last_before_rtl is None:
            rospy.logwarn_throttle_identical(
                5, "Waiting for /last_before_rtl to be set..."
            )
            last_before_rtl = rospy.get_param("/last_before_rtl", None)
        self.last_before_rtl = int(last_before_rtl)

    def sim_image_callback(self, msg):
        self.latest_image_msg = msg

    def yolo_image_callback(self, msg):
        self.latest_yolo_image_msg = msg

    def pull_waypoints(self):
        """Update the mission on the fcu"""
        rospy.wait_for_service("/mavros/mission/pull")
        try:
            response = self.waypoint_pull_client()
            if response.success:
                rospy.loginfo(f"Waypoint pull success: {response.success}")
            else:
                rospy.logwarn("Failed to pull waypoints.")
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")

    def restart_callback(self, msg):
        if "restart" in msg.text.lower():
            message = f"Restarting code"
            self.send_status(message, False)
            self.waypoint_reached = 0
            self.lap = 0
            self.run_detection_once = False
            self.detected_object_waypoints.clear_queue()
            rospy.loginfo(f"Waypoint reached = {self.waypoint_reached}")
            rospy.loginfo(f"Lap =  {self.lap}")
            rospy.loginfo(f"Run Detection Once = {self.run_detection_once}")
            q = self.detected_object_waypoints.get_detected_objects()
            rospy.loginfo(f"Queue Size = {len(q)}, Objects in Queue: {q}")
            self.update_mission_data()
            self.fetch_mission_indices()

    def trigger_camera(self):
        rospy.loginfo("Object Detected. Triggering Jetson-side camera")
        self.detected_photo_pub.publish(Bool(data=True))

    def update_waypoint_reached(self, msg):
        self.waypoint_reached = msg.wp_seq
        q = self.detected_object_waypoints.get_detected_objects()

        if self.waypoint_reached == self.last_before_rtl:
            self.lap += 1

            if len(q) == 0:
                rospy.logwarn("Queue Empty")
                index = self.rtl_index
                self.delete_waypoint_data(index)
                return

            name = q[0]["name"]
            lat = q[0]["latitude"]
            long = q[0]["longitude"]
            index = self.last_before_rtl + 1
            message = f"{name} INSERTED at index: {index}"
            self.send_status(message, False)
            self.change_mode("GUIDED")
            self.send_waypoint_data(lat, long, ALT, index)
            self.change_mode("AUTO")

            self.rtl_index = index
            self.last_before_rtl = -1
            return

        if self.waypoint_reached == self.rtl_index:
            rospy.loginfo(f"{GREEN}Object waypoint reached{RESET}")
            message = f"Object waypoint reached. Payload dropping"
            self.send_status(message, False)
            self.change_mode("GUIDED")
            self.delete_waypoint_data(self.rtl_index + 1)
            self.change_mode("AUTO")
            self.change_mode("GUIDED")
            self.servo_controller.run_sequence()
            rospy.loginfo("Stopping detection...")
            self.subscriber.unregister()
            
        rospy.loginfo(f"{GREEN}Lap Updated: {self.lap}{RESET}")

    def speed_cb(self, msg):
        rospy.loginfo_throttle(10, f"{BLUE}Current airspeed: {msg.airspeed:.2f}{RESET}")

    def gps_calc(
        self, gps_lat, gps_lon, target_x, target_y, img_width, img_height, yaw_degrees
    ):
        """
        Calculate GPS coordinates assuming max shift of 100ft (~0.00030 deg) from image center to edge.
        """

        # Max GPS shift from center to edge (in degrees)
        max_deg_shift = 0.00001373  # ~5 feet

        # Compute center of the image
        image_center_x = img_width / 2.0
        image_center_y = img_height / 2.0

        # Pixel displacement from center
        dx_pixels = target_x - image_center_x
        dy_pixels = (
            target_y - image_center_y
        )  # don't invert; use image convention consistently

        # Max possible pixel distance (diagonal from center to corner)
        max_pixel_distance = math.sqrt((image_center_x) ** 2 + (image_center_y) ** 2)

        # Actual pixel distance from center to target
        actual_pixel_distance = math.sqrt(dx_pixels**2 + dy_pixels**2)

        # Normalize displacement (0 to 1 scale)
        norm_dx = dx_pixels / max_pixel_distance
        norm_dy = dy_pixels / max_pixel_distance

        # Scale normalized values to GPS degree shift (max 0.00030 degrees)
        raw_shift_lon = norm_dx * max_deg_shift
        raw_shift_lat = norm_dy * max_deg_shift

        # Apply yaw rotation (so direction matches drone orientation)
        yaw_rad = math.radians(yaw_degrees)
        rotated_lon = raw_shift_lon * math.cos(yaw_rad) - raw_shift_lat * math.sin(
            yaw_rad
        )
        rotated_lat = raw_shift_lon * math.sin(yaw_rad) + raw_shift_lat * math.cos(
            yaw_rad
        )

        # Apply shift to original GPS coordinates
        new_gps_lat = gps_lat - rotated_lat
        new_gps_lon = gps_lon + rotated_lon

        return new_gps_lat, new_gps_lon

    def callback(self, msg):
        if msg.detections.detections:
            rospy.loginfo(f"{len(msg.detections.detections)} object(s) detected!")
            gps_response = self.request_drone_data()

            # print(type(gps_response))
            bbox_coords = msg.detections.detections
            rospy.logdebug(
                gps_response.latitude, gps_response.longitude, gps_response.altitude
            )
            rospy.loginfo_throttle(5, f"Current wp_reached {self.waypoint_reached}")

            if self.within_geofence:  # time.time() - self.lasttime > 10 :
                # self.lasttime = time.time()
                # self.run_detection_once = True
                for i in range(len(bbox_coords)):
                    # rospy.loginfo(bbox_coords[i].bbox.center)
                    # print(gps_response.latitude, gps_response.longitude, gps_response.altitude)
                    lat, long = self.gps_calc(
                        gps_response.latitude,
                        gps_response.longitude,
                        bbox_coords[i].bbox.center.x,
                        bbox_coords[i].bbox.center.y,
                        640,
                        480,
                        gps_response.yaw,
                    )
                    # rospy.loginfo("calling waypoint service")
                    # waypoint_response = self.send_waypoint_data(
                    #     lat, long, 50
                    # )
                    # rospy.loginfo(f"Waypoint: {waypoint_response.success}")
                    detected_name = self.class_names[bbox_coords[i].results[0].id]
                    index = max(self.next_after_takeoff, self.waypoint_reached + 1)
                    if not self.compare_object_names(detected_name):
                        message = f"{detected_name} DETECTED at index {index}"
                        self.send_status(message, False)
                        rospy.loginfo(f"Calculated Position: LAT: {lat}, LONG:{long}")
                        self.detected_object_waypoints.add_object(
                            self.class_names[bbox_coords[i].results[0].id],
                            lat,
                            long,
                            ALT,
                            index,
                        )
                        # self.trigger_camera()
                        queue_length = len(
                            self.detected_object_waypoints.get_detected_objects()
                        )
                        if self.latest_yolo_image_msg is not None and queue_length <= 2:
                            timestamp = time.strftime("%Y%m%d-%H%M%S")
                            yolo_image = self.bridge.imgmsg_to_cv2(
                                self.latest_yolo_image_msg, desired_encoding="bgr8"
                            )
                            detected_filename = os.path.join(
                                self.detected_object_path,
                                f"detected_photo_{timestamp}.jpg",
                            )
                            cv2.imwrite(detected_filename, yolo_image)
                            rospy.loginfo(f"Photo saved to {detected_filename}")

                        # message = (
                        #     f"'{detected_name}' at " f"LAT: {lat:.6f}, LON: {long:.6f}"
                        # )
                        # self.send_status(message)
                        # message = f"WP added at {index}"
                        # self.send_status(message)
                        # self.change_mode("GUIDED")
                        # self.send_waypoint_data(lat, long, ALT, index)
                        # self.change_mode("AUTO")
                        rospy.loginfo(
                            self.detected_object_waypoints.get_detected_objects()
                        )

        else:
            rospy.loginfo_throttle(5, "No objects detected.")

    def compare_object_names(self, object_name):
        q = self.detected_object_waypoints.get_detected_objects()
        return any(item["name"] == object_name for item in q)

    def set_servo(self, channel, pwm_value):
        try:
            command = CommandLongRequest()
            command.command = 183  # MAV_CMD_DO_SET_SERVO
            command.param1 = channel
            command.param2 = pwm_value
            response = self.command_service(command)
            if response.success:
                rospy.loginfo(f"Servo on channel {channel} set to PWM {pwm_value}")
            else:
                rospy.logerr("Failed to set servo.")
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")

    def request_drone_data(self):
        rospy.wait_for_service("/get_drone_data")
        try:
            get_data = rospy.ServiceProxy("/get_drone_data", GetGPSData)
            response = get_data()
            return GetGPSDataResponse(
                response.latitude, response.longitude, response.altitude, response.yaw
            )
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")

    def update_mission_data(self):
        rospy.loginfo("called mission update function")
        rospy.wait_for_service("/UpdateMission")
        try:
            get_data = rospy.ServiceProxy("/UpdateMission", UpdateMission)
            response = get_data()
            return UpdateMissionResponse(response.success)
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")

    def delete_waypoint_data(self, index):
        rospy.loginfo("called waypoint deletion function")
        rospy.wait_for_service("/DelWaypoint")
        rospy.loginfo("deletion service loaded")
        try:
            del_data = rospy.ServiceProxy("/DelWaypoint", DelWaypoint)
            request = DelWaypointRequest()
            request.index = index
            response = del_data(request)
            return DelWaypointResponse(response.success)
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")

    def send_waypoint_data(self, lat, long, alt, index):
        rospy.loginfo("called waypoint function")
        rospy.wait_for_service("/AddWaypoint")
        rospy.loginfo("addition service loaded")
        try:
            send_data = rospy.ServiceProxy("/AddWaypoint", AddWaypoint)
            request = AddWaypointRequest()
            request.altitude = alt
            request.longitude = long
            request.latitude = lat
            request.index = index
            response = send_data(request)
            return AddWaypointResponse(response.success)
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")

    def activate_servo(self):
        servo_channel = 9
        self.set_servo(servo_channel, 1500)
        time.sleep(1)
        self.set_servo(servo_channel, 1000)

    def change_mode(self, mode):
        rospy.loginfo(f"Setting mode to {mode}...")
        response = self.set_mode(custom_mode=mode)

        if response.mode_sent:
            rospy.loginfo(f"Mode changed to {mode}")
        else:
            rospy.logerr("Failed to change mode")

    # TODO: fix throttle. function still works but throttling needs to be fixed
    def send_status(self, text, throttle):
        now = time.time()

        if throttle and (now - self.last_status_time <= self.status_interval):
            return
        status_msg = StatusText()
        status_msg.severity = 6  # 6 = NOTICE
        status_msg.text = text
        self.status_pub.publish(status_msg)
        self.last_status_time = now

    def set_mission_index(self, index):
        rospy.loginfo(f"Setting current mission waypoint to {index}")
        req = WaypointSetCurrentRequest()
        req.wp_seq = index
        response = self.set_wp_srv(req)
        if response.success:
            rospy.loginfo(f"Mission set to waypoint {index}")
        else:
            rospy.logerr("Failed to set mission index.")


if __name__ == "__main__":
    rospy.init_node("yolo_result_subscriber")
    node = YoloResultSubscriber()
    rospy.spin()