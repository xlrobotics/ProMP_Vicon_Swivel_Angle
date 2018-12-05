import numpy as np
from viconlib import *
# from vicon_bridge.msg import Markers
import math


class MotionMapping():
    def __init__(self):
        # setup gesture control parameters.
        self.y_limit = 80 * np.pi / 180
        self.x_limit = 10 * np.pi / 180
        # max velocity for translation is 0.05, for rotation is 0.05, unit m/s
        self.max_trans_vel = 0.05
        self.max_rot_vel = 0.05
        self.engage_distance = 200  # unit is mm, 0.2 m
        self.engaee_angle = 30 * np.pi / 180  # shoulder angle bigger than this angle, rotate
        self.time_to_top = 2  # unit is second
        self.bend_dist = 150  # unit is mm
        self.last_translation_time = 0
        self.last_rotation_time = 0

        # setup gripper control parameters.
        self.gripper_close = 0.2
        self.gripper_openn = 1.0
        self.palm_open = 120.0  # unit mm
        self.palm_close = 70.0  # unit mm
        self.preshape_limit = 1.4

        # initialize subject body paramters
        self.arm_length = 0
        self.leg_length = 0
        self.calf_length = 0
        self.left_foot = None
        self.right_foot = None

        # set parameter for arm mapping
        self.robot_arm_length = 0.745
        self.l1 = 0.069
        self.l2 = 0.37082
        self.l4 = 0.229525
        self.l3 = 0.37442
        self.robot_joint_max = [np.pi / 2, 1.047, 3.0541, 2.61, 3.059, 2.094, 3.059]
        self.robot_joint_min = [-1.7016, -2.147, -3.0541, -0.05, -3.059, -1.5707, -3.059]

        # save last frame markers
        self.last_markers = []

    def update_vicon_marker(self, markers):
        if len(markers) != 23:
            print
            "Marker size is invalid."
            return

        # check marker lost
        markers = self.data_lost_detection(markers)
        # for upper body
        # for the right arm
        self.shoulder_right = np.matrix([[markers[0].x], [markers[0].y], [markers[0].z]])
        self.elbow_right = np.matrix([[markers[1].x], [markers[1].y], [markers[1].z]]) - self.shoulder_right
        self.wrist_right_1 = np.matrix([[markers[2].x], [markers[2].y], [markers[2].z]]) - self.shoulder_right
        self.wrist_right_2 = np.matrix([[markers[3].x], [markers[3].y], [markers[3].z]]) - self.shoulder_right
        self.wrist_right = (self.wrist_right_1 + self.wrist_right_2) / 2
        self.hand_right = np.matrix([[markers[4].x], [markers[4].y], [markers[4].z]]) - self.shoulder_right
        self.index_right = np.matrix([[markers[13].x], [markers[13].y], [markers[13].z]]) - self.shoulder_right
        self.thumb_right = np.matrix([[markers[10].x], [markers[10].y], [markers[10].z]]) - self.shoulder_right

        # for the left arm
        self.shoulder_left = np.matrix([[markers[5].x], [markers[5].y], [markers[5].z]])
        self.elbow_left = np.matrix([[markers[6].x], [markers[6].y], [markers[6].z]]) - self.shoulder_left
        self.wrist_left_1 = np.matrix([[markers[7].x], [markers[7].y], [markers[7].z]]) - self.shoulder_left
        self.wrist_left_2 = np.matrix([[markers[8].x], [markers[8].y], [markers[8].z]]) - self.shoulder_left
        self.wrist_left = (self.wrist_left_1 + self.wrist_left_2) / 2
        self.hand_left = np.matrix([[markers[9].x], [markers[9].y], [markers[9].z]]) - self.shoulder_left
        self.index_left = np.matrix([[markers[14].x], [markers[14].y], [markers[14].z]]) - self.shoulder_left
        self.thumb_left = np.matrix([[markers[11].x], [markers[11].y], [markers[11].z]]) - self.shoulder_left

        # rotate from starting vicon frame to torso frame attached on human body, torse frame on human body is y
        # facing direction, x towards right arm direction, z upwards.
        shoulder_vector = self.shoulder_right - self.shoulder_left  # is the torse frame x positive direction.
        shoulder_unit = shoulder_vector / np.linalg.norm(shoulder_vector)
        unit_vector_x = np.matrix([1, 0, 0])
        rotation_angle = np.arccos(np.dot(unit_vector_x, shoulder_unit).item(0))
        rotation_direction = np.cross(shoulder_unit.transpose(), unit_vector_x)
        rotation_angle = (1 if rotation_direction[0][2] > 0 else -1) * rotation_angle
        # print 'rotation angle is :', rotation_angle  # this is the rotation angle from shoulder to [1, 0, 0], in z positive direction.
        # calculate rotation frames from torso frame to right arm/left arm.
        v2r = rotz(-np.pi / 2 + rotation_angle)
        v2l = rotz(np.pi / 2 + rotation_angle)
        r_v2r = v2r[0:3, 0:3]
        r_v2l = v2l[0:3, 0:3]

        # in right arm's local world frame
        self.elbow_right_r = 0.001 * np.matmul(r_v2r, self.elbow_right)
        self.wrist_right_r1 = 0.001 * np.matmul(r_v2r, self.wrist_right_1)
        self.wrist_right_r2 = 0.001 * np.matmul(r_v2r, self.wrist_right_2)
        self.hand_right_r = 0.001 * np.matmul(r_v2r, self.hand_right)
        self.index_right_r = 0.001 * np.matmul(r_v2r, self.index_right)

        # in left arm's local world frame
        self.elbow_left_l = 0.001 * np.matmul(r_v2l, self.elbow_left)
        self.wrist_left_l1 = 0.001 * np.matmul(r_v2l, self.wrist_left_1)
        self.wrist_left_l2 = 0.001 * np.matmul(r_v2l, self.wrist_left_2)
        self.hand_left_l = 0.001 * np.matmul(r_v2l, self.hand_left)
        self.index_left_l = 0.001 * np.matmul(r_v2l, self.index_left)

        # for lower body, translate from vicon to fixed torse frame,
        v2t = rotz(np.pi)
        v2t = v2t[0:3, 0:3]
        '''
        self.pelvis_left = np.matmul(v2t, np.matrix([[markers[15].x], [markers[15].y], [markers[15].z]]))
        self.knee_left = np.matmul(v2t, np.matrix([[markers[16].x], [markers[16].y], [markers[16].z]]))
        self.heel_left = np.matmul(v2t, np.matrix([[markers[17].x], [markers[17].y], [markers[17].z]]))
        self.pelvis_right = np.matmul(v2t, np.matrix([[markers[18].x], [markers[18].y], [markers[18].z]]))
        self.knee_right = np.matmul(v2t, np.matrix([[markers[19].x], [markers[19].y], [markers[19].z]]))
        self.heel_right = np.matmul(v2t, np.matrix([[markers[20].x], [markers[20].y], [markers[20].z]]))
        self.shoulder_right_t = np.matmul(v2t, self.shoulder_right)
        self.shoulder_left_t = np.matmul(v2t, self.shoulder_left)
        self.toe_left = np.matmul(v2t, np.matrix([[markers[21].x], [markers[21].y], [markers[21].z]]))
        self.toe_right = np.matmul(v2t, np.matrix([[markers[22].x], [markers[22].y], [markers[22].z]]))
        '''
        # updata last frame markers
        self.last_markers = markers

    def data_lost_detection(self, markers):
        if self.last_markers == []:
            return markers
        if len(markers) != 23:
            return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for i in range(len(markers)):
            if self.isZero(markers[i]):
                markers[i].x = self.last_markers[i].x
                markers[i].y = self.last_markers[i].y
                markers[i].z = self.last_markers[i].z
        return markers

    def isZero(self, data_point):
        if data_point.x == 0.0 and data_point.y == 0.0 and data_point.z == 0.0:
            return True

    def update_arm_length(self):
        self.arm_length = np.linalg.norm(self.elbow_right) + np.linalg.norm(self.elbow_right - self.wrist_right)
        self.arm_length = self.arm_length / 1000

    def update_body_paramters(self):
        self.arm_length = np.linalg.norm(self.elbow_right) + np.linalg.norm(self.elbow_right - self.wrist_right)
        self.arm_length = self.arm_length / 1000
        # print "Arm length is detected as: ", self.arm_length
        temp_length = np.linalg.norm(self.pelvis_right - self.knee_right) + np.linalg.norm(
            self.knee_right - self.heel_right)
        if temp_length > 1800.0 or temp_length < 400:
            print ("someting is wrong, do not resume movement.")
            temp_length = 0
        self.leg_length = temp_length
        # print "Leg length is detected as: ", self.leg_length
        self.calf_length = np.linalg.norm(self.knee_right - self.heel_right)
        # print "calf length is detected as: ", self.calf_length
        self.left_foot = self.heel_left
        self.right_foot = self.heel_right
        # print "Left and right feet initial position renewed."

    def gripper_mapping_left(self):
        dist = math.sqrt((self.index_left[0] - self.thumb_left[0]) ** 2 + (self.index_left[1] - self.thumb_left[1]) ** 2
                         + (self.index_left[2] - self.thumb_left[2]) ** 2)
        gripper_pose = self.map_dist(dist)
        return gripper_pose

    def map_dist(self, dist):
        if dist > self.palm_open:
            dist = self.palm_open
        k = (self.gripper_openn - self.gripper_close) / (self.palm_open - self.palm_close)
        b = self.gripper_close - k * self.palm_close
        pose = k * dist + b
        return pose

    def gripper_mapping_right(self):
        dist = math.sqrt(
            (self.index_right[0] - self.thumb_right[0]) ** 2 + (self.index_right[1] - self.thumb_right[1]) ** 2
            + (self.index_right[2] - self.thumb_right[2]) ** 2)
        gripper_pose = self.map_dist(dist)
        return gripper_pose

    def gripper_preshape_left(self):
        foot_vector_left = [self.toe_left[0] - self.heel_left[0], self.toe_left[1] - self.heel_left[1]]
        foot_vector_left = np.matrix(foot_vector_left / np.linalg.norm(foot_vector_left))
        signal_left = np.arccos(np.dot(foot_vector_left, np.matrix([[0], [1]])).item(0))
        if abs(signal_left) >= self.preshape_limit:
            return True
        else:
            return False

    def gripper_preshape_right(self):
        foot_vector_right = [self.toe_right[0] - self.heel_right[0], self.toe_right[1] - self.heel_right[1]]
        foot_vector_right = np.matrix(foot_vector_right / np.linalg.norm(foot_vector_right))
        signal_right = np.arccos(np.dot(foot_vector_right, np.matrix([[0], [1]])).item(0))
        if abs(signal_right) >= self.preshape_limit:
            return True
        else:
            return False

    def detect_high_knee_left(self):
        # if z distance between one of the knee and pelvis, we say its high knee motion
        # calculate z distance between knee and pelvis
        left_dist = np.abs(self.left_foot[2] - self.heel_left[2]).item(0)
        if left_dist > self.calf_length * 2.0 / 3.0:
            return True
        else:
            return False

    def detect_high_knee_right(self):
        # if z distance between one of the knee and pelvis, we say its high knee motion
        # calculate z distance between knee and pelvis
        right_dist = np.abs(self.right_foot[2] - self.heel_right[2]).item(0)
        if right_dist > self.calf_length * 2.0 / 3.0:
            return True
        else:
            return False

    def detect_bend_down(self):
        # if both left and right z distance between pelvis and heel is lower than length_leg minus bend_dist
        # the person is bending
        left_dist = np.abs(self.pelvis_left[2] - self.heel_left[2]).item(0)
        right_dist = np.abs(self.pelvis_right[2] - self.heel_right[2]).item(0)
        limit = self.leg_length - self.bend_dist
        if left_dist < limit and right_dist < limit:
            # print("bending down detected, pause movement")
            return True
        else:
            return False

    def detect_translation(self):
        # given left heel, right heel start position, left heel, right heel positon, calculate
        # the velocity the base should go.
        left_planner_position = self.heel_left[0:2] - self.left_foot[0:2]
        right_planner_position = self.heel_right[0:2] - self.right_foot[0:2]
        length_left = np.linalg.norm(left_planner_position)
        length_right = np.linalg.norm(right_planner_position)
        # print length_left
        if length_right > self.engage_distance:
            vel_x, vel_y = self.get_trans_vel(right_planner_position)
            return vel_y, -vel_x
        elif length_left > self.engage_distance:
            vel_x, vel_y = self.get_trans_vel(left_planner_position)
            return vel_y, vel_x
        else:
            self.last_translation_time = 0.0
            return 0, 0

    def get_trans_vel(self, vec):
        unit_vector_y = np.matrix([0, 1])
        unit = vec / np.linalg.norm(vec)
        angle = np.arccos(np.dot(unit_vector_y, unit).item(0))
        angle = self.filter_angles(angle)
        self.last_translation_time = self.last_translation_time + 0.02
        # calculate next min jerk velocity
        vel = self.min_jerk(self.max_trans_vel, self.last_translation_time)
        vel_x = vel * np.sin(angle)
        vel_y = vel * np.cos(angle)
        return vel_x, vel_y

    def min_jerk(self, goal, t):
        # generate the next step min jerk velocity
        # the velocity trajectory needs 4 second to reach to the top velocity
        # based on minjerk.pdf in documents, equation (1)
        if t / 0.02 > 100:
            return 0.0938
        d = 2.0 * self.time_to_top
        vel = 2 * goal * (10 / np.power(d, 3) * 3 * np.power(t, 2)
                          - 15 / np.power(d, 4) * 4 * np.power(t, 3)
                          + 6 / np.power(d, 5) * 5 * np.power(t, 4))
        return vel

    def filter_angles(self, angle):
        if np.pi / 2 - self.x_limit < angle and angle < np.pi / 2 + self.x_limit:
            angle = np.pi / 2
        if angle < self.x_limit:
            angle = 0
        if angle > np.pi - self.x_limit:
            angle = np.pi
        return angle

    def detect_rotation(self):
        shoulder_vector = self.shoulder_right_t - self.shoulder_left_t  # shoulder in torso frame.
        shoulder_unit = shoulder_vector / np.linalg.norm(shoulder_vector)
        unit_vector_x = np.matrix([1, 0, 0])
        rotation_angle = np.arccos(np.dot(unit_vector_x, shoulder_unit).item(0))
        rotation_direction = np.cross(unit_vector_x, shoulder_unit.transpose())
        rotation_angle = (1 if rotation_direction[0][2] > 0 else -1) * rotation_angle
        if np.abs(rotation_angle) > self.engaee_angle:
            self.last_rotation_time = self.last_rotation_time + 0.02
            rot_vel = self.min_jerk(self.max_rot_vel, self.last_rotation_time)
            if rotation_angle < 0:
                rot_vel = -rot_vel
            return rot_vel
        else:
            self.last_rotation_time = 0
            return 0

    def get_angles(self):
        # left hand and right hand calculation of approach vector is opposite. wrist 1 and wirst 2 for calculate right angle
        # is flipped intentionally.
        angles_right = self.cal_angles(self.elbow_right_r, self.wrist_right_r2, self.wrist_right_r1, self.hand_right_r,
                                       self.index_right_r)
        angles_left = self.cal_angles(self.elbow_left_l, self.wrist_left_l1, self.wrist_left_l2, self.hand_left_l,
                                      self.index_left_l)
        robot_right = self.mapping_right(angles_right)
        robot_left = self.mapping_left(angles_left)
        robot_left[6] = robot_left[6] - np.pi
        return robot_right + robot_left

    def cal_angles(self, h_elbow, h_wrist1, h_wrist2, h_palm, h_index):
        h_wrist = (h_wrist1 + h_wrist2) / 2.0
        swivel = self.swivel_angle(h_elbow, h_wrist)
        # print("swivel angles is")
        # print swivel
        # get robot wrist position using task space mapping
        wrist = self.wrist_mapping(h_wrist)
        approach = self.get_approach(h_palm, h_wrist1, h_wrist2)
        angles = self.ik_baxter_full(wrist, approach, swivel)
        palm_angle = self.get_palm_angle(h_index, h_wrist1, h_wrist2)
        angles[6] = palm_angle
        # print palm_angle
        return swivel
        # return angles

    def ik_baxter_full(self, ee, approach, swivel):
        theta = [0, 0, 0, 0, 0, 0, 0]
        n = approach / np.linalg.norm(approach)
        pw = ee - self.l4 * n
        # calculate first four angles of baxter using swivel angle and wrist position
        theta[0:4] = self.ik_swivel(pw, swivel)

        # solve for the last three joints using
        T01 = rotz(theta[0]) * transy(-self.l1)
        T12 = rotx(theta[1])
        T23 = roty(theta[2]) * transy(-self.l2)
        T34 = rotx(theta[3])
        T04 = np.matmul(np.matmul(T01, T12), np.matmul(T23, T34))
        ee_p = inver_t(T04) * np.matrix([[ee[0].item(0)], [ee[1].item(0)], [ee[2].item(0)], [1]])
        p1 = ee_p[0].item(0)
        p2 = ee_p[1].item(0)
        p3 = ee_p[2].item(0)
        cos6 = -(p2 + self.l3) / self.l4
        sin6 = np.sqrt((p1 * p1 + p3 * p3) / (self.l4 * self.l4))
        theta6 = np.arctan2(sin6, cos6)
        sin5 = -p1 / np.sin(theta6) / self.l4
        cos5 = -p3 / np.sin(theta6) / self.l4
        theta5 = -np.arctan2(sin5, cos5)
        theta7 = 0
        theta[4:] = [theta5, theta6, theta7]
        return theta

    @staticmethod
    def get_palm_angle(h_index, h_wrist1, h_wrist2):
        # print h_wrist1, h_wrist2, h_index
        wrist_vector = h_wrist1 - h_wrist2
        wrist_unit = wrist_vector / np.linalg.norm(wrist_vector)
        h_wrist = (h_wrist1 + h_wrist2) / 2.0
        index_vector = h_index - h_wrist
        index_unit = index_vector / np.linalg.norm(index_vector)
        dot_p = np.dot(wrist_unit.transpose(), index_unit).item(0)
        # print dot_p
        angle = np.arccos(dot_p).item(0)
        # print angle
        return np.pi - angle

    @staticmethod
    def get_approach(h_palm, h_wrist1, h_wrist2):
        wrist = (h_wrist1 + h_wrist2) / 2.0
        v1 = h_wrist2 - h_wrist1
        v2 = h_palm - wrist
        a = [v1[0].item(0), v1[1].item(0), v1[2].item(0)]
        b = [v2[0].item(0), v2[1].item(0), v2[2].item(0)]
        approach = np.cross(a, b)
        approach = approach / np.linalg.norm(approach)
        # print approach
        return np.matrix([[approach[0].item(0)], [approach[1].item(0)], [approach[2].item(0)]])

    def mapping_left(self, angles):
        # define robot joint limits [s0, s1, e0, e1, w0, w1, w2]
        # check joint limits
        angles[0] = angles[0] + np.pi / 4.0  # jonit 0 has different zero configurations.
        angles[2] = -angles[2]
        angles[6] = angles[6] - 10.0 * np.pi / 180.0
        # print "left,", angles[6]
        for i in range(0, 7):
            if angles[i] > self.robot_joint_max[i]:
                angles[i] = self.robot_joint_max[i]
            if angles[i] < self.robot_joint_min[i]:
                angles[i] = self.robot_joint_min[i]
        return angles

    def mapping_right(self, angles):
        # define robot joint limits [s0, s1, e0, e1, w0, w1, w2]
        # check joint limits
        angles[0] = angles[0] - np.pi / 4.0  # jonit 0 has different zero configurations.
        angles[2] = -angles[2]
        # print "right,", angles[6]
        angles[6] = angles[6] + 10.0 * np.pi / 180.0  # for mapping to the right finger orientation
        for i in range(0, 7):
            if angles[i] > self.robot_joint_max[i]:
                angles[i] = self.robot_joint_max[i]
            if angles[i] < self.robot_joint_min[i]:
                angles[i] = self.robot_joint_min[i]
        return angles

    def ik_baxter(self, Pe, Pw):
        l1 = self.l1
        l2 = self.l2
        l3 = self.l3
        sin2 = -Pe[2].item(0) / l2
        l_pe = np.linalg.norm(Pe)
        cos2 = (l_pe * l_pe - l1 * l1 - l2 * l2) / (2 * l1 * l2)
        theta2 = np.arctan2(sin2, cos2)
        sin1 = Pe[0] / (l1 + np.cos(theta2) * l2)
        cos1 = -Pe[1] / (l1 + np.cos(theta2) * l2)
        theta1 = np.arctan2(sin1.item(0), cos1.item(0))
        T01 = rotz(theta1) * transy(-l1)
        T12 = rotx(theta2)
        T01_inver = inver_t(T01)
        T12_inver = inver_t(T12)
        Pw_2 = np.matmul(T12_inver, np.matmul(T01_inver, np.matrix(
            [Pw[0].item(0), Pw[1].item(0), Pw[2].item(0), 1]).transpose()))  # Pw in frame 2
        p1 = Pw_2[0].item(0)
        p2 = Pw_2[1].item(0)
        p3 = Pw_2[2].item(0)
        cos4 = -(p2 + l2) / l3
        sin4 = math.sqrt((p1 * p1 + p3 * p3) / (l3 * l3))
        theta4 = np.arctan2(sin4, cos4)
        sin3 = -p1 / np.sin(theta4) / l3
        cos3 = -p3 / np.sin(theta4) / l3
        theta3 = np.arctan2(sin3, cos3)
        theta = [theta1, theta2, theta3, theta4, 0, 0, 0]
        # print(theta)
        return theta

    # use baxter wrist position and human swivel angle to solve baxter ik. pw 3x1
    def ik_swivel(self, Pw, swivel):
        l1 = self.l1 + self.l2
        l2 = self.l3
        Ps = np.matrix([[0], [0], [0]])
        n = (Pw - Ps) / np.linalg.norm(Pw - Ps)
        a = np.matrix([[0], [0], [-1]])
        u_temp = a - np.dot(a.transpose(), n).item(0) * n
        u = u_temp / np.linalg.norm(u_temp)
        R = self.rot_axis_angle(n, -swivel)
        f = np.matmul(R, u)
        l_ws = np.linalg.norm(Pw - Ps)
        cos_theta = (l1 * l1 + l_ws * l_ws - l2 * l2) / (2 * l1 * np.linalg.norm(Pw - Ps))
        sin_theta = np.sqrt(1 - cos_theta ** 2)
        f_prime = cos_theta * n + sin_theta * f
        V_se = l1 * f_prime
        Pe = V_se + Ps
        angles = self.ik_baxter(Pe, Pw)
        return angles

    @staticmethod
    def rot_axis_angle(a, theta):
        ct = np.cos(theta)
        st = np.sin(theta)
        x = a[0].item(0)
        y = a[1].item(0)
        z = a[2].item(0)
        R11 = ct + x * x * (1 - ct)
        R12 = x * y * (1 - ct) - z * st
        R13 = x * z * (1 - ct) + y * st
        R21 = y * x * (1 - ct) + z * st
        R22 = ct + y * y * (1 - ct)
        R23 = y * z * (1 - ct) - x * st
        R31 = z * x * (1 - ct) - y * st
        R32 = z * y * (1 - ct) + x * st
        R33 = ct + z * z * (1 - ct)
        R = np.matrix([[R11, R12, R13],
                       [R21, R22, R23],
                       [R31, R32, R33]])
        return R

    # calculates human swivel angle
    @staticmethod
    def swivel_angle(pe, pw):
        ps = np.matrix([[0], [0], [0]])
        a = np.matrix([[0], [0], [-1]])
        n = (pw - ps) / np.linalg.norm(pw - ps)
        f = pe - ps
        f_ec = f - np.dot(f.transpose(), n).item(0) * n
        f_norm = f_ec / np.linalg.norm(f_ec)
        a_projection = a - np.dot(a.transpose(), n).item(0) * n
        u = a_projection / np.linalg.norm(a_projection)
        phi_sin = np.dot(n.transpose(), np.cross(f_norm.transpose(), u.transpose()).transpose()).item(0)
        phi_cos = np.dot(f_norm.transpose(), u).item(0)
        phi = np.arctan2(phi_sin, phi_cos).item(0)
        swivel_angle = phi
        return swivel_angle

    def wrist_mapping(self, human_wrist):
        length_wrist = np.linalg.norm(human_wrist)
        n = (human_wrist) / length_wrist
        ratio = self.robot_arm_length / self.arm_length
        robot_wrist = length_wrist * ratio * n
        robot_wrist[2] = robot_wrist[2] + 0.15
        # print robot_wrist
        return robot_wrist

