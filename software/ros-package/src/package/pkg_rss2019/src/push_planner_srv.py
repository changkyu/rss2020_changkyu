#!/usr/bin/env python
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import os, sys
dir_current, _ = os.path.split(os.path.abspath(__file__))
dir_software = os.path.join(dir_current,'../../../../..')
if os.path.abspath(dir_software) not in sys.path:
    sys.path.append(dir_software)

from cellphysics import DiffPhysics
from kinoplanner import CellPhysicsPlanner
from simulator import distance_angle

from pkg_rss2019.srv import add_object_srv
from pkg_rss2019.srv import estimate_prop_srv
from pkg_rss2019.srv import plan_pushing_srv,plan_pushing_srvResponse
from pkg_rss2019.srv import save_history_srv

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose2D
from cv_bridge import CvBridge,CvBridgeError

bridge = CvBridge()
planner = {}
pub_plan_debug = None
pub_prop_debug = None

def handle_add_object(req):

    print('\n[Add Object] %s' % req.name)
    if req.name not in planner:
        try:
            image = bridge.imgmsg_to_cv2(req.image, "8UC1")
        except CvBridgeError as e:
            print(e)
        planner[req.name] = CellPhysicsPlanner(req.name,image,[4.9*0.5]*4,req.meter_per_pixel)
        #planner[req.name] = CellPhysicsPlanner(req.name,image,[2.5,1],req.meter_per_pixel)
    else:
        print('Already exists.')

    img_prop = planner[req.name].drawProperties()
    msg_prop = bridge.cv2_to_imgmsg(img_prop)
    pub_prop_debug.publish(msg_prop)
    
    print('done')
    return []

def handle_estimate_prop(req):

    print('\n[Estimate Properties] %s' % req.name)
    if req.name not in planner:
        print('[Error] Please add object first')
        return []

    planner[req.name].addObservation(req.pos_begin, 
                                     req.pos_final, 
                                     req.pos_push, 
                                     req.vec_push, 
                                     req.duration)
    img_prop = planner[req.name].drawProperties()
    msg_prop = bridge.cv2_to_imgmsg(img_prop)
    pub_prop_debug.publish(msg_prop)
    print('done')
    return []

def handle_plan_pushing(req):

    global bridge
    global planner
    global pub_plan_debug

    init = [req.init.x, req.init.y, req.init.theta]
    goal = [req.goal.x, req.goal.y, req.goal.theta]

    print('\n[Planning] %s' % req.name)
    print('{} to {}'.format(init,goal))
    if req.name not in planner:
        print('[Error] Please add object first')
        return []

    #path, actions = planner[req.name].plan(init,goal,pred_step=req.pred_step if req.pred_step>0 else 20)
    path, actions = planner[req.name].plan(init,goal,pred_step=20)

    begs = []
    ends = []
    for i, action in enumerate(actions):
        x,y,yaw = path[i]
        for key, value in action.items() if type(action)==dict else action:
            if key=='pos':
                pos = [0,0]
                pos[0] = np.cos(yaw)*value[0]-np.sin(yaw)*value[1] + x
                pos[1] = np.sin(yaw)*value[0]+np.cos(yaw)*value[1] + y
            elif key=='velocity':
                velocity = [0,0]
                velocity[0] = np.cos(yaw)*value[0]-np.sin(yaw)*value[1]
                velocity[1] = np.sin(yaw)*value[0]+np.cos(yaw)*value[1]
            elif key=='duration':
                duration = value

        pose0 = Pose2D()
        pose0.x, pose0.y = pos
        pose1 = Pose2D()
        pose1.x, pose1.y = np.array(pos) + np.array(velocity)
        begs.append(pose0)
        ends.append(pose1)

    path_res = []
    for i in range(len(path)):
        pose = Pose2D()
        pose.x, pose.y, pose.theta = path[i]
        path_res.append(pose)

    img_plan = planner[req.name].drawPlan(path, actions)
    img_plan = cv2.cvtColor(img_plan, cv2.COLOR_RGBA2BGR)
    msg_plan = bridge.cv2_to_imgmsg(img_plan)
    pub_plan_debug.publish(msg_plan)

    print('done')

    return plan_pushing_srvResponse(path_res,begs,ends)

def handle_save_history(req):

    print('\n[Save] %s to %s' % (req.name,req.fp_save))
    planner[req.name].save(req.fp_save)
    return []

if __name__=='__main__':
    rospy.init_node('push_planner_srv')

    rospy.Service('/pkg_rss2019/add_object',    add_object_srv,    handle_add_object)
    rospy.Service('/pkg_rss2019/estimate_prop', estimate_prop_srv, handle_estimate_prop)
    rospy.Service('/pkg_rss2019/plan_pushing',  plan_pushing_srv,  handle_plan_pushing)
    rospy.Service('/pkg_rss2019/save_history',  save_history_srv,  handle_save_history)
    
    pub_plan_debug = rospy.Publisher('/pkg_rss2019/markers/plan/image', Image, queue_size=10)
    pub_prop_debug = rospy.Publisher('/pkg_rss2019/markers/prop/image', Image, queue_size=10)

    print "[Ready]"
    rospy.spin()