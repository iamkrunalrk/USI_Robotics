import rclpy
from rclpy.node import Node
from thymio_controller import ControllerNode
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Range
from enum import Enum
from math import inf
import random
import sys
import sys
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())
from models.experimental import attempt_load
from utils.augmentations import  letterbox 
from utils.general import check_img_size, check_imshow, non_max_suppression,scale_coords, set_logging, increment_path
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, time_synchronized

bridge = CvBridge()


class ThymioState(Enum):
    FORWARD = 1
    BACK = 2
    ROTATING = 3
    COMPLETED = 4

class ThymioController(ControllerNode):
    UPDATE_STEP = 1/20
    OUT_OF_RANGE = 0.12
    TARGET_DISTANCE = OUT_OF_RANGE
    TOO_CLOSE = 0.05
    TARGET_ERROR = 0.001
    
    def __init__(self):
        super().__init__('thymio_controller', update_step=self.UPDATE_STEP)

        weights='yolov5s.pt'  # model.pt path(s)
        self.imgsz=640  # inference size (pixels)
        self.conf_thres=0.25  # confidence threshold
        self.iou_thres=0.45  # NMS IOU threshold
        self.max_det=200  # maximum detections per image
        self.classes=None  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms=False  # class-agnostic NMS
        self.augment=False  # augmented inference
        self.visualize=False  # visualize features
        self.line_thickness=2  # bounding box thickness (pixels)
        self.hide_labels=False  # hide labels
        self.hide_conf=False  # hide confidences
        self.half=False  # use FP16 half-precision inference
        self.stride = 32
        device_num=''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False  # show results
        save_crop=False  # save cropped prediction boxes
        nosave=False  # do not save images/videos
        update=False  # update all models
        name='exp'  # save results to project/name

        # Initialize
        set_logging()
        self.device = select_device(device_num)
        self.half &= self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
        stride = int(self.model.stride.max())  # model stride
        imgsz = check_img_size(self.imgsz, s=stride)  # check image size
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names
        if self.half:
            self.model.half()  # to FP16

        # Dataloader
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        self.subscription = self.create_subscription(Image,f'/{self.name}/camera',self.camera_callback,10)

        #Thymio state management
        self.current_state = None
        self.next_state = ThymioState.FORWARD
        
        # Proximity sensors subscription
        self.front_sensors = ["center_left", "center", "center_right"]
        self.lateral_sensors = ["left", "right"]
        self.rear_sensors = ["rear_left", "rear_right"]
        self.proximity_sensors = self.front_sensors + self.lateral_sensors + self.rear_sensors
        self.proximity_distances = dict()
        self.proximity_subscribers = [
            self.create_subscription(Range, f'/{self.name}/proximity/{sensor}', self.create_proximity_callback(sensor), 10)
            for sensor in self.proximity_sensors
        ]
    
    def create_proximity_callback(self, sensor): # Callback that manages thymio proximity sensors
        def proximity_callback(msg):
            self.proximity_distances[sensor] = msg.range if msg.range >= 0.0 else inf
            
        return proximity_callback            
    
    def refresh_callback(self): # Main callback that checks the state of the Thymio
        if self.current_state != ThymioState.COMPLETED:
            if self.odom_pose is None or len(self.proximity_distances) < len(self.proximity_sensors):
                return
    
            if self.next_state != self.current_state:
                
                if self.next_state == ThymioState.FORWARD:
                    self.init_forward()
                elif self.next_state == ThymioState.BACK:
                    self.init_backup()
                elif self.next_state == ThymioState.ROTATING:
                    self.init_rotating()
                
                self.current_state = self.next_state
            
            if self.current_state == ThymioState.FORWARD:
                self.update_forward()
            elif self.current_state == ThymioState.BACK:
                self.update_backup()
            elif self.current_state == ThymioState.ROTATING:
                self.update_rotating()
        else:
            self.get_logger().info("Trash detected! Task completed")
            self.stop()
            cmd_vel = Twist() 
            cmd_vel.linear.x  = 0.0
            cmd_vel.angular.z = -0.7
            self.vel_publisher.publish(cmd_vel)
    
    def init_forward(self):
        self.stop()
    
    def update_forward(self):       
        if any(self.proximity_distances[sensor] < self.TARGET_DISTANCE for sensor in self.front_sensors):
            self.next_state = ThymioState.BACK
            return
            
        cmd_vel = Twist() 
        cmd_vel.linear.x  = 0.1
        cmd_vel.angular.z = 0.0
        self.vel_publisher.publish(cmd_vel)
    
    def init_backup(self):
        self.stop()

    def update_backup(self): # Thymio is turning back after obstacle detection
        if all(self.proximity_distances[sensor] > self.TOO_CLOSE for sensor in self.front_sensors):
            self.next_state = ThymioState.ROTATING
            return
            
        cmd_vel = Twist() 
        cmd_vel.linear.x  = -0.1
        cmd_vel.angular.z =  0.0
        self.vel_publisher.publish(cmd_vel)
        
    def init_rotating(self):
        self.stop()
        self.turn_direction = random.sample([-1, 1], 1)[0] # Random rotation when obstacle is detected
    
    def update_rotating(self): # Rotation after obstacle is detected
        if all(self.proximity_distances[sensor] == inf for sensor in self.front_sensors):
            self.next_state = ThymioState.FORWARD
            return

        cmd_vel = Twist() 
        cmd_vel.linear.x  = 0.0
        cmd_vel.angular.z = self.turn_direction * 3.0
        self.vel_publisher.publish(cmd_vel)
    
    def camera_callback(self, data): # Callback that manages detection of objects
        t0 = time.time()
        img = bridge.imgmsg_to_cv2(data, "rgb8")

        # check for common shapes
        s = np.stack([letterbox(x, self.imgsz, stride=self.stride)[0].shape for x in img], 0)  # shapes
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not self.rect:
            print('WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.')

        # Letterbox
        img0 = img.copy()
        img = img[np.newaxis, :, :, :]        

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        save_dir = '/home/usi/Desktop/random_ars'
        pred = self.model(img,
                     augment=self.augment,
                     visualize=increment_path(save_dir, mkdir=True) if self.visualize else False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
        t2 = time_synchronized()


        # Process detections
        for i, det in enumerate(pred):  # detections per image
            s = f'{i}: '
            s += '%gx%g ' % img.shape[2:]  # print string

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = None if self.hide_labels else (self.names[c] if self.hide_conf else f'{self.names[c]} {conf:.2f}')
                    plot_one_box(xyxy, img0, label=label, color=colors(c, True), line_thickness=self.line_thickness)
                    if (self.names[c] in ['paper', 'glass', 'metal', 'trash', 'biodegradable', 'cardboard'] ) and conf >= 0.60:
                        self.current_state = ThymioState.COMPLETED
                
        
        cv2.imshow("Detection", img0)
        cv2.waitKey(4)    


def main():
    rclpy.init(args=sys.argv)
  
    node = ThymioController()
    node.start()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.stop()


if __name__ == '__main__':
    main()

