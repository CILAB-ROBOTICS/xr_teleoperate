import numpy as np
import time
import argparse
import cv2
from multiprocessing import shared_memory, Value, Array, Lock
import threading
import logging_mp

from tactile_collecting.sensors.app.serial_util import get_available_serial_port
from tactile_collecting.sensors.app.stage import DummyStage
from tactile_collecting.sensors.sensors import MultiSensors, SensorEnv

logging_mp.basic_config(level=logging_mp.INFO)
logger_mp = logging_mp.get_logger(__name__)

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from televuer import TeleVuerWrapper
from teleop.robot_control.robot_arm import G1_29_ArmController, G1_23_ArmController, H1_2_ArmController, H1_ArmController
from teleop.robot_control.robot_arm_ik import G1_29_ArmIK, G1_23_ArmIK, H1_2_ArmIK, H1_ArmIK
from teleop.robot_control.robot_hand_inspire import Inspire_Controller
from teleop.image_server.image_client import ImageClient
from teleop.utils.episode_writer import EpisodeWriter
from sshkeyboard import listen_keyboard, stop_listening

# for simulation
from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_

import multiprocessing as mp
mp.set_start_method('fork', force=True)

# multi-camera (RealSense) helpers
try:
    import teleop.multi_camera as mc
except Exception:
    mc = None


def _select_realsense_serials():
    if mc is None:
        return []
    serials = mc.list_realsense_serials()
    if len(serials) == 0:
        serials = ["SYNTH0", "SYNTH1", "SYNTH2"]
    elif len(serials) == 1:
        serials = [serials[0], "SYNTH1", "SYNTH2"]
    elif len(serials) == 2:
        serials = [serials[0], serials[1], "SYNTH2"]
    else:
        serials = serials[:3]
    return serials


def _init_realsense_manager():
    if mc is None:
        logger_mp.error("multi_camera module not available; RealSense streaming disabled.")
        return None
    serials = _select_realsense_serials()
    specs = [
        mc.RSSpec("wrist_left", serials[0], width=640, height=480, fps=30, need_depth=False),
        mc.RSSpec("wrist_right", serials[1], width=640, height=480, fps=30, need_depth=False),
        mc.RSSpec("front", serials[2], width=640, height=480, fps=30, need_depth=True),
    ]
    logger_mp.info(f"Initializing RealSense manager with serials: {serials}")
    return mc.RSManager(specs)


def _colorize_depth(depth_u16: np.ndarray):
    if mc is None:
        return None
    dm = depth_u16.astype(np.float32) * mc.DEPTH_SCALE
    dm = np.clip(dm, mc.DEPTH_MIN_M, mc.DEPTH_MAX_M)
    norm = ((dm - mc.DEPTH_MIN_M) / (mc.DEPTH_MAX_M - mc.DEPTH_MIN_M) * 255.0).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)


def publish_reset_category(category: int, publisher):  # Scene Reset signal
    msg = String_(data=str(category))
    publisher.Write(msg)
    logger_mp.info(f"published reset category: {category}")

num_tactile_per_hand = 1062 # add

# state transition
start_signal = True
running = True
should_toggle_recording = False
is_recording = False


def on_press(key):
    global running, start_signal, should_toggle_recording
    if key == 'r':
        start_signal = True
        logger_mp.info("Program start signal received.")
    elif key == 'q':
        stop_listening()
        running = False
    elif key == 's':
        should_toggle_recording = True
    else:
        logger_mp.info(f"{key} was pressed, but no action is defined for this key.")


listen_keyboard_thread = threading.Thread(target=listen_keyboard,
                                          kwargs={"on_press": on_press, "until": None, "sequential": False, },
                                          daemon=True)
listen_keyboard_thread.start()



class Dummy_ArmController:

    def speed_gradual_max(self):
        pass

    def get_current_dual_arm_q(self):
        return np.zeros(14)

    def get_current_dual_arm_dq(self):
        return np.zeros(14)

    def ctrl_dual_arm(self, dual_arm_q, dual_arm_tauff):
        pass

    def ctrl_dual_arm_go_home(self):
        pass

class Dummy_ArmIK:

    def solve_ik(self, left_arm_pose, right_arm_pose, current_lr_arm_q, current_lr_arm_dq):
        return np.zeros(14), np.zeros(14)


class DummyHand_Controller:

    def ctrl_dual_hand(self, left_hand_pos, right_hand_pos):
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_dir', type=str, default='./utils/data', help='path to save data')
    parser.add_argument('--frequency', type=float, default=60.0, help='save data\'s frequency')

    # basic control parameters
    parser.add_argument('--xr-mode', type=str, choices=['hand', 'controller'], default='hand',
                        help='Select XR device tracking source')
    parser.add_argument('--arm', type=str, choices=['G1_29', 'G1_23', 'H1_2', 'H1', 'dummy'], default='G1_29',
                        help='Select arm controller')
    parser.add_argument('--ee', type=str, choices=['dex1', 'dex3', 'inspire1', 'brainco'],
                        help='Select end effector controller')
    # mode flags
    parser.add_argument('--record', action='store_true', help='Enable data recording')
    parser.add_argument('--motion', action='store_true', help='Enable motion control mode')
    parser.add_argument('--headless', action='store_true', help='Enable headless mode (no display)')
    parser.add_argument('--sim', action='store_true', help='Enable isaac simulation mode')
    parser.add_argument('--carpet_tactile', action='store_true', help='Enable carpet tactile sensor data collection')
    parser.add_argument('--third_camera', action='store_true', help='Enable third camera (RealSense color via UVC)')
    parser.add_argument('--realsense', action='store_true', help='Enable 3 RealSense cameras (front depth + wrists) logging')

    args = parser.parse_args()
    logger_mp.info(f"args: {args}")

    # image client: img_config should be the same as the configuration in image_server.py (of Robot's development computing unit)
    if args.sim:
        img_config = {
            'fps': 30,
            'head_camera_type': 'opencv',
            'head_camera_image_shape': [480, 848],  # Head camera resolution
            'head_camera_id_numbers': ['339222071291'],
            #'wrist_camera_type': 'opencv',
            #'wrist_camera_image_shape': [480, 640],  # Wrist camera resolution
            #'wrist_camera_id_numbers': [2, 4],
            'third_camera_type': 'opencv',
            'third_camera_image_shape': [480, 640],  # Third camera resolution
            'third_camera_id_numbers': ['239622301900'],
        }
    else:
        img_config = {
            'fps': 30,
            'head_camera_type': 'opencv',
            'head_camera_image_shape': [480, 848],  # Head camera resolution
            'head_camera_id_numbers': ['339222071291'],
            #'wrist_camera_type': 'opencv',
            #'wrist_camera_image_shape': [480, 640],  # Wrist camera resolution
            #'wrist_camera_id_numbers': [2, 4],
            'third_camera_type': 'opencv',
            'third_camera_image_shape': [480, 640],  # Third camera resolution
            'third_camera_id_numbers': ['918512072592'], # TODO: change the camera id
        }


    ASPECT_RATIO_THRESHOLD = 2.0  # If the aspect ratio exceeds this value, it is considered binocular
    if len(img_config['head_camera_id_numbers']) > 1 or (
            img_config['head_camera_image_shape'][1] / img_config['head_camera_image_shape'][
        0] > ASPECT_RATIO_THRESHOLD):
        BINOCULAR = True
    else:
        BINOCULAR = False
    if 'wrist_camera_type' in img_config:
        WRIST = True
    else:
        WRIST = False

    THIRD = bool(args.third_camera)
    USE_RS = bool(args.realsense)

    if BINOCULAR and not (img_config['head_camera_image_shape'][1] / img_config['head_camera_image_shape'][
        0] > ASPECT_RATIO_THRESHOLD):
        tv_img_shape = (img_config['head_camera_image_shape'][0], img_config['head_camera_image_shape'][1] * 2, 3)
    else:
        tv_img_shape = (img_config['head_camera_image_shape'][0], img_config['head_camera_image_shape'][1], 3)

    tv_img_shm = shared_memory.SharedMemory(create=True, size=np.prod(tv_img_shape) * np.uint8().itemsize)
    tv_img_array = np.ndarray(tv_img_shape, dtype=np.uint8, buffer=tv_img_shm.buf)

    if WRIST and args.sim:
        wrist_img_shape = (img_config['wrist_camera_image_shape'][0], img_config['wrist_camera_image_shape'][1] * 2, 3)
        wrist_img_shm = shared_memory.SharedMemory(create=True, size=np.prod(wrist_img_shape) * np.uint8().itemsize)
        wrist_img_array = np.ndarray(wrist_img_shape, dtype=np.uint8, buffer=wrist_img_shm.buf)
        img_client = ImageClient(tv_img_shape=tv_img_shape, tv_img_shm_name=tv_img_shm.name,
                                 wrist_img_shape=wrist_img_shape, wrist_img_shm_name=wrist_img_shm.name,
                                 server_address="127.0.0.1")
    elif WRIST and not args.sim:
        wrist_img_shape = (img_config['wrist_camera_image_shape'][0], img_config['wrist_camera_image_shape'][1] * 2, 3)
        wrist_img_shm = shared_memory.SharedMemory(create=True, size=np.prod(wrist_img_shape) * np.uint8().itemsize)
        wrist_img_array = np.ndarray(wrist_img_shape, dtype=np.uint8, buffer=wrist_img_shm.buf)
        img_client = ImageClient(tv_img_shape=tv_img_shape, tv_img_shm_name=tv_img_shm.name,
                                 wrist_img_shape=wrist_img_shape, wrist_img_shm_name=wrist_img_shm.name)
    elif THIRD and not args.sim:
        third_img_shape = (img_config['third_camera_image_shape'][0], img_config['third_camera_image_shape'][1] * 2, 3)
        third_img_shm = shared_memory.SharedMemory(create=True, size=np.prod(third_img_shape) * np.uint8().itemsize)
        third_img_array = np.ndarray(third_img_shape, dtype=np.uint8, buffer=third_img_shm.buf)
        img_client = ImageClient(tv_img_shape=tv_img_shape, tv_img_shm_name=tv_img_shm.name,
                                 third_img_shape=third_img_shape, third_img_shm_name=third_img_shm.name)
    else:
        img_client = ImageClient(tv_img_shape=tv_img_shape, tv_img_shm_name=tv_img_shm.name)


    image_receive_thread = threading.Thread(target=img_client.receive_process, daemon=True)
    image_receive_thread.daemon = True
    image_receive_thread.start()
    logger_mp.info("Image client started.")

    # television: obtain hand pose data from the XR device and transmit the robot's head camera image to the XR device.
    tv_wrapper = TeleVuerWrapper(binocular=BINOCULAR, use_hand_tracking=args.xr_mode == "hand", img_shape=tv_img_shape,
                                 img_shm_name=tv_img_shm.name,
                                 return_state_data=True, return_hand_rot_data=False)
    logger_mp.info("TeleVuer wrapper started.")

    rs_manager = _init_realsense_manager() if USE_RS else None

    # arm
    if args.arm == "G1_29":
        print("Using G1.29 arm controller and IK solver.")
        arm_ctrl = G1_29_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
        arm_ik = G1_29_ArmIK()
        print("G1.29 arm controller and IK solver initialized.")
    elif args.arm == "G1_23":
        arm_ctrl = G1_23_ArmController(simulation_mode=args.sim)
        arm_ik = G1_23_ArmIK()
    elif args.arm == "dummy":
        arm_ctrl = Dummy_ArmController()
        arm_ik = Dummy_ArmIK()
    else:
        raise NotImplementedError(f"Arm {args.arm} is not implemented yet.")

    if args.ee == "inspire1":
        left_hand_pos_array = Array('d', 75, lock=True)  # [input]
        right_hand_pos_array = Array('d', 75, lock=True)  # [input]
        dual_hand_data_lock = Lock()
        dual_hand_state_array = Array('d', 12, lock=False)  # [output] current left, right hand state(12) data.
        dual_hand_action_array = Array('d', 36, lock=False)  # [output] current left, right hand action(12) data.
        dual_hand_force_array = Array('d', 12, lock=False)
        dual_hand_touch_array = Array('d', 1062 * 2, lock=False) # add
        hand_ctrl = Inspire_Controller(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock,
                                       dual_hand_state_array, dual_hand_action_array, dual_hand_touch_array,
                                       dual_hand_force_array)
    else:
        hand_ctrl = DummyHand_Controller()

    # controller + motion mode
    if args.xr_mode == "controller" and args.motion:
        from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient # NOQA

        sport_client = LocoClient()
        sport_client.SetTimeout(0.0001)
        sport_client.Init()

    if args.carpet_tactile:
        tactile_env = SensorEnv(
            ports=[get_available_serial_port()],
            stack_num=1,
            adaptive_calibration=True,
            stage=DummyStage(),
            normalize=True
        )
        tactile_env.set_resistance(40)  # 초기 저항값 설정

    # record + headless mode
    if args.record and args.headless:
        recorder = EpisodeWriter(task_dir=args.task_dir, frequency=args.frequency, rerun_log=False)
    elif args.record and not args.headless:
        recorder = EpisodeWriter(task_dir=args.task_dir, frequency=args.frequency, rerun_log=True)

    logger_mp.info("Initialization complete. Entering main control loop.")

    if args.arm == "dummy":
        start_signal = True

    try:
        logger_mp.info("Please enter the start signal (enter 'r' to start the subsequent program)")
        while not start_signal:
            time.sleep(0.01)
        arm_ctrl.speed_gradual_max()

        s_t = time.time()
        while running:

            start_time = time.time()

            rs_colors = {}
            rs_depth = None
            if USE_RS and rs_manager is not None:
                if rs_manager.ready("wrist_left"):
                    rs_colors["rs_wrist_left"] = rs_manager.get_rgb("wrist_left").copy()
                if rs_manager.ready("wrist_right"):
                    rs_colors["rs_wrist_right"] = rs_manager.get_rgb("wrist_right").copy()
                if rs_manager.ready("front"):
                    rs_colors["rs_front_rgb"] = rs_manager.get_rgb("front").copy()
                    depth_frame = rs_manager.get_depth("front")
                    if depth_frame is not None:
                        rs_depth = _colorize_depth(depth_frame.copy())

            if not args.headless:
                tv_resized_image = cv2.resize(tv_img_array, (tv_img_shape[1] // 2, tv_img_shape[0] // 2))
                cv2.imshow("record image", tv_resized_image)
                if THIRD:
                    third_resized = cv2.resize(third_img_array, (third_img_shape[1] // 2, third_img_shape[0] // 2))
                    cv2.imshow("third camera", third_resized)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    running = False
                elif key == ord('s'):
                    should_toggle_recording = True

            if args.record and should_toggle_recording:
                should_toggle_recording = False
                if not is_recording:
                    if recorder.create_episode():
                        is_recording = True
                    else:
                        logger_mp.error("Failed to create episode. Recording not started.")
                else:
                    is_recording = False
                    recorder.save_episode()


            # get input data
            tele_data = tv_wrapper.get_motion_state_data()
            if (args.ee == "dex3" or args.ee == "inspire1" or args.ee == "brainco") and args.xr_mode == "hand":
                with left_hand_pos_array.get_lock():
                    left_hand_pos_array[:] = tele_data.left_hand_pos.flatten()
                with right_hand_pos_array.get_lock():
                    right_hand_pos_array[:] = tele_data.right_hand_pos.flatten()
            else:
                pass

            # high level control
            if args.xr_mode == "controller" and args.motion:
                # quit teleoperate
                if tele_data.tele_state.right_aButton:
                    running = False
                # command robot to enter damping mode. soft emergency stop function
                if tele_data.tele_state.left_thumbstick_state and tele_data.tele_state.right_thumbstick_state:
                    sport_client.Damp()
                # control, limit velocity to within 0.3
                sport_client.Move(-tele_data.tele_state.left_thumbstick_value[1] * 0.3,
                                  -tele_data.tele_state.left_thumbstick_value[0] * 0.3,
                                  -tele_data.tele_state.right_thumbstick_value[0] * 0.3)
                logger_mp.info(f"{tele_data.tele_state}")

                if tele_data.tele_state.left_aButton:
                    sport_client.StopMove()

                if tele_data.tele_state.right_bButton:
                    sport_client.Squat2StandUp()

            # get current robot state data.
            current_lr_arm_q = arm_ctrl.get_current_dual_arm_q()
            current_lr_arm_dq = arm_ctrl.get_current_dual_arm_dq()

            # solve ik using motor data and wrist pose, then use ik results to control arms.
            time_ik_start = time.time()
            sol_q, sol_tauff = arm_ik.solve_ik(tele_data.left_arm_pose, tele_data.right_arm_pose, current_lr_arm_q,
                                               current_lr_arm_dq)
            time_ik_end = time.time()
            logger_mp.debug(f"ik:\t{round(time_ik_end - time_ik_start, 6)}")
            arm_ctrl.ctrl_dual_arm(sol_q, sol_tauff)

            # record data
            if args.record:
                # dex hand or gripper
                if args.ee == "inspire1" and args.xr_mode == 'hand':
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:6]
                        right_ee_state = dual_hand_state_array[-6:]
                        left_hand_force_state = dual_hand_force_array[:6]
                        right_hand_force_state = dual_hand_force_array[-6:]
                        left_hand_action = dual_hand_action_array[:6]
                        right_hand_action = dual_hand_action_array[6:12]
                        left_hand_force_action = dual_hand_action_array[12:18]
                        right_hand_force_action = dual_hand_action_array[18:24]
                        left_hand_speed_action = dual_hand_action_array[24:30]
                        right_hand_speed_action = dual_hand_action_array[30:36]
                        left_hand_touch = dual_hand_touch_array[:1062]
                        right_hand_touch = dual_hand_touch_array[-1062:]
                        current_body_state = []
                        current_body_action = []
                elif args.ee == 'inspire1' and args.xr_mode == 'controller':
                     with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:6]
                        right_ee_state = dual_hand_state_array[-6:]
                        left_hand_force_state = dual_hand_force_array[:6]
                        right_hand_force_state = dual_hand_force_array[-6:]
                        left_hand_action = dual_hand_action_array[:6]
                        right_hand_action = dual_hand_action_array[6:12]
                        left_hand_force_action = dual_hand_action_array[12:18]
                        right_hand_force_action = dual_hand_action_array[18:24]
                        left_hand_speed_action = dual_hand_action_array[24:30]
                        right_hand_speed_action = dual_hand_action_array[30:36]
                        left_hand_touch = dual_hand_touch_array[:1062]
                        right_hand_touch = dual_hand_touch_array[-1062:]
                        current_body_state = arm_ctrl.get_current_motor_q().tolist()
                        current_body_action = [-tele_data.tele_state.left_thumbstick_value[1]  * 0.1,
                                                -tele_data.tele_state.left_thumbstick_value[0]  * 0.1,
                                                -tele_data.tele_state.right_thumbstick_value[0] * 0.1]
                else:
                    left_ee_state = []
                    right_ee_state = []
                    left_hand_action = []
                    right_hand_action = []
                    current_body_state = []
                    current_body_action = []

                    left_hand_force_state = []
                    right_hand_force_state = []
                    left_hand_speed_action = []
                    right_hand_speed_action = []
                    left_hand_force_action = []
                    right_hand_force_action = []
                    left_hand_touch = []
                    right_hand_touch = []

                # head image
                current_tv_image = tv_img_array.copy()
                # wrist image
                if WRIST:
                    current_wrist_image = wrist_img_array.copy()
                if THIRD:
                    current_third_image = third_img_array.copy()
                # arm state and action
                left_arm_state  = current_lr_arm_q[:7]
                left_arm_dq_state = current_lr_arm_dq[:7]
                right_arm_state = current_lr_arm_q[-7:]
                right_arm_dq_state = current_lr_arm_dq[-7:]
                left_arm_action = sol_q[:7]
                left_arm_torque_action = sol_tauff[:7]
                right_arm_action = sol_q[-7:]
                right_arm_torque_action = sol_tauff[-7:]

                if is_recording:
                    colors = {}
                    depths = {}
                    if BINOCULAR:
                        colors[f"color_{0}"] = current_tv_image[:, :tv_img_shape[1] // 2]
                        colors[f"color_{1}"] = current_tv_image[:, tv_img_shape[1] // 2:]
                        if WRIST:
                            colors[f"color_{2}"] = current_wrist_image[:, :wrist_img_shape[1] // 2]
                            colors[f"color_{3}"] = current_wrist_image[:, wrist_img_shape[1] // 2:]
                    else:
                        colors[f"color_{0}"] = current_tv_image
                        #if WRIST:
                        #    colors[f"color_{1}"] = current_wrist_image[:, :wrist_img_shape[1] // 2]
                        #    colors[f"color_{2}"] = current_wrist_image[:, wrist_img_shape[1] // 2:]
                        if THIRD:
                            colors[f"color_{3}"] = current_third_image[:, third_img_shape[1] // 2:]
                            #colors[f"color_{3}"] = current_third_image
                    colors.update(rs_colors)
                    if rs_depth is not None:
                        depths["rs_front_depth"] = rs_depth
                    states = {
                        "left_arm": {
                            "qpos": left_arm_state.tolist(),  # numpy.array -> list
                            "qvel": left_arm_dq_state.tolist(),
                            "torque": [],
                        },
                        "right_arm": {
                            "qpos": right_arm_state.tolist(),
                            "qvel": right_arm_dq_state.tolist(),
                            "torque": [],
                        },
                        "left_hand": {
                            "qpos": left_ee_state,
                            "qvel": [],
                            "torque": left_hand_force_state,
                        },
                        "right_hand": {
                            "qpos": right_ee_state,
                            "qvel": [],
                            "torque": right_hand_force_state,
                        },
                        "body": {
                            "qpos": current_body_state,
                        },
                    }
                    actions = {
                        "left_arm": {
                            "qpos": left_arm_action.tolist(),
                            "qvel": [],
                            "torque": left_arm_torque_action.tolist(),
                        },
                        "right_arm": {
                            "qpos": right_arm_action.tolist(),
                            "qvel": [],
                            "torque": right_arm_torque_action.tolist(),
                        },
                        "left_hand": {
                            "qpos": left_hand_action,
                            "qvel": left_hand_speed_action,
                            "torque": left_hand_force_action,
                        },
                        "right_hand": {
                            "qpos": right_hand_action,
                            "qvel": right_hand_speed_action,
                            "torque": right_hand_force_action,
                        },
                        "body": {
                            "qpos": current_body_action,
                        },
                    }
                    tactiles = {
                        "left_tactile": left_hand_touch,
                        "right_tactile": right_hand_touch,
                    }


                    if args.carpet_tactile:
                        carpet_tactiles = tactile_env.get()
                        carpet_tactiles = {
                            "carpet_0": carpet_tactiles,
                        }

                    else:
                        carpet_tactiles = None

                    recorder.add_item(colors=colors, depths=depths, states=states, actions=actions,
                                      tactiles=tactiles, carpet_tactiles=carpet_tactiles)

            current_time = time.time()
            time_elapsed = current_time - start_time
            sleep_time = max(0, (1 / args.frequency) - time_elapsed)
            time.sleep(sleep_time)
            logger_mp.debug(f"main process sleep: {sleep_time}")

    except KeyboardInterrupt:
        logger_mp.info("KeyboardInterrupt, exiting program...")
    except Exception as e:
        logger_mp.error(f"An error occurred: {e}")
        logger_mp.info("Exiting program due to an error...")
    finally:
        arm_ctrl.ctrl_dual_arm_go_home()
        tv_img_shm.close()
        tv_img_shm.unlink()
        if WRIST:
            wrist_img_shm.close()
            wrist_img_shm.unlink()
        if THIRD:
            third_img_shm.close()
            third_img_shm.unlink()
        if rs_manager:
            rs_manager.close()
        if args.record:
            recorder.close()
        # listen_keyboard_thread.join()
        logger_mp.info("Finally, exiting program...")
        exit(0)