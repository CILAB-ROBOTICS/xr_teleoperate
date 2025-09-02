# this file is legacy, need to fix.
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize # dds
# from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_ # OLD IDL, REMOVED
# from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_ # OLD IDL, REMOVED

# NEW IMPORTS for Inspire Hand SDK
from inspire_sdkpy import inspire_dds, inspire_hand_defaut

from teleop.robot_control.hand_retargeting import HandRetargeting, HandType # Assuming this remains the same
import numpy as np
# from enum import IntEnum # Old Enums for joint indexing might not be needed for new DDS messages
import threading
import time
from multiprocessing import Process, Array, Lock # Removed shared_memory as Array is used

inspire_tip_indices = [4, 9, 14, 19, 24] # Assuming this remains relevant for hand_retargeting
Inspire_Num_Motors = 6 # Number of motors per hand


# NEW DDS TOPIC NAMES (assuming these are the correct topics based on SDK examples)
kTopicInspireCtrlLeft = "rt/inspire_hand/ctrl/l"
kTopicInspireCtrlRight = "rt/inspire_hand/ctrl/r"
kTopicInspireStateLeft = "rt/inspire_hand/state/l"
kTopicInspireStateRight = "rt/inspire_hand/state/r"
kTopicInspireTouchLeft = "rt/inspire_hand/touch/l" #추가
kTopicInspireTouchRight = "rt/inspire_hand/touch/r"#추가

#추가
touch_dict = {
    "fingerone_tip_touch": 9,
    "fingerone_top_touch": 96,
    "fingerone_palm_touch": 80,
    "fingertwo_tip_touch": 9,
    "fingertwo_top_touch": 96,
    "fingertwo_palm_touch": 80,
    "fingerthree_tip_touch": 9,
    "fingerthree_top_touch": 96,
    "fingerthree_palm_touch": 80,
    "fingerfour_tip_touch": 9,
    "fingerfour_top_touch": 96,
    "fingerfour_palm_touch": 80,
    "fingerfive_tip_touch": 9,
    "fingerfive_top_touch": 96,
    "fingerfive_middle_touch": 9,
    "fingerfive_palm_touch": 96,
    "palm_touch": 112
}
Inspire_Num_Tactile = 1062


alpha = 0.1                            # 속도 LPF 계수 (0~1, 0이면 완전 평균)
Kp = 0.4                               # PID 비례계수
Kd = 0.02                              # PID 미분계수

class Inspire_Controller:
    def __init__(self, left_hand_array, right_hand_array, dual_hand_data_lock=None, dual_hand_state_array=None,
                 dual_hand_action_array=None, dual_hand_touch_array = None, dual_hand_force_array = None, fps=100.0, Unit_Test=False, network_interface=""): # Added network_interface
        print("Initialize Inspire_Controller...")
        self.fps = fps
        self.Unit_Test = Unit_Test
        self.debug_counter = 0 # 디버깅 카운터 추가
        # self.dual_hand_touch_array = dual_hand_touch_array #추가Inspire_Num_Tactile

        # Initialize DDS Channel Factory
        # This should ideally be called once per process.
        # If multiple controllers or DDS entities run in the same process, ensure this is handled.
        try:
            ChannelFactoryInitialize(0, network_interface)
            print(f"DDS ChannelFactory initialized with interface: '{network_interface if network_interface else 'default'}'")
        except Exception as e:
            print(f"Warning: DDS ChannelFactoryInitialize failed or already initialized: {e}")


        if not self.Unit_Test:
            self.hand_retargeting = HandRetargeting(HandType.INSPIRE_HAND)
        else:
            # Assuming a different config for unit tests if necessary
            self.hand_retargeting = HandRetargeting(HandType.INSPIRE_HAND_Unit_Test)
        # Initialize hand command publishers
        self.LeftHandCmd_publisher = ChannelPublisher(kTopicInspireCtrlLeft, inspire_dds.inspire_hand_ctrl)
        self.LeftHandCmd_publisher.Init()
        self.RightHandCmd_publisher = ChannelPublisher(kTopicInspireCtrlRight, inspire_dds.inspire_hand_ctrl)
        self.RightHandCmd_publisher.Init()

        # Initialize hand state subscribers
        self.LeftHandState_subscriber = ChannelSubscriber(kTopicInspireStateLeft, inspire_dds.inspire_hand_state)
        self.LeftHandState_subscriber.Init() # Consider using callback if preferred: Init(callback_func, period_ms)
        self.RightHandState_subscriber = ChannelSubscriber(kTopicInspireStateRight, inspire_dds.inspire_hand_state)
        self.RightHandState_subscriber.Init()
        
        ## 추가
        # Initialize hand touch subscribers
        self.LeftTouch_subscriber = ChannelSubscriber(kTopicInspireTouchLeft, inspire_dds.inspire_hand_touch)
        self.LeftTouch_subscriber.Init()
        self.RightTouch_subscriber = ChannelSubscriber(kTopicInspireTouchRight, inspire_dds.inspire_hand_touch)
        self.RightTouch_subscriber.Init()
        
        # Shared Arrays for hand states ([0,1] normalized values)
        self.left_hand_state_array = Array('d', Inspire_Num_Motors, lock=True)
        self.right_hand_state_array = Array('d', Inspire_Num_Motors, lock=True)
        self.left_hand_force_array = Array('d', Inspire_Num_Motors, lock=True)
        self.right_hand_force_array = Array('d', Inspire_Num_Motors, lock=True)

        ## 추가
        self.left_tactile_array = Array('d', Inspire_Num_Tactile, lock=True)
        self.right_tactile_array = Array('d', Inspire_Num_Tactile, lock=True)


        # Initialize subscribe thread
        self.subscribe_state_thread = threading.Thread(target=self._subscribe_hand_state_loop)
        self.subscribe_state_thread.daemon = True
        self.subscribe_state_thread.start()
    

        # Wait for initial DDS messages (optional, but good for ensuring connection)
        wait_count = 0
        while not (any(self.left_hand_state_array.get_obj()) or any(self.right_hand_state_array.get_obj())):
            if wait_count % 100 == 0: # Print every second
                print(f"[Inspire_Controller] Waiting to subscribe to hand states from DDS (L: {any(self.left_hand_state_array.get_obj())}, R: {any(self.right_hand_state_array.get_obj())})...")
            time.sleep(0.01)
            wait_count +=1
            if wait_count > 500: # Timeout after 5 seconds
                print("[Inspire_Controller] Warning: Timeout waiting for initial hand states. Proceeding anyway.")
                break
        print("[Inspire_Controller] Initial hand states received or timeout.")

        hand_control_process = Process(target=self.control_process_loop, args=(
            left_hand_array, right_hand_array, self.left_hand_state_array, self.right_hand_state_array,
            self.left_tactile_array, self.right_tactile_array, 
            self.left_hand_force_array, self.right_hand_force_array,
            dual_hand_data_lock, dual_hand_state_array, 
            dual_hand_action_array, dual_hand_touch_array, dual_hand_force_array))
        
        hand_control_process.daemon = True
        hand_control_process.start()

        print("Initialize Inspire_Controller OK!\n")
        
        #추가
        self._touch_offsets = {}
        start = 0
        for name, length in touch_dict.items():       # touch_dict는 9~112 길이 정보
            self._touch_offsets[name] = (start, start + length)
            start += length                           # 다음 필드의 시작 인덱스
        
    # 추가
    # def _subscribe_hand_touch_loop(self):
    #     print("[Inspire_Controller] Subscribe thread2 started.")
    #     local_debug_counter = 0
    #     while True:
    #         # Left
    #         left_touch_msg = self.LeftTouch_subscriber.Read()
    #         if left_touch_msg is not None:
    #             if hasattr(left_touch_msg, 'fingerone_tip_touch') and len(left_touch_msg.fingerone_tip_touch) == Inspire_Num_Tactile:
    #                 with self.left_tactile_array.get_lock():
    #                     for i in range(Inspire_Num_Tactile):
    #                         self.left_tactile_array[i] = left_touch_msg.fingerone_tip_touch[i]
                            
    #         # Right
    #         right_touch_msg = self.RightTouch_subscriber.Read()
    #         if right_touch_msg is not None:
    #             if hasattr(right_touch_msg, 'fingerone_tip_touch') and len(right_touch_msg.fingerone_tip_touch) == Inspire_Num_Tactile:
    #                 with self.right_tactile_array.get_lock():
    #                     for i in range(Inspire_Num_Tactile):
    #                         self.right_tactile_array[i] = right_touch_msg.fingerone_tip_touch[i]
                            
    #         local_debug_counter +=1
    #         time.sleep(0.002)

    def _subscribe_hand_state_loop(self):
        print("[Inspire_Controller] Subscribe thread started.")
        local_debug_counter = 0
        while True:
            # Left Hand
            left_state_msg = self.LeftHandState_subscriber.Read()
            if left_state_msg is not None:
                if hasattr(left_state_msg, 'angle_act') and len(left_state_msg.angle_act) == Inspire_Num_Motors:
                    with self.left_hand_state_array.get_lock():
                        for i in range(Inspire_Num_Motors):
                            self.left_hand_state_array[i] = left_state_msg.angle_act[i] / 1000.0
                    #if local_debug_counter % 100 == 0: # Log every 100 cycles
                        #print(f"[Subscribe L] angle_act: {left_state_msg.angle_act}, norm_state: {[f'{x:.3f}' for x in self.left_hand_state_array[:]]}")
                if hasattr(left_state_msg, 'force_act') and len(left_state_msg.force_act) == Inspire_Num_Motors:
                    with self.left_hand_force_array.get_lock():
                        for i in range(Inspire_Num_Motors):
                            self.left_hand_force_array[i] = left_state_msg.force_act[i] / 3000.0
                
                else:
                    print(f"[Inspire_Controller] Warning: Received left_state_msg but attributes are missing or incorrect. Type: {type(left_state_msg)}, Content: {str(left_state_msg)[:100]}")
            # Right Hand
            right_state_msg = self.RightHandState_subscriber.Read()
            if right_state_msg is not None:
                if hasattr(right_state_msg, 'angle_act') and len(right_state_msg.angle_act) == Inspire_Num_Motors:
                    with self.right_hand_state_array.get_lock():
                        for i in range(Inspire_Num_Motors):
                            self.right_hand_state_array[i] = right_state_msg.angle_act[i] / 1000.0
                    #if local_debug_counter % 100 == 0: # Log every 100 cycles
                        #print(f"[Subscribe R] angle_act: {right_state_msg.angle_act}, norm_state: {[f'{x:.3f}' for x in self.right_hand_state_array[:]]}")
                if hasattr(right_state_msg, 'force_act') and len(right_state_msg.force_act) == Inspire_Num_Motors:
                    with self.right_hand_force_array.get_lock():
                        for i in range(Inspire_Num_Motors):
                            self.right_hand_force_array[i] = right_state_msg.force_act[i] / 3000.0
                else:
                    print(f"[Inspire_Controller] Warning: Received right_state_msg but attributes are missing or incorrect. Type: {type(right_state_msg)}, Content: {str(right_state_msg)[:100]}")
            
            #추가
            # Left
            left_touch_msg = self.LeftTouch_subscriber.Read()
            if left_touch_msg is not None:
                # start_idx = 0
                # for key, value in touch_dict.items():
                #     if hasattr(left_touch_msg, key) and len(getattr(left_touch_msg, key)) == value:
                #         tactile_values = getattr(left_touch_msg, key)
                #         # with self.right_hand_state_array.get_lock():
                #         #     self.dual_hand_touch_array[:inspire_num_tactile] = left_touch_msg.tactile
                #         with self.left_tactile_array.get_lock():
                #             for i in range(value):
                #                 self.left_tactile_array[start_idx+i] = left_touch_msg.tactile_values[i]
                                
                #             # print("Dehug Tactile Sensor Value For finger_tip_Left" )
                #             # print(type(self.left_tactile_array))
                #             # print(len(list(self.left_tactile_array)))
                #             # print("MAX value:", max(list(self.left_tactile_array)))
                                #             # print("MIN value:", min(list(self.left_tactile_array)))
                                #             # print("#"*50)

                if hasattr(left_touch_msg, 'fingerone_tip_touch') and len(left_touch_msg.fingerone_tip_touch) == 9:
                    with self.left_tactile_array.get_lock():
                        for i in range(9):
                            self.left_tactile_array[i] = left_touch_msg.fingerone_tip_touch[i]

                if hasattr(left_touch_msg, 'fingerone_top_touch') and len(left_touch_msg.fingerone_top_touch) == 96:
                    with self.left_tactile_array.get_lock():
                        for i in range(96):
                            self.left_tactile_array[9 + i] = left_touch_msg.fingerone_top_touch[i]

                if hasattr(left_touch_msg, 'fingerone_palm_touch') and len(left_touch_msg.fingerone_palm_touch) == 80:
                    with self.left_tactile_array.get_lock():
                        for i in range(80):
                            self.left_tactile_array[105 + i] = left_touch_msg.fingerone_palm_touch[i]

                if hasattr(left_touch_msg, 'fingertwo_tip_touch') and len(left_touch_msg.fingertwo_tip_touch) == 9:
                    with self.left_tactile_array.get_lock():
                        for i in range(9):
                            self.left_tactile_array[185 + i] = left_touch_msg.fingertwo_tip_touch[i]

                if hasattr(left_touch_msg, 'fingertwo_top_touch') and len(left_touch_msg.fingertwo_top_touch) == 96:
                    with self.left_tactile_array.get_lock():
                        for i in range(96):
                            self.left_tactile_array[194 + i] = left_touch_msg.fingertwo_top_touch[i]

                if hasattr(left_touch_msg, 'fingertwo_palm_touch') and len(left_touch_msg.fingertwo_palm_touch) == 80:
                    with self.left_tactile_array.get_lock():
                        for i in range(80):
                            self.left_tactile_array[290 + i] = left_touch_msg.fingertwo_palm_touch[i]

                if hasattr(left_touch_msg, 'fingerthree_tip_touch') and len(left_touch_msg.fingerthree_tip_touch) == 9:
                    with self.left_tactile_array.get_lock():
                        for i in range(9):
                            self.left_tactile_array[370 + i] = left_touch_msg.fingerthree_tip_touch[i]

                if hasattr(left_touch_msg, 'fingerthree_top_touch') and len(left_touch_msg.fingerthree_top_touch) == 96:
                    with self.left_tactile_array.get_lock():
                        for i in range(96):
                            self.left_tactile_array[379 + i] = left_touch_msg.fingerthree_top_touch[i]

                if hasattr(left_touch_msg, 'fingerthree_palm_touch') and len(left_touch_msg.fingerthree_palm_touch) == 80:
                    with self.left_tactile_array.get_lock():
                        for i in range(80):
                            self.left_tactile_array[475 + i] = left_touch_msg.fingerthree_palm_touch[i]

                if hasattr(left_touch_msg, 'fingerfour_tip_touch') and len(left_touch_msg.fingerfour_tip_touch) == 9:
                    with self.left_tactile_array.get_lock():
                        for i in range(9):
                            self.left_tactile_array[555 + i] = left_touch_msg.fingerfour_tip_touch[i]

                if hasattr(left_touch_msg, 'fingerfour_top_touch') and len(left_touch_msg.fingerfour_top_touch) == 96:
                    with self.left_tactile_array.get_lock():
                        for i in range(96):
                            self.left_tactile_array[564 + i] = left_touch_msg.fingerfour_top_touch[i]

                if hasattr(left_touch_msg, 'fingerfour_palm_touch') and len(left_touch_msg.fingerfour_palm_touch) == 80:
                    with self.left_tactile_array.get_lock():
                        for i in range(80):
                            self.left_tactile_array[660 + i] = left_touch_msg.fingerfour_palm_touch[i]

                if hasattr(left_touch_msg, 'fingerfive_tip_touch') and len(left_touch_msg.fingerfive_tip_touch) == 9:
                    with self.left_tactile_array.get_lock():
                        for i in range(9):
                            self.left_tactile_array[740 + i] = left_touch_msg.fingerfive_tip_touch[i]

                if hasattr(left_touch_msg, 'fingerfive_top_touch') and len(left_touch_msg.fingerfive_top_touch) == 96:
                    with self.left_tactile_array.get_lock():
                        for i in range(96):
                            self.left_tactile_array[749 + i] = left_touch_msg.fingerfive_top_touch[i]

                if hasattr(left_touch_msg, 'fingerfive_middle_touch') and len(left_touch_msg.fingerfive_middle_touch) == 9:
                    with self.left_tactile_array.get_lock():
                        for i in range(9):
                            self.left_tactile_array[845 + i] = left_touch_msg.fingerfive_middle_touch[i]

                if hasattr(left_touch_msg, 'fingerfive_palm_touch') and len(left_touch_msg.fingerfive_palm_touch) == 96:
                    with self.left_tactile_array.get_lock():
                        for i in range(96):
                            self.left_tactile_array[854 + i] = left_touch_msg.fingerfive_palm_touch[i]

                if hasattr(left_touch_msg, 'palm_touch') and len(left_touch_msg.palm_touch) == 112:
                    with self.left_tactile_array.get_lock():
                        for i in range(112):
                            self.left_tactile_array[950 + i] = left_touch_msg.palm_touch[i]  

                        # print("Dehug Tactile Sensor Value For finger_tip_Left" )
                        # print(type(self.left_tactile_array))
                        # print(list(self.left_tactile_array))
                        # print("MAX value:", max(list(self.left_tactile_array)))
                        # print("MIN value:", min(list(self.left_tactile_array)))
                        # print("#"*50)

                            
            # Right
            right_touch_msg = self.RightTouch_subscriber.Read()
            if right_touch_msg is not None:
                # start_idx = 0
                # for key, value in touch_dict.items():
                #     if hasattr(right_touch_msg, key) and len(getattr(right_touch_msg, key)) == value:
                #         tactile_values = getattr(right_touch_msg, key)
                #         # with self.right_hand_state_array.get_lock():
                #         #     self.dual_hand_touch_array[:inspire_num_tactile] = left_touch_msg.tactile
                #         with self.right_tactile_array.get_lock():
                #             for i in range(value):
                #                 self.right_tactile_array[start_idx+i] = right_touch_msg.tactile_values[i]
                
                # if hasattr(right_touch_msg, 'fingerone_tip_touch') and len(right_touch_msg.fingerone_tip_touch) == Inspire_Num_Tactile:
                #     # tactile_values = getattr(left_touch_msg, key)
                #     # with self.right_hand_state_array.get_lock():
                #         # self.dual_hand_touch_array[:inspire_num_tactile] = left_touch_msg.tactile
                #     with self.right_tactile_array.get_lock():
                #         for i in range(Inspire_Num_Tactile):
                #             self.right_tactile_array[i] = right_touch_msg.fingerone_tip_touch[i]
                            
                                
                            # print("Dehug Tactile Sensor Value For finger_tip_Left" )
                            # print(type(self.left_tactile_array))
                            # print(len(list(self.left_tactile_array)))
                            # print("MAX value:", max(list(self.left_tactile_array)))
                            # print("MIN value:", min(list(self.left_tactile_array)))
                            # print("#"*50)
                            
                if hasattr(right_touch_msg, 'fingerone_tip_touch') and len(right_touch_msg.fingerone_tip_touch) == 9:
                    with self.right_tactile_array.get_lock():
                        for i in range(9):
                            self.right_tactile_array[i] = right_touch_msg.fingerone_tip_touch[i]

                if hasattr(right_touch_msg, 'fingerone_top_touch') and len(right_touch_msg.fingerone_top_touch) == 96:
                    with self.right_tactile_array.get_lock():
                        for i in range(96):
                            self.right_tactile_array[9 + i] = right_touch_msg.fingerone_top_touch[i]

                if hasattr(right_touch_msg, 'fingerone_palm_touch') and len(right_touch_msg.fingerone_palm_touch) == 80:
                    with self.right_tactile_array.get_lock():
                        for i in range(80):
                            self.right_tactile_array[105 + i] = right_touch_msg.fingerone_palm_touch[i]

                if hasattr(right_touch_msg, 'fingertwo_tip_touch') and len(right_touch_msg.fingertwo_tip_touch) == 9:
                    with self.right_tactile_array.get_lock():
                        for i in range(9):
                            self.right_tactile_array[185 + i] = right_touch_msg.fingertwo_tip_touch[i]

                if hasattr(right_touch_msg, 'fingertwo_top_touch') and len(right_touch_msg.fingertwo_top_touch) == 96:
                    with self.right_tactile_array.get_lock():
                        for i in range(96):
                            self.right_tactile_array[194 + i] = right_touch_msg.fingertwo_top_touch[i]

                if hasattr(right_touch_msg, 'fingertwo_palm_touch') and len(right_touch_msg.fingertwo_palm_touch) == 80:
                    with self.right_tactile_array.get_lock():
                        for i in range(80):
                            self.right_tactile_array[290 + i] = right_touch_msg.fingertwo_palm_touch[i]

                if hasattr(right_touch_msg, 'fingerthree_tip_touch') and len(right_touch_msg.fingerthree_tip_touch) == 9:
                    with self.right_tactile_array.get_lock():
                        for i in range(9):
                            self.right_tactile_array[370 + i] = right_touch_msg.fingerthree_tip_touch[i]

                if hasattr(right_touch_msg, 'fingerthree_top_touch') and len(right_touch_msg.fingerthree_top_touch) == 96:
                    with self.right_tactile_array.get_lock():
                        for i in range(96):
                            self.right_tactile_array[379 + i] = right_touch_msg.fingerthree_top_touch[i]

                if hasattr(right_touch_msg, 'fingerthree_palm_touch') and len(right_touch_msg.fingerthree_palm_touch) == 80:
                    with self.right_tactile_array.get_lock():
                        for i in range(80):
                            self.right_tactile_array[475 + i] = right_touch_msg.fingerthree_palm_touch[i]

                if hasattr(right_touch_msg, 'fingerfour_tip_touch') and len(right_touch_msg.fingerfour_tip_touch) == 9:
                    with self.right_tactile_array.get_lock():
                        for i in range(9):
                            self.right_tactile_array[555 + i] = right_touch_msg.fingerfour_tip_touch[i]

                if hasattr(right_touch_msg, 'fingerfour_top_touch') and len(right_touch_msg.fingerfour_top_touch) == 96:
                    with self.right_tactile_array.get_lock():
                        for i in range(96):
                            self.right_tactile_array[564 + i] = right_touch_msg.fingerfour_top_touch[i]

                if hasattr(right_touch_msg, 'fingerfour_palm_touch') and len(right_touch_msg.fingerfour_palm_touch) == 80:
                    with self.right_tactile_array.get_lock():
                        for i in range(80):
                            self.right_tactile_array[660 + i] = right_touch_msg.fingerfour_palm_touch[i]

                if hasattr(right_touch_msg, 'fingerfive_tip_touch') and len(right_touch_msg.fingerfive_tip_touch) == 9:
                    with self.right_tactile_array.get_lock():
                        for i in range(9):
                            self.right_tactile_array[740 + i] = right_touch_msg.fingerfive_tip_touch[i]

                if hasattr(right_touch_msg, 'fingerfive_top_touch') and len(right_touch_msg.fingerfive_top_touch) == 96:
                    with self.right_tactile_array.get_lock():
                        for i in range(96):
                            self.right_tactile_array[749 + i] = right_touch_msg.fingerfive_top_touch[i]

                if hasattr(right_touch_msg, 'fingerfive_middle_touch') and len(right_touch_msg.fingerfive_middle_touch) == 9:
                    with self.right_tactile_array.get_lock():
                        for i in range(9):
                            self.right_tactile_array[845 + i] = right_touch_msg.fingerfive_middle_touch[i]

                if hasattr(right_touch_msg, 'fingerfive_palm_touch') and len(right_touch_msg.fingerfive_palm_touch) == 96:
                    with self.right_tactile_array.get_lock():
                        for i in range(96):
                            self.right_tactile_array[854 + i] = right_touch_msg.fingerfive_palm_touch[i]

                if hasattr(right_touch_msg, 'palm_touch') and len(right_touch_msg.palm_touch) == 112:
                    with self.right_tactile_array.get_lock():
                        for i in range(112):
                            self.right_tactile_array[950 + i] = right_touch_msg.palm_touch[i] 

            local_debug_counter +=1
            time.sleep(0.002)

    def _send_hand_command(self, left_angle_cmd_scaled, right_angle_cmd_scaled, 
                           left_force_cmd_scaled, right_force_cmd_scaled, left_speed_cmd_scaled, right_speed_cmd_scaled):
        """
        Send scaled angle commands [0-1000] to both hands.
        """
        # Left Hand Command
        left_cmd_msg = inspire_hand_defaut.get_inspire_hand_ctrl()
        left_cmd_msg.angle_set = left_angle_cmd_scaled
        left_cmd_msg.force_set = left_force_cmd_scaled
        left_cmd_msg.speed_set = left_speed_cmd_scaled
        left_cmd_msg.mode = 0b0001 # Mode 13: Angle control
        self.LeftHandCmd_publisher.Write(left_cmd_msg)

        # Right Hand Command
        right_cmd_msg = inspire_hand_defaut.get_inspire_hand_ctrl()
        right_cmd_msg.angle_set = right_angle_cmd_scaled
        right_cmd_msg.force_set = right_force_cmd_scaled
        right_cmd_msg.speed_set = right_speed_cmd_scaled
        right_cmd_msg.mode = 0b0001 # Mode 1: Angle  control
        self.RightHandCmd_publisher.Write(right_cmd_msg)
        
        # if self.debug_counter % 50 == 0: # Log every 50 control cycles
        # print(f"[Send Cmd L] Scaled: {left_angle_cmd_scaled}")
        # print(f"[Send Cmd R] Scaled: {right_angle_cmd_scaled}")

    def control_process_loop(self, left_hand_input_array, right_hand_input_array, 
                             shared_left_hand_state_array, shared_right_hand_state_array,
                             shared_left_hand_touch_array, shared_right_hand_touch_array,
                             shared_left_hand_force_array, shared_right_hand_force_array, 
                             dual_hand_data_lock=None, 
                             dual_hand_state_array_shm=None, dual_hand_action_array_shm=None, 
                             dual_hand_touch_array_shm=None, dual_hand_force_array_shm=None):
        
        print("[Inspire_Controller] Control process started.")
        running = True
        current_left_q_target_norm = np.ones(Inspire_Num_Motors, dtype=float) 
        current_right_q_target_norm = np.ones(Inspire_Num_Motors, dtype=float)
        prev_left_angle_norm = current_left_q_target_norm.copy()
        prev_right_angle_norm = current_right_q_target_norm.copy()
        prev_err_left  = np.zeros(6)
        prev_err_right = np.zeros(6)
        
        try:
            while running:
                start_time = time.time()
                self.debug_counter +=1

                left_hand_mat = np.array(left_hand_input_array[:]).reshape(25, 3).copy()
                right_hand_mat = np.array(right_hand_input_array[:]).reshape(25, 3).copy()

                #if self.debug_counter % 100 == 0: # Log every 100 cycles
                    #print(f"\n[Ctrl Loop {self.debug_counter}] Input L Hand Mat (brief): {left_hand_mat[inspire_tip_indices[0]]}")
                    #print(f"[Ctrl Loop {self.debug_counter}] Input R Hand Mat (brief): {right_hand_mat[inspire_tip_indices[0]]}")


                with shared_left_hand_state_array.get_lock():
                    current_left_q_state_norm = np.array(shared_left_hand_state_array[:])
                with shared_right_hand_state_array.get_lock():
                    current_right_q_state_norm = np.array(shared_right_hand_state_array[:])
                
                state_data_for_logging = np.concatenate((current_left_q_state_norm, current_right_q_state_norm))
                
                # 추가
                with shared_left_hand_touch_array.get_lock():
                    current_left_q_touch_norm = np.array(shared_left_hand_touch_array[:])
                with shared_right_hand_touch_array.get_lock():
                    current_right_q_touch_norm = np.array(shared_right_hand_touch_array[:])
                touch_data_for_logging = np.concatenate((current_left_q_touch_norm, current_right_q_touch_norm))

                with shared_left_hand_force_array.get_lock():
                    current_left_q_force_norm = np.array(shared_left_hand_force_array[:])
                with shared_right_hand_force_array.get_lock():
                    current_right_q_force_norm = np.array(shared_right_hand_force_array[:])    
                force_data_for_logging = np.concatenate((current_left_q_force_norm, current_right_q_force_norm))


                human_hand_data_valid = not np.all(right_hand_mat == 0.0) and \
                                        not np.all(left_hand_mat[4] == np.array([-1.13, 0.3, 0.15]))
                
                #if self.debug_counter % 100 == 0:
                    #print(f"[Ctrl Loop] human_hand_data_valid: {human_hand_data_valid}")

                if human_hand_data_valid:
                    ref_left_value = left_hand_mat[inspire_tip_indices]
                    ref_right_value = right_hand_mat[inspire_tip_indices]

                    #if self.debug_counter % 100 == 0:
                        #print(f"[Ctrl Loop] Ref L Value (brief): {ref_left_value[0]}")
                        #print(f"[Ctrl Loop] Ref R Value (brief): {ref_right_value[0]}")

                    raw_left_q_target_rad = self.hand_retargeting.left_retargeting.retarget(ref_left_value)[
                        self.hand_retargeting.left_dex_retargeting_to_hardware]
                    raw_right_q_target_rad = self.hand_retargeting.right_retargeting.retarget(ref_right_value)[
                        self.hand_retargeting.right_dex_retargeting_to_hardware]
                    
                    #if self.debug_counter % 100 == 0:
                        #print(f"[Ctrl Loop] Raw L Radian: {[f'{x:.3f}' for x in raw_left_q_target_rad]}")
                        #print(f"[Ctrl Loop] Raw R Radian: {[f'{x:.3f}' for x in raw_right_q_target_rad]}")
                    
                    def normalize_radians_to_01(val, min_val, max_val, joint_name_for_debug=""):
                        # original: (max_val - val) / (max_val - min_val) -> if val is min_val (open), result is 1. if val is max_val (closed), result is 0.
                        # This assumes: Inspire Hand 0 = closed, 1 = open for normalized values.
                        # And Inspire SDK 0 = closed, 1000 = open for scaled commands.
                        # Let's test: (val - min_val) / (max_val - min_val) -> if val is min_val (open), result is 0. if val is max_val (closed), result is 1.
                        # This means 0=open, 1=closed. This would need inversion before sending to SDK.

                        # Sticking to original interpretation for now: 0=closed, 1=open (normalized)
                        # Check if min_val and max_val are the same
                        if (max_val - min_val) == 0:
                            # print(f"[Normalize Debug - {joint_name_for_debug}] Warning: min_val and max_val are equal ({min_val}). Returning 0.5.")
                            return 0.5 # Or handle as an error, or return based on convention (e.g. 0 if closed, 1 if open)
                        
                        normalized_val = np.clip((max_val - val) / (max_val - min_val), 0.0, 1.0)
                        # if self.debug_counter % 100 == 0:
                        # print(f"[Normalize Debug - {joint_name_for_debug}] val: {val:.3f}, min: {min_val}, max: {max_val}, norm_val: {normalized_val:.3f}")
                        return normalized_val

                    for idx in range(Inspire_Num_Motors):
                        joint_debug_name_l = f"L_idx{idx}"
                        joint_debug_name_r = f"R_idx{idx}"
                        if idx <= 3: # Pinky, Ring, Middle, Index
                            min_r, max_r = 0.0, 1.7 # Retargeting output: 0.0 (open) to 1.7 (closed)
                        elif idx == 4: # Thumb Bend
                            min_r, max_r = 0.0, 0.5  # Retargeting output: 0.0 (open) to 0.5 (closed)
                        elif idx == 5: # Thumb Rotation
                            min_r, max_r = -0.1, 1.3 # Retargeting output: -0.1 ( adduct? open-ish) to 1.3 (abduct? closed-ish)
                        
                        current_left_q_target_norm[idx] = normalize_radians_to_01(raw_left_q_target_rad[idx], min_r, max_r, joint_debug_name_l)
                        current_right_q_target_norm[idx] = normalize_radians_to_01(raw_right_q_target_rad[idx], min_r, max_r, joint_debug_name_r)
                    
                    #if self.debug_counter % 100 == 0:
                        #print(f"[Ctrl Loop] Norm L Target: {[f'{x:.3f}' for x in current_left_q_target_norm]}")
                        #print(f"[Ctrl Loop] Norm R Target: {[f'{x:.3f}' for x in current_right_q_target_norm]}")
                
                scaled_left_cmd = [int(np.clip(val * 1000, 0, 1000)) for val in current_left_q_target_norm]
                scaled_right_cmd = [int(np.clip(val * 1000, 0, 1000)) for val in current_right_q_target_norm]
                
                #if self.debug_counter % 100 == 0: # Log every 100 control cycles
                    #print(f"[Ctrl Loop] Scaled L Cmd: {scaled_left_cmd}")
                    #print(f"[Ctrl Loop] Scaled R Cmd: {scaled_right_cmd}")
                # Send commands to hands
                dt = 1.0 / self.fps
                raw_left_speed  = (current_left_q_target_norm  - prev_left_angle_norm)  / dt
                raw_right_speed = (current_right_q_target_norm - prev_right_angle_norm) / dt
                
                
                left_speed_cmd_scaled  = np.clip(raw_left_speed,  0, 1) * 1000
                right_speed_cmd_scaled = np.clip(raw_right_speed, 0, 1) * 1000
                
                left_speed_cmd_scaled  = left_speed_cmd_scaled.astype(int).tolist()
                right_speed_cmd_scaled = right_speed_cmd_scaled.astype(int).tolist()
                
                prev_left_angle_norm  = current_left_q_target_norm.copy()
                prev_right_angle_norm = current_right_q_target_norm.copy()
                
                # --- force_set (각도 오차 기반 PD) ----------------------------------------
                err_left  = current_left_q_target_norm  - current_left_q_state_norm   # (-1 ~ +1)
                err_right = current_right_q_target_norm - current_right_q_state_norm

                d_err_left  = (err_left  - prev_err_left)  / dt
                d_err_right = (err_right - prev_err_right) / dt


                u_left  = Kp * err_left  + Kd * d_err_left
                u_right = Kp * err_right + Kd * d_err_right

                desired_force_left  = np.clip(u_left,  0, 1) 
                desired_force_right = np.clip(u_right, 0, 1)

                # 5. 스케일:  0 ~ +3000   (Inspire force 범위)
                left_force_cmd_scaled  = (u_left  * 3000).astype(int).tolist()
                right_force_cmd_scaled = (u_right * 3000).astype(int).tolist()

                # 6. 루프 끝에서 오차 갱신
                prev_err_left  = err_left
                prev_err_right = err_right
                
                
                action_data_for_logging = np.concatenate((current_left_q_target_norm, current_right_q_target_norm,
                                                          desired_force_left, desired_force_right,
                                                          raw_left_speed, raw_right_speed,
                                                        ))
                
                if dual_hand_state_array_shm and dual_hand_action_array_shm and dual_hand_data_lock:
                    with dual_hand_data_lock:
                        dual_hand_state_array_shm[:] = state_data_for_logging
                        dual_hand_action_array_shm[:] = action_data_for_logging
                        
                #추가        
                if dual_hand_force_array_shm and dual_hand_data_lock:
                    with dual_hand_data_lock:
                        dual_hand_force_array_shm[:] = force_data_for_logging 
                # 추가
                if dual_hand_touch_array_shm and dual_hand_data_lock:
                    with dual_hand_data_lock:
                        dual_hand_touch_array_shm[:] = touch_data_for_logging

                self._send_hand_command(scaled_left_cmd, scaled_right_cmd,
                                        left_force_cmd_scaled, right_force_cmd_scaled,
                                        left_speed_cmd_scaled, right_speed_cmd_scaled,
                                        )
                current_time = time.time()
                time_elapsed = current_time - start_time
                sleep_time = max(0, (1.0 / self.fps) - time_elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("[Inspire_Controller] Control process received KeyboardInterrupt. Exiting.")
        finally:
            running = False
            print("[Inspire_Controller] Control process has been closed.")

if __name__ == '__main__':
    print("Starting Inspire_Controller example...")
    mock_left_hand_input = Array('d', 75, lock=True)
    mock_right_hand_input = Array('d', 75, lock=True)
    
    with mock_right_hand_input.get_lock():
        for i in range(len(mock_right_hand_input)):
            mock_right_hand_input[i] = (i % 10) * 0.01 
            
    with mock_left_hand_input.get_lock():
        temp_left_mat = np.zeros((25,3))
        temp_left_mat[4] = np.array([-1.13, 0.3, 0.15])
        mock_left_hand_input[:] = temp_left_mat.flatten()

    shared_lock = Lock()
    shared_state = Array('d', Inspire_Num_Motors * 2, lock=False)
    shared_action = Array('d', Inspire_Num_Motors * 2, lock=False)

    try:
        controller = Inspire_Controller(
            left_hand_array=mock_left_hand_input,
            right_hand_array=mock_right_hand_input,
            dual_hand_data_lock=shared_lock,
            dual_hand_state_array=shared_state,
            dual_hand_action_array=shared_action,
            fps=50.0, 
            Unit_Test=False, # True로 설정 시 inspire_hand_Unit_Test.yml을 로드하려고 시도
            network_interface=""
        )

        count = 0
        main_loop_running = True
        while main_loop_running:
            try:
                time.sleep(1.0) 
                # Simulate a slight change in human hand input
                with mock_right_hand_input.get_lock():
                    # Make a noticeable change to one coordinate (e.g., y-coord of thumb tip)
                    # inspire_tip_indices[0] is thumb tip, index 1 is y-coordinate
                    mock_right_hand_input[inspire_tip_indices[0]*3 + 1] = 0.1 + (count % 10) * 0.02 
                
                with shared_lock:
                    print(f"Cycle {count} - Logged State: {[f'{x:.3f}' for x in shared_state[:]]}, Logged Action: {[f'{x:.3f}' for x in shared_action[:]]}")
                count +=1
                if count > 3000 : # Increased run time for more observation
                    print("Example finished after 3000 cycles.")
                    main_loop_running = False
            except KeyboardInterrupt:
                print("Main loop interrupted. Finishing example.")
                main_loop_running = False

    except Exception as e:
        print(f"An error occurred in the example: {e}")
    finally:
        print("Exiting main program.")