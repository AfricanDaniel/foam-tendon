#!/usr/bin/env python3

import time
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

# Import your custom service here
from actuator_interfaces.srv import MoveByDegrees

from dynamixel_sdk import *

# --- DYNAMIXEL SETTINGS ---
ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE = 64
ADDR_PROFILE_VELOCITY = 112
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132

PROTOCOL_VERSION = 2.0
DXL_ID = 2
BAUDRATE = 1000000
DEVICENAME = '/dev/ttyUSB0'

TORQUE_ENABLE = 1
TORQUE_DISABLE = 0
EXTENDED_POSITION_CONTROL_MODE = 4
DEFAULT_VELOCITY = 20
DXL_MOVING_STATUS_THRESHOLD = 10

# Positional Constants
HOME_POSE = 3914
MAX_POSE = 6060
PULSES_PER_DEGREE = 4096 / 360.0
PULSES_PER_1080_DEGREES = 12288


class DynamixelServiceNode(Node):
    def __init__(self):
        super().__init__('dynamixel_service_node')

        # 1. Initialize Dynamixel Port and Packet Handlers
        self.portHandler = PortHandler(DEVICENAME)
        self.packetHandler = PacketHandler(PROTOCOL_VERSION)

        if not self.portHandler.openPort():
            self.get_logger().error("Failed to open the port")
            quit()

        if not self.portHandler.setBaudRate(BAUDRATE):
            self.get_logger().error("Failed to change the baudrate")
            quit()

        # 2. Configure Motor (Torque Off -> Set Mode -> Torque On -> Set Speed)
        self.packetHandler.write1ByteTxRx(self.portHandler, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
        self.packetHandler.write1ByteTxRx(self.portHandler, DXL_ID, ADDR_OPERATING_MODE, EXTENDED_POSITION_CONTROL_MODE)
        self.packetHandler.write1ByteTxRx(self.portHandler, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
        self.packetHandler.write4ByteTxRx(self.portHandler, DXL_ID, ADDR_PROFILE_VELOCITY, DEFAULT_VELOCITY)

        self.get_logger().info("Dynamixel initialized and waiting for service calls...")

        # 3. Create Services
        self.srv_home = self.create_service(Trigger, 'home_motor', self.home_callback)
        self.srv_max = self.create_service(Trigger, 'max_motor', self.max_callback)
        self.srv_seq = self.create_service(Trigger, 'sequence_motor', self.sequence_callback)
        self.srv_move = self.create_service(MoveByDegrees, 'move_degrees', self.move_degrees_callback)
        self.srv_pose = self.create_service(Trigger, 'get_pose', self.get_pose_callback)

    # --- HELPER FUNCTIONS ---
    def get_current_position(self):
        for _ in range(5):
            pos, dxl_comm_result, dxl_error = self.packetHandler.read4ByteTxRx(
                self.portHandler, DXL_ID, ADDR_PRESENT_POSITION)
            if dxl_comm_result == COMM_SUCCESS and dxl_error == 0:
                # Convert unsigned 32-bit to signed for extended position mode
                if pos > 0x7FFFFFFF:
                    pos -= 0x100000000
                return pos
            self.get_logger().warn(
                f"Position read failed (comm={dxl_comm_result}, err={dxl_error}), retrying...")
            time.sleep(0.01)
        raise RuntimeError("Failed to read motor position after 5 attempts")

    def wait_for_motor(self, target_pos):
        """Blocks execution until the motor reaches the target position."""
        while True:
            current_pos = self.get_current_position()
            if not abs(target_pos - current_pos) > DXL_MOVING_STATUS_THRESHOLD:
                break
            time.sleep(0.01)  # Small sleep to prevent maxing out CPU

    def send_motor_command(self, target_pos):
        """Writes the goal position to the motor."""
        self.packetHandler.write4ByteTxRx(self.portHandler, DXL_ID, ADDR_GOAL_POSITION, int(target_pos))

    # --- SERVICE CALLBACKS ---
    def home_callback(self, request, response):
        current_pos = self.get_current_position()

        self.get_logger().info(f"Current Angle: {current_pos / PULSES_PER_DEGREE:.2f}°")
        self.get_logger().info(f"Target Angle: {HOME_POSE / PULSES_PER_DEGREE:.2f}°")
        self.get_logger().info(f"Moving to HOME pose: {HOME_POSE}")

        self.send_motor_command(HOME_POSE)
        self.wait_for_motor(HOME_POSE)

        response.success = True
        response.message = "Arrived at home position."
        return response

    def max_callback(self, request, response):
        current_pos = self.get_current_position()

        self.get_logger().info(f"Current Angle: {current_pos / PULSES_PER_DEGREE:.2f}°")
        self.get_logger().info(f"Target Angle: {MAX_POSE / PULSES_PER_DEGREE:.2f}°")
        self.get_logger().info(f"Moving to MAX pose: {MAX_POSE}")

        self.send_motor_command(MAX_POSE)
        self.wait_for_motor(MAX_POSE)

        response.success = True
        response.message = "Arrived at max position."
        return response

    def sequence_callback(self, request, response):
        start_pos = self.get_current_position()
        target_ccw = start_pos + int(PULSES_PER_DEGREE * 360)

        self.get_logger().info(f"Sequence start: {start_pos / PULSES_PER_DEGREE:.2f}°")

        # Step 1: 3 full rotations CCW
        self.get_logger().info(f"--- Step 1: 1 rotation CCW -> {target_ccw / PULSES_PER_DEGREE:.2f}° ---")
        self.send_motor_command(target_ccw)
        self.wait_for_motor(target_ccw)

        # Step 2: Pause
        self.get_logger().info("Target reached. Pausing for 5 seconds...")
        time.sleep(5.0)

        # Step 3: Return to start
        self.get_logger().info(f"--- Step 2: Returning to {start_pos / PULSES_PER_DEGREE:.2f}° ---")
        self.send_motor_command(start_pos)
        self.wait_for_motor(start_pos)

        response.success = True
        response.message = "Sequence complete."
        return response

    def get_pose_callback(self, request, response):
        current_pos = self.get_current_position()
        degrees = current_pos / PULSES_PER_DEGREE
        self.get_logger().info(f"Current pose: {current_pos} pulses ({degrees:.2f}°)")
        response.success = True
        response.message = f"pulses={current_pos}, degrees={degrees:.2f}"
        return response

    def move_degrees_callback(self, request, response):
        current_pos = self.get_current_position()

        # Positive degrees = CCW (adding pulses), Negative degrees = CW (subtracting pulses)
        pulses_to_move = request.degrees * PULSES_PER_DEGREE
        target_pos = int(current_pos + pulses_to_move)

        self.get_logger().info(f"Current Angle: {current_pos / PULSES_PER_DEGREE:.2f}°")
        self.get_logger().info(f"Target Angle: {target_pos / PULSES_PER_DEGREE:.2f}°")
        self.get_logger().info(f"Moving {request.degrees} degrees to pulse position: {target_pos}")

        self.send_motor_command(target_pos)
        self.wait_for_motor(target_pos)

        response.success = True
        response.message = f"Successfully moved {request.degrees} degrees."
        return response

    def destroy_node(self):
        """Safely disable torque and close port when node is killed via Ctrl+C"""
        self.packetHandler.write1ByteTxRx(self.portHandler, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
        self.portHandler.closePort()
        self.get_logger().info("Torque disabled and port closed.")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DynamixelServiceNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()