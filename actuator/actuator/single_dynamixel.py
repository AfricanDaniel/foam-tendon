#!/usr/bin/env python3

import os
import time
from dynamixel_sdk import * # --- DYNAMIXEL XL430-W250 SETTINGS ---

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

# MODIFIED: Value 4 is Extended Position Control Mode (allows multi-turn)
EXTENDED_POSITION_CONTROL_MODE = 4
DEFAULT_VELOCITY = 20

# MODIFIED: Slightly loosened the threshold so it doesn't get stuck in the loop
DXL_MOVING_STATUS_THRESHOLD = 10

# MODIFIED: 1080 degrees = 3 full revolutions (3 * 4096 = 12288 pulses)
PULSES_PER_1080_DEGREES = 12288


def main():
    portHandler = PortHandler(DEVICENAME)
    packetHandler = PacketHandler(PROTOCOL_VERSION)

    if not portHandler.openPort():
        print("Failed to open the port")
        quit()

    if not portHandler.setBaudRate(BAUDRATE):
        print("Failed to change the baudrate")
        quit()

    # Ensure Torque is OFF so we can change settings
    packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)

    # MODIFIED: Set to Extended Position Control Mode for multi-turn
    packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_OPERATING_MODE, EXTENDED_POSITION_CONTROL_MODE)
    print("Operating Mode set to Extended Position Control (Multi-Turn).")

    # Enable Dynamixel Torque
    packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
    print("Dynamixel has been successfully connected and torque is ON.")

    # Set the slower speed profile
    packetHandler.write4ByteTxRx(portHandler, DXL_ID, ADDR_PROFILE_VELOCITY, DEFAULT_VELOCITY)

    try:
        # Read present position
        dxl_present_position, _, _ = packetHandler.read4ByteTxRx(portHandler, DXL_ID, ADDR_PRESENT_POSITION)
        print(f"Current Position: {dxl_present_position}")

        # Move 1080 Degrees CCW
        target_pos_ccw = dxl_present_position + PULSES_PER_1080_DEGREES
        print(f"Moving 1080 degrees CCW to position: {target_pos_ccw}")
        packetHandler.write4ByteTxRx(portHandler, DXL_ID, ADDR_GOAL_POSITION, target_pos_ccw)

        while True:
            dxl_present_position, _, _ = packetHandler.read4ByteTxRx(portHandler, DXL_ID, ADDR_PRESENT_POSITION)
            if not abs(target_pos_ccw - dxl_present_position) > DXL_MOVING_STATUS_THRESHOLD:
                break

        # Wait 5 seconds before moving back
        print("Target reached. Pausing for 5 seconds...")
        time.sleep(5)

        # Move 1080 Degrees CW
        target_pos_cw = target_pos_ccw - PULSES_PER_1080_DEGREES
        print(f"Moving 1080 degrees CW to position: {target_pos_cw}")
        packetHandler.write4ByteTxRx(portHandler, DXL_ID, ADDR_GOAL_POSITION, target_pos_cw)

        while True:
            dxl_present_position, _, _ = packetHandler.read4ByteTxRx(portHandler, DXL_ID, ADDR_PRESENT_POSITION)
            if not abs(target_pos_cw - dxl_present_position) > DXL_MOVING_STATUS_THRESHOLD:
                break

        print("Returned to start. Pausing for 1 second...")
        time.sleep(1)

    finally:
        packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
        print("Torque disabled.")
        portHandler.closePort()
        print("Port closed.")


if __name__ == '__main__':
    main()