from pymavlink import mavutil

# Connect to Pixhawk
master = mavutil.mavlink_connection('udpin:localhost:14550')  # Use appropriate port

# Arm the ROV (send command to Pixhawk)
master.mav.command_long_send(
    master.target_system,
    master.target_component,
    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
    0,  # Arm (1) or disarm (0)
    1,  # 1 for arm
    0, 0, 0, 0, 0, 0
)
