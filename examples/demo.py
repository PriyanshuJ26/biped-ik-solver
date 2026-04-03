from src.biped_ik import BipedRobotIK
import numpy as np

def main():
    robot = BipedRobotIK()

    # Target ankle position (in meters)
    target = [0.1, 0.05, -0.5]

    print("=== Biped IK Demo ===")
    print("Target position:", target)

    # Solve IK
    angles = robot.inverse_kinematics_left_leg(target)

    print("\nJoint angles (degrees):")
    print([round(np.degrees(a), 2) for a in angles])

    # Verify with FK
    T = robot.forward_kinematics_left_leg(angles)
    achieved = T[:3, 3]

    print("\nAchieved position:", achieved)

    # Compute error
    error = np.linalg.norm(np.array(target) - achieved)
    print("Position error:", round(error, 6), "m")


if __name__ == "__main__":
    main()