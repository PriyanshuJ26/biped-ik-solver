"""
Unit tests for BipedRobotIK
Run with:  python -m pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from src.biped_ik import BipedRobotIK


@pytest.fixture
def robot():
    return BipedRobotIK()


# ---------------------------------------------------------------------------
# D-H transform
# ---------------------------------------------------------------------------

class TestDHTransform:
    def test_identity_at_zero(self, robot):
        T = robot.dh_transform(0, 0, 0, 0)
        np.testing.assert_allclose(T, np.eye(4), atol=1e-10)

    def test_pure_rotation(self, robot):
        T = robot.dh_transform(0, 0, 0, np.pi / 2)
        assert T.shape == (4, 4)
        # Last row must be [0,0,0,1]
        np.testing.assert_allclose(T[3], [0, 0, 0, 1], atol=1e-10)

    def test_pure_translation_d(self, robot):
        T = robot.dh_transform(0, 0, 0.5, 0)
        assert pytest.approx(T[2, 3], abs=1e-10) == 0.5

    def test_pure_translation_a(self, robot):
        T = robot.dh_transform(0, 0.4, 0, 0)
        assert pytest.approx(T[0, 3], abs=1e-10) == 0.4


# ---------------------------------------------------------------------------
# Forward kinematics
# ---------------------------------------------------------------------------

class TestForwardKinematics:
    def test_zero_angles_returns_4x4(self, robot):
        T = robot.forward_kinematics_left_leg([0]*6)
        assert T.shape == (4, 4)

    def test_homogeneous_last_row(self, robot):
        T = robot.forward_kinematics_left_leg([0.1, -0.2, 0.3, 0.4, -0.1, 0.0])
        np.testing.assert_allclose(T[3], [0, 0, 0, 1], atol=1e-10)

    def test_right_leg_zero_angles(self, robot):
        T = robot.forward_kinematics_right_leg([0]*6)
        assert T.shape == (4, 4)
        np.testing.assert_allclose(T[3], [0, 0, 0, 1], atol=1e-10)


# ---------------------------------------------------------------------------
# Inverse kinematics
# ---------------------------------------------------------------------------

class TestInverseKinematics:
    def test_returns_six_angles(self, robot):
        angles = robot.inverse_kinematics_left_leg([0.0, 0.0, -0.50])
        assert len(angles) == 6

    def test_all_finite(self, robot):
        angles = robot.inverse_kinematics_left_leg([0.05, 0.0, -0.45])
        assert all(np.isfinite(a) for a in angles)

    def test_unreachable_target_clamped(self, robot):
        # Target far beyond max reach – should warn but not crash
        angles = robot.inverse_kinematics_left_leg([10.0, 0.0, 0.0])
        assert len(angles) == 6
        assert all(np.isfinite(a) for a in angles)

    def test_near_singularity_no_nan(self, robot):
        # Fully extended leg
        angles = robot.inverse_kinematics_left_leg([0.0, 0.0, -(robot.l4 + robot.l5)])
        assert all(np.isfinite(a) for a in angles)


# ---------------------------------------------------------------------------
# Squat motion
# ---------------------------------------------------------------------------

class TestSquatMotion:
    def test_returns_two_lists(self, robot):
        left, right = robot.squat_motion_ik(0.5)
        assert len(left) == 6
        assert len(right) == 6

    def test_symmetric(self, robot):
        left, right = robot.squat_motion_ik(0.7)
        np.testing.assert_allclose(left, right, atol=1e-10)

    @pytest.mark.parametrize("depth", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_all_finite_for_depths(self, robot, depth):
        left, right = robot.squat_motion_ik(depth)
        assert all(np.isfinite(a) for a in left + right)


# ---------------------------------------------------------------------------
# Joint limit validation
# ---------------------------------------------------------------------------

class TestJointLimits:
    def test_zero_angles_valid(self, robot):
        assert robot.validate_joint_limits([0.0]*6) is True

    def test_extreme_angle_invalid(self, robot):
        angles = [0, 0, 0, np.radians(160), 0, 0]   # knee 160° > 140°
        assert robot.validate_joint_limits(angles) is False


# ---------------------------------------------------------------------------
# Walking trajectory
# ---------------------------------------------------------------------------

class TestWalkingTrajectory:
    def test_correct_length(self, robot):
        traj = robot.generate_walking_trajectory(num_points=15)
        assert len(traj) == 15

    def test_each_point_has_three_coords(self, robot):
        traj = robot.generate_walking_trajectory(num_points=5)
        for pt in traj:
            assert len(pt) == 3

    def test_first_point_at_origin_x(self, robot):
        traj = robot.generate_walking_trajectory(step_length=0.20, num_points=10)
        assert pytest.approx(traj[0][0], abs=1e-9) == 0.0

    def test_last_point_x_equals_step_length(self, robot):
        traj = robot.generate_walking_trajectory(step_length=0.20, num_points=10)
        assert pytest.approx(traj[-1][0], abs=1e-9) == 0.20
