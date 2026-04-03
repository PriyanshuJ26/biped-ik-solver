"""
Biped Robot Inverse Kinematics Solver
======================================
Based on the research paper:
    "Inverse Kinematics Solution for Biped Robot"
    Thavai & Kadam, IOSR-JMCE, Volume 12, Issue 1 Ver. IV (Jan-Feb 2015), PP 57-62

Robot configuration:
    - 11 DOF total: 5 DOF per leg + 1 DOF for torso (shared)
    - Per leg: hip (2 DOF), knee (1 DOF), ankle (2 DOF)
    - Links modelled using Denavit-Hartenberg (D-H) convention
"""

import numpy as np


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _safe_arccos(value: float) -> float:
    """arccos with automatic clamping to [-1, 1]."""
    return float(np.arccos(np.clip(value, -1.0, 1.0)))


def _safe_arcsin(value: float) -> float:
    """arcsin with automatic clamping to [-1, 1]."""
    return float(np.arcsin(np.clip(value, -1.0, 1.0)))


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class BipedRobotIK:
    """
    Analytical Inverse Kinematics solver for an 11-DOF biped robot.

    Joint numbering (left leg):  θ1 (torso) → θ2 (hip pitch) → θ3 (hip roll)
                                 → θ4 (knee) → θ5 (ankle pitch) → θ6 (ankle roll)
    Joint numbering (right leg): θ1 (torso) → θ7 (hip pitch) → θ8 (hip roll)
                                 → θ9 (knee) → θ10 (ankle pitch) → θ11 (ankle roll)
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self,
                 l2: float = 0.10,
                 l4: float = 0.30,
                 l5: float = 0.30,
                 l6: float = 0.05,
                 l7: float = 0.10,
                 l9: float = 0.30,
                 l10: float = 0.30):
        """
        Parameters
        ----------
        l2  : Hip lateral offset for left leg  (m)
        l4  : Thigh length for left leg         (m)
        l5  : Shank length for left leg         (m)
        l6  : Foot offset for left leg          (m)
        l7  : Hip lateral offset for right leg  (m)
        l9  : Thigh length for right leg        (m)
        l10 : Shank length for right leg        (m)
        """
        self.l2  = l2
        self.l4  = l4
        self.l5  = l5
        self.l6  = l6
        self.l7  = l7
        self.l9  = l9
        self.l10 = l10

        # Joint angle limits in degrees (from paper, Section II)
        self.joint_limits = {
            'hip_sagittal':   (-50,  70),
            'hip_lateral':    (-60,  50),
            'knee':           (  0, 140),
            'ankle_sagittal': (-50,  70),
            'ankle_lateral':  (-60,  50),
        }

    # ------------------------------------------------------------------
    # D-H transformation matrix  (Equation 1)
    # ------------------------------------------------------------------

    @staticmethod
    def dh_transform(alpha: float, a: float, d: float, theta: float) -> np.ndarray:
        """
        Build the 4×4 D-H homogeneous transformation matrix.

        Parameters
        ----------
        alpha : twist angle  (rad)
        a     : link length  (m)
        d     : link offset  (m)
        theta : joint angle  (rad)

        Returns
        -------
        T : (4, 4) ndarray
        """
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)

        return np.array([
            [ct,  -st * ca,  st * sa,  a * ct],
            [st,   ct * ca, -ct * sa,  a * st],
            [ 0,        sa,       ca,       d],
            [ 0,         0,        0,       1],
        ], dtype=float)

    # ------------------------------------------------------------------
    # Forward kinematics
    # ------------------------------------------------------------------

    def forward_kinematics_left_leg(self, joint_angles) -> np.ndarray:
        """
        Forward kinematics for the left leg (Equation 2).

        Parameters
        ----------
        joint_angles : array-like of 6 floats
            [θ1, θ2, θ3, θ4, θ5, θ6] in radians

        Returns
        -------
        T06 : (4, 4) ndarray – torso → left ankle transform
        """
        t1, t2, t3, t4, t5, t6 = joint_angles

        T01 = self.dh_transform(0,           0,       0,  t1)
        T12 = self.dh_transform(np.pi / 2,   self.l2, 0,  t2 + np.pi / 2)
        T23 = self.dh_transform(np.pi / 2,   0,       0,  t3)
        T34 = self.dh_transform(0,           self.l4, 0,  t4)
        T45 = self.dh_transform(0,           self.l5, 0,  t5)
        T56 = self.dh_transform(-np.pi / 2,  0,       0,  t6)

        return T01 @ T12 @ T23 @ T34 @ T45 @ T56

    def forward_kinematics_right_leg(self, joint_angles) -> np.ndarray:
        """
        Forward kinematics for the right leg.

        Parameters
        ----------
        joint_angles : array-like of 6 floats
            [θ1, θ7, θ8, θ9, θ10, θ11] in radians

        Returns
        -------
        T011 : (4, 4) ndarray – torso → right ankle transform
        """
        t1, t7, t8, t9, t10, t11 = joint_angles

        T01   = self.dh_transform(0,           0,        0, t1)
        T17   = self.dh_transform(-np.pi / 2,  self.l7,  0, t7 + np.pi / 2)
        T78   = self.dh_transform(-np.pi / 2,  0,        0, t8)
        T89   = self.dh_transform(0,           self.l9,  0, t9)
        T910  = self.dh_transform(0,           self.l10, 0, t10)
        T1011 = self.dh_transform(np.pi / 2,   0,        0, t11)

        return T01 @ T17 @ T78 @ T89 @ T910 @ T1011

    # ------------------------------------------------------------------
    # Intermediate geometric helpers  (Equations 7–11)
    # ------------------------------------------------------------------

    def _hip_to_ankle_distance(self, Px: float, Py: float, Pz: float) -> float:
        """l06 = sqrt(Px² + Py² + Pz²)  (Equation 7)."""
        return float(np.sqrt(Px**2 + Py**2 + Pz**2))

    def _clamp_l06(self, l06: float) -> float:
        """Clamp target distance to the robot's reachable range."""
        max_reach = self.l4 + self.l5
        min_reach = abs(self.l4 - self.l5) + 1e-4   # small epsilon avoids singularity

        if l06 > max_reach:
            print(f"[Warning] Target {l06:.4f} m exceeds max reach {max_reach:.4f} m. Clamping.")
            return max_reach - 1e-4
        if l06 < min_reach:
            print(f"[Warning] Target {l06:.4f} m below min reach {min_reach:.4f} m. Clamping.")
            return min_reach
        return l06

    def _calc_l06yz(self, theta3: float, theta4: float, theta5: float) -> float:
        """
        Projected hip-to-ankle distance in the YZ plane (Equation 10).
        Simplification: equals (l4 + l5)|sin(θ3)| when joints are aligned,
        here computed from the full paper expression.
        """
        l4, l5 = self.l4, self.l5
        c3, s3 = np.cos(theta3), np.sin(theta3)
        c4, s4 = np.cos(theta4), np.sin(theta4)
        c5, s5 = np.cos(theta5), np.sin(theta5)

        # --- term A (numerator squared, from paper eq. 10) ---
        num_a = (l4 * (c5 * c4 - s4 * s5) * s3
                 + l5 * (c4**2 * c5 + c5 * s4**2) * s3) ** 2

        den_shared_8 = (c4**2 * c5**2 * c3**2 + c4**2 * c5**2 * s3**2
                        + c4**2 * c3**2 * s5**2 + c4**2 * s5**2 * s3**2
                        + c5**2 * c3**2 * s4**2 + c5**2 * s4**2 * s3**2
                        + c3**2 * s4**2 * s5**2 + s4**2 * s5**2 * s3**2) ** 2

        # --- term B (from paper eq. 10) ---
        num_b = (l5 * s5 * c4**2
                 + l4 * s5 * c4
                 + l5 * s5 * s4**2
                 + l4 * c5 * s4) ** 2

        den_4 = (c4**2 * c5**2 + c4**2 * s5**2
                 + c5**2 * s4**2 + s4**2 * s5**2) ** 2

        # guard against near-zero denominators
        if den_shared_8 < 1e-12 or den_4 < 1e-12:
            # fallback: project using simple geometry
            return float(abs((l4 + l5) * np.sin(theta3)))

        return float(np.sqrt(num_a / den_shared_8 + num_b / den_4))

    def _calc_l05yz(self, theta3: float, theta4: float, theta5: float) -> float:
        """
        Projected hip-to-knee distance in the YZ plane (Equation 11).
        Uses the same structural formula as l06yz but for the thigh only.
        """
        # In the paper eq.11 shares the same structure as eq.10 for l06yz,
        # so we reuse _calc_l06yz with only l4 contributing (l5=0).
        l4 = self.l4
        c3, s3 = np.cos(theta3), np.sin(theta3)
        c4, s4 = np.cos(theta4), np.sin(theta4)
        c5, s5 = np.cos(theta5), np.sin(theta5)

        num_a = (l4 * (c5 * c4 - s4 * s5) * s3) ** 2

        den_8 = (c4**2 * c5**2 * c3**2 + c4**2 * c5**2 * s3**2
                 + c4**2 * c3**2 * s5**2 + c4**2 * s5**2 * s3**2
                 + c5**2 * c3**2 * s4**2 + c5**2 * s4**2 * s3**2
                 + c3**2 * s4**2 * s5**2 + s4**2 * s5**2 * s3**2) ** 2

        num_b = (l4 * s5 * c4 + l4 * c5 * s4) ** 2
        den_4 = (c4**2 * c5**2 + c4**2 * s5**2
                 + c5**2 * s4**2 + s4**2 * s5**2) ** 2

        if den_8 < 1e-12 or den_4 < 1e-12:
            return float(abs(l4 * np.sin(theta3)))

        return float(np.sqrt(num_a / den_8 + num_b / den_4))

    # ------------------------------------------------------------------
    # Individual joint solvers
    # ------------------------------------------------------------------

    def _solve_theta4(self, l06: float) -> float:
        """
        Knee angle via law of cosines (Equation 8).

        θ4 = arccos((l06² - l4² - l5²) / (2·l4·l5))
        """
        cos_t4 = (l06**2 - self.l4**2 - self.l5**2) / (2 * self.l4 * self.l5)
        return _safe_arccos(cos_t4)

    def _solve_theta5_squat(self, l06: float) -> float:
        """
        Ankle pitch angle for squat motion (Section V of paper).

        l = sqrt(l4² - l06²) + sqrt(l5² - l06²)
        θ5 = arcsin(l / l4)
        """
        val_a = max(self.l4**2 - l06**2, 0.0)
        val_b = max(self.l5**2 - l06**2, 0.0)
        l = np.sqrt(val_a) + np.sqrt(val_b)
        return _safe_arcsin(l / self.l4)

    def _solve_theta6(self, l06yz: float, l05yz: float,
                      P_inv: np.ndarray) -> float:
        """
        Ankle roll angle via law of cosines (Equation 12).

        θ6 = sign(P⁻¹[2,4]) · arccos((l06yz² - l05yz² - l6²) / (2·l05yz·l6))
        """
        l6 = self.l6
        denom = 2.0 * l05yz * l6
        if abs(denom) < 1e-10:
            return 0.0
        cos_arg = (l06yz**2 - l05yz**2 - l6**2) / denom
        sign_val = 1.0 if P_inv[1, 3] >= 0 else -1.0
        return sign_val * _safe_arccos(cos_arg)

    def _solve_theta2(self, theta3: float, theta4: float,
                      theta5: float, theta6: float) -> float:
        """
        Hip pitch angle (Equation 15).
        """
        c3, s3 = np.cos(theta3), np.sin(theta3)
        c4, s4 = np.cos(theta4), np.sin(theta4)
        c5, s5 = np.cos(theta5), np.sin(theta5)
        c6, s6 = np.cos(theta6), np.sin(theta6)

        # Intermediate direction-cosine products
        expr1 = c5 * s3 * s4 - c5 * c3 * c4   # corrected signs from paper eq. 15
        expr2 = c5 * c3 * s4 + c5 * c4 * s3

        inner = c5 * expr1 + s5 * expr2
        first  = c6 * (c6 * s5 + s6 * inner)
        second = s6 * (s5 * s6 + c6 * inner)
        denom  = c6**2 - s6**2

        if abs(denom) < 1e-10:
            return 0.0

        sin_arg = (first - second) / denom
        return _safe_arcsin(sin_arg)

    def _solve_theta1(self, theta2: float, theta3: float, theta4: float,
                      theta5: float, theta6: float) -> float:
        """
        Torso angle (Equation 16).
        """
        c2, s2 = np.cos(theta2), np.sin(theta2)
        c3, s3 = np.cos(theta3), np.sin(theta3)
        c4, s4 = np.cos(theta4), np.sin(theta4)
        c5, s5 = np.cos(theta5), np.sin(theta5)
        c6, s6 = np.cos(theta6), np.sin(theta6)

        # Expressions from paper eq. 16
        ea = s4 * s3 - c4 * c3 * s2
        eb = c3 * s4 + c4 * s2 * s3
        ec = c4 * s3 + c3 * s4 * s2
        ed = c4 * c3 - s4 * s2 * s3

        inner_cos5 = c5 * (c4 * ea + s4 * eb)
        inner_sin5 = s5 * (c4 * eb - s4 * ea)
        combined   = inner_cos5 + inner_sin5

        first_complex  = c6 * combined - c4 * c2 * s6
        second_complex = s6 * combined - c4 * c2 * c6
        denom = c6**2 - s6**2

        if abs(denom) < 1e-10 or abs(c2) < 1e-10:
            return 0.0

        cos_arg = (s6 * first_complex / denom - c6 * second_complex / denom) / c2
        return _safe_arccos(cos_arg)

    def _solve_theta3(self, theta1: float, theta2: float,
                      theta4: float, theta5: float) -> float:
        """
        Hip roll angle (Equation 19).
        """
        l4, l5 = self.l4, self.l5
        c1, s1 = np.cos(theta1), np.sin(theta1)
        c2, s2 = np.cos(theta2), np.sin(theta2)
        c4, s4 = np.cos(theta4), np.sin(theta4)
        c5, s5 = np.cos(theta5), np.sin(theta5)

        # (cos θ1 sin θ2 + cos θ2 sin θ1) = sin(θ1+θ2)
        sum12  = c1 * s2 + c2 * s1   # sin(θ1+θ2)
        # (cos θ1 cos θ2 - sin θ1 sin θ2) = cos(θ1+θ2)
        diff12 = c1 * c2 - s1 * s2   # cos(θ1+θ2)

        denom_sq = (c1**2 * c2**2 + c1**2 * s2**2
                    + c2**2 * s1**2 + s1**2 * s2**2)   # always = 1

        eff = l5 * (c5 * c4 - s5 * s4) + l4 * c5      # effective reach factor

        term_sum  = eff * sum12
        term_diff = eff * diff12

        # numerator for cos θ3 (paper eq. 19)
        numerator = (term_sum * sum12 + term_diff * diff12) / max(denom_sq, 1e-12) - l4 / l5
        cos_arg = np.clip(numerator, -1.0, 1.0)
        return float(np.arccos(cos_arg))

    # ------------------------------------------------------------------
    # Inverse kinematics – left leg
    # ------------------------------------------------------------------

    def inverse_kinematics_left_leg(self, target_pos,
                                    target_orient=None) -> list:
        """
        Compute IK for the left leg.

        Parameters
        ----------
        target_pos    : (3,) array-like  [Px, Py, Pz] of ankle in torso frame (m)
        target_orient : ignored (orientation handled analytically via paper eqs.)

        Returns
        -------
        angles : list of 6 floats [θ1, θ2, θ3, θ4, θ5, θ6] in radians
        """
        Px, Py, Pz = float(target_pos[0]), float(target_pos[1]), float(target_pos[2])

        # Step 1 – distance from hip to ankle (eq. 7)
        l06 = self._clamp_l06(self._hip_to_ankle_distance(Px, Py, Pz))

        # Step 2 – knee angle (eq. 8)
        theta4 = self._solve_theta4(l06)

        # Step 3 – ankle pitch (squat formula, used as initial estimate for θ5)
        theta5 = self._solve_theta5_squat(l06)

        # Step 4 – projected distances (eqs. 9-11), use θ3=0 as initial estimate
        theta3_init = 0.0
        l06yz = self._calc_l06yz(theta3_init, theta4, theta5)
        l05yz = self._calc_l05yz(theta3_init, theta4, theta5)

        # Step 5 – ankle roll θ6 (eq. 12); P_inv = identity at initial step
        P_inv = np.eye(4)
        theta6 = self._solve_theta6(l06yz, l05yz, P_inv)

        # Step 6 – hip pitch θ2 (eq. 15)
        theta2 = self._solve_theta2(theta3_init, theta4, theta5, theta6)

        # Step 7 – hip roll θ3 (eq. 19), now with θ1=0 as estimate
        theta3 = self._solve_theta3(0.0, theta2, theta4, theta5)

        # Step 8 – torso angle θ1 (eq. 16)
        theta1 = self._solve_theta1(theta2, theta3, theta4, theta5, theta6)

        return [theta1, theta2, theta3, theta4, theta5, theta6]

    # ------------------------------------------------------------------
    # Squat motion IK  (Section V)
    # ------------------------------------------------------------------

    def squat_motion_ik(self, squat_depth: float = 0.5):
        """
        IK for symmetric squat-down motion (simplified 3-R chain, Section V).

        Parameters
        ----------
        squat_depth : float in [0, 1]
            0 = standing, 1 = maximum squat

        Returns
        -------
        left_angles, right_angles : each a list of 6 floats (rad)
        """
        squat_depth = float(np.clip(squat_depth, 0.0, 1.0))
        l06 = self._clamp_l06((self.l4 + self.l5) * (1.0 - 0.45 * squat_depth))

        theta5 = self._solve_theta5_squat(l06)
        theta3 = -theta5 / 2.0
        theta4 = -(theta3 + theta5)
        theta1 = theta2 = theta6 = 0.0

        angles = [theta1, theta2, theta3, theta4, theta5, theta6]
        return list(angles), list(angles)   # symmetric

    # ------------------------------------------------------------------
    # Joint limit validation
    # ------------------------------------------------------------------

    def validate_joint_limits(self, joint_angles) -> bool:
        """
        Check all joint angles against the mechanical limits defined in the paper.

        Parameters
        ----------
        joint_angles : array-like of 6 floats (rad)
            Order: [torso, hip_pitch, hip_roll, knee, ankle_pitch, ankle_roll]

        Returns
        -------
        bool : True if all angles are within limits
        """
        keys = ['hip_sagittal', 'hip_lateral', 'hip_sagittal',
                'knee', 'ankle_sagittal', 'ankle_lateral']
        # index 0 is torso – check with a wide ±90° range
        wide = (-90, 90)
        limits = [wide] + [self.joint_limits[k] for k in keys[1:]]

        all_ok = True
        for i, (angle_rad, (lo, hi)) in enumerate(zip(joint_angles, limits)):
            angle_deg = np.degrees(angle_rad)
            if not (lo <= angle_deg <= hi):
                print(f"  [Limit violation] Joint {i+1}: {angle_deg:.2f}° "
                      f"(allowed {lo}° to {hi}°)")
                all_ok = False
        return all_ok

    # ------------------------------------------------------------------
    # Walking trajectory generator
    # ------------------------------------------------------------------

    def generate_walking_trajectory(self,
                                    step_length: float = 0.20,
                                    step_height: float = 0.05,
                                    num_points: int = 20) -> list:
        """
        Generate a simple straight-line walking trajectory for one step.

        Parameters
        ----------
        step_length : forward distance per step (m)
        step_height : maximum foot lift height  (m)
        num_points  : number of via-points

        Returns
        -------
        list of [x, y, z] waypoints
        """
        trajectory = []
        for i in range(num_points):
            t = i / max(num_points - 1, 1)
            x = step_length * t
            y = 0.0
            # Foot lifts during the first half of the swing phase
            z = -0.55 + step_height * np.sin(np.pi * t) if t < 0.5 else -0.55
            trajectory.append([x, y, z])
        return trajectory


# ---------------------------------------------------------------------------
# Demo / self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    robot = BipedRobotIK()

    separator = "=" * 52

    print(separator)
    print("   Biped Robot Inverse Kinematics Solver – Demo")
    print(separator)

    # ------------------------------------------------------------------
    # 1. Forward kinematics
    # ------------------------------------------------------------------
    print("\n[1] Forward Kinematics Test")
    test_angles = [0, 0, np.pi / 6, np.pi / 4, -np.pi / 6, 0]
    T_left = robot.forward_kinematics_left_leg(test_angles)
    pos = T_left[:3, 3]
    print(f"    Input angles (deg): {[round(np.degrees(a), 2) for a in test_angles]}")
    print(f"    Left ankle position: x={pos[0]:.4f}  y={pos[1]:.4f}  z={pos[2]:.4f}  (m)")

    # ------------------------------------------------------------------
    # 2. Inverse kinematics + verification
    # ------------------------------------------------------------------
    print("\n[2] Inverse Kinematics Test")
    target_position = [0.05, 0.0, -0.50]
    print(f"    Target ankle position: {target_position}")

    ik_angles = robot.inverse_kinematics_left_leg(target_position)
    ik_deg = [round(np.degrees(a), 3) for a in ik_angles]
    print(f"    IK solution (deg)  : {ik_deg}")

    T_verify = robot.forward_kinematics_left_leg(ik_angles)
    achieved = T_verify[:3, 3]
    error = np.linalg.norm(np.array(target_position) - achieved)
    print(f"    Achieved position  : x={achieved[0]:.4f}  y={achieved[1]:.4f}  z={achieved[2]:.4f}  (m)")
    print(f"    Position error     : {error:.6f} m")

    print("\n    Joint limit check:")
    valid = robot.validate_joint_limits(ik_angles)
    print(f"    All within limits  : {valid}")

    # ------------------------------------------------------------------
    # 3. Squat motion
    # ------------------------------------------------------------------
    print("\n[3] Squat Motion Test")
    for depth in [0.0, 0.5, 1.0]:
        left, _ = robot.squat_motion_ik(squat_depth=depth)
        left_deg = [round(np.degrees(a), 2) for a in left]
        print(f"    depth={depth:.1f} -> angles (deg): {left_deg}")

    # ------------------------------------------------------------------
    # 4. Walking trajectory
    # ------------------------------------------------------------------
    print("\n[4] Walking Trajectory Generation")
    traj = robot.generate_walking_trajectory(step_length=0.20,
                                             step_height=0.05,
                                             num_points=10)
    print(f"    Generated {len(traj)} waypoints")
    for i, pt in enumerate(traj):
        print(f"    [{i:02d}]  x={pt[0]:.3f}  y={pt[1]:.3f}  z={pt[2]:.3f}")

    print(f"\n{separator}")
    print("   Demo complete.")
    print(separator)
