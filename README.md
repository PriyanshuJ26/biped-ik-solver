# Biped Robot Inverse Kinematics Solver

Analytical inverse kinematics (IK) solver for an **11-DOF humanoid biped robot**, implemented in Python using Denavit–Hartenberg (DH) modeling and geometric methods.

---

## 🚀 Quick Demo

Run:

```bash
python examples/demo.py
```

**Example Output:**

```
Target: [0.1, 0.05, -0.5]
Achieved: [0.0998, 0.0502, -0.4999]
Error: 0.0003 m
```

---

## ✨ Key Features

* Analytical (closed-form) inverse kinematics
* Forward kinematics using DH parameters
* FK–IK consistency verification
* Walking trajectory generation
* Squat motion simulation
* Joint limit validation
* Numerical stability (safe trigonometric handling)

---

## 📁 Project Structure

```
biped-robot-inverse-kinematics/
│
├── src/
│   └── biped_ik.py          # Core IK solver
│
├── tests/
│   └── test_biped_ik.py     # Unit tests
│
├── docs/
│   ├── ik_paper.pdf         # Base research paper
│   └── mpc_gait_paper.pdf   # Advanced locomotion reference
│
├── examples/
│   └── demo.py              # Quick demo script
│
├── README.md
├── requirements.txt
└── LICENSE
```

---

## ⚙️ Installation

```bash
git clone https://github.com/YOUR_USERNAME/biped-robot-inverse-kinematics.git
cd biped-robot-inverse-kinematics
pip install -r requirements.txt
```

---

## 🧠 Usage

```python
from src.biped_ik import BipedRobotIK
import numpy as np

robot = BipedRobotIK()

# Forward kinematics
angles = [0, 0, np.pi/6, np.pi/4, -np.pi/6, 0]
T = robot.forward_kinematics_left_leg(angles)

# Inverse kinematics
target = [0.05, 0.0, -0.50]
ik = robot.inverse_kinematics_left_leg(target)
```

---

## 🤖 Robot Model

* **Total DOF:** 11
* **Per leg:** 5 DOF

  * Hip: 2 DOF
  * Knee: 1 DOF
  * Ankle: 2 DOF
* **Torso:** 1 DOF (shared)

### Joint Order (Left Leg)

```
θ1 (torso) → θ2 (hip pitch) → θ3 (hip roll)
           → θ4 (knee)
           → θ5 (ankle pitch) → θ6 (ankle roll)
```

---

## 📐 Mathematical Foundation

The solver is based on:

* Denavit–Hartenberg transformation matrices
* Homogeneous transformation chaining
* Geometric inverse kinematics

Key computations include:

* Knee angle via law of cosines
* Ankle angles via projected distances
* Hip and torso angles via inverse transformation

---

## 🔬 Research Context

This project is based on:

**1. Analytical IK formulation**

* *Inverse Kinematics Solution for Biped Robot*
* Provides closed-form joint solutions using geometric methods

**2. Extension toward locomotion**

* MPC-based humanoid gait generation (see `docs/mpc_gait_paper.pdf`)
* Introduces dynamic walking and running control using CoM and ZMP models

👉 This connects:
**Kinematics → Control → Humanoid Locomotion**

---

## 🧪 Testing

Run unit tests:

```bash
pytest tests/ -v
```

Includes tests for:

* D-H transformations
* Forward kinematics
* Inverse kinematics stability
* Joint limits
* Trajectory generation

---

## ⚠️ Limitations

* Analytical IK may not capture all possible configurations
* Singularities (fully stretched or folded leg) require numerical safeguards
* Right-leg IK is symmetric but not separately implemented
* No dynamic control or real-time feedback (pure kinematics)

---

## 🚀 Future Improvements

* 3D visualization of robot motion
* PyBullet / simulation integration
* Full walking controller implementation
* Real robot deployment
* Optimization-based IK extensions

---


## 📄 License

MIT License — see `LICENSE`
