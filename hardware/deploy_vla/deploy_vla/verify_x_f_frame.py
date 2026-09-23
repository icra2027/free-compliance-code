"""Diagnostic (read-only, no actuation) for the deploy_smolvla.py wipe-delta sign
flip: checks whether the follower's real /current_pose topic -- what deploy_smolvla.py
feeds the policy directly as x_f's position -- already includes the flange->wiper-tip
offset that src/compliance_vla/policy/labels.py bakes into x_f via panda_fk.fk_batch(q) + tool_offset.npy
(see compliance-vla/scripts/calibrate_tool_offset.py).

If it doesn't, the state the policy sees at deployment is shifted from the training
distribution by R_flange(q) @ tool_offset (~10.4cm along the tool's own z-axis,
rotated into base frame by whatever orientation the tool is held at) -- a
plausible root cause for a reproducible, orientation-dependent Δx/Δy error, as
opposed to deploy_smolvla.py's own frame handling (which applies no transform at
all to position anywhere, so is not itself where an axis-flip bug could live).

Run it while the real follower is just sitting still, then again while jogging it
through a representative wipe orientation, and watch which of the three candidates
below stays closest to the real /current_pose reading.
"""

import os
import sys

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped

BOOKISH_SCRIPTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "compliance-vla", "scripts",
)
sys.path.insert(0, BOOKISH_SCRIPTS)
import panda_fk as fk  # noqa: E402

TOOL_OFFSET = np.load(os.path.join(BOOKISH_SCRIPTS, "tool_offset.npy"))


class VerifyXFFrame(Node):
    def __init__(self, follower_ns: str = "follower"):
        super().__init__("verify_x_f_frame")
        self.joint_q = None
        self.create_subscription(
            JointState,
            f"/{follower_ns}/franka_robot_state_broadcaster/measured_joint_states",
            self._joint_cb, 10,
        )
        self.create_subscription(
            PoseStamped,
            f"/{follower_ns}/franka_robot_state_broadcaster/current_pose",
            self._pose_cb, 10,
        )

    def _joint_cb(self, msg):
        if len(msg.position) >= 7:
            self.joint_q = np.asarray(msg.position[:7], dtype=np.float64)

    def _pose_cb(self, msg):
        if self.joint_q is None:
            return
        real_pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])

        pos_flange, rot_flange = fk.fk_batch(self.joint_q[None, :])
        pos_flange, rot_flange = pos_flange[0], rot_flange[0]
        pos_tip = pos_flange + rot_flange @ TOOL_OFFSET

        d_flange = real_pos - pos_flange
        d_tip = real_pos - pos_tip

        self.get_logger().info(
            f"real={real_pos} | real-FK_flange={d_flange} (|.|={np.linalg.norm(d_flange):.4f}) | "
            f"real-FK_tip={d_tip} (|.|={np.linalg.norm(d_tip):.4f})",
            throttle_duration_sec=1.0,
        )


def main():
    rclpy.init()
    node = VerifyXFFrame()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
