"""Target-centric controller (portable core/)."""

import numpy as np

from core.interfaces import PoseEstimate, VelocityCmd, DTYPE


class TargetCentricController:
    def __init__(self, target_world, kp: float = 1.0, v_max: float = 2.0):
        self.target_world = np.array(target_world, dtype=DTYPE).reshape(3)
        self.kp = float(kp)
        self.v_max = float(v_max)

    def command(self, estimate: PoseEstimate, evasion=None) -> VelocityCmd:
        pos_world = estimate.pose_world[:3, 3]
        R_world_body = estimate.pose_world[:3, :3]
        e_world = self.target_world - pos_world
        e_body = R_world_body.T @ e_world
        v_body = self.kp * e_body
        if evasion is not None:
            v_body = v_body + np.asarray(evasion, dtype=DTYPE).reshape(3)
        speed = float(np.linalg.norm(v_body))
        if speed > self.v_max:
            v_body = (v_body / speed) * self.v_max
        yaw_rate = 0.5 * float(np.arctan2(e_body[1], e_body[0]))
        return VelocityCmd(timestamp=estimate.timestamp, linear_vel=v_body, yaw_rate=yaw_rate)
