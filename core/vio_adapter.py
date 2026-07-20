"""
core/vio_adapter.py
-------------------
The SEAM of the whole system. The scheduler must not care WHERE its signals come
from -- the kinematic sim, a drift scaffold over a real dataset, or (later) a real
OpenVINS/ORB-SLAM3 build. They all implement the same tiny interface, so Parth can
drop the real VIO in later by writing one more subclass, and nothing downstream changes.

This file is PURE and portable (no numpy, no OpenCV) -- it belongs in core/.

A VioSource produces, each update, a `VioSignals` bundle:
  - est:   the (x, y) position estimate in the target-centric frame
  - cues:  the glass-box dict the UncertaintyScheduler consumes
  - gt:    ground truth (x, y) when available (datasets/sim have it; real flight won't)
It can also apply_fix() (an absolute correction) and report whether it has arrived.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class VioSignals:
    t: float
    est: tuple                      # (x, y) estimate
    cues: dict                      # keys: sigma_pos, sigma_head, feature_loss, blur,
                                    #       imu_bias, active_features
    gt: tuple | None = None         # (x, y) ground truth, or None on real flight
    extra: dict = field(default_factory=dict)

    def error(self):
        """True estimate error if ground truth is available, else None."""
        if self.gt is None:
            return None
        return ((self.est[0] - self.gt[0]) ** 2 + (self.est[1] - self.gt[1]) ** 2) ** 0.5


class VioSource(ABC):
    """Anything that can feed the scheduler. Implement these four methods."""

    @abstractmethod
    def update(self) -> VioSignals | None:
        """Advance one step; return signals, or None when the stream is exhausted."""

    @abstractmethod
    def apply_fix(self) -> bool:
        """Attempt an absolute correction (e.g. a landmark in view). True if applied."""

    def arrived(self) -> bool:
        return False

    def name(self) -> str:
        return self.__class__.__name__


# A real VIO would be added later WITHOUT touching the scheduler, like:
#
#   class OpenVinsSource(VioSource):
#       def update(self):     # pull pose+covariance from the OpenVINS process
#           ...
#       def apply_fix(self):  # fuse an ArUco measurement into the OpenVINS state
#           ...
#
# That is the entire point of this seam.
