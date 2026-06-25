import os
import tempfile
import numpy as np

from harness.groundtruth import read_euroc_groundtruth

def test_read_euroc_groundtruth():
    # Mock a temporary EuRoC directory
    with tempfile.TemporaryDirectory() as tmpdir:
        gt_dir = os.path.join(tmpdir, "mav0", "state_groundtruth_estimate0")
        os.makedirs(gt_dir)
        csv_path = os.path.join(gt_dir, "data.csv")

        with open(csv_path, "w") as f:
            f.write("#timestamp,p_RS_R_x,p_RS_R_y,p_RS_R_z,q_RS_w,q_RS_x,q_RS_y,q_RS_z\n")
            # Row 1: 1 billion ns = 1.0 sec, pos = (1, 2, 3), unit quaternion = (1, 0, 0, 0) -> Identity
            f.write("1000000000,1.0,2.0,3.0,1.0,0.0,0.0,0.0\n")
            # Row 2: 2 billion ns = 2.0 sec, pos = (4, 5, 6), valid quaternion for testing
            f.write("2000000000,4.0,5.0,6.0,0.7071068,0.0,0.0,0.7071068\n")

        ts, poses = read_euroc_groundtruth(tmpdir)

        # 1. Assert timestamps converted correctly to seconds
        assert len(ts) == 2
        assert ts[0] == 1.0
        assert ts[1] == 2.0

        # 2. Assert first pose is correct (Identity rotation, translation [1, 2, 3])
        assert np.allclose(poses[0][:3, 3], [1.0, 2.0, 3.0])
        assert np.allclose(poses[0][:3, :3], np.eye(3))

        # 3. Assert second pose has correct translation and an orthonormal rotation block
        assert np.allclose(poses[1][:3, 3], [4.0, 5.0, 6.0])
        R2 = poses[1][:3, :3]
        assert np.allclose(R2 @ R2.T, np.eye(3), atol=1e-6)
        assert abs(np.linalg.det(R2) - 1.0) < 1e-6
