"""Verify camera convention in Dy_train_meta.json.

Tests two hypotheses:
  A) md['w2c'] stores real world-to-camera matrices
  B) md['w2c'] stores camera-to-world (DUSt3R convention)
"""

import json
import numpy as np
import sys


def main(meta_path):
    with open(meta_path) as f:
        md = json.load(f)

    n_cams = len(md["hw"])
    w2c0 = np.array(md["w2c"][0][0])
    R = w2c0[:3, :3]
    det = np.linalg.det(R)
    ortho_err = np.max(np.abs(R.T @ R - np.eye(3)))
    print(f"Matrix shape: {w2c0.shape}, R det: {det:.6f}, ortho_err: {ortho_err:.8f}")

    # Camera centers under both hypotheses
    centers_A = []  # Hypothesis A: real w2c → C = -R^T @ t
    centers_B = []  # Hypothesis B: c2w → C = mat[:3,3]
    for c in range(n_cams):
        mat = np.array(md["w2c"][0][c])
        R_c, t_c = mat[:3, :3], mat[:3, 3]
        centers_A.append(-R_c.T @ t_c)
        centers_B.append(t_c)

    centers_A = np.array(centers_A)
    centers_B = np.array(centers_B)

    print(f"\n--- Hypothesis A: md[w2c] = REAL w2c ---")
    for i, cc in enumerate(centers_A):
        print(f"  cam{i}: [{cc[0]:.4f}, {cc[1]:.4f}, {cc[2]:.4f}]")
    spread_A = np.linalg.norm(centers_A.max(0) - centers_A.min(0))
    print(f"  Spread: {spread_A:.4f}")

    print(f"\n--- Hypothesis B: md[w2c] = c2w ---")
    for i, cc in enumerate(centers_B):
        print(f"  cam{i}: [{cc[0]:.4f}, {cc[1]:.4f}, {cc[2]:.4f}]")
    spread_B = np.linalg.norm(centers_B.max(0) - centers_B.min(0))
    print(f"  Spread: {spread_B:.4f}")

    # Projection test A
    scene_A = centers_A.mean(axis=0)
    print(f"\n--- Projection (Hyp A, center={np.round(scene_A, 4)}) ---")
    all_in_A = True
    for c in range(n_cams):
        mat = np.array(md["w2c"][0][c])
        K_c = np.array(md["k"][0][c])
        h_c, w_c = md["hw"][c]
        p = mat @ np.append(scene_A, 1.0)
        z = p[2]
        if z > 0:
            px = K_c[0, 0] * p[0] / z + K_c[0, 2]
            py = K_c[1, 1] * p[1] / z + K_c[1, 2]
            ok = 0 <= px <= w_c and 0 <= py <= h_c
            label = "IN" if ok else "OUT"
            if not ok:
                all_in_A = False
            print(f"  cam{c}: ({px:.1f}, {py:.1f}) z={z:.3f} [{label}]")
        else:
            all_in_A = False
            print(f"  cam{c}: BEHIND z={z:.3f}")

    # Projection test B
    scene_B = centers_B.mean(axis=0)
    print(f"\n--- Projection (Hyp B, center={np.round(scene_B, 4)}) ---")
    all_in_B = True
    for c in range(n_cams):
        c2w = np.array(md["w2c"][0][c])
        w2c_actual = np.linalg.inv(c2w)
        K_c = np.array(md["k"][0][c])
        h_c, w_c = md["hw"][c]
        p = w2c_actual @ np.append(scene_B, 1.0)
        z = p[2]
        if z > 0:
            px = K_c[0, 0] * p[0] / z + K_c[0, 2]
            py = K_c[1, 1] * p[1] / z + K_c[1, 2]
            ok = 0 <= px <= w_c and 0 <= py <= h_c
            label = "IN" if ok else "OUT"
            if not ok:
                all_in_B = False
            print(f"  cam{c}: ({px:.1f}, {py:.1f}) z={z:.3f} [{label}]")
        else:
            all_in_B = False
            print(f"  cam{c}: BEHIND z={z:.3f}")

    # Verdict
    print(f"\n=== VERDICT ===")
    print(f"Hypothesis A (real w2c): spread={spread_A:.2f}, all_in_frame={all_in_A}")
    print(f"Hypothesis B (c2w):      spread={spread_B:.2f}, all_in_frame={all_in_B}")

    if spread_A > spread_B * 3 and all_in_A:
        print("CONCLUSION: md[w2c] stores REAL w2c (Hypothesis A)")
    elif all_in_B and not all_in_A:
        print("CONCLUSION: md[w2c] stores c2w (Hypothesis B)")
    else:
        print("INCONCLUSIVE: manual inspection needed")

    print(f"\nMeta keys: {list(md.keys())}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/node_data/joon/data/monofusion/m5t2_v5/_raw_data/m5t2/trajectory/Dy_train_meta.json"
    main(path)
