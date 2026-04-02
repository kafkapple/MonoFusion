"""Inspect the full data pipeline: raw -> FaceLift -> MonoFusion."""
import json
import os
import numpy as np
import h5py
from PIL import Image

def main():
    # === 1. RAW DATA ===
    raw = "/node_data/joon/data/raw/markerless_mouse_1_nerf"
    print("=== RAW DATA ===")
    print("Contents:", sorted(os.listdir(raw)))

    h5 = h5py.File(os.path.join(raw, "camera_params.h5"), "r")
    print("\nH5 keys:", list(h5.keys()))
    for k in list(h5.keys()):
        print("  %s: shape=%s dtype=%s" % (k, h5[k].shape, h5[k].dtype))

    # Extract camera info
    for k in ["w2c", "extrinsic", "ext", "camera_matrix"]:
        if k in h5:
            data = np.array(h5[k])
            print("\nRaw '%s' shape: %s" % (k, data.shape))
            if data.ndim >= 2:
                print("cam0:\n", data[0])
                if data[0].shape == (4, 4) or data[0].shape == (3, 4):
                    R = data[0][:3, :3]
                    t = data[0][:3, 3]
                    pos = -R.T @ t
                    print("cam0 position:", pos)
                    print("det(R):", np.linalg.det(R))

    for k in ["intrinsic", "K", "camera_intrinsic"]:
        if k in h5:
            data = np.array(h5[k])
            print("\nRaw '%s' shape: %s" % (k, data.shape))
            print("cam0:\n", data[0])

    h5.close()

    # Check raw images
    raw_imgs = os.path.join(raw, "simpleclick_undist")
    if os.path.exists(raw_imgs):
        cams = sorted(os.listdir(raw_imgs))
        print("\nRaw image dirs:", cams[:6])
        if cams:
            first_cam = os.path.join(raw_imgs, cams[0])
            imgs = sorted(os.listdir(first_cam))
            if imgs:
                sample = Image.open(os.path.join(first_cam, imgs[0]))
                print("Raw image: %s, size=%s, mode=%s" % (imgs[0], sample.size, sample.mode))

    # === 2. FACELIFT PREPROCESSED ===
    fl_dir = "/node_data/joon/data/preprocessed/FaceLift_mouse/M5/000000"
    print("\n=== FACELIFT PREPROCESSED ===")
    cam_json = json.load(open(os.path.join(fl_dir, "opencv_cameras.json")))
    print("Num frame entries:", len(cam_json["frames"]))
    for f in cam_json["frames"][:6]:
        w2c_fl = np.array(f["w2c"])
        R = w2c_fl[:3, :3]
        t = w2c_fl[:3, 3]
        cam_pos_fl = -R.T @ t
        print("cam%d: fx=%.1f fy=%.1f cx=%.1f cy=%.1f w=%d h=%d pos=[%.2f,%.2f,%.2f] det(R)=%.4f" % (
            f["view_id"], f["fx"], f["fy"], f["cx"], f["cy"], f["w"], f["h"],
            cam_pos_fl[0], cam_pos_fl[1], cam_pos_fl[2],
            np.linalg.det(R)))

    # FL image info
    imgs_dir = os.path.join(fl_dir, "images")
    samples = sorted(os.listdir(imgs_dir))
    if samples:
        img = Image.open(os.path.join(imgs_dir, samples[0]))
        print("\nFaceLift image: %s, size=%s, mode=%s" % (samples[0], img.size, img.mode))
        print("FaceLift image files:", samples[:6])

    # === 3. MONOFUSION CONVERTED ===
    mf_dir = "/node_data/joon/data/monofusion/m5t2_v5"
    print("\n=== MONOFUSION CONVERTED ===")
    meta = json.load(open(os.path.join(mf_dir, "_raw_data/m5t2/trajectory/Dy_train_meta.json")))
    print("n_frames=%d, n_cams=%d, convention=%s" % (
        len(meta["k"]), len(meta["hw"]), meta.get("camera_convention", "NOT SET")))

    for ci in range(len(meta["hw"])):
        K = np.array(meta["k"][0][ci])
        w2c_mf = np.array(meta["w2c"][0][ci])
        R = w2c_mf[:3, :3]
        t = w2c_mf[:3, 3]
        cam_pos_mf = -R.T @ t
        hw = meta["hw"][ci]
        print("cam%d: fx=%.1f fy=%.1f cx=%.1f cy=%.1f h=%d w=%d pos=[%.2f,%.2f,%.2f] det(R)=%.4f" % (
            ci, K[0,0], K[1,1], K[0,2], K[1,2], hw[0], hw[1],
            cam_pos_mf[0], cam_pos_mf[1], cam_pos_mf[2],
            np.linalg.det(R)))

    # MF image info
    mf_img_dir = os.path.join(mf_dir, "images/m5t2_cam00")
    mf_samples = sorted(os.listdir(mf_img_dir))
    if mf_samples:
        mf_img = Image.open(os.path.join(mf_img_dir, mf_samples[0]))
        print("\nMF image: %s, size=%s, mode=%s" % (mf_samples[0], mf_img.size, mf_img.mode))

    # === 4. DEPTH DATA ===
    print("\n=== DEPTH DATA (MoGe per-camera) ===")
    for cam in ["m5t2_cam00", "m5t2_cam01", "m5t2_cam02", "m5t2_cam03"]:
        cd = os.path.join(mf_dir, "aligned_moge_depth", cam)
        if os.path.exists(cd):
            d_files = sorted(os.listdir(cd))
            d = np.load(os.path.join(cd, d_files[0]))
            print("%s: %d files, shape=%s, range=[%.3f, %.3f], mean=%.3f" % (
                cam, len(d_files), d.shape, d.min(), d.max(), d.mean()))

    # === 5. CONVERSION INFO ===
    conv_info = os.path.join(mf_dir, "conversion_info.json")
    if os.path.exists(conv_info):
        print("\n=== CONVERSION INFO ===")
        info = json.load(open(conv_info))
        for k, v in info.items():
            print("  %s: %s" % (k, v))

    # === 6. COMPARE RAW vs FL vs MF cameras ===
    print("\n=== CAMERA COMPARISON ===")
    h5 = h5py.File(os.path.join(raw, "camera_params.h5"), "r")
    for k in h5.keys():
        if "w2c" in k.lower() or "ext" in k.lower():
            raw_cams = np.array(h5[k])
            break
    else:
        raw_cams = None
    raw_K_all = None
    for k in h5.keys():
        if "int" in k.lower() or k == "K":
            raw_K_all = np.array(h5[k])
            break
    h5.close()

    if raw_cams is not None:
        print("Raw cameras shape:", raw_cams.shape)
        print("Raw n_cameras:", raw_cams.shape[0])
        if raw_K_all is not None:
            print("Raw K shape:", raw_K_all.shape)
            print("Raw cam0 K:\n", raw_K_all[0])
            print("\nFL cam0 K: fx=%.1f fy=%.1f cx=%.1f cy=%.1f (512x512)" % (
                cam_json["frames"][0]["fx"], cam_json["frames"][0]["fy"],
                cam_json["frames"][0]["cx"], cam_json["frames"][0]["cy"]))
            print("MF cam0 K: fx=%.1f fy=%.1f cx=%.1f cy=%.1f (512x512)" % (
                np.array(meta["k"][0][0])[0,0], np.array(meta["k"][0][0])[1,1],
                np.array(meta["k"][0][0])[0,2], np.array(meta["k"][0][0])[1,2]))

if __name__ == "__main__":
    main()
