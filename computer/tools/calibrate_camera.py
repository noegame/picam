#!/usr/bin/env python3
"""
calibrate_camera.py

Usage:
  python calibrate_camera.py --images_dir ./calib_images --pattern 9x6 --square_size 0.025 --out camera_calibration.npz
  python calibrate_camera.py --images_dir "C:\HOME\WORK\raspberrypi\ZOD\picam\data\calibration" --pattern 9x6 --square_size 0.025 --out camera_calibration.npz

- images_dir : dossier contenant les photos du damier (formats jpg/png)
- pattern : nombre de coins intérieurs du damier "colsxrows" (ex: 9x6 -> 9 colonnes intérieures, 6 lignes intérieures)
- square_size : taille d'une case en mètres (ou en unités cohérentes) ; ex 0.025 pour 25 mm
- out : fichier de sortie (.npz) contenant camera_matrix, dist_coeffs, rvecs, tvecs

Description:
    Calibre une caméra en utilisant des images d'un damier (chessboard) avec OpenCV.
    Détecte les coins intérieurs du damier dans les images fournies, puis calcule la matrice de la caméra et les coefficients de distorsion.
    Sauvegarde les résultats dans un fichier .npz.

"""

import argparse
import glob
import os
import sys
import cv2
import numpy as np

def parse_args():
    p = argparse.ArgumentParser(description="Calibrate camera using chessboard images (OpenCV).")
    p.add_argument("--images_dir", required=True, help="Folder with calibration images")
    p.add_argument("--pattern", default="9x6", help="Chessboard inner corners as COLSxROWS (default 9x6)")
    p.add_argument("--square_size", type=float, default=0.025, help="Size of one square in meters (default 0.025)")
    p.add_argument("--out", default="camera_calibration.npz", help="Output file (npz) to save calibration")
    p.add_argument("--show", action="store_true", help="Show corner detection images (for debugging)")
    return p.parse_args()

def collect_image_paths(images_dir):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff")
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(images_dir, e)))
    paths.sort()
    return paths

def calibrate_from_images(image_paths, pattern_size, square_size, show=False):
    cols, rows = pattern_size
    objp = np.zeros((rows*cols, 3), np.float32)
    # Note: grid coordinates: (0,0,0), (1,0,0), ... (cols-1,rows-1,0) scaled by square_size
    objp[:,:2] = np.mgrid[0:cols, 0:rows].T.reshape(-1,2) * square_size

    objpoints = []  # 3d points in world space
    imgpoints = []  # 2d points in image plane

    used_images = []
    for p in image_paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Warning: cannot read {p}", file=sys.stderr)
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, (cols, rows),cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if found:
            # refine corners for subpixel accuracy
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
            corners_refined = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), term)
            objpoints.append(objp)
            imgpoints.append(corners_refined)
            used_images.append(p)
            if show:
                vis = img.copy()
                cv2.drawChessboardCorners(vis, (cols, rows), corners_refined, found)
                cv2.imshow("corners", vis)
                cv2.waitKey(500)
        else:
            print(f"Chessboard not found in {p}", file=sys.stderr)

    if show:
        cv2.destroyAllWindows()

    if len(objpoints) < 5:
        raise RuntimeError(f"Not enough valid calibration images ({len(objpoints)} found). Need >=5 (prefer >=10-20).")

    # Calibration
    img_shape = cv2.cvtColor(cv2.imread(used_images[0]), cv2.COLOR_BGR2GRAY).shape[::-1]  # (w,h)
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape, None, None,
        flags=cv2.CALIB_RATIONAL_MODEL  # includes k4,k5,k6
    )

    # Compute reprojection error
    total_error = 0.0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        err = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += err
    mean_error = total_error / len(objpoints)

    return {
        "ret": ret,
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs,
        "rvecs": rvecs,
        "tvecs": tvecs,
        "reprojection_error": mean_error,
        "image_size": img_shape,
        "used_images": used_images
    }

def undistort_example(image_path, camera_matrix, dist_coeffs, out_path=None):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read {image_path}")
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w,h), 1, (w,h))
    und = cv2.undistort(img, camera_matrix, dist_coeffs, None, newcameramtx)
    if out_path:
        cv2.imwrite(out_path, und)
    return und

def main():
    args = parse_args()
    try:
        cols_rows = args.pattern.lower().split('x')
        if len(cols_rows) != 2:
            raise ValueError()
        cols = int(cols_rows[0]); rows = int(cols_rows[1])
    except Exception:
        print("pattern must be like 9x6 (COLSxROWS of inner corners).", file=sys.stderr)
        sys.exit(1)

    image_paths = collect_image_paths(args.images_dir)
    if not image_paths:
        print(f"No images found in {args.images_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(image_paths)} images, trying to detect chessboard {cols}x{rows}...")
    result = calibrate_from_images(image_paths, (cols, rows), args.square_size, show=args.show)

    # Print & save
    print("Calibration RMS error (returned by calibrateCamera):", result["ret"])
    print("Reprojection mean error (per-image average):", result["reprojection_error"])
    print("Image size (w,h):", result["image_size"])
    print("Camera matrix:\n", result["camera_matrix"])
    print("Distortion coefficients:\n", result["dist_coeffs"].ravel())
    print("Used images:", len(result["used_images"]))

    np.savez(args.out,
             camera_matrix=result["camera_matrix"],
             dist_coeffs=result["dist_coeffs"],
             rvecs=np.array(result["rvecs"], dtype=object),
             tvecs=np.array(result["tvecs"], dtype=object),
             reprojection_error=result["reprojection_error"],
             image_size=result["image_size"],
             used_images=result["used_images"])
    print(f"Saved calibration to {args.out}")

    # Optional: undistort and save first used image for quick check
    try:
        ex_path = result["used_images"][0]
        und = undistort_example(ex_path, result["camera_matrix"], result["dist_coeffs"], out_path="undistorted_example.png")
        print("Saved undistorted example to undistorted_example.png")
    except Exception as e:
        print("Could not create undistorted example:", e, file=sys.stderr)

if __name__ == "__main__":
    main()
