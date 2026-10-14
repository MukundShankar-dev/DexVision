"""Float64 polygon features for differentiable calibrated silhouette fitting."""

from __future__ import annotations

import cv2
import numpy as np


def convex_polygon(points: np.ndarray) -> np.ndarray:
    """Return hull vertices retaining the original [N,2] float64 coordinates.

    OpenCV chooses discrete vertex indices only. Continuous optimization must
    not round the selected projected coordinates to float32 afterwards.
    """
    points = np.asarray(points, np.float64).reshape(-1, 2)
    if len(points) < 3 or not np.isfinite(points).all():
        raise ValueError("A finite polygon needs at least three points.")
    indices = cv2.convexHull(points.astype(np.float32), returnPoints=False).ravel()
    return points[indices]


def polygon_features(points: np.ndarray) -> np.ndarray:
    """Centroid and covariance features [cx,cy,std_x,std_y,cov/max_std].

    All values use pixels. Shift before integrating to avoid catastrophic
    cancellation between raw moments at large image-coordinate offsets.
    """
    points = np.asarray(points, np.float64).reshape(-1, 2)
    origin = points.mean(0)
    local = points-origin
    x, y = local.T
    nx, ny = np.roll(local, -1, axis=0).T
    cross = x*ny-nx*y
    twice_area = cross.sum()
    if abs(twice_area) < 1e-10:
        raise ValueError("Degenerate silhouette polygon.")
    cx = ((x+nx)*cross).sum()/(3*twice_area)
    cy = ((y+ny)*cross).sum()/(3*twice_area)
    variance_x = ((x*x+x*nx+nx*nx)*cross).sum()/(6*twice_area)-cx*cx
    variance_y = ((y*y+y*ny+ny*ny)*cross).sum()/(6*twice_area)-cy*cy
    covariance = ((2*x*y+x*ny+nx*y+2*nx*ny)*cross).sum()/(12*twice_area)-cx*cy
    std_x, std_y = np.sqrt(np.maximum([variance_x, variance_y], 1e-12))
    return np.array([cx+origin[0], cy+origin[1], std_x, std_y,
                     covariance/max(std_x, std_y)], np.float64)
