import numpy as np
import cv2

# Define facial regions based on dlib's 68-point or 81-point model
# These indices work for both standard 68 and extended 81 point models
FACIAL_REGIONS = {
    "jaw": list(range(0, 17)),
    "eyebrows": list(range(17, 27)),
    "nose": list(range(27, 36)),
    "eyes": list(range(36, 48)),
    "mouth": list(range(48, 60)),
    # Forehead is only available in 81-point models
    "forehead": list(range(68, 81)),
}


def analyze_attention_polygon(heatmap, landmarks, threshold=0.3):
    """
    Analyze which facial regions are activated by the heatmap using precise polygons.

    Args:
        heatmap: GradCAM-generated heatmap (H x W matrix), unnormalized or normalized.
        landmarks: Coordinates of facial feature points, shape (N, 2).
        threshold: Activation threshold (0.0 to 1.0).

    Returns:
        results: Dict with region names as keys and 1 (activated) or 0 (not activated) as values.
    """
    if landmarks is None:
        return {region: 0 for region in FACIAL_REGIONS.keys()}

    # Normalize heatmap to [0, 1] if not already
    if heatmap.max() - heatmap.min() > 0:
        heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    else:
        heatmap_norm = heatmap

    results = {}
    h, w = heatmap_norm.shape

    for region_name, indices in FACIAL_REGIONS.items():
        # Handle case where landmarks model doesn't support the region (e.g. forehead in 68-point)
        valid_indices = [idx for idx in indices if idx < len(landmarks)]
        if not valid_indices:
            results[region_name] = 0
            continue

        region_points = landmarks[valid_indices]

        # Skip if insufficient points to form a polygon/line
        if len(region_points) < 3:
            results[region_name] = 0
            continue

        # Create a binary mask for the current facial region
        mask = np.zeros((h, w), dtype=np.uint8)

        # Convert points to integer for cv2
        points = region_points.astype(np.int32)

        # For 'jaw' and 'eyebrows', they are lines/curves, not closed polygons naturally.
        # However, to check attention 'on' them, we can treat their convex hull
        # or a slightly thickened line as the region.
        # Using convexHull is generally robust for gathering the area 'around' the features.
        hull = cv2.convexHull(points)
        cv2.fillConvexPoly(mask, hull, 1)

        # Calculate mean or max attention within the mask
        masked_heatmap = heatmap_norm * mask

        # Check if the region is activated
        # Option A: Max value in region > threshold (Optimistic)
        # Option B: Mean value in region > threshold (Strict)
        # Using Max here to be consistent with previous logic ("is there *any* strong attention here?")
        if np.sum(mask) > 0:  # Avoid division by zero or empty regions
            # Get values only inside the mask
            values_in_region = heatmap_norm[mask == 1]
            max_attention = np.max(values_in_region) if len(values_in_region) > 0 else 0
            results[region_name] = 1 if max_attention > threshold else 0
        else:
            results[region_name] = 0

    return results