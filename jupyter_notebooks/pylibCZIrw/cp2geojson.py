import numpy as np
import cv2
import json
from shapely.geometry import mapping, Polygon


def seg_npy_to_geojson(seg_npy_path, geojson_path):
    # Load segmentation data
    data = np.load(seg_npy_path, allow_pickle=True).item()
    masks = data['masks']

    # Prepare GeoJSON structure
    geojson = {
        "type": "FeatureCollection",
        "features": []
    }

    # Extract contours and convert to GeoJSON features
    for label in np.unique(masks):
        if label == 0:  # Skip background
            continue
        # Find contours
        _, contours, _ = cv2.findContours((masks == label).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            # Convert contour to polygon and skip invalid polygons
            if contour.shape[0] < 3:
                continue
            polygon = Polygon(contour.squeeze())
            if not polygon.is_valid:
                continue
            # Add polygon to GeoJSON
            feature = {
                "type": "Feature",
                "properties": {
                    "label": int(label)
                },
                "geometry": mapping(polygon)
            }
            geojson["features"].append(feature)

    # Save GeoJSON to file
    with open(geojson_path, 'w') as f:
        json.dump(geojson, f)


# Example usage
seg_npy_path = 'HS3ST1___CD8 12175 HS3ST1___Snap-155268_c1+2+3_seg.npy'
geojson_path = 'HS3ST1___CD8 12175 HS3ST1___Snap-155268_c1+2+3_seg.geojson'
seg_npy_to_geojson(seg_npy_path, geojson_path)