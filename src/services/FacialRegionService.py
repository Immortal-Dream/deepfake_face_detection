from src.config.path_config import *
import dlib
import cv2
import pandas as pd
import numpy as np
from pathlib import Path

from utils.region_utils import analyze_attention_polygon


class FacialRegionService:
    def __init__(self, threshold_quantile=0.7, min_threshold=0.1, experiment_name='MobileNetV3Large_baseline', dataset_name='rvf10k'):
        """
        initialize facial region analysis service

        args:
            threshold_quantile: quantile to use as attention threshold (e.g., 0.7 = 70th percentile)
            min_threshold: minimum threshold value to use even if quantile is lower (default 0.1)
            experiment_name: name of the experiment for organizing output
            dataset_name: name of the dataset being analyzed
        """
        # threshold parameters for determining region activation
        self.threshold_quantile = threshold_quantile
        self.min_threshold = min_threshold

        # setup output path
        self.experiment_name = experiment_name
        self.dataset_name = dataset_name
        self.output_folder = OUTPUT_FOLDER / experiment_name / dataset_name
        self.output_folder.mkdir(parents=True, exist_ok=True)

        # define facial regions using dlib's 81-point model
        self.FACIAL_REGIONS = {
            "jaw": list(range(0, 17)),  # jawline: points 0-16
            "eyebrows": list(range(17, 27)),  # eyebrows: points 17-26
            "nose": list(range(27, 36)),  # nose: points 27-35
            "eyes": list(range(36, 48)),  # eyes: points 36-47
            "mouth": list(range(48, 60)),  # mouth: points 48-59
            "forehead": list(range(68, 81)),  # forehead: points 68-80
        }

        # accumulated facial region results for batch processing
        self.accumulated_result = []

        # initialize dlib face detector and landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        landmark_model_path = DATA_ROOT / "shape_predictor_81_face_landmarks.dat"

        if not landmark_model_path.exists():
            print(f"Warning: landmark model not found at {landmark_model_path}")
            print("facial landmark detection will be skipped")
            self.predictor = None
        else:
            self.predictor = dlib.shape_predictor(str(landmark_model_path))
            print(f"loaded 81-point facial landmark predictor from {landmark_model_path}")

    def detect_facial_landmarks(self, image):
        """
        detect 81 facial landmarks in an image

        args:
            image: rgb image as numpy array

        returns:
            landmarks: numpy array of shape (81, 2) with (x, y) coordinates
                      or None if no face detected
        """
        if self.predictor is None:
            return None

        # convert to grayscale for dlib
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # detect faces
        faces = self.detector(gray)

        if len(faces) == 0:
            print("warning: no face detected in image")
            return None

        # get landmarks from first detected face
        shape = self.predictor(gray, faces[0])
        landmarks = np.array([[shape.part(i).x, shape.part(i).y] for i in range(81)])

        return landmarks

    def analyze_facial_region(self, heatmap, landmarks=None, threshold_quantile=None, min_threshold=None):
        """
        analyze which facial regions are activated by the heatmap using polygon masks.
        Wrapper around the utility function.
        """
        # use instance threshold if not specified
        tq = threshold_quantile if threshold_quantile is not None else self.threshold_quantile
        mt = min_threshold if min_threshold is not None else self.min_threshold

        # Call the new polygon-based utility function
        return analyze_attention_polygon(heatmap, landmarks, threshold_quantile=tq, min_threshold=mt)

    def add_result(self, filename, heatmap, image=None, landmarks=None,
                   prediction=None, true_label=None, confidence=None):
        """
        analyze a single image's heatmap and add to accumulated results

        args:
            filename: image filename
            heatmap: gradcam heatmap
            image: original image (rgb numpy array), used to detect landmarks if landmarks not provided
            landmarks: pre-computed 81 facial landmarks (optional)
            prediction: model prediction (0 or 1)
            true_label: ground truth label (0 or 1)
            confidence: prediction confidence (0-1)
        """
        # detect landmarks if not provided
        if landmarks is None and image is not None:
            landmarks = self.detect_facial_landmarks(image)

        # analyze facial regions
        region_attention = self.analyze_facial_region(heatmap, landmarks)

        # construct result row
        result_row = {
            "filename": Path(filename).name if isinstance(filename, (str, Path)) else filename,
            "prediction": prediction if prediction is not None else -1,
            "true_label": true_label if true_label is not None else -1,
            "confidence": confidence if confidence is not None else 0.0,
        }

        # add regional attention flags
        for region in self.FACIAL_REGIONS.keys():
            result_row[f"attention_{region}"] = region_attention.get(region, 0)

        # add to accumulated results
        self.accumulated_result.append(result_row)

        return region_attention

    def save_results_to_csv(self, filename="layercam_analysis.csv"):
        """
        save accumulated results to csv file

        args:
            filename: output csv filename (default: "layercam_analysis.csv")
        """
        if not self.accumulated_result:
            print("warning: no results to save")
            return

        # create dataframe
        df = pd.DataFrame(self.accumulated_result)

        # define output path
        csv_path = self.output_folder / filename

        # save to csv
        df.to_csv(csv_path, index=False)

        print(f"\ncsv saved to: {csv_path}")
        print(f"total rows: {len(df)}")
        print(f"\ncolumns: {list(df.columns)}")

        # print summary statistics
        if len(df) > 0:
            print("\n=== summary statistics ===")
            print(f"total images analyzed: {len(df)}")

            if 'prediction' in df.columns and df['prediction'].max() >= 0:
                print(f"predictions: {df['prediction'].value_counts().to_dict()}")

            if 'true_label' in df.columns and df['true_label'].max() >= 0:
                print(f"true labels: {df['true_label'].value_counts().to_dict()}")

            print("\n=== region activation rates ===")
            for region in self.FACIAL_REGIONS.keys():
                col = f"attention_{region}"
                if col in df.columns:
                    rate = df[col].mean() * 100
                    print(f"{region:12s}: {rate:5.1f}%")

        return csv_path

    def clear_results(self):
        """
        clear accumulated results (useful for processing multiple datasets)
        """
        self.accumulated_result = []
        print("accumulated results cleared")

    def get_statistics(self):
        """
        get statistics from accumulated results without saving

        returns:
            dict with summary statistics
        """
        if not self.accumulated_result:
            return {"error": "no results available"}

        df = pd.DataFrame(self.accumulated_result)

        stats = {
            "total_images": len(df),
            "region_activation_rates": {}
        }

        # calculate activation rates for each region
        for region in self.FACIAL_REGIONS.keys():
            col = f"attention_{region}"
            if col in df.columns:
                stats["region_activation_rates"][region] = df[col].mean()

        # add prediction statistics if available
        if 'prediction' in df.columns and df['prediction'].max() >= 0:
            stats["prediction_distribution"] = df['prediction'].value_counts().to_dict()

        if 'true_label' in df.columns and df['true_label'].max() >= 0:
            stats["true_label_distribution"] = df['true_label'].value_counts().to_dict()

        # calculate accuracy if both prediction and true_label available
        if 'prediction' in df.columns and 'true_label' in df.columns:
            if df['prediction'].max() >= 0 and df['true_label'].max() >= 0:
                accuracy = (df['prediction'] == df['true_label']).mean()
                stats["accuracy"] = accuracy

        return stats
