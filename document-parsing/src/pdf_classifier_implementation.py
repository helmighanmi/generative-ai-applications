"""
PDF Page Classification System
==============================
Based on: "Classification Automatique des Pages PDF : Approche Hybride"
Original: Deterministic geometric + visual approach
Improved: Adaptive ML-enhanced approach (Binary only - no ambiguous class)

Author: GHANMI Helmi
Date: 2026-02-22
"""

import fitz
import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from enum import Enum
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from collections import deque
import json

class PageType(Enum):
    DOCUMENT = "document"
    DIAGRAM = "diagram"
    # AMBIGUOUS removed - binary classification only

@dataclass
class ClassificationResult:
    page_type: PageType
    confidence: float
    features: Dict
    method: str

# ============================================================================
# ORIGINAL IMPLEMENTATION (From French Report)
# ============================================================================

class LayoutClassifier:
    """
    Original LayoutClassifier - Geometric text block analysis
    """
    def __init__(self, word_threshold=50, occupancy_threshold=0.3, cluster_threshold=0.6):
        self.word_threshold = word_threshold
        self.occupancy_threshold = occupancy_threshold
        self.cluster_threshold = cluster_threshold

    def classify(self, page: fitz.Page) -> ClassificationResult:
        # Extract geometric features
        words = page.get_text("words")
        blocks = page.get_text("blocks")

        word_count = len(words)
        block_count = len(blocks)

        # Spatial occupancy
        page_rect = page.rect
        grid_size = 20
        cell_w, cell_h = page_rect.width/grid_size, page_rect.height/grid_size
        occupied = set()
        for b in blocks:
            x0,y0,x1,y1 = b[:4]
            for r in range(int(y0/cell_h), int(y1/cell_h)+1):
                for c in range(int(x0/cell_w), int(x1/cell_w)+1):
                    occupied.add((r,c))
        occupancy = len(occupied)/(grid_size**2)

        # Cluster ratio
        text_area = sum((b[2]-b[0])*(b[3]-b[1]) for b in blocks)
        page_area = page_rect.width * page_rect.height
        cluster_ratio = text_area/page_area if page_area>0 else 0

        # Decision
        is_doc = (word_count > self.word_threshold and 
                  occupancy > self.occupancy_threshold and
                  cluster_ratio < self.cluster_threshold)

        return ClassificationResult(
            page_type=PageType.DOCUMENT if is_doc else PageType.DIAGRAM,
            confidence=0.7,
            features={'word_count': word_count, 'occupancy': occupancy, 'cluster_ratio': cluster_ratio},
            method='layout'
        )

class VisualLayoutClassifier:
    """
    Original VisualLayoutClassifier - Raster analysis
    """
    def __init__(self, resolution=150, edge_thresh=50):
        self.resolution = resolution
        self.edge_thresh = edge_thresh

    def classify(self, page: fitz.Page) -> ClassificationResult:
        # Rasterize
        mat = fitz.Matrix(self.resolution/72, self.resolution/72)
        pix = page.get_pixmap(matrix=mat)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if pix.n>1 else img

        # Edge density
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(sobelx**2 + sobely**2)
        edge_density = np.mean(edges > self.edge_thresh)

        # Dark pixels
        dark_ratio = np.mean(gray < 200)

        # Complexity
        row_trans = np.sum(np.abs(np.diff(gray, axis=1)) > 20)
        col_trans = np.sum(np.abs(np.diff(gray, axis=0)) > 20)
        complexity = row_trans + col_trans

        is_diagram = edge_density > 0.05 or complexity > 1000

        return ClassificationResult(
            page_type=PageType.DIAGRAM if is_diagram else PageType.DOCUMENT,
            confidence=0.6,
            features={'edge_density': edge_density, 'dark_ratio': dark_ratio, 'complexity': complexity},
            method='visual'
        )

class HybridClassifier:
    """
    Original Hybrid - Combines layout + visual
    """
    def __init__(self):
        self.layout = LayoutClassifier()
        self.visual = VisualLayoutClassifier()

    def classify(self, page: fitz.Page) -> ClassificationResult:
        l_res = self.layout.classify(page)
        v_res = self.visual.classify(page)

        # Weighted voting
        l_score = l_res.confidence if l_res.page_type==PageType.DOCUMENT else -l_res.confidence
        v_score = v_res.confidence if v_res.page_type==PageType.DIAGRAM else -v_res.confidence
        combined = 0.6*l_score + 0.4*v_score

        # REMOVED: Ambiguous handling - now binary only
        ptype = PageType.DOCUMENT if combined > 0 else PageType.DIAGRAM

        return ClassificationResult(
            page_type=ptype,
            confidence=abs(combined),
            features={'layout': l_res.features, 'visual': v_res.features},
            method='hybrid'
        )

# ============================================================================
# IMPROVED IMPLEMENTATION - BINARY ONLY (NO AMBIGUOUS)
# ============================================================================

class ImprovedPDFClassifier:
    """
    Improved classifier with adaptive calibration - BINARY CLASSIFICATION ONLY
    No ambiguous class - forces decision between Document and Diagram
    """
    def __init__(self, calibration_pages: int = 5):
        self.scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.calibrated = False
        self.calibration_pages = calibration_pages
        self.thresholds = {}
        self.history = deque(maxlen=100)

    def extract_features(self, page: fitz.Page) -> Dict:
        """Comprehensive multi-scale feature extraction"""
        features = {}

        # Text features
        words = page.get_text("words")
        blocks = page.get_text("blocks")
        features['word_count'] = len(words)
        features['block_count'] = len(blocks)

        # Multi-scale occupancy
        for grid in [10, 20, 40]:
            occ = self._occupancy(page, blocks, grid)
            features[f'occ_{grid}'] = occ

        # Cluster features
        if blocks:
            clusters = self._cluster(blocks)
            features['n_clusters'] = len(clusters)
            features['max_cluster'] = max(len(c) for c in clusters)
        else:
            features['n_clusters'] = 0
            features['max_cluster'] = 0

        # Visual features (multi-res)
        for res in [100, 200]:
            mat = fitz.Matrix(res/72, res/72)
            pix = page.get_pixmap(matrix=mat)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if pix.n>1 else img

            edges = cv2.Canny(gray, 50, 150)
            features[f'edges_{res}'] = np.sum(edges>0)/edges.size
            features[f'contrast_{res}'] = np.std(gray)

        return features

    def _occupancy(self, page, blocks, grid_size):
        if not blocks: return 0
        rect = page.rect
        cell_w, cell_h = rect.width/grid_size, rect.height/grid_size
        occupied = set()
        for b in blocks:
            x0,y0,x1,y1 = b[:4]
            for r in range(int(y0/cell_h), int(y1/cell_h)+1):
                for c in range(int(x0/cell_w), int(x1/cell_w)+1):
                    occupied.add((r,c))
        return len(occupied)/(grid_size**2)

    def _cluster(self, blocks):
        if len(blocks) < 2: return [[b] for b in blocks]
        centers = np.array([[(b[0]+b[2])/2, (b[1]+b[3])/2] for b in blocks])
        clustering = DBSCAN(eps=20, min_samples=1).fit(centers)
        clusters = {}
        for i, lbl in enumerate(clustering.labels_):
            clusters.setdefault(lbl, []).append(blocks[i])
        return list(clusters.values())

    def calibrate(self, pages: List[fitz.Page]):
        """Calibrate thresholds on sample pages"""
        print(f"Calibrating on {len(pages)} pages...")
        features = [self.extract_features(p) for p in pages]

        # Statistical thresholds
        self.thresholds['word_count'] = np.median([f['word_count'] for f in features])
        self.thresholds['occupancy'] = np.median([f['occ_20'] for f in features])
        self.thresholds['edges'] = np.percentile([f['edges_200'] for f in features], 75)

        # Fit anomaly detector (used for confidence, not for ambiguous class)
        X = np.array([[f[k] for k in sorted(f.keys())] for f in features])
        self.scaler.fit(X)
        self.anomaly_detector.fit(self.scaler.transform(X))

        self.calibrated = True
        print(f"Thresholds: {self.thresholds}")

    def classify(self, page: fitz.Page) -> ClassificationResult:
        """
        Binary classification - forces decision between DOCUMENT and DIAGRAM
        No ambiguous class, but includes confidence score
        """
        features = self.extract_features(page)

        # Anomaly detection (used for confidence adjustment, not classification)
        if self.calibrated:
            X = self.scaler.transform([[features[k] for k in sorted(features.keys())]])
            anomaly_score = self.anomaly_detector.decision_function(X)[0]
            is_anomaly = self.anomaly_detector.predict(X)[0] == -1
        else:
            anomaly_score = 0
            is_anomaly = False

        # Scoring
        doc_score, diag_score = 0, 0

        if features['word_count'] > self.thresholds.get('word_count', 50):
            doc_score += 1
        else:
            diag_score += 0.5

        if features['occ_20'] > self.thresholds.get('occupancy', 0.3):
            doc_score += 1
        else:
            diag_score += 0.5

        if features['edges_200'] > self.thresholds.get('edges', 0.05):
            diag_score += 1

        if features['n_clusters'] > 3:
            doc_score += 0.5

        # BINARY DECISION ONLY - No ambiguous option
        total = doc_score + diag_score
        if total > 0:
            doc_prob = doc_score / total
            # Confidence is how far from 0.5 (binary certainty)
            confidence = abs(doc_prob - 0.5) * 2  # Scale to 0-1
        else:
            doc_prob = 0.5
            confidence = 0.0

        # FORCE BINARY CHOICE - No threshold for ambiguous
        if doc_prob >= 0.5:
            ptype = PageType.DOCUMENT
        else:
            ptype = PageType.DIAGRAM

        # Adjust confidence if anomaly detected (lower confidence but same decision)
        if is_anomaly:
            confidence *= 0.8  # Reduce confidence but keep decision

        result = ClassificationResult(
            page_type=ptype,
            confidence=confidence,
            features={
                **features, 
                'doc_probability': doc_prob,
                'anomaly_score': anomaly_score,
                'is_anomaly': is_anomaly  # Kept for info but doesn't change class
            },
            method='improved_binary'
        )

        self.history.append(result)
        return result

    def get_confidence_stats(self) -> Dict:
        """Get statistics about classification confidence"""
        if not self.history:
            return {}
        
        confidences = [r.confidence for r in self.history]
        anomalies = sum(1 for r in self.history if r.features.get('is_anomaly', False))
        
        return {
            'mean_confidence': np.mean(confidences),
            'min_confidence': np.min(confidences),
            'low_confidence_count': sum(1 for c in confidences if c < 0.3),
            'anomaly_count': anomalies,
            'total_classified': len(self.history)
        }

# ============================================================================
# USAGE
# ============================================================================
"""
if __name__ == "__main__":
    # Example usage
    doc = fitz.open("example.pdf")

    # Original approach
    hybrid = HybridClassifier()

    # Improved approach - BINARY ONLY
    improved = ImprovedPDFClassifier(calibration_pages=5)
    improved.calibrate([doc[i] for i in range(min(5, len(doc)))])

    for i in range(len(doc)):
        page = doc[i]
        
        orig = hybrid.classify(page)
        imp = improved.classify(page)
        
        # Both return only DOCUMENT or DIAGRAM (no AMBIGUOUS)
        print(f"Page {i+1}: Original={orig.page_type.name}, Improved={imp.page_type.name}, "
              f"Confidence={imp.confidence:.2f}")
    
    # Get confidence statistics
    stats = improved.get_confidence_stats()
    print(f"\nStats: {stats}")

    doc.close()
"""