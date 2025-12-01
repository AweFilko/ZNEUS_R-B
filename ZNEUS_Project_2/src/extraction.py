import cv2
import numpy as np
from skimage.feature import local_binary_pattern, hog


class FeatureExtractor:
    def __init__(self, use_color_hist=True, use_hog=True, use_lbp=True, use_orb=True):
        self.use_color_hist = use_color_hist
        self.use_hog = use_hog
        self.use_lbp = use_lbp
        self.use_orb = use_orb

        self.orb = cv2.ORB_create(nfeatures=500)

    def extract(self, img):
        """Extracts features and returns a single 1D vector."""
        features = []

        # Ensure grayscale & RGB versions
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # ----- 1. COLOR HISTOGRAM -----
        if self.use_color_hist:
            hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8],
                                [0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            features.append(hist)

        # ----- 2. HOG FEATURES -----
        if self.use_hog:
            hog_feats = hog(gray,
                            orientations=9,
                            pixels_per_cell=(16, 16),
                            cells_per_block=(2, 2),
                            block_norm='L2-Hys')
            features.append(hog_feats)

        # ----- 3. LBP TEXTURE --------
        if self.use_lbp:
            lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
            (hist, _) = np.histogram(lbp.ravel(),
                                     bins=np.arange(0, 59),
                                     range=(0, 58))
            hist = hist.astype("float")
            hist /= (hist.sum() + 1e-7)
            features.append(hist)

        # ----- 4. ORB DESCRIPTORS -----
        if self.use_orb:
            keypoints, descriptors = self.orb.detectAndCompute(gray, None)

            if descriptors is None:
                descriptors = np.zeros((1, 32))

            # Strategy: average descriptor vector = robust + fixed size
            orb_feat = descriptors.mean(axis=0)
            features.append(orb_feat)

        # Concatenate ALL features into one vector
        return np.concatenate(features)
