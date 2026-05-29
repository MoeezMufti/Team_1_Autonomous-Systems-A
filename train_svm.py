import cv2
import numpy as np
import glob
import os
import joblib
import time
import sys
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- THE UNIFIED PIPELINE ---
def extract_ground_truth_and_features(img, is_training=True):
    h, w = img.shape[:2]

    # 1. Crank Saturation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255)
    hsv_cranked = hsv.astype(np.uint8)

    # 2. Color Masking
    lower_green = np.array([40, 100, 100]) 
    upper_green = np.array([80, 255, 255])
    color_mask = cv2.inRange(hsv_cranked, lower_green, upper_green)

    # 3. ROI Constraint (Bottom 50%, Center 70%)
    roi_mask = np.zeros_like(color_mask)
    y_min, y_max = int(h * 0.5), h                 
    x_min, x_max = int(w * 0.15), int(w * 0.85)     
    roi_mask[y_min:y_max, x_min:x_max] = 255

    constrained_mask = cv2.bitwise_and(color_mask, roi_mask)

    # 4. Blob Filtering
    contours, _ = cv2.findContours(constrained_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_contour = None
    min_distance = float('inf')
    target_point = (w // 2, h) 

    if contours:
        for cnt in contours:
            if cv2.contourArea(cnt) < 50: continue
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                dist = np.sqrt((cx - target_point[0])**2 + (cy - target_point[1])**2)
                if dist < min_distance:
                    min_distance = dist
                    best_contour = cnt

    if best_contour is None:
        return None, None

    # --- EXTRACTING THE BINARY MASK FOR THE SVM ---
    pure_mask = np.zeros_like(constrained_mask)
    cv2.drawContours(pure_mask, [best_contour], -1, 255, cv2.FILLED)
    
    roi_cropped_mask = pure_mask[y_min:y_max, x_min:x_max]
    svm_feature_img = cv2.resize(roi_cropped_mask, (64, 64))
    
    features = svm_feature_img.flatten()

    # 5. Label Calculation
    label = None
    if is_training:
        topmost = tuple(best_contour[best_contour[:, :, 1].argmin()][0])
        bottommost = tuple(best_contour[best_contour[:, :, 1].argmax()][0])
        
        sensitivity = 3.0
        offset_weight = 2.0
        max_shift = (x_max - x_min)

        heading_error = (topmost[0] - bottommost[0]) / max_shift
        offset_error = (bottommost[0] - (w // 2)) / max_shift
        
        label = np.clip((heading_error + (offset_error * offset_weight)) * sensitivity, -1.0, 1.0)

    return label, features

def main():
    input_dir = "C:\\Users\\hamas\\Documents\\Official Documents\\University\\SVM\\Team_1_Autonomous-Systems-A\\extracted_frames"
    image_paths = sorted(glob.glob(os.path.join(input_dir, '*.png')))
    
    if not image_paths:
        print(f"Error: No .png images found in {input_dir}")
        sys.exit(1)

    X, y = [], []
    valid_frames = 0
    total_files = len(image_paths)

    print(f"Starting extraction for {total_files} images (plus augmentations)...")
    
    start_time = time.time()

    for i, path in enumerate(image_paths):
        img = cv2.imread(path)
        if img is None: continue

        # Original Frame
        label, features = extract_ground_truth_and_features(img, is_training=True)
        if label is not None:
            X.append(features)
            y.append(label)
            valid_frames += 1

        # Mirrored Frame Augmentation
        img_flipped = cv2.flip(img, 1) 
        label_flipped, features_flipped = extract_ground_truth_and_features(img_flipped, is_training=True)
        if label_flipped is not None:
            X.append(features_flipped)
            y.append(label_flipped)
            valid_frames += 1

        # --- TERMINAL PROGRESS TRACKER ---
        # Update terminal every 10 frames or on the last frame
        if (i + 1) % 10 == 0 or (i + 1) == total_files:
            elapsed_time = time.time() - start_time
            frames_processed = i + 1
            
            # Math for ETA
            time_per_frame = elapsed_time / frames_processed
            remaining_frames = total_files - frames_processed
            eta_seconds = remaining_frames * time_per_frame
            
            eta_mins, eta_secs = divmod(int(eta_seconds), 60)
            
            # Visual Progress Bar
            percent = frames_processed / total_files
            bar_length = 30
            filled_length = int(bar_length * percent)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            
            # Print to stdout with carriage return (\r) to overwrite the line
            sys.stdout.write(f"\r[{bar}] {percent*100:.1f}% | {frames_processed}/{total_files} | ETA: {eta_mins:02d}:{eta_secs:02d} | Valid Samples: {valid_frames}")
            sys.stdout.flush()

    print(f"\n\nExtraction complete in {time.time() - start_time:.1f} seconds.")
    print(f"Stratifying and training on {valid_frames} valid samples...")
    
    train_start = time.time()
    
    X = np.array(X)
    y = np.array(y)

    bins = np.array([-1.0, -0.6, -0.15, 0.15, 0.6, 1.0])
    y_binned = np.digitize(y, bins)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y_binned)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    model.fit(X_train_scaled, y_train)
    
    print(f"Training finished in {time.time() - train_start:.1f} seconds.")
    print(f"Final R^2 Score on Validation Set: {model.score(X_test_scaled, y_test):.4f}")
    
    joblib.dump(model, 'svm_model.pkl')
    joblib.dump(scaler, 'svm_scaler.pkl')
    print("\nExported: svm_model.pkl and svm_scaler.pkl")

if __name__ == '__main__':
    main()