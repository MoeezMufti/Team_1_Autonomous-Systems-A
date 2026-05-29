import cv2
import numpy as np
import glob
import os
import argparse
import sys

def get_steering_class(label):
    if label <= -0.6:
        return "Hard Left"
    elif label <= -0.15:
        return "Slight Left"
    elif label < 0.15:
        return "Straight"
    elif label < 0.6:
        return "Slight Right"
    else:
        return "Hard Right"

def process_and_label_frame(img, sensitivity, offset_weight):
    h, w = img.shape[:2]

    # 1. Crank Saturation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255)
    hsv_cranked = hsv.astype(np.uint8)

    # 2. Color Masking
    lower_green = np.array([40, 100, 100]) 
    upper_green = np.array([80, 255, 255])
    color_mask = cv2.inRange(hsv_cranked, lower_green, upper_green)

    # 3. ROI Constraint
    roi_mask = np.zeros_like(color_mask)
    y_min = int(h * 0.5)      
    y_max = h                 
    x_min = int(w * 0.15)     
    x_max = int(w * 0.85)     
    roi_mask[y_min:y_max, x_min:x_max] = 255

    constrained_mask = cv2.bitwise_and(color_mask, roi_mask)

    # 4. Blob Filtering
    contours, _ = cv2.findContours(constrained_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_contour = None
    min_distance = float('inf')
    target_point = (w // 2, h) 

    if contours:
        for cnt in contours:
            if cv2.contourArea(cnt) < 50: 
                continue
                
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                dist = np.sqrt((cx - target_point[0])**2 + (cy - target_point[1])**2)
                if dist < min_distance:
                    min_distance = dist
                    best_contour = cnt

    # 5. Combined Error Calculation
    label = 0.0
    final_mask_vis = np.zeros_like(img) 

    if best_contour is not None:
        cv2.drawContours(final_mask_vis, [best_contour], -1, (0, 255, 0), cv2.FILLED)
        
        topmost = tuple(best_contour[best_contour[:, :, 1].argmin()][0])
        bottommost = tuple(best_contour[best_contour[:, :, 1].argmax()][0])
        center_x = w // 2
        
        max_possible_shift = (x_max - x_min)

        # A. Heading Error (The Gradient Lookahead)
        heading_error = (topmost[0] - bottommost[0]) / max_possible_shift
        
        # B. Cross-Track Offset Error (Distance from bottom of line to center screen)
        offset_error = (bottommost[0] - center_x) / max_possible_shift

        # C. Combine and apply Sensitivity
        raw_label = heading_error + (offset_error * offset_weight)
        label = np.clip(raw_label * sensitivity, -1.0, 1.0)
        
        # --- VISUALIZATIONS ---
        # Magenta line: Heading Error (Lookahead vector)
        cv2.line(final_mask_vis, bottommost, topmost, (255, 0, 255), 3)
        cv2.circle(final_mask_vis, topmost, 8, (0, 0, 255), -1) 
        
        # Yellow line: Cross-Track Error (Offset to center)
        cv2.line(final_mask_vis, (center_x, bottommost[1]), bottommost, (0, 255, 255), 3)
        cv2.circle(final_mask_vis, (center_x, bottommost[1]), 8, (255, 255, 0), -1)
        
        # Blue dot: The car's anchor point on the line
        cv2.circle(final_mask_vis, bottommost, 8, (255, 0, 0), -1) 

    return label, final_mask_vis


def main():
    parser = argparse.ArgumentParser(description="Generate validation MP4 with Heading and Offset calculations.")
    parser.add_argument('-i', '--input_dir', type=str, required=True, help="Path to the folder containing .png files.")
    parser.add_argument('-o', '--output', type=str, default='validation_output.mp4', help="Output .mp4 filename.")
    parser.add_argument('--fps', type=int, default=30, help="Framerate for the output video.")
    parser.add_argument('--limit', type=int, default=0, help="Limit frames to process (0 = all).")
    
    # New Hyperparameters
    parser.add_argument('--sensitivity', type=float, default=1.5, help="Multiplier for the final steering label.")
    parser.add_argument('--offset_weight', type=float, default=0.7, help="Weight applied to the cross-track error.")
    
    args = parser.parse_args()

    image_paths = sorted(glob.glob(os.path.join(args.input_dir, '*.png')))
    
    if not image_paths:
        print(f"Error: No .png images found in {args.input_dir}")
        sys.exit(1)

    if args.limit > 0:
        image_paths = image_paths[:args.limit]

    first_frame = cv2.imread(image_paths[0])
    h, w = first_frame.shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, args.fps, (w, h))

    total_frames = len(image_paths)
    print(f"Exporting to {args.output} | Sensitivity: {args.sensitivity} | Offset Weight: {args.offset_weight}")

    for i, path in enumerate(image_paths):
        img = cv2.imread(path)
        if img is None: continue

        label, final_mask_vis = process_and_label_frame(img, args.sensitivity, args.offset_weight)
        steering_class = get_steering_class(label)

        text_val = f"Value: {label:+.3f}"
        text_class = f"Command: {steering_class}"
        
        if "Hard" in steering_class:
            color = (0, 0, 255) 
        elif "Slight" in steering_class:
            color = (255, 255, 0) 
        else:
            color = (0, 255, 0) 
        
        cv2.putText(final_mask_vis, text_val, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        cv2.putText(final_mask_vis, text_class, (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        out.write(final_mask_vis)
        
        sys.stdout.write(f"\rProcessing frame {i+1}/{total_frames} ({(i+1)/total_frames*100:.1f}%)")
        sys.stdout.flush()

    out.release()
    print(f"\nDone! Clean validation video saved to: {args.output}")

if __name__ == "__main__":
    main()