#!/usr/bin/env python3
# RetinaFace Face Marker - Uses state-of-the-art RetinaFace detector
# Implementation based on the RetinaFace paper and insightface implementation

import cv2
import numpy as np
import os
import argparse
import time
from PIL import Image

try:
    from retinaface import RetinaFace
    RETINAFACE_AVAILABLE = True
except ImportError:
    RETINAFACE_AVAILABLE = False

def ensure_retinaface():
    """
    Make sure RetinaFace is available, providing installation instructions if not.
    
    Returns:
        Boolean indicating whether RetinaFace is available
    """
    if not RETINAFACE_AVAILABLE:
        print("\n⚠️ RetinaFace is not installed. Install it with:")
        print("pip install retina-face")
        print("or")
        print("pip install -U git+https://github.com/serengil/retinaface.git\n")
        return False
    return True

def detect_faces(image_path, conf_threshold=0.9):
    """
    Detect faces using RetinaFace
    
    Args:
        image_path: Path to the input image
        conf_threshold: Confidence threshold for face detection
        
    Returns:
        Dictionary of detected faces with landmarks and bounding boxes
    """
    if not ensure_retinaface():
        return []
    
    try:
        # RetinaFace works directly with the image path
        faces = RetinaFace.detect_faces(image_path, threshold=conf_threshold, allow_upscaling=True)
        return faces
    except Exception as e:
        print(f"Error detecting faces: {e}")
        return []

def mark_faces(image_path, output_path=None, mark_color=(0, 255, 0), 
               box_thickness=2, label_faces=True, conf_threshold=0.9,
               draw_landmarks=True, return_image=False):
    """
    Detect faces in an image using RetinaFace and mark them with rectangles and landmarks.
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the output image (if None, will use 'marked_' + original name)
        mark_color: Color for the face rectangles (B,G,R)
        box_thickness: Thickness of the rectangle lines
        label_faces: Whether to label detected faces with numbers
        conf_threshold: Confidence threshold for face detection
        draw_landmarks: Whether to draw facial landmarks
        return_image: Whether to return the marked image
        
    Returns:
        If return_image is True, returns the marked image as a numpy array
        Otherwise returns path to the saved image
    """
    if not ensure_retinaface():
        return None
    
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Create a copy for drawing
    marked_img = img.copy()
    
    # Detect faces
    start_time = time.time()
    faces = detect_faces(image_path, conf_threshold)
    detection_time = time.time() - start_time
    
    if faces is None or len(faces) == 0:
        print(f"No faces detected in {image_path}")
        if output_path is None:
            base, ext = os.path.splitext(image_path)
            output_path = f"{base}_no_faces{ext}"
        cv2.imwrite(output_path, marked_img)
        return output_path if not return_image else marked_img
    
    print(f"Found {len(faces)} faces using RetinaFace in {detection_time:.3f} seconds")
    
    # Landmark colors for different facial features
    landmark_colors = {
        "left_eye": (255, 0, 0),    # Blue
        "right_eye": (255, 0, 0),   # Blue
        "nose": (0, 0, 255),        # Red
        "left_lip": (255, 0, 255),  # Magenta
        "right_lip": (255, 0, 255)  # Magenta
    }
    
    # Draw rectangles and landmarks for each face
    for i, (face_idx, face_data) in enumerate(faces.items()):
        # Get bounding box coordinates
        x1, y1, x2, y2 = (
            int(face_data['facial_area'][0]),
            int(face_data['facial_area'][1]),
            int(face_data['facial_area'][2]),
            int(face_data['facial_area'][3])
        )
        
        # Draw bounding box
        cv2.rectangle(marked_img, (x1, y1), (x2, y2), mark_color, box_thickness)
        
        # Draw confidence score label
        if label_faces:
            confidence = face_data.get('score', 0)
            label_text = f"Face {i+1} ({confidence:.2f})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(marked_img, label_text, (x1, y1-10), font, 0.5, mark_color, 2)
        
        # Draw facial landmarks
        if draw_landmarks and 'landmarks' in face_data:
            landmarks = face_data['landmarks']
            for landmark_name, coords in landmarks.items():
                x, y = int(coords[0]), int(coords[1])
                color = landmark_colors.get(landmark_name, (0, 255, 255))  # Default to yellow
                cv2.circle(marked_img, (x, y), 3, color, -1)
    
    # Save the output image if requested
    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_marked_retinaface{ext}"
    
    cv2.imwrite(output_path, marked_img)
    print(f"Saved marked image to {output_path}")
    
    if return_image:
        return marked_img
    return output_path

def process_directory(input_dir, output_dir=None, **kwargs):
    """
    Process all images in a directory
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save output images
        **kwargs: Additional arguments for mark_faces
    """
    if not os.path.isdir(input_dir):
        print(f"Error: {input_dir} is not a directory")
        return
    
    if output_dir is None:
        output_dir = os.path.join(input_dir, "marked")
    
    os.makedirs(output_dir, exist_ok=True)
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    count = 0
    
    for filename in os.listdir(input_dir):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"marked_{filename}")
            
            try:
                mark_faces(input_path, output_path, **kwargs)
                count += 1
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    print(f"Processed {count} images. Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detect and mark faces in images using RetinaFace')
    parser.add_argument('input', help='Path to the input image or directory of images')
    parser.add_argument('--output', '-o', help='Path to save the marked image or directory')
    parser.add_argument('--color', nargs=3, type=int, default=[0, 255, 0],
                        help='BGR color for marking faces (default: 0 255 0)')
    parser.add_argument('--thickness', '-t', type=int, default=2,
                        help='Thickness of box lines (default: 2)')
    parser.add_argument('--no-labels', action='store_true',
                        help='Disable face numbering and confidence scores')
    parser.add_argument('--no-landmarks', action='store_true',
                        help='Disable drawing facial landmarks')
    parser.add_argument('--confidence', '-c', type=float, default=0.9,
                        help='Confidence threshold for detection (default: 0.9)')
    
    args = parser.parse_args()
    
    if os.path.isdir(args.input):
        # Process directory
        process_directory(
            args.input,
            args.output,
            mark_color=tuple(args.color),
            box_thickness=args.thickness,
            label_faces=not args.no_labels,
            draw_landmarks=not args.no_landmarks,
            conf_threshold=args.confidence
        )
    else:
        # Process single image
        mark_faces(
            args.input,
            args.output,
            mark_color=tuple(args.color),
            box_thickness=args.thickness,
            label_faces=not args.no_labels,
            draw_landmarks=not args.no_landmarks,
            conf_threshold=args.confidence
        )