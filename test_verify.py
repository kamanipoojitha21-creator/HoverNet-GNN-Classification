import os
import sys
import torch
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Add the current directory to sys.path to import app.py
sys.path.append('c:/Users/HP/Desktop/breast')

try:
    from app import analyze_image, app
    import cv2
    
    print("Testing analyze_image with a sample image...")
    
    # Use one of the uploaded files if available, otherwise just check if a dummy run works
    upload_dir = 'c:/Users/HP/Desktop/breast/static/uploads'
    files = [f for f in os.listdir(upload_dir) if f.endswith(('.jpg', '.png'))]
    
    if not files:
        print("No image found in uploads. Creating a dummy image for testing.")
        dummy_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        test_path = 'c:/Users/HP/Desktop/breast/static/uploads/test_dummy.jpg'
        cv2.imwrite(test_path, dummy_img)
    else:
        test_path = os.path.join(upload_dir, files[0])
        print(f"Using existing file: {test_path}")

    results = analyze_image(test_path)
    
    print("\n--- RESULTS ---")
    print(f"Prediction Class: {results['prediction']['class']}")
    print(f"Nuclei Count: {results['metrics']['num_nuclei']}")
    print(f"Cell Density: {results['metrics']['cell_density']}")
    print(f"Tumor Grade: {results['metrics']['tumor_grade']}")
    print(f"Size Variation: {results['metrics']['size_variation']:.2f}%")
    
    print("\nVerification successful!")

except Exception as e:
    print(f"\nVerification failed with error: {e}")
    import traceback
    traceback.print_exc()
