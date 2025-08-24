import os

import cv2
from ultralytics import YOLO

def run_inference(image_path: str, model_name: str = 'yolov8n.pt') -> None:

    # Load the pre-trained YOLOv8 model
    print(f"Loading model: {model_name}...")
    model = YOLO(model_name)

    # Read the input image using OpenCV
    print(f"Reading image from: {image_path}...")
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return

    # Run inference on the image
    print("Running inference...")
    results = model(image)
    result = results[0]

    # Draw the bounding boxes and labels on the image
    annotated_image = result.plot()

    print(f"Found {len(result.boxes)} objects.")

    # Display the annotated image in a window
    cv2.imshow("YOLOv8 Inference", annotated_image)

    # Wait for a key press and then close the window
    print("Press any key to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # --- Configuration --- 
    # Make sure you have an image named 'test_image.jpg' in the same directory.
    TEST_IMAGE_PATH = 'test_image.jpg'

    # --- Run the Inference Script ---
    if os.path.exists(TEST_IMAGE_PATH):
        run_inference(TEST_IMAGE_PATH)
    else:
        print(f"Error: Test image not found at: '{TEST_IMAGE_PATH}'")
        print("Please add an image to your project directory to test the script.")