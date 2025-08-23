import os

from ultralytics import YOLO
from ultralytics .engine.results import Results

def fine_tune_yolo_model(data_yaml_path: str, epochs: int, monde_name: str = 'yolov8n.pt') -> None:
    """
    Fine-tunes a pre-trained YOLOv8 model on a custom dataset.

    Args:
        data_yaml_path (str): The path to the dataset's .yaml configuration file.
        epochs (int): The number of epochs to train for.
        model_name (str, optional): The name of the pre-trained YOLO model to start with. 
                                    Defaults to 'yolov8n.pt' (the smallest and fastest version).
    """

    # Load a pretrained YOLOv8 model
    model = YOLO(monde_name)

    # Train the model on the dataset created by user reports
    results = model.train(data=data_yaml_path, epochs=epochs, imgsz=640)

    if not isinstance(results, Results):
        raise TypeError("The training process did not return a valid Results object.")

    print("Fine-tunning complete.")
    print("Model performance metrics:")
    print(results.metrics)

    print(f"Best model saved to: {results.save_dir}/weights/best.pt")

if __name__ == '__main__':

    # --- Configuration ---
    DATASET_YAML_PATH = '../data/tools_dataset.yaml'
    NUM_EPOCHS = 10

    # --- Run the Trainning ---
    if os.path.exists(DATASET_YAML_PATH):
        fine_tune_yolo_model(DATASET_YAML_PATH, NUM_EPOCHS)
    else:
        print(f"Error: Dataset configuration file not found at '{DATASET_YAML_PATH}'")