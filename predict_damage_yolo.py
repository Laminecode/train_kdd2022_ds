# YOLOv8 Road Damage Prediction Script
# Usage: python predict_damage.py --model_path yolov8_road_damage_best.pt --image_path path/to/image.jpg


from ultralytics import YOLO
import cv2

# Set these variables directly
model_path = "outputs/yolo/best-yolo.pt"  # Change to your model path
image_path = "dataset/raw/test/images/China_Drone_000738.jpg"  # Change to your image path
conf = 0.25  # Confidence threshold

# Class names (must match training)
CLASS_NAMES = [
    'Longitudinal_crack',   # D00
    'Transverse_crack',     # D10
    'Alligator_crack',      # D20
    'Pothole',              # D40
    'Other_damage'          # Other
]


def main():
    model = YOLO(model_path)

    # Read and resize image to (640, 640) as in processed2
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return
    img_resized = cv2.resize(img, (640, 640), interpolation=cv2.INTER_AREA)

    # Run prediction on resized image
    results = model.predict(img_resized, conf=conf, imgsz=640, verbose=False)

    # Print results
    for i, box in enumerate(results[0].boxes):
        cls_id = int(box.cls[0])
        conf_score = float(box.conf[0])
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        class_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f'Class_{cls_id}'
        print(f"Detection {i+1}: {class_name} at {xyxy} (conf={conf_score:.2f})")

    # Visualize
    result_img = results[0].plot()
    # Save the result image
    cv2.imwrite("prediction_result.jpg", result_img)
    print("Result image saved as prediction_result.jpg")

    # Optionally, display with matplotlib (works in most environments)
    import matplotlib.pyplot as plt
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.title('YOLOv8 Road Damage Detection')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
