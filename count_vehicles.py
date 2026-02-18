# This line brings in the main YOLO tool.
from ultralytics import YOLO

# --- STEP 1: LOAD YOUR TRAINED MODEL ---
# We load 'best.pt', which is your smart model trained in Google Colab.
# Make sure 'best.pt' is in your PyCharm project folder!
print("Loading your trained model...")
model = YOLO('best.pt')
print("Model loaded successfully!")

# --- STEP 2: DEFINE THE IMAGE YOU WANT TO ANALYZE ---
# This is the exact file path to the image you want to check.
image_path = "D:\\THESIS\\Images_1sec_interval\\frame_25.jpg"

# --- STEP 3: RUN THE PREDICTION ---
# The model will look at the image and find all the objects it can.
# save=True will also save a new image with boxes drawn on it.
print(f"Analyzing the image: {image_path}")
results = model.predict(source=image_path, save=True)
print("Analysis complete!")

# --- STEP 4: COUNT THE SPECIFIC OBJECTS ---
# These are the objects we want to count.
# Note: The dataset uses 'motor' for motorcycle.
objects_to_count = ['car', 'motor', 'truck', 'bus']
object_counts = {obj: 0 for obj in objects_to_count}

# The results are a list, but we only analyzed one image, so we take the first one.
result = results[0]

# Get all the names the model knows (e.g., 'person', 'car', 'bus', etc.)
class_names = result.names

# Go through every single object the model found in the image.
for box in result.boxes:
    # Get the ID of the object class (e.g., car might be ID 2)
    class_id = int(box.cls[0])

    # Get the name of that class from the ID.
    class_name = class_names[class_id]

    # Check if this object is one of the ones we want to count.
    if class_name in objects_to_count:
        object_counts[class_name] += 1  # Add 1 to the count for that object.

# --- STEP 5: SHOW THE FINAL COUNTS ---
print("\n--- DETECTION RESULTS ---")
print(f"Found {object_counts['car']} car(s).")
print(f"Found {object_counts['motor']} motorcycle(s).")
print(f"Found {object_counts['truck']} truck(s).")
print(f"Found {object_counts['bus']} bus(es).")
print("\nAn image with the detections has been saved in the 'runs/detect/predict' folder.")
