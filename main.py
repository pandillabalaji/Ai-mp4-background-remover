import cv2
import os
import torch
from ultralytics import YOLO

# Paths for input and output video directories
input_folder = "inputvideo"
output_folder = "outputvideo"
os.makedirs(output_folder, exist_ok=True)

# Load YOLOv8 segmentation model (use smaller model like yolov8n-seg for faster performance)
model = YOLO('yolov8x-seg.pt')  # Change to yolov8n-seg.pt for better speed at lower accuracy
device = 0 if torch.cuda.is_available() else 'cpu'  # Use GPU if available

# Get the first video file in the input folder
video_files = [f for f in os.listdir(input_folder) if f.endswith((".mp4", ".mov", ".avi"))]
if not video_files:
    print("[ERROR] No video found in inputvideo folder.")
    exit()

video_path = os.path.join(input_folder, video_files[0])
print(f"[INFO] Processing video: {video_path}")

# Load the video using OpenCV
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize VideoWriter to save output video (Green screen background)
output_path = os.path.join(output_folder, "output.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8 Segmentation (GPU enabled)
    results = model.predict(source=frame, device=device, classes=[0], conf=0.4, verbose=False)
    masks = results[0].masks

    if masks is not None:
        mask = masks.data[0].cpu().numpy()
        mask = (mask * 255).astype("uint8")
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))  # Resize mask to match frame size

        # Convert mask to 3-channel format for bitwise operations
        mask_3ch = cv2.merge([mask, mask, mask])
        inverse_mask = cv2.bitwise_not(mask)
        inverse_mask_3ch = cv2.merge([inverse_mask, inverse_mask, inverse_mask])

        # Create green screen background
        green_bg = cv2.merge([frame[:, :, 0]*0, frame[:, :, 1]*0 + 255, frame[:, :, 2]*0])

        # Apply the mask to remove the background and combine with the green screen
        masked_person = cv2.bitwise_and(frame, mask_3ch)
        background = cv2.bitwise_and(green_bg, inverse_mask_3ch)
        final_frame = cv2.add(masked_person, background)
    else:
        final_frame = frame

    # Write the processed frame directly to the output video
    out.write(final_frame)

# Release resources
cap.release()
out.release()

print(f"[INFO] Video processing complete. Output saved to: {output_path}")
