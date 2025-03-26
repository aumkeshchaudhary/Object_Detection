import torch
import numpy as np
import gradio as gr
import cv2
import time
import os
from pathlib import Path
from PIL import Image

# Create cache directory for models
os.makedirs("models", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load YOLOv5 Nano model
model_path = Path("models/yolov5n.pt")
if model_path.exists():
    print(f"Loading model from cache: {model_path}")
    model = torch.hub.load("ultralytics/yolov5", "custom", path=str(model_path), source="local").to(device)
else:
    print("Downloading YOLOv5n model and caching...")
    model = torch.hub.load("ultralytics/yolov5", "yolov5n", pretrained=True).to(device)
    torch.save(model.state_dict(), model_path)

# Optimize model for speed
model.conf = 0.25  # Lower confidence threshold for speed
model.iou = 0.45   # Better IoU threshold
model.classes = None  
model.max_det = 100  # Limit maximum detections

if device.type == "cuda":
    model.half()  # Use FP16 precision
else:
    torch.set_num_threads(os.cpu_count())

model.eval()

# Pre-generate colors for bounding boxes
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(model.names), 3), dtype=np.uint8)

def process_video(video_path):
    # Check if video_path is None or empty
    if video_path is None or video_path == "":
        return None
    
    # Handle the case when Gradio passes a tuple (file, None)
    if isinstance(video_path, tuple) and len(video_path) >= 1:
        video_path = video_path[0]
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return "Error: Could not open video file."

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Use mp4v codec which is more widely supported
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = "output_video.mp4"
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # For FPS calculation
    frame_count = 0
    start_time = time.time()
    
    # Skip frames for faster processing if needed
    frame_skip = 0
    if device.type != "cuda":  # Skip more frames on CPU
        frame_skip = 1
    
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_idx += 1
        if frame_skip > 0 and frame_idx % (frame_skip + 1) != 0:
            out.write(frame)  # Write original frame
            continue
        
        # Convert frame for YOLOv5
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Use smaller inference size for speed
        results = model(img, size=384)  # Reduced from 640 to 384
        
        detections = results.xyxy[0].cpu().numpy()
        
        # Draw bounding boxes
        for *xyxy, conf, cls in detections:
            x1, y1, x2, y2 = map(int, xyxy)
            class_id = int(cls)
            color = colors[class_id].tolist()
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)
            label = f"{model.names[class_id]} {conf:.2f}"
            
            # Black text
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 0, 0), 2, cv2.LINE_AA)
        
        # Update frame count for FPS calculation
        frame_count += 1
        
        # Calculate and display FPS every 10 frames
        if frame_count % 10 == 0:
            elapsed_time = time.time() - start_time
            fps_calc = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            # Add FPS counter with black text
            cv2.putText(frame, f"FPS: {fps_calc:.2f}", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        
        out.write(frame)
    
    cap.release()
    out.release()
    
    return output_path

def process_image(image):
    if image is None:
        return None
        
    img = np.array(image)
    
    # Process with smaller size for speed
    results = model(img, size=512)

    detections = results.pred[0].cpu().numpy()

    for *xyxy, conf, cls in detections:
        x1, y1, x2, y2 = map(int, xyxy)
        class_id = int(cls)
        color = colors[class_id].tolist()
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)
        label = f"{model.names[class_id]} {conf:.2f}"
        # Black text
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

    return Image.fromarray(img)


css = """
#title {
    text-align: center;
    color: #2C3E50;
    font-size: 2.5rem;
    margin: 1.5rem 0;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
}
.gradio-container {
    background-color: #F5F7FA;
}
.tab-item {
    background-color: white;
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    margin: 10px;
}
.button-row {
    display: flex;
    justify-content: space-around;
    margin: 1rem 0;
}
#video-process-btn, #submit-btn {
    background-color: #3498DB;
    border: none;
}
#clear-btn {
    background-color: #E74C3C;
    border: none;
}
.output-container {
    margin-top: 1.5rem;
    border: 2px dashed #3498DB;
    border-radius: 10px;
    padding: 10px;
}
.footer {
    text-align: center;
    margin-top: 2rem;
    font-size: 0.9rem;
    color: #7F8C8D;
}
"""

with gr.Blocks(css=css, title="Video & Image Object Detection by YOLOv5") as demo:
    gr.Markdown("""# YOLOv5 Object Detection""", elem_id="title")
    
    with gr.Tabs():
        with gr.TabItem("Video Detection", elem_classes="tab-item"):
            with gr.Row():
                video_input = gr.Video(
                    label="Upload Video", 
                    interactive=True, 
                    elem_id="video-input"
                )
            
            with gr.Row(elem_classes="button-row"):
                process_button = gr.Button(
                    "Process Video", 
                    variant="primary", 
                    elem_id="video-process-btn"
                )
            
            with gr.Row(elem_classes="output-container"):
                video_output = gr.Video(
                    label="Processed Video", 
                    elem_id="video-output"
                )
            
            process_button.click(
                fn=process_video, 
                inputs=video_input, 
                outputs=video_output
            )
            
        with gr.TabItem("Image Detection", elem_classes="tab-item"):
            with gr.Row():
                image_input = gr.Image(
                    type="pil", 
                    label="Upload Image", 
                    interactive=True
                )
            
            with gr.Row(elem_classes="button-row"):
                clear_button = gr.Button(
                    "Clear", 
                    variant="secondary", 
                    elem_id="clear-btn"
                )
                submit_button = gr.Button(
                    "Detect Objects", 
                    variant="primary", 
                    elem_id="submit-btn"
                )
            
            with gr.Row(elem_classes="output-container"):
                image_output = gr.Image(
                    label="Detected Objects", 
                    elem_id="image-output"
                )
            
            clear_button.click(
                fn=lambda: None, 
                inputs=None, 
                outputs=image_output
            )
            
            submit_button.click(
                fn=process_image, 
                inputs=image_input, 
                outputs=image_output
            )
    
    gr.Markdown("""
    ### Powered by YOLOv5.
    This application enables seamless object detection using the YOLOv5 model, allowing users to analyze images and videos with high accuracy and efficiency.
    """, elem_classes="footer")

if __name__ == "__main__":
    demo.launch()