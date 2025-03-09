# Object Detection with YOLOv5
[![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-yellow)](https://huggingface.co/spaces/Aumkeshchy2003/Object_Detection)
## Overview
This repository contains an object detection model utilizing YOLOv5. The model is implemented in a Jupyter Notebook and designed to detect objects in videos and images captured through a webcam or uploaded manually.

## Features
- Uses **YOLOv5** for object detection.
- Applies **non-maximum suppression (NMS)** for better detection results.
- Displays bounding boxes with labels and confidence scores.

## Installation
To set up the environment, run the following commands:

```bash
pip install -U torch torchvision cython
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

# Clone YOLOv5 repository and install dependencies
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
```

## Usage
1. **Import necessary libraries:**
   ```python
   import torch
   from torchvision.models.detection import fasterrcnn_resnet50_fpn
   from yolov5.models.yolo import Model
   ```

2. **Load the YOLOv5 model:**
   ```python
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)
   model.eval()
   ```

3. **Capture an image using the webcam in Google Colab:**
   ```python
   from IPython.display import display, Javascript
   from google.colab.output import eval_js
   from base64 import b64decode
   ```
   (The script allows users to take a live photo and process it.)

4. **Preprocess the image and make predictions:**
   ```python
   from PIL import Image
   from torchvision.transforms import functional as F

   def preprocess_image(image_path):
       image = Image.open(image_path)
       image_tensor = F.to_tensor(image)
       return image_tensor.unsqueeze(0).to(device)

   image_tensor = preprocess_image('photo.jpg')
   outputs = model(image_tensor)[0]
   ```

5. **Draw bounding boxes on detected objects:**
   ```python
   import cv2
   from google.colab.patches import cv2_imshow

   def draw_boxes(image_path, outputs, threshold=0.3):
       image = cv2.imread(image_path)
       image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
       for box in outputs:
           score, label, x1, y1, x2, y2 = box[4].item(), int(box[5].item()), box[0].item(), box[1].item(), box[2].item(), box[3].item()
           if score > threshold:
               cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
               text = f"{model.names[label]}: {score:.2f}"
               cv2.putText(image, text, (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
       return image

   result_image = draw_boxes('photo.jpg', outputs)
   cv2_imshow(result_image)
   ```

## Results
- The model successfully detects objects and labels them with confidence scores.
- Bounding boxes are drawn around detected objects.

## Acknowledgments
- **YOLOv5** by Ultralytics: [GitHub Repo](https://github.com/ultralytics/yolov5)
- **Google Colab** for easy cloud execution.

## Contributing
Feel free to fork the repository and improve the model! Open issues for suggestions or bugs.

## License
This project is released under the MIT License.

