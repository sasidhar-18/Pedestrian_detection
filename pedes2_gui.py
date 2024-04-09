import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import ImageTk, Image
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import cv2

# Load the pre-trained model
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Function to perform pedestrian detection
def detect_pedestrians(image_path):
    # Read the image
    img = Image.open(image_path)

    # Run inference
    with torch.no_grad():
        # Preprocess the image
        transform = transforms.Compose([transforms.ToTensor()])
        img = transform(img)
        img = img.unsqueeze(0)  # Add batch dimension

        # Run inference
        predictions = model(img)

    # Process predictions and draw bounding boxes
    img = cv2.imread(image_path)
    for box, label in zip(predictions[0]['boxes'], predictions[0]['labels']):
        box = [int(coord) for coord in box]
        # Filter out non-person objects (label for person detection is typically 1)
        if label == 1:
            img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), thickness=2)

    # Display the image with bounding boxes
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img.thumbnail((400, 400))  # Resize image for display
    img_tk = ImageTk.PhotoImage(img)
    label_image.configure(image=img_tk)
    label_image.image = img_tk

# Function to open a file dialog and get the image path
def open_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        detect_pedestrians(file_path)
    else:
        messagebox.showerror("Error", "No image selected.")

# Create the main window
root = tk.Tk()
root.title("Pedestrian Detection")

# Create a button to open the image
open_button = tk.Button(root, text="Open Image", command=open_image)
open_button.pack(padx=10, pady=10)

# Create a label to display the image
label_image = tk.Label(root)
label_image.pack()

# Run the GUI main loop
root.mainloop()
