# 1. What is image segmentation, and why is it important?
# Image segmentation involves dividing an image into segments or regions to simplify its representation. It is crucial for precise analysis, helping in applications like object detection, medical imaging, and autonomous driving.

# 2. Explain the difference between image classification, object detection, and image segmentation.
# Image classification categorizes the entire image. Object detection detects and locates objects in an image using bounding boxes. Image segmentation classifies each pixel, creating masks for object regions.

# 3. What is Mask R-CNN, and how is it different from traditional object detection models?
# Mask R-CNN extends Faster R-CNN by adding a segmentation mask branch to generate pixel-level masks for each detected object, making it capable of both object detection and instance segmentation.

# 4. What role does the "RoIAlign" layer play in Mask R-CNN?
# RoIAlign helps accurately align region proposals to a fixed-size grid to preserve spatial information, improving the performance of mask predictions in Mask R-CNN.

# 5. What are semantic, instance, and panoptic segmentation?
# Semantic segmentation labels every pixel with a class. Instance segmentation additionally differentiates between object instances. Panoptic segmentation combines both by labeling each pixel and distinguishing between object instances.

# 6. Describe the role of bounding boxes and masks in image segmentation models.
# Bounding boxes define the object's location, while masks provide pixel-level boundaries for the objects, which is critical for instance segmentation.

# 7. What is the purpose of data annotation in image segmentation?
# Data annotation provides labeled training data, marking the objects and their regions in an image, enabling models to learn and make predictions.

# 8. How does Detectron2 simplify model training for object detection and segmentation tasks?
# Detectron2 simplifies model training with modular components, pre-trained models, and easy integration for various tasks like object detection and segmentation.

# 9. Why is transfer learning valuable in training segmentation models?
# Transfer learning allows models to leverage pre-trained weights, speeding up training and improving performance, especially when limited data is available.

# 10. How does Mask R-CNN improve upon the Faster R-CNN model architecture?
# Mask R-CNN adds a segmentation mask branch to Faster R-CNN’s object detection framework, allowing for pixel-level object instance segmentation.

# 11. What is meant by "from bounding box to polygon masks" in image segmentation?
# It refers to the process of converting the rectangular bounding box around an object into a more precise polygon-shaped mask for better object segmentation.

# 12. How does data augmentation benefit image segmentation model training?
# Data augmentation creates varied versions of images, enhancing the model's ability to generalize and improving its robustness to different image conditions.

# 13. Describe the architecture of Mask R-CNN, focusing on the backbone, region proposal network (RPN), and segmentation mask head.
# Mask R-CNN uses a backbone (e.g., ResNet) for feature extraction, RPN for generating region proposals, and a mask head for generating segmentation masks for each detected object.

# 14. Explain the process of registering a custom dataset in Detectron2 for model training.
# To register a custom dataset in Detectron2, you need to define the dataset format, load annotations, and use the `DatasetCatalog` and `MetadataCatalog` APIs to register and prepare the dataset.

# 15. What challenges arise in scene understanding for image segmentation, and how can Mask R-CNN address them?
# Challenges include occlusion, overlapping objects, and complex backgrounds. Mask R-CNN handles them through its precise mask generation and RoIAlign layer for accurate spatial feature alignment.

# 16. How is the "IoU (Intersection over Union)" metric used in evaluating segmentation models?
# IoU measures the overlap between the predicted mask and the ground truth mask, providing a metric to evaluate the accuracy of object segmentation.

# 17. Discuss the use of transfer learning in Mask R-CNN for improving segmentation on custom datasets.
# Transfer learning helps Mask R-CNN fine-tune pre-trained weights, improving performance on small or specialized datasets by starting from a model already trained on large datasets.

# 18. What is the purpose of evaluation curves, such as precision-recall curves, in segmentation model assessment?
# Precision-recall curves help assess the performance of segmentation models, especially for imbalanced datasets, by showing the trade-off between precision and recall.

# 19. How do Mask R-CNN models handle occlusions or overlapping objects in segmentation?
# Mask R-CNN handles occlusions by using instance segmentation, generating separate masks for overlapping objects, and improving detection with the RoIAlign layer.

# 20. Explain the impact of batch size and learning rate on Mask R-CNN model training.
# Larger batch sizes lead to more stable gradients, while smaller batch sizes provide more updates per epoch. The learning rate affects convergence speed and model performance.

# 21. Describe the challenges of training segmentation models on custom datasets, particularly in the context of Detectron2.
# Challenges include data annotation, dataset imbalance, and generalization to unseen scenarios. Detectron2’s modularity and pre-trained models help address some of these challenges.

# 22. How does Mask R-CNN's segmentation head output differ from a traditional object detector’s output?
# Mask R-CNN's segmentation head outputs a pixel-level mask for each detected object, whereas traditional detectors output bounding boxes and class labels without pixel-wise segmentation.


import cv2
import numpy as np
import torch
import torchvision
from torchvision import transforms
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 1. Basic color-based segmentation to separate the blue color in an image
image = cv2.imread('image.jpg')
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the blue color range
lower_blue = np.array([100, 150, 0])
upper_blue = np.array([140, 255, 255])

# Create mask for blue color
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# Segment blue color
result = cv2.bitwise_and(image, image, mask=mask)

# Display the result
cv2.imshow('Blue Color Segmentation', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 2. Edge detection with Canny to highlight object edges in an image
image_gray = cv2.imread('image.jpg', 0)  # Load in grayscale
edges = cv2.Canny(image_gray, 100, 200)

# Display the edges
cv2.imshow('Canny Edge Detection', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 3. Load a pretrained Mask R-CNN model from PyTorch and use it for object detection and segmentation on an image
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Load image and preprocess
image_pil = Image.open('image.jpg')
transform = transforms.Compose([transforms.ToTensor()])
image_tensor = transform(image_pil).unsqueeze(0)

# Get predictions
with torch.no_grad():
    prediction = model(image_tensor)

# 4. Generate bounding boxes for each object detected by Mask R-CNN in an image
boxes = prediction[0]['boxes']
labels = prediction[0]['labels']
scores = prediction[0]['scores']

# Display bounding boxes on the image
image_bboxes = np.array(image)
for box in boxes:
    x1, y1, x2, y2 = box
    cv2.rectangle(image_bboxes, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

cv2.imshow('Image with Bounding Boxes', image_bboxes)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 5. Convert an image to grayscale and apply Otsu's thresholding method for segmentation
image_gray = cv2.imread('image.jpg', 0)
_, otsu_thresh = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Display Otsu's thresholding result
cv2.imshow('Otsu Thresholding', otsu_thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 6. Perform contour detection in an image to detect distinct objects or shapes
image_contours = cv2.imread('image.jpg')
gray_contours = cv2.cvtColor(image_contours, cv2.COLOR_BGR2GRAY)
_, thresh_contours = cv2.threshold(gray_contours, 127, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(thresh_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the image
cv2.drawContours(image_contours, contours, -1, (0, 255, 0), 2)

# Display the result
cv2.imshow('Contours', image_contours)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 7. Apply Mask R-CNN to detect objects and their segmentation masks in a custom image and display them
image = cv2.imread('image.jpg')
image_tensor = transform(image_pil).unsqueeze(0)

with torch.no_grad():
    prediction = model(image_tensor)

# Display segmentation masks
masks = prediction[0]['masks']
for i in range(len(masks)):
    mask = masks[i, 0]
    mask = mask.mul(255).byte().cpu().numpy()
    _, thresh_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Apply the mask to the image
    segmented_image = cv2.bitwise_and(image, image, mask=thresh_mask)
    cv2.imshow(f'Segmented Object {i+1}', segmented_image)
    cv2.waitKey(0)

cv2.destroyAllWindows()

# 8. Apply k-means clustering for segmenting regions in an image
image_kmeans = cv2.imread('image.jpg')
image_kmeans = cv2.cvtColor(image_kmeans, cv2.COLOR_BGR2RGB)

# Reshape the image to 2D
pixels = image_kmeans.reshape((-1, 3))

# Apply k-means clustering
kmeans = KMeans(n_clusters=3)
kmeans.fit(pixels)
segmented_image = kmeans.cluster_centers_[kmeans.labels_].reshape(image_kmeans.shape)

# Display the result
plt.imshow(segmented_image)
plt.title('K-means Clustering Segmentation')
plt.show()
