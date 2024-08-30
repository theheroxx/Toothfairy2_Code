import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torch import nn
import matplotlib.pyplot as plt
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import json


class CTScanDataset(Dataset):
    def __init__(self, image_dir, label_dir, json_path, transforms=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transforms = transforms
        self.image_filenames = sorted(os.listdir(image_dir))
        self.label_filenames = sorted(os.listdir(label_dir))

        # Load label information from JSON
        with open(json_path, 'r') as f:  # Corrected the mode to 'r'
            self.label_info = json.load(f)["labels"]
            

    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        label_path = os.path.join(self.label_dir, self.label_filenames[idx])

        # Load image as grayscale (1 channel)
        image = Image.open(img_path).convert("L")
        
        # Load label image and map it to class indices
        label = Image.open(label_path)
        label = torch.tensor(np.array(label), dtype=torch.uint8)
        
        # Ensure the label is in the range of class indices [0, 41]
        label = torch.clamp(label, min=0, max=41).long()

        if self.transforms is not None:
            image = self.transforms(image)

        # Generate target
        target = {
            "masks": label.unsqueeze(0),  # Add channel dimension
            "labels": label.unique(),      # Extract unique class labels from mask
            "boxes": self._get_bounding_boxes(label) # Compute bounding boxes
        }

        return image, target
    
    def _get_bounding_boxes(self, label):
        """Generate bounding boxes for each class in the label."""
        boxes = []
        unique_classes = label.unique()

        for cls in unique_classes:
            if cls == 0:
                continue  # Skip background class

            pos = torch.nonzero(label == cls)
            if pos.numel() > 0:
                xmin = torch.min(pos[:, 1])
                xmax = torch.max(pos[:, 1])
                ymin = torch.min(pos[:, 0])
                ymax = torch.max(pos[:, 0])
                boxes.append([xmin, ymin, xmax, ymax])

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)

        return boxes



def get_data_loaders(image_dir, label_dir, json_path, batch_size=4):
    dataset = CTScanDataset(
        image_dir=image_dir,
        label_dir=label_dir,
        json_path=json_path,
        transforms=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalization for grayscale images
        ])
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))
    return dataloader

def get_model(num_classes):
    model = maskrcnn_resnet50_fpn(pretrained=True)

    # Replace the pre-trained head with a new one for the specific number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Replace the mask predictor with a new one
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model

from torch.utils.tensorboard import SummaryWriter

def train_model(model, dataloader, num_epochs, device="cuda"):
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    writer = SummaryWriter()

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        print(f"Starting Epoch {epoch + 1}/{num_epochs}")

        for batch_idx, (images, targets) in enumerate(dataloader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass
            loss_dict = model(images, targets)
            
            # Sanity check for losses
            losses = sum(loss for loss in loss_dict.values())
            if torch.any(torch.isnan(losses)) or torch.any(losses < 0): 
                print(f"Warning: Invalid loss encountered at batch {batch_idx + 1}. Skipping this batch.") #if negative, show error
                continue

            print(f"Batch {batch_idx + 1}/{len(dataloader)}, Loss: {losses.item()}")

            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += losses.item()
            writer.add_scalar('Training Loss', losses.item(), epoch * len(dataloader) + batch_idx)

        print(f"Epoch {epoch + 1} completed with average loss: {epoch_loss / len(dataloader)}")

        lr_scheduler.step()

        
    writer.close()


def save_predictions(model, dataloader, output_dir, device="cuda"):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for i, (images, targets) in enumerate(dataloader):
            images = list(img.to(device) for img in images)
            outputs = model(images)

            for j, output in enumerate(outputs):
                masks_pred = output['masks'].squeeze().cpu().numpy()

                fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                # Plot original image
                image = images[j].squeeze().cpu().numpy()  # For grayscale, remove channel dimension
                axes[0].imshow(image, cmap='gray')
                axes[0].set_title('Original Image')
                axes[0].axis('off')

                # Plot original mask
                mask = targets[j]['masks'].squeeze().cpu().numpy()
                axes[1].imshow(mask, cmap='gray')
                axes[1].set_title('Original Mask')
                axes[1].axis('off')

                # Plot predicted mask
                pred_mask = masks_pred[0] if masks_pred.ndim == 3 else masks_pred
                axes[2].imshow(pred_mask, cmap='gray')
                axes[2].set_title('Predicted Mask')
                axes[2].axis('off')

                plt.show()

                plt.savefig(os.path.join(output_dir, f"result_{i}_{j}.png"), bbox_inches='tight', pad_inches=0)

if __name__ == "__main__":
    # Paths
    image_dir = './data/sample/slices/test/im1'
    label_dir = './data/sample/slices/test/lb1'
    json_path = './data/sample/dataset.json'  
    output_dir = './data/sample/slices/test/rs1'

    # Parameters
    num_classes = 42  # Background + 41 classes
    num_epochs = 3
    batch_size = 4

    # Get data loaders
    dataloader = get_data_loaders(image_dir, label_dir, json_path, batch_size)

    # Initialize model
    model = get_model(num_classes)

    # Train the model
    train_model(model, dataloader, num_epochs=num_epochs)

    # Save predictions and plot
    save_predictions(model, dataloader, output_dir)
