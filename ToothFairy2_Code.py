import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from torchvision.models.detection import maskrcnn_resnet50_fpn
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.utils.tensorboard import SummaryWriter

class CTScanDataset(Dataset):
    def __init__(self, image_dir, label_dir, transforms=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transforms = transforms
        self.image_filenames = sorted(os.listdir(image_dir))
        self.label_filenames = sorted(os.listdir(label_dir))

    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        label_path = os.path.join(self.label_dir, self.label_filenames[idx])

        # Load image as RGB (3 channels expected by Mask R-CNN)
        image = Image.open(img_path).convert("RGB")
        label = Image.open(label_path)

        if self.transforms is not None:
            image = self.transforms(image)

        # Convert the label to a tensor, assuming labels are already in the range 0 to num_classes-1
        label = torch.tensor(np.array(label), dtype=torch.uint8)

        # Find all unique classes in the label
        unique_classes = torch.unique(label)
        
        # Prepare boxes, labels, and masks
        boxes = []
        labels = []
        masks = []

        for cls in unique_classes:
            if cls == 0:  # Skip background class if it exists
                continue
            
            cls_mask = (label == cls).float()  # Binary mask for the class
            pos = torch.nonzero(cls_mask)
            if pos.numel() == 0:
                continue
            
            xmin = torch.min(pos[:, 1])
            xmax = torch.max(pos[:, 1])
            ymin = torch.min(pos[:, 0])
            ymax = torch.max(pos[:, 0])

            boxes.append([xmin.item(), ymin.item(), xmax.item(), ymax.item()])
            labels.append(cls.item())
            masks.append(cls_mask)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        masks = torch.stack(masks)  # Stack all class masks

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks
        }

        return image, target
    

# Updated DataLoader with Data Augmentation and Train/Test Split
def get_data_loaders(image_dir, label_dir, batch_size=4, split_ratio=0.75):
    dataset = CTScanDataset(
        image_dir=image_dir,
        label_dir=label_dir,
        transforms=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalization for RGB images
        ])
    )

    # Calculate split sizes
    train_size = int(split_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))
    
    return train_loader, test_loader

# Updated Model with ResNet-50 Backbone and Mask Predictor
def get_model(num_classes):
    model = maskrcnn_resnet50_fpn(pretrained=True)  # Using ResNet-50 backbone

    # Replace the pre-trained head with a new one for the specific number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Replace the mask predictor with a new one
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model

# Updated Training with Adam Optimizer
def train_model(model, dataloader, num_epochs, device="cuda"):
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    
    # Replaced SGD with Adam optimizer
    optimizer = torch.optim.Adam(params, lr=0.0001)  # Adam optimizer with learning rate of 0.0001
    
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

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
                print(f"Warning: Invalid loss encountered at batch {batch_idx + 1}. Skipping this batch.")
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

# Updated Save Predictions with Simple Thresholding Post-Processing
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
                axes[0].imshow(np.transpose(image, (1, 2, 0)))  # Transpose for RGB images
                axes[0].set_title('Original Image')
                axes[0].axis('off')

                # Plot original mask
                mask = targets[j]['masks'].squeeze().cpu().numpy()
                axes[1].imshow(mask, cmap='gray')
                axes[1].set_title('Original Mask')
                axes[1].axis('off')

                # Plot predicted mask with thresholding post-processing
                pred_mask = masks_pred[0] if masks_pred.ndim == 3 else masks_pred
                pred_mask = (pred_mask > 0.5).astype(np.uint8)  # Simple thresholding
                axes[2].imshow(pred_mask, cmap='gray')
                axes[2].set_title('Predicted Mask')
                axes[2].axis('off')

                plt.show()

                plt.savefig(os.path.join(output_dir, f"result_{i}_{j}.png"), bbox_inches='tight', pad_inches=0)

if __name__ == "__main__":
    # Paths
    image_dir = './data/sample/slices/test/im1'
    label_dir = './data/sample/slices/test/lb1'
    output_dir = './data/sample/slices/test/rs1'

    # Parameters
    num_classes = 42  # Background + 41 classes
    num_epochs = 8  # Number of epochs for training
    batch_size = 4  # Batch size

    # Get data loaders
    train_loader, test_loader = get_data_loaders(image_dir, label_dir, batch_size)

    # Initialize model
    model = get_model(num_classes)

    # Train the model
    train_model(model, train_loader, num_epochs=num_epochs)

    # Save predictions and plot
    save_predictions(model, test_loader, output_dir)
