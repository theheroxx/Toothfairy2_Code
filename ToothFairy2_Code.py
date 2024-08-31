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

        # Load image as grayscale (1 channel)
        image = Image.open(img_path).convert("L")
        label = Image.open(label_path)

        if self.transforms is not None:
            image = self.transforms(image)

        label = torch.tensor(np.array(label), dtype=torch.uint8)

        # Generate boxes and labels for each class separately
        boxes = []
        labels = []
        masks = []

        for class_id in range(1, 42):  # Assuming class IDs are 1 to 41
            class_mask = (label == class_id).float()

            pos = torch.nonzero(class_mask)
            if pos.numel() == 0:
                continue

            xmin = torch.min(pos[:, 1])
            xmax = torch.max(pos[:, 1])
            ymin = torch.min(pos[:, 0])
            ymax = torch.max(pos[:, 0])

            # Ensure the bounding box has positive height and width
            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(class_id)
                masks.append(class_mask)

        if len(boxes) == 0:
            # If no class was found, return a dummy target
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, label.shape[0], label.shape[1]), dtype=torch.uint8)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            masks = torch.stack(masks)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks
        }

        return image, target

    

# DataLoader with Data Augmentation and Train/Test Split
def get_data_loaders(image_dir, label_dir, batch_size=4, split_ratio=0.75):
    dataset = CTScanDataset(
        image_dir=image_dir,
        label_dir=label_dir,
        transforms=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalization for grayscale images
        ])
    )

    # Calculate split sizes
    train_size = int(split_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))
    
    return train_loader, test_loader

# Model with ResNet-50 Backbone
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




# Training with Adam Optimizer
def train_model(model, dataloader, num_epochs, device="cuda"):
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    
    optimizer = torch.optim.Adam(params, lr=0.0001)
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

# Save Predictions with Simple Thresholding Post-Processing
def save_predictions(model, dataloader, output_dir, device="cuda"):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for i, (images, targets) in enumerate(dataloader):
            images = list(img.to(device) for img in images)
            outputs = model(images)

            for j, output in enumerate(outputs):
                masks_pred = output['masks'].cpu().numpy()

                # Combine all predicted masks into a single mask
                combined_mask = np.zeros(masks_pred.shape[2:], dtype=np.uint8)  # Remove the channel dimension
                for k in range(masks_pred.shape[0]):
                    pred_mask = (masks_pred[k].squeeze() > 0.5).astype(np.uint8)
                    combined_mask = np.maximum(combined_mask, pred_mask * (k + 1))  # Assign each class a unique value

                fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                # Plot original image
                image = images[j].squeeze().cpu().numpy()  # For grayscale, remove channel dimension
                axes[0].imshow(image, cmap='gray')
                axes[0].set_title('Original Image')
                axes[0].axis('off')

                # Plot original mask (combined across all channels)
                if 'masks' in targets[j] and len(targets[j]['masks']) > 0:
                    original_mask_combined = torch.max(targets[j]['masks'], dim=0).values.squeeze().cpu().numpy()
                else:
                    original_mask_combined = np.zeros_like(image)
                axes[1].imshow(original_mask_combined, cmap='gray')
                axes[1].set_title('Original Mask (Combined)')
                axes[1].axis('off')

                # Plot combined predicted mask (now 2D)
                axes[2].imshow(combined_mask, cmap='gray')
                axes[2].set_title('Combined Predicted Mask')
                axes[2].axis('off')

                plt.show()

                plt.savefig(os.path.join(output_dir, f"result_{i}_{j}.png"), bbox_inches='tight', pad_inches=0)
                plt.close()





if __name__ == "__main__":
    # Paths
    image_dir = './data/sample/slices/test/im1'
    label_dir = './data/sample/slices/test/lb1'
    output_dir = './data/sample/slices/test/rs1'

    # Parameters
    num_classes = 42  # Background + 41 classes
    num_epochs = 1
    batch_size = 4

    # Get data loaders
    train_loader, test_loader = get_data_loaders(image_dir, label_dir, batch_size)

    # Initialize model
    model = get_model(num_classes)

    # Train the model
    train_model(model, train_loader, num_epochs=num_epochs)

    # Save predictions and plot
    save_predictions(model, test_loader, output_dir)
