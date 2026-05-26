import torch
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import glob
import numpy as np
import matplotlib.pyplot as plt
import os
import tifffile as tiff


# https://github.com/qubvel-org/segmentation_models.pytorch


device = "cuda" if torch.cuda.is_available() else "cpu"

num_classes = 2

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=1,
    classes=num_classes,
).to(device)


class SegDataset(Dataset):
    def __init__(self, image_paths, mask_paths=None, transform=None, preprocess_fn=None, downsample=4):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.preprocess_fn = preprocess_fn
        self.downsample = downsample

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = tiff.imread(image_path).astype(np.float32)

        image = (image - image.min()) / (image.max() - image.min() + 1e-8)

        mask = None
        if self.mask_paths is not None:
            mask_path = self.mask_paths[idx]
            mask = np.load(mask_path, allow_pickle=True) # tiff.imread(mask_path)
            mask = mask.item()['masks'].astype(np.float32)
            # print(mask.shape, np.unique(mask))

            if mask.ndim == 3:
                mask = mask[..., 0]

            mask = mask.astype(np.int64)   # 保留 0/1/2

        if self.preprocess_fn is not None:
            image = self.preprocess_fn(image)

        if self.transform is not None:
            if mask is not None:
                augmented = self.transform(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]
            else:
                augmented = self.transform(image=image)
                image = augmented["image"]

        image = image[::self.downsample, ::self.downsample]
        image = image[None, ...]
        image = torch.tensor(image, dtype=torch.float32)

        if mask is not None:
            mask = mask[::self.downsample, ::self.downsample]
            mask = torch.tensor(mask, dtype=torch.long)   # [H, W]
            return image, mask

        return image, image_path


train_image_paths = sorted(glob.glob('./data/annotation/train/bleo/*.tif'))
train_mask_paths = sorted(glob.glob('./data/annotation/train/bleo/*_seg.npy'))
# print(train_image_paths)
# print(train_mask_paths)

train_dataset = SegDataset(
    image_paths=train_image_paths,
    mask_paths=train_mask_paths,
    downsample=1
)

image, mask = train_dataset[0]
plt.figure()
plt.imshow(image[0], cmap='gray')
plt.figure()
plt.imshow(mask)

train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

loss_fn = smp.losses.DiceLoss(mode="multiclass", from_logits=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for images, masks in train_loader:
        images = images.to(device, dtype=torch.float32)   # [B,1,H,W]
        masks = masks.to(device, dtype=torch.long)        # [B,H,W]
        # print(images.shape, masks.shape)

        optimizer.zero_grad()
        logits = model(images)                            # [B,C,H,W]
        loss = loss_fn(logits, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss/len(train_loader):.6f}')

torch.save(model.state_dict(), './models/model_multiclass_unet.pth')



classes = 'multi'
root_path = './data/Rat MIR/Rat 1_bleo'
root_path = './data/Rat MIR/Bleo'

save_mask_dir = os.path.join(root_path, f'2_tif_mask_{classes}')
save_overlap_dir = os.path.join(root_path, f'2_tif_mask_overlap_{classes}')
os.makedirs(save_mask_dir, exist_ok=True)
os.makedirs(save_overlap_dir, exist_ok=True)

pred_image_paths = sorted(glob.glob(os.path.join(root_path,'2_tif/*.tif')))

pred_dataset = SegDataset(
    image_paths=pred_image_paths,
    mask_paths=None,
    downsample=4
)

pred_loader = DataLoader(
    pred_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)


model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=1,
    classes=num_classes,
).to(device)

model.load_state_dict(torch.load('./models/model_multiclass_unet.pth', map_location=device))

model.eval()

with torch.no_grad():
    for images, image_paths in pred_loader:        
        images = images.to(device, dtype=torch.float32)

        logits = model(images)                  # [B,3,H,W]
        preds = torch.argmax(logits, dim=1)     # [B,H,W]

        images_np = images.cpu().numpy()
        preds_np = preds.cpu().numpy()

        for b in range(images_np.shape[0]):
            img = images_np[b, 0]
            pred_mask = preds_np[b].astype(np.uint8)

            base_name = os.path.splitext(os.path.basename(image_paths[b]))[0]

            # 保存整张类别图：0=background, 1=class1, 2=class2
            tiff.imwrite(
                os.path.join(save_mask_dir, f'{base_name}_mask.tif'),
                pred_mask
            )

            # 分别保存两类二值mask
            mask1 = (pred_mask == 1).astype(np.uint8) * 255
            mask2 = (pred_mask == 2).astype(np.uint8) * 255

            tiff.imwrite(
                os.path.join(save_mask_dir, f'{base_name}_class1.tif'),
                mask1
            )
            tiff.imwrite(
                os.path.join(save_mask_dir, f'{base_name}_class2.tif'),
                mask2
            )

            # overlap
            plt.figure(figsize=(6, 6))
            plt.imshow(img, cmap='gray')
            plt.imshow(pred_mask == 0, alpha=0.25, cmap='Greens')
            plt.imshow(pred_mask == 1, alpha=0.25, cmap='Reds')
            plt.imshow(pred_mask == 2, alpha=0.25, cmap='Blues')
            plt.axis('off')
            plt.savefig(
                os.path.join(save_overlap_dir, f'{base_name}_overlap.png'),
                bbox_inches='tight',
                pad_inches=0
            )
            plt.close()