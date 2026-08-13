import sys
sys.path.insert(0, 'Anti_UAV_Localization/src')

from dataset.uav_dataset import UAVSegmentationDataset

ds = UAVSegmentationDataset(
    images_dir='Anti_UAV_Localization/data/raw/images/val',
    masks_dir='Anti_UAV_Localization/data/raw/masks/val',
    img_size=(512, 512),
    stride=20,
)

print(f'Val samples (stride=20): {len(ds)}')
img, mask = ds[0]
print(f'Image: {img.shape}, mask: {mask.shape}, mask sum: {mask.sum().item():.0f}')