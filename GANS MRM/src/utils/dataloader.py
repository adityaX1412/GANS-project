from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageDraw
import numpy as np


class CelebADataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']

        # Ensure the image is in PIL format
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

     # Create masked image
    def mask_image(self,image):
        """
        Apply a random mask to the image.
        Here we use a simple rectangular mask for demonstration purposes.
        """
        # Convert PIL Image to NumPy array
        img_array = np.array(image)
    
        # Define mask size and position
        mask_height, mask_width = img_array.shape[0] // 4, img_array.shape[1] // 4
        mask_x = np.random.randint(0, img_array.shape[1] - mask_width)
        mask_y = np.random.randint(0, img_array.shape[0] - mask_height)
    
        # Apply mask (black rectangle)
        img_array[mask_y:mask_y + mask_height, mask_x:mask_x + mask_width, :] = 0
    
        # Convert back to PIL Image
        masked_image = Image.fromarray(img_array)
        return masked_image
       
    masked_image = mask_image(image)

    # Apply transformations
    if self.transform:
        image = self.transform(image)
        masked_image = self.transform(masked_image)
        return masked_image, image


    


