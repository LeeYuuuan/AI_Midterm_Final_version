

import zipfile
from tqdm import tqdm
from PIL import Image
import numpy as np
import io
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import os
import torchvision.utils as utils



def preprocess(input_zip, size=(128, 128)):
    images_processed = []

    with zipfile.ZipFile(input_zip, 'r') as archive_zip:
        archives = archive_zip.namelist()
        images = []
        labels = []
        for i, archive in enumerate(archives):
            if archive.endswith(('.png', '.jpg', '.jpeg')):
                images.append(archive)
                if 'cat' in archive:
                    labels.append(0)
                if 'dog' in archive:
                    labels.append(1)
        # images = [archive for archive in archives if archive.endswith(('.png', '.jpg', '.jpeg'))]
        imgs = []
        for img_path in tqdm(images):
            with archive_zip.open(img_path) as image_zip:
                img = Image.open(io.BytesIO(image_zip.read()))


                imgs.append(img)
                # images_processed.append(img_array)

    # dataset = np.array(images_processed)

    return imgs, np.array(labels)


class Lr_Hr_dataset(Dataset):
    def __init__(self, cat_and_dog_dataset, labels, transform=None):
        self.cat_and_dog_dataset = cat_and_dog_dataset
        self.transform = transform
        self.labels = labels

        self.lr_images = []
        self.hr_images = []
        lr_transform = transforms.Resize((32, 32))
        hr_transform = transforms.Resize((128, 128))
        for img in tqdm(cat_and_dog_dataset):
            self.lr_images.append(lr_transform(img))
            self.hr_images.append(hr_transform(img))

    def __len__(self):
        return len(self.cat_and_dog_dataset)

    def __getitem__(self, idx):
        lr_image = self.lr_images[idx]
        hr_image = self.hr_images[idx]
        if self.transform:
            lr_image = self.transform(lr_image)
            hr_image = self.transform(hr_image)

        label = self.labels[idx]

        return lr_image, hr_image, label
    


    # prompt: zip floder fake_data

import zipfile
import os

def zip_folder(folder_path, output_zip_path):
    """Zips the specified folder.

    Args:
        folder_path: Path to the folder to be zipped.
        output_zip_path: Path to the output zip file.
    """

    try:
        with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, folder_path)
                    zipf.write(file_path, arcname=arcname)
        print(f"Folder '{folder_path}' successfully zipped to '{output_zip_path}'")
    except FileNotFoundError:
        print(f"Error: Folder '{folder_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


# Example usage: Replace 'training_results/fake_data' and 'fake_data.zip' with your desired paths
zip_folder('training_results/fake_data', 'training_results/fake_data.zip')



def generate_dataset(netG, test_loader):
    netG.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_path = 'training_results/fake_data' + '/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    count = 0
    with torch.no_grad():
        for val_lr, _, labels in tqdm(test_loader):
            if count >= 1000:
                break
            batch_size = val_lr.size(0)
            lr = val_lr
            if device == 'cuda':
                lr = lr.float().cuda()
            fake_img = netG(lr)
            for i in range(batch_size):
                label = labels[i].item()
                str_label = 'cat' if label == 0 else 'dog'
                utils.save_image(fake_img[i], out_path + str(count)  + '_' + str_label + '.png')
                count += 1

class Cat_dog_dataset(Dataset):
    def __init__(self, cat_and_dog_images, labels, transform=None):
        self.transform = transform
        self.images = cat_and_dog_images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
    
from matplotlib import pyplot as plt
def plot_loss(losses, name, test_losses=None):
    out_path = 'training_results/figures' + '/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Train Loss', marker='o')
    if test_losses:
        plt.plot(test_losses, label='Test Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(out_path + name + '_loss.png')
    plt.show()