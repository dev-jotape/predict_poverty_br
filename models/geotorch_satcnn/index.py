import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image 
from geotorchai.models.raster import SatCNN
from torchvision.models import resnet50, ResNet50_Weights

from train_model import PytorchTrainingAndTest

# Defina o caminho para as pastas de treinamento
data_dir = '../../dataset/google_images'

# Transformações para redimensionar e normalizar as imagens
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.3295, 0.3599, 0.3076), (0.1815, 0.1401, 0.1258))  # Normalização
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalização
])

# Crie um conjunto de dados personalizado
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for label, class_name in enumerate(os.listdir(root_dir)):
            class_dir = os.path.join(root_dir, class_name)
            for image_name in os.listdir(class_dir):
                self.image_paths.append(os.path.join(class_dir, image_name))
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label

# Crie um DataLoader para o conjunto de dados
dataset = CustomDataset(data_dir, transform=transform)

dataset_size = len(dataset)
print(dataset_size)
train_size = int(0.85 * dataset_size)
# val_size = int(0.15 * dataset_size)
test_size = dataset_size - train_size

train_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, test_size])
print(len(train_dataset))
print(len(test_dataset))

# Crie os dataloaders
batch_size = 32
learning_rate = 0.0001

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# channels_sum, channels_squared_sum, num_batches = 0, 0, 0
# for i, sample in enumerate(train_loader):
#     channels_sum += torch.mean(data_temp, dim=[0, 2, 3])
#     channels_squared_sum += torch.mean(data_temp**2, dim=[0, 2, 3])
#     num_batches += 1

# mean = channels_sum / num_batches
# std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

# norm_params = {"mean": mean, "std": std}

# print(norm_params)

model = SatCNN(3, 128, 128, 3)

# model = resnet50(weights=ResNet50_Weights.DEFAULT, progress=True)
# model.fc = nn.Linear(2048, 3)

print(model)
trainer = PytorchTrainingAndTest()

trainer.run_model(1, model, 'SatCNN', 'google_images', train_loader, test_loader, learning_rate, 30, 3)
# trainer.run_model(2, model, 'ResNet50_finetuning', 'google_images', train_loader, test_loader, learning_rate, 30, 3)