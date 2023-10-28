from geotorchai.models.raster import SatCNN
import torch
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image 
from torchvision import transforms
import torch.nn as nn
import torchmetrics
import lightning as L

class SatCNNModel(L.LightningModule):
    def __init__(self):
        super(SatCNNModel, self).__init__()
        self.model = SatCNN(3, 128, 128, 3)

    def load_model(self, path):
        state_dict = torch.load(path)['state_dict']
        new_weights = self.model.state_dict()
        for k, v in state_dict.items():
            name = k[6:]
            print(name)
            new_weights[name] = v
        self.model.load_state_dict(new_weights)
        print('Model Created!')

    def forward(self, x):
        return self.model(x)

path = './trained-weights/SatCNN-google_images-exp1-v1.ckpt'
model = SatCNNModel()
model.load_model(path)

model.model.sequences_part2 = model.model.sequences_part2[:-3]
print(model)

data_dir = '../../dataset/google_images'
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.3295, 0.3599, 0.3076), (0.1815, 0.1401, 0.1258))  # Normalização
])
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
test_size = dataset_size - train_size

train_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, test_size])
print(len(train_dataset))
print(len(test_dataset))


train_loader = DataLoader(test_dataset, batch_size=32)

images,labels = next(iter(train_loader))

for images,labels in iter(train_loader):
    print(len(labels))
    # val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=3)
    logits = model(images)
    print(logits.shape)
    break
    # y_pred =torch.argmax(logits, dim=1)
    # print('y_pred => ', y_pred)
    # result = val_accuracy(y_pred, labels)
    # print(result)
