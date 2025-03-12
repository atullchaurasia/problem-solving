import torch
import torchvision.datasets as datasets
from tqdm import tqdm
from torch import nn, optim
from model import VariatioalAutoEncoder
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_DIM = 64 * 64 * 3 
H_DIM = 200
Z_DIM = 20
NUM_EPOCHS = 10
BATCH_SIZE = 32
LR_RATE = 3e-4


class CustomImageDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.image_paths = glob.glob(os.path.join(folder_path, "*"))  
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB") 
        if self.transform:
            image = self.transform(image)
        return image 

transform = transforms.Compose([
    transforms.Resize((64, 64)), 
    transforms.ToTensor() 
])

dataset_path = "D:/VAE/data"

train_dataset = CustomImageDataset(dataset_path, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = VariatioalAutoEncoder(INPUT_DIM, H_DIM, Z_DIM).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR_RATE)
loss_fn = nn.BCELoss(reduction="sum")


for epoch in range(NUM_EPOCHS):
    loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
    
    for i, x in loop:
        x = x.to(DEVICE).view(x.shape[0], -1) 
        x_reconstructed, mu, log_var = model(x) 
        reconstruction_loss = loss_fn(x_reconstructed, x)
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        loss = reconstruction_loss + kl_div
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loop.set_postfix(loss=loss.item())
        
model = model.to("cpu")


def inference(model, dataset, anime_idx, num_example=1):
    model.eval() 
    
    encoding_anime = []
    images = []

    for i in range(min(10, len(dataset))):
        image = dataset[i].view(1, -1) 
        images.append(image)

        with torch.no_grad():
            mu, log_var = model.encode(image)
            sigma = torch.exp(0.5 * log_var) 
        encoding_anime.append((mu, sigma))

    mu, sigma = encoding_anime[anime_idx]

    for example in range(num_example):
        epsilon = torch.randn_like(sigma)  
        z = mu + sigma * epsilon 

        with torch.no_grad():
            out = model.decode(z)

        out = out.view(-1, 3, 64, 64)
        save_image(out, f'generated_anime{anime_idx}_ex{example}.png')

for idx in range(10):
    inference(model, train_dataset, anime_idx=idx, num_example=1)
