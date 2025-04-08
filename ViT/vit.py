import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from torch import nn, Tensor
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import Compose, Resize, ToTensor, ToPILImage

from einops.layers.torch import Rearrange
from einops import rearrange, repeat

# ============================================================
# 1. DATA LOADING & VISUALIZATION
# ============================================================

to_tensor = Compose([Resize((144, 144)), ToTensor()])
dataset = OxfordIIITPet(root=".", download=True, transform=to_tensor, target_types="category")

dataloader = DataLoader(dataset, batch_size=20, shuffle=True)

def show_images(dataloader, num_samples=20, cols=4):
    plt.figure(figsize=(15, 15))
    to_pil = ToPILImage()
    images, _ = next(iter(dataloader))
    for i in range(min(num_samples, len(images))):
        plt.subplot(int(num_samples / cols) + 1, cols, i + 1)
        plt.imshow(to_pil(images[i]))
        plt.axis("off")
    plt.show()

show_images(dataloader)

# ============================================================
# 2. MODEL COMPONENTS
# ============================================================

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=8, emb_size=128):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, emb_size)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x)
        return x

sample_datapoint = torch.unsqueeze(dataset[0][0], 0)
print("Initial shape:", sample_datapoint.shape)
embedding = PatchEmbedding()(sample_datapoint)
print("Patches shape:", embedding.shape)

class Attention(nn.Module):
    def __init__(self, dim, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.att = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, dropout=dropout)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

    def forward(self, x):

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
        attn_output, _ = self.att(q, k, v)
        attn_output = attn_output.transpose(0, 1)
        return attn_output

att_test = Attention(dim=128, n_heads=4, dropout=0.)
print("Attention output shape:", att_test(torch.ones((1, 5, 128))).shape)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Sequential):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

ff = FeedForward(dim=128, hidden_dim=256)
print("FeedForward output shape:", ff(torch.ones((1, 5, 128))).shape)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

residual_attn = Residual(Attention(dim=128, n_heads=4, dropout=0.))
print("Residual Attention output shape:", residual_attn(torch.ones((1, 5, 128))).shape)

# ============================================================
# 3. VISION TRANSFORMER (ViT) MODEL
# ============================================================

class ViT(nn.Module):
    def __init__(self, ch=3, img_size=144, patch_size=4, emb_dim=32,
                 n_layers=6, out_dim=37, dropout=0.1, heads=2):
        super(ViT, self).__init__()

        self.channels = ch
        self.height = img_size
        self.width = img_size
        self.patch_size = patch_size
        self.n_layers = n_layers

        self.patch_embedding = PatchEmbedding(in_channels=ch, patch_size=patch_size, emb_size=emb_dim)

        num_patches = (img_size // patch_size) ** 2
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, emb_dim))
        self.cls_token = nn.Parameter(torch.rand(1, 1, emb_dim))

        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            transformer_block = nn.Sequential(
                Residual(PreNorm(emb_dim, Attention(emb_dim, n_heads=heads, dropout=dropout))),
                Residual(PreNorm(emb_dim, FeedForward(emb_dim, emb_dim, dropout=dropout)))
            )
            self.layers.append(transformer_block)

        self.head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, out_dim)
        )

    def forward(self, img):
        x = self.patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.pos_embedding[:, :(n + 1)]

        for layer in self.layers:
            x = layer(x)

        return self.head(x[:, 0, :])

model = ViT()
print(model)
print("ViT model output shape:", model(torch.ones((1, 3, 144, 144))).shape)

# ============================================================
# 4. TRAINING SETUP
# ============================================================

train_split = int(0.8 * len(dataset))
train_dataset, test_dataset = random_split(dataset, [train_split, len(dataset) - train_split])

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ViT().to(device)

optimizer = optim.AdamW(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# ============================================================
# 5. TRAINING LOOP WITH LOSS GRAPH
# ============================================================

num_epochs = 1000
train_loss_history = []
test_loss_history = []

for epoch in range(num_epochs):
    model.train()
    train_losses = []

    for inputs, labels in train_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    avg_train_loss = np.mean(train_losses)
    train_loss_history.append(avg_train_loss)

    if epoch % 5 == 0:
        print(f">>> Epoch {epoch} train loss: {avg_train_loss:.4f}")

        model.eval()
        test_losses = []
        with torch.no_grad():
            for inputs, labels in test_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_losses.append(loss.item())

        avg_test_loss = np.mean(test_losses)
        test_loss_history.append(avg_test_loss)
        print(f">>> Epoch {epoch} test loss: {avg_test_loss:.4f}")

# ============================================================
# 6. TRAINING LOSS vs TEST LOSS GRAPH
# ============================================================

plt.figure(figsize=(10, 5))
plt.plot(range(0, num_epochs, 5), train_loss_history, label="Train Loss", marker="o")
plt.plot(range(0, num_epochs, 5), test_loss_history, label="Test Loss", marker="s")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Test Loss")
plt.legend()
plt.grid()
plt.show()

# ============================================================
# 7. EVALUATION ON A TEST BATCH
# ============================================================

inputs, labels = next(iter(test_dataloader))
inputs, labels = inputs.to(device), labels.to(device)
outputs = model(inputs)

print("Predicted classes:", outputs.argmax(-1))
print("Actual classes:", labels)