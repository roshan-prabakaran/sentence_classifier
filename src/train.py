import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
from src.model import SimpleSentenceEncoder

def contrastive_loss(x1, x2, label, margin=1.0):
    dist = torch.norm(x1 - x2, dim=1)
    loss = label * dist.pow(2) + (1 - label) * torch.clamp(margin - dist, min=0).pow(2)
    return loss.mean()

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleSentenceEncoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Load data
    with open('C:/INTERN/data/pairs.json') as f:
        pairs = json.load(f)

    sentences1 = [p["sentence1"] for p in pairs]
    sentences2 = [p["sentence2"] for p in pairs]
    labels = torch.tensor([p["label"] for p in pairs], dtype=torch.float32, device=device)

    # Training
    model.train()
    for epoch in range(30):
        optimizer.zero_grad()
        embed1 = model(sentences1).to(device)
        embed2 = model(sentences2).to(device)
        loss = contrastive_loss(embed1, embed2, labels)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    # Save model
    torch.save(model.state_dict(), "sentence_encoder.pth")

if __name__ == "__main__":
    train()
