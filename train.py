import numpy as np
import torch

def train(model, dataloader, loss_function, optimizer, device, epochs=10, print_every=100):
    model.train()
    model.to(device)
    for epoch in range(epochs):
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            if i % print_every == 0:
                print(f"Epoch {epoch + 1} Batch {i} Loss: {loss.item()}")
    return model

def evaluate(model, dataloader, loss_function, device):
    model.eval()
    model.to(device)
    total_loss = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            total_loss += loss.item()
            total += len(labels)
    return total_loss / total