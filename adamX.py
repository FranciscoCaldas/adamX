# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 00:46:06 2025

@author: Francisco
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from adan import Adan
torch.manual_seed(42)

# === Load and Preprocess CIFAR-10 and MNIST ===
def get_dataloaders(dataset="cifar10", batch_size=68):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    if dataset == "mnist":
        train_set = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
        test_set = torchvision.datasets.MNIST(root="./data", train=False, transform=transform, download=True)
        input_dim = 28 * 28
    elif dataset == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_set = torchvision.datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
        test_set = torchvision.datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)
        input_dim = 32 * 32 * 3
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, input_dim

# === Define a Simple Neural Network ===
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 classes for MNIST and CIFAR-10
        self.c1 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    #    self.fc1.weight.data.fill_(0.01)  # Fill with constant value (0.01)
    #    self.fc1.bias.data.fill_(0.1)     # Fill biases with constant (0.1)
    #    self.fc2.weight.data.normal_(mean=0.0, std=0.02)  # Fill weights with normal distribution
    #    self.fc2.bias.data.fill_(0.2)  
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x
    
     # Set biases to zero
     
class ImprovedCNN(nn.Module):
    def __init__(self,input_dim):
        super(ImprovedCNN, self).__init__()
        
        # First Convolutional Layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 3 input channels (RGB), 32 output channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 32 input channels, 64 output channels
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 64 input channels, 128 output channels
        
        # Max Pooling layer
        self.pool = nn.MaxPool2d(2, 2)  # Pooling kernel size 2
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 512)  # Flattened image size is 128 * 4 * 4 after 3 conv layers and 2 pool layers
        self.fc2 = nn.Linear(512, 10)  # 10 output classes for CIFAR-10
        
        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(0.1)
        
        # Batch Normalization layers
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        # Apply Convolution + BatchNorm + ReLU + Max Pooling
        x = self.pool(torch.nn.functional.relu(self.bn1(self.conv1(x))))  # conv1 -> BN -> ReLU -> Pooling
        x = self.pool(torch.nn.functional.relu(self.bn2(self.conv2(x))))  # conv2 -> BN -> ReLU -> Pooling
        x = self.pool(torch.nn.functional.relu(self.bn3(self.conv3(x))))  # conv3 -> BN -> ReLU -> Pooling
        
        # Flatten the output of the convolutional layers
        x = x.view(-1, 128 * 4 * 4)  # Flatten the output
        
        # Fully connected layers with Dropout
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.dropout(x)  # Apply Dropout
        x = self.fc2(x)
        
        x = self.softmax(x)
        
        return x

# === AdamX Optimizer ===
class AdamX(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, alpha=0.99, lambda_exp=1,cc=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, alpha=alpha, lambda_exp=lambda_exp,cc=cc)
        super(AdamX, self).__init__(params, defaults)
    
    def step(self, closure=None):
        if closure is not None:
            loss = closure()
            
        
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]
                
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                    state["prev_grad"] = torch.ones_like(p.data)
                    
                m, v = state["m"], state["v"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                alpha = group["alpha"]
                lambda_exp = group["lambda_exp"]
                c = group['cc']
                
                state["step"] += 1
                
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                m_hat = m / (1 - beta1 ** state["step"])
                v_hat = v / (1 - beta2 ** state["step"])
                
                #x = alpha * m_hat# + (1 - alpha) * v_hat.sqrt()
                x = m_hat 
                gamma = torch.exp(lambda_exp * torch.nn.functional.cosine_similarity(grad, state["prev_grad"], dim=0))
                #print(gamma)
                state["prev_grad"] = grad.clone()
                
                v_tilde = beta2 * v_hat + (1 - beta2) * max(v_hat.mean().item(), torch.var(grad).item())
                
                prev_params = p.data.clone()  
                # Parameter update
                #print(-gamma)
                p.data.addcdiv_(-x.mul_(gamma).mul_(group['lr']), (v_tilde.sqrt() + eps))
               # plt.plot(p)
                #cena = -x.mul_(gamma) / (v_tilde.sqrt() + eps)
                #plt.plot(cena.cpu().detach().numpy())
                #plt.show()
                #-group["lr"] * gamma

                if closure is not None:
                    new_loss = closure() 
                   # print('new_loss:', new_loss.item())
                   # print('loss:',loss.item())
                    if new_loss > loss:  # Revert if loss increased
                        #print('nanana')
                        p.data.copy_(prev_params + c * (p.data - prev_params)) 
                        
                        #probability
                        #new_loss = closure()
                    #loss = new_loss

        return loss

# === Training Function ===
def train(model, optimizer, train_loader, test_loader, epochs=10):
    criterion = nn.CrossEntropyLoss()
    train_losses, test_losses = [], []
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(torch.float32), labels.to(torch.long)
            
            def closure():
                #optimizer.zero_grad()  # Reset gradients
                outputs = model(images)  # Forward pass
                loss = criterion(outputs, labels)  # Compute loss
                #loss.backward()  # Backpropagation
                return loss  
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step(closure=closure)
            total_train_loss += loss.item()
        if epoch % 10 == 0:
            model.eval()
            total_test_loss = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(torch.float32), labels.to(torch.long)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    total_test_loss += loss.item()
            test_losses.append(total_test_loss / len(test_loader))
        
        train_losses.append(total_train_loss / len(train_loader))
        
        if epoch% 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}")
        else:
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_losses[-1]:.4f}")
    
    return train_losses, test_losses

# === Run Experiments ===
for dataset in ["cifar10"]:
    print(f"\nTraining on {dataset.upper()}...")
    train_loader, test_loader, input_dim = get_dataloaders(dataset)
    model_adam = ImprovedCNN(input_dim)
    model_adan = ImprovedCNN(input_dim)
    model_adamx = ImprovedCNN(input_dim)
    
    optimizer_adam = optim.Adam(model_adam.parameters(), lr=1e-3)
    optimizer_adan = Adan(model_adamx.parameters())

    #optimizer_adamx = AdamX(model_adamx.parameters(), lr=1e-3)
    
    loss_adam_train, loss_adam_test = train(model_adam, optimizer_adam,train_loader,test_loader, epochs=100)
    loss_adamx_train, loss_adam_test = train(model_adan,optimizzer_adan,train_loader,test_loader,epochs=100)
    loss_adamx_train, loss_adamx_test = train(model_adamx, optimizer_adan,train_loader,test_loader,epochs=100)
    
    
 #%%   
import numpy as np
plt.figure(dpi=600)
plt.plot(loss_adam_train, label=f"Adam ({dataset}) Train", linestyle="solid")
plt.plot(np.arange(0,100,10),loss_adam_test, label=f"Adam ({dataset}) Test", linestyle="dashed")
plt.plot(loss_adamx_train, label=f"AdamX ({dataset}) Train", linestyle="solid")
plt.plot(np.arange(0,100,10),loss_adamx_test, label=f"AdamX ({dataset}) Test", linestyle="dashed")
plt.xlabel("Epochs")
plt.ylabel("Loss") 
plt.legend()
plt.title(f"Adam vs. AdamX Loss Curve on {dataset.upper()}")
plt.show()
