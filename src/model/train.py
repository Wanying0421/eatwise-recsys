# src/model/train.py

import torch
import torch.optim as optim
import numpy as np
import random

def set_seed(seed=42):
    """Sets a fixed seed for reproducibility as requested in Week 2."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def train_loop(model, train_loader, val_loader, epochs=50, lr=0.001, alpha=0.5, beta=0.5):
    """
    Training loop skeleton for Algorithm 1.
    Loss = alpha * L_rating + beta * L_click
    """
    set_seed(42)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    print("Starting training loop skeleton...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        # --- TRAINING BATCH ITERATION SKELTON ---
        # for batch in train_loader:
        #     optimizer.zero_grad()
        #     loss_rating, loss_click = model.compute_loss(batch)
        #     loss = alpha * loss_rating + beta * loss_click
        #     loss.backward()
        #     optimizer.step()
        #     total_loss += loss.item()
        
        # Validation and Early Stopping Check (Skeleton)
        val_loss = total_loss # Placeholder for validation loss calculation
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model checkpoint here later
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break
                
    print("Training loop setup complete.")
