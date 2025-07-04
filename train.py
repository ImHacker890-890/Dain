import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CocoCaptions
from torchvision import transforms
from .model import DainDiffusionModel, DiffusionProcess
import os
from tqdm import tqdm


def load_coco_dataset(root='./data', annFile='./data/annotations/captions_train2017.json', 
                      image_size=64, batch_size=32):
    """Load COCO dataset with transformations"""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Create directories if they don't exist
    os.makedirs(root, exist_ok=True)
    
    dataset = CocoCaptions(
        root=os.path.join(root, 'train2017'),
        annFile=annFile,
        transform=transform,
        download=True
    )
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                           num_workers=4, pin_memory=True)
    return dataloader


def train_on_coco(epochs=50, lr=1e-4, image_size=64, batch_size=32, 
                  model_save_path='dain_weights.pth', device=None):
    """
    Train the diffusion model on COCO dataset
    
    Args:
        epochs: Number of training epochs
        lr: Learning rate
        image_size: Size of input images
        batch_size: Batch size for training
        model_save_path: Path to save trained model weights
        device: Device to use ('cuda' or 'cpu'), auto-detects if None
    
    Returns:
        Trained model and training loss history
    """
    # Device setup
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model and diffusion process
    model = DainDiffusionModel(image_size=image_size).to(device)
    diffusion = DiffusionProcess()
    
    # Load COCO dataset
    try:
        dataloader = load_coco_dataset(image_size=image_size, batch_size=batch_size)
    except Exception as e:
        raise RuntimeError(f"Failed to load COCO dataset: {str(e)}")
    
    # Optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training loop
    loss_history = []
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch_idx, (images, _) in enumerate(progress_bar):
            images = images.to(device)
            
            # Sample random timesteps
            t = torch.randint(0, diffusion.num_timesteps, (images.shape[0],), 
                            device=device).long()
            
            # Calculate loss
            optimizer.zero_grad(set_to_none=True)
            loss = diffusion.p_losses(model, images, t)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        
        # Update learning rate
        scheduler.step()
        
        # Calculate average epoch loss
        avg_epoch_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_epoch_loss)
        
        print(f'Epoch {epoch+1}/{epochs} | Loss: {avg_epoch_loss:.4f} | '
              f'LR: {scheduler.get_last_lr()[0]:.2e}')
        
        # Save best model
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(model.state_dict(), model_save_path)
            print(f'Saved model weights to {model_save_path}')
    
    print('Training completed!')
    return model, loss_history
