import torch
from .model import DainDiffusionModel, DiffusionProcess
from PIL import Image
import numpy as np
from transformers import CLIPTextModel, CLIPTokenizer


def load_pretrained_model(weights_path='dain_weights.pth', image_size=64, device='cuda'):
    """Load pretrained model weights"""
    model = DainDiffusionModel(image_size=image_size).to(device)
    if torch.cuda.is_available() and device == 'cuda':
        model.load_state_dict(torch.load(weights_path))
    else:
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.eval()
    return model


def generate_from_prompt(prompt=None, num_images=1, image_size=64, 
                         weights_path='dain_weights.pth', device='cuda'):
    """Generate images in memory without saving to disk"""
    # Load model
    model = load_pretrained_model(weights_path, image_size, device)
    
    # Initialize diffusion process
    diffusion = DiffusionProcess()
    
    # Generate random noise
    shape = (num_images, 3, image_size, image_size)
    
    with torch.no_grad():
        # Generate samples
        samples = diffusion.sample(model, shape, device)
        generated_images = samples[-1]  # Get final denoised images
        
        # Convert to PIL images
        pil_images = []
        for img in generated_images:
            img = img.cpu().numpy().transpose(1, 2, 0)
            img = (img * 127.5 + 127.5).astype(np.uint8)
            pil_images.append(Image.fromarray(img))
            
        return pil_images
