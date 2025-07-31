"""
EfficientNet-B3 Skin Lesion Classification - Model Definitions and Loading

This module contains the EfficientNet model class and loading functions
for use in feature extraction and inference.
"""

import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet


class EfficientNetB3SkinLesionClassifier(nn.Module):
    """EfficientNet-B3 based transfer learning model for skin lesion classification"""
    
    def __init__(self, num_classes=8, pretrained=True, freeze_backbone=True):
        super(EfficientNetB3SkinLesionClassifier, self).__init__()
        self.num_classes = num_classes
        
        # Load EfficientNet-B3 backbone
        if pretrained:
            self.efficientnet = EfficientNet.from_pretrained('efficientnet-b3')
        else:
            self.efficientnet = EfficientNet.from_name('efficientnet-b3')
        
        # Optionally freeze backbone parameters
        if freeze_backbone:
            for param in self.efficientnet.parameters():
                param.requires_grad = False
        
        # Replace classifier head
        num_features = self.efficientnet._fc.in_features
        self.efficientnet._fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
        # Feature extractor (backbone without classifier)
        self.feature_extractor = nn.Sequential(*list(self.efficientnet.children())[:-1])
    
    def forward(self, x):
        """Forward pass for classification"""
        return self.efficientnet(x)
    
    def extract_features(self, x):
        """Extract features from the backbone (before classification layer)"""
        return self.efficientnet.extract_features(x)
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters for fine-tuning"""
        for param in self.efficientnet.parameters():
            param.requires_grad = True
        print("EfficientNet-B3 backbone unfrozen for fine-tuning")


def load_pretrained_model(checkpoint_path=None, device='cpu', num_classes=8):
    """
    Load a pre-trained EfficientNet-B3 model
    
    Args:
        checkpoint_path: Path to checkpoint file (optional)
        device: Device to load model on
        num_classes: Number of output classes
    
    Returns:
        model: Loaded EfficientNet model
    """
    if checkpoint_path and checkpoint_path.exists():
        # Load from checkpoint
        print(f"Loading model from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model = EfficientNetB3SkinLesionClassifier(
            num_classes=checkpoint.get('num_classes', num_classes),
            pretrained=False,
            freeze_backbone=False
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Model loaded from checkpoint")
    else:
        # Load pretrained backbone only
        print("Loading EfficientNet-B3 with pretrained ImageNet weights")
        model = EfficientNetB3SkinLesionClassifier(
            num_classes=num_classes,
            pretrained=True,
            freeze_backbone=False
        )
        print("✓ Model loaded with pretrained weights")
    
    model = model.to(device)
    model.eval()
    return model
