import numpy as np
import torch
import torch.nn as nn


class Transformer(nn.Module):
    """
    Transformer Encoder for Action Recognition
    
    Advantages:
    - Captures long-range temporal dependencies
    - Parallel processing capability
    - Self-attention mechanism for important frame detection
    """
    def __init__(self, input_size, hidden_size, num_layers, num_classes, 
                 nhead=4, dropout_prob=0.3):
        super(Transformer, self).__init__()
        self.hidden_size = hidden_size
        
        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_size, dropout_prob)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            dropout=dropout_prob,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            logits: (batch, num_classes)
        """
        # Project to hidden size
        x = self.input_proj(x)  # (batch, seq_len, hidden_size)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        x = self.transformer(x)  # (batch, seq_len, hidden_size)
        
        # Global average pooling
        x = torch.mean(x, dim=1)  # (batch, hidden_size)
        
        # Classification
        out = self.classifier(x)
        
        return out


class PositionalEncoding(nn.Module):
    """Positional Encoding for Transformer"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
