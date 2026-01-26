import torch
import torch.nn as nn


class BiLSTMAttention(nn.Module):
    """
    BiLSTM with Attention Mechanism
    
    Architecture:
    1. Bidirectional LSTM - Captures temporal information in both directions
    2. Attention Mechanism - Learns weighted importance of each frame
    3. Deep Classifier - Enhanced feature extraction with LayerNorm and Dropout
    """
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_prob=0.4):
        super(BiLSTMAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # BiLSTM (bidirectional, output size: 2*hidden_size)
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_prob if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(hidden_size * 2)
        self.ln2 = nn.LayerNorm(hidden_size)
        
        # Deep classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout_prob),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob * 0.5),
            
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.fill_(0.01)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            logits: (batch, num_classes)
        """
        # BiLSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size*2)
        lstm_out = self.ln1(lstm_out)
        
        # Attention weights
        attn_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        
        # Weighted sum
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, hidden_size*2)
        
        # Classification
        out = self.classifier(context)
        
        return out
