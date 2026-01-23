import torch
import torch.nn as nn


class ImprovedLSTM(nn.Module):
    """
    改进版LSTM，包含以下优化：
    1. BiLSTM（双向LSTM）- 捕获前后时序信息
    2. 注意力机制 - 学习重要帧的权重
    3. 更深的分类器 - 增强特征提取
    4. 残差连接 - 缓解梯度消失
    """
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_prob=0.4):
        super(ImprovedLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # BiLSTM（双向，所以输出是 2*hidden_size）
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers,
            batch_first=True,
            bidirectional=True,  # 双向
            dropout=dropout_prob if num_layers > 1 else 0
        )
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # 归一化
        self.ln1 = nn.LayerNorm(hidden_size * 2)
        self.ln2 = nn.LayerNorm(hidden_size)
        
        # 更深的分类器
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
        x: (batch, seq_len, input_size)
        """
        batch_size = x.size(0)
        
        # BiLSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size*2)
        lstm_out = self.ln1(lstm_out)
        
        # 注意力权重
        attn_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        
        # 加权求和
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, hidden_size*2)
        
        # 分类
        out = self.classifier(context)
        
        return out  # 返回logits，不要softmax


class TransformerEncoder(nn.Module):
    """
    基于Transformer Encoder的动作识别模型
    适合捕获长距离时序依赖
    """
    def __init__(self, input_size, hidden_size, num_layers, num_classes, 
                 nhead=4, dropout_prob=0.3):
        super(TransformerEncoder, self).__init__()
        self.hidden_size = hidden_size
        
        # 输入投影
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(hidden_size, dropout_prob)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            dropout=dropout_prob,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 分类头
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
        x: (batch, seq_len, input_size)
        """
        # 投影到hidden_size
        x = self.input_proj(x)  # (batch, seq_len, hidden_size)
        
        # 位置编码
        x = self.pos_encoder(x)
        
        # Transformer编码
        x = self.transformer(x)  # (batch, seq_len, hidden_size)
        
        # 全局平均池化
        x = torch.mean(x, dim=1)  # (batch, hidden_size)
        
        # 分类
        out = self.classifier(x)
        
        return out


class PositionalEncoding(nn.Module):
    """位置编码"""
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
        x: (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


import numpy as np
