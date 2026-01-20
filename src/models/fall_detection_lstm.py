import torch
import torch.nn as nn

class FallDetectionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes=1, dropout_prob=0.5):
        super(FallDetectionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout_prob if num_layers > 1 else 0
        )
        
        # BatchNorm
        self.bn = nn.BatchNorm1d(hidden_size)
        
        # MLP
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc3 = nn.Linear(hidden_size // 4, num_classes)
        
        self.dropout = nn.Dropout(dropout_prob)
        self.relu = nn.ReLU()

        # 自动根据类别数选择激活函数
        if num_classes == 1:
            self.final_act = nn.Sigmoid()
        else:
            self.final_act = nn.Softmax(dim=1)

        self._init_weights()
    
    def _init_weights(self):
        # 初始化 LSTM
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        
        # 初始化线性层
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                m.bias.data.fill_(0.01)
    
    def forward(self, x):
        batch = x.size(0)
        h0 = torch.zeros(self.num_layers, batch, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch, self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # last time step
        
        out = self.bn(out)
        
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        
        out = self.fc3(out)
        
        out = self.final_act(out)
        return out
