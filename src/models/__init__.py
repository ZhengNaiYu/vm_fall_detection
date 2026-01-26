from .lstm import LSTM
from .gru import GRU
from .bilstm_attention import BiLSTMAttention
from .transformer import Transformer
from .vae import VAE
from .physical_rules import detect_fall_by_physics

# Backward compatibility aliases (deprecated, use new names instead)
FallDetectionLSTM = LSTM
FallDetectionGRU = GRU
ImprovedLSTM = BiLSTMAttention
TransformerEncoder = Transformer