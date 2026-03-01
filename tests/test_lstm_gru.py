"""Tests for LSTM/GRU shape propagation and verification."""
import pytest
from src.model_checker import verify_model


class TestLSTMShapePropagation:
    """Test LSTM shape verification."""

    def test_lstm_safe_basic(self):
        """Basic LSTM with matching input_size."""
        result = verify_model('''
import torch.nn as nn
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(128, 256)
        self.fc = nn.Linear(256, 10)
    def forward(self, x):
        out = self.lstm(x)
        return self.fc(out)
''', input_shapes={"x": ("seq_len", "batch", 128)})
        assert result.safe, f"Expected safe, got violations: {result.pretty()}"

    def test_lstm_input_size_mismatch(self):
        """LSTM with wrong input_size should be caught."""
        result = verify_model('''
import torch.nn as nn
class LSTMBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(128, 256)
        self.fc = nn.Linear(256, 10)
    def forward(self, x):
        out = self.lstm(x)
        return self.fc(out)
''', input_shapes={"x": ("seq_len", "batch", 64)})
        assert not result.safe, "Expected bug: input_size=128 but got dim=64"

    def test_lstm_hidden_to_linear_mismatch(self):
        """LSTM hidden_size doesn't match next linear layer."""
        result = verify_model('''
import torch.nn as nn
class LSTMLinearBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(128, 256)
        self.fc = nn.Linear(512, 10)  # BUG: 256 != 512
    def forward(self, x):
        out = self.lstm(x)
        return self.fc(out)
''', input_shapes={"x": ("seq_len", "batch", 128)})
        assert not result.safe, "Expected bug: LSTM output 256 != Linear input 512"

    def test_lstm_bidirectional_safe(self):
        """Bidirectional LSTM doubles the output size."""
        result = verify_model('''
import torch.nn as nn
class BiLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(128, 256, bidirectional=True)
        self.fc = nn.Linear(512, 10)  # 256 * 2 = 512
    def forward(self, x):
        out = self.lstm(x)
        return self.fc(out)
''', input_shapes={"x": ("seq_len", "batch", 128)})
        assert result.safe, f"Expected safe (bidir 256*2=512): {result.pretty()}"

    def test_lstm_bidirectional_mismatch(self):
        """Bidirectional LSTM output size wrong."""
        result = verify_model('''
import torch.nn as nn
class BiLSTMBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(128, 256, bidirectional=True)
        self.fc = nn.Linear(256, 10)  # BUG: should be 512
    def forward(self, x):
        out = self.lstm(x)
        return self.fc(out)
''', input_shapes={"x": ("seq_len", "batch", 128)})
        assert not result.safe, "Expected bug: bidir output 512 != Linear input 256"

    def test_lstm_batch_first(self):
        """LSTM with batch_first=True."""
        result = verify_model('''
import torch.nn as nn
class BatchFirstLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(128, 256, batch_first=True)
        self.fc = nn.Linear(256, 10)
    def forward(self, x):
        out = self.lstm(x)
        return self.fc(out)
''', input_shapes={"x": ("batch", "seq_len", 128)})
        assert result.safe, f"Expected safe: {result.pretty()}"

    def test_lstm_embedding_to_lstm(self):
        """Embedding → LSTM pipeline."""
        result = verify_model('''
import torch.nn as nn
class EmbLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(10000, 256)
        self.lstm = nn.LSTM(256, 128)
        self.fc = nn.Linear(128, 10)
    def forward(self, x):
        e = self.emb(x)
        out = self.lstm(e)
        return self.fc(out)
''', input_shapes={"x": ("seq_len", "batch")})
        assert result.safe, f"Expected safe: {result.pretty()}"

    def test_lstm_embedding_size_mismatch(self):
        """Embedding dim doesn't match LSTM input_size."""
        result = verify_model('''
import torch.nn as nn
class EmbLSTMBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(10000, 256)
        self.lstm = nn.LSTM(128, 64)  # BUG: emb_dim=256 != input_size=128
        self.fc = nn.Linear(64, 10)
    def forward(self, x):
        e = self.emb(x)
        out = self.lstm(e)
        return self.fc(out)
''', input_shapes={"x": ("seq_len", "batch")})
        assert not result.safe, "Expected bug: embedding_dim=256 != LSTM input_size=128"

    def test_lstm_multilayer(self):
        """Multi-layer LSTM (output shape same as single layer)."""
        result = verify_model('''
import torch.nn as nn
class MultiLayerLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(128, 256, num_layers=3)
        self.fc = nn.Linear(256, 10)
    def forward(self, x):
        out = self.lstm(x)
        return self.fc(out)
''', input_shapes={"x": ("seq_len", "batch", 128)})
        assert result.safe, f"Expected safe: {result.pretty()}"

    def test_lstm_stacked_different_sizes(self):
        """Two LSTM layers with different hidden sizes."""
        result = verify_model('''
import torch.nn as nn
class StackedLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(128, 256)
        self.lstm2 = nn.LSTM(256, 64)
        self.fc = nn.Linear(64, 10)
    def forward(self, x):
        h1 = self.lstm1(x)
        h2 = self.lstm2(h1)
        return self.fc(h2)
''', input_shapes={"x": ("seq_len", "batch", 128)})
        assert result.safe, f"Expected safe: {result.pretty()}"

    def test_lstm_stacked_mismatch(self):
        """Stacked LSTM with mismatched sizes."""
        result = verify_model('''
import torch.nn as nn
class StackedLSTMBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(128, 256)
        self.lstm2 = nn.LSTM(128, 64)  # BUG: expects 256 from lstm1
        self.fc = nn.Linear(64, 10)
    def forward(self, x):
        h1 = self.lstm1(x)
        h2 = self.lstm2(h1)
        return self.fc(h2)
''', input_shapes={"x": ("seq_len", "batch", 128)})
        assert not result.safe, "Expected bug: lstm1 output 256 != lstm2 input 128"


class TestGRUShapePropagation:
    """Test GRU shape verification."""

    def test_gru_safe_basic(self):
        """Basic GRU with matching input_size."""
        result = verify_model('''
import torch.nn as nn
class GRUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(128, 256)
        self.fc = nn.Linear(256, 10)
    def forward(self, x):
        out = self.gru(x)
        return self.fc(out)
''', input_shapes={"x": ("seq_len", "batch", 128)})
        assert result.safe, f"Expected safe: {result.pretty()}"

    def test_gru_input_size_mismatch(self):
        """GRU with wrong input_size."""
        result = verify_model('''
import torch.nn as nn
class GRUBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(128, 256)
        self.fc = nn.Linear(256, 10)
    def forward(self, x):
        out = self.gru(x)
        return self.fc(out)
''', input_shapes={"x": ("seq_len", "batch", 64)})
        assert not result.safe, "Expected bug: GRU input_size=128 but got dim=64"

    def test_gru_hidden_to_linear_mismatch(self):
        """GRU hidden_size doesn't match next linear layer."""
        result = verify_model('''
import torch.nn as nn
class GRULinearBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(128, 256)
        self.fc = nn.Linear(512, 10)  # BUG: 256 != 512
    def forward(self, x):
        out = self.gru(x)
        return self.fc(out)
''', input_shapes={"x": ("seq_len", "batch", 128)})
        assert not result.safe, "Expected bug: GRU output 256 != Linear input 512"

    def test_gru_bidirectional_safe(self):
        """Bidirectional GRU doubles the output size."""
        result = verify_model('''
import torch.nn as nn
class BiGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(128, 256, bidirectional=True)
        self.fc = nn.Linear(512, 10)  # 256 * 2 = 512
    def forward(self, x):
        out = self.gru(x)
        return self.fc(out)
''', input_shapes={"x": ("seq_len", "batch", 128)})
        assert result.safe, f"Expected safe (bidir 256*2=512): {result.pretty()}"

    def test_gru_bidirectional_mismatch(self):
        """Bidirectional GRU output size wrong."""
        result = verify_model('''
import torch.nn as nn
class BiGRUBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(128, 256, bidirectional=True)
        self.fc = nn.Linear(256, 10)  # BUG: should be 512
    def forward(self, x):
        out = self.gru(x)
        return self.fc(out)
''', input_shapes={"x": ("seq_len", "batch", 128)})
        assert not result.safe, "Expected bug: bidir output 512 != Linear input 256"

    def test_gru_embedding_pipeline(self):
        """Embedding → GRU → Linear pipeline."""
        result = verify_model('''
import torch.nn as nn
class EmbGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(5000, 128)
        self.gru = nn.GRU(128, 64)
        self.fc = nn.Linear(64, 5)
    def forward(self, x):
        e = self.emb(x)
        out = self.gru(e)
        return self.fc(out)
''', input_shapes={"x": ("seq_len", "batch")})
        assert result.safe, f"Expected safe: {result.pretty()}"


class TestRNNCertificates:
    """Test that LSTM/GRU models produce valid certificates."""

    def test_lstm_safe_has_certificate(self):
        """Safe LSTM model should produce a verification condition."""
        result = verify_model('''
import torch.nn as nn
class LSTMSafe(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(64, 128)
        self.fc = nn.Linear(128, 10)
    def forward(self, x):
        out = self.lstm(x)
        return self.fc(out)
''', input_shapes={"x": ("seq_len", "batch", 64)})
        assert result.safe
        assert result.certificate is not None
        cert_str = result.certificate.smtlib_certificate()
        assert len(cert_str) > 0

    def test_lstm_bug_has_counterexample(self):
        """Buggy LSTM model should produce a counterexample."""
        result = verify_model('''
import torch.nn as nn
class LSTMBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(64, 128)
        self.fc = nn.Linear(256, 10)  # BUG
    def forward(self, x):
        out = self.lstm(x)
        return self.fc(out)
''', input_shapes={"x": ("seq_len", "batch", 64)})
        assert not result.safe
        assert result.counterexample is not None


class TestLSTMGRUWithCEGAR:
    """Test LSTM/GRU with CEGAR contract discovery."""

    def test_lstm_cegar_finds_real_bug_with_symbolic(self):
        """CEGAR correctly identifies real violation with symbolic dims."""
        from src.shape_cegar import run_shape_cegar
        result = run_shape_cegar('''
import torch.nn as nn
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(128, 256)
        self.fc = nn.Linear(256, 10)
    def forward(self, x):
        out = self.lstm(x)
        return self.fc(out)
''', input_shapes={"x": ("seq_len", "batch", "features")})
        # With no guards, CEGAR correctly finds features != 128 as real bug
        assert result.has_real_bugs

    def test_gru_cegar_finds_real_bug_with_symbolic(self):
        """CEGAR correctly identifies real violation with symbolic dims."""
        from src.shape_cegar import run_shape_cegar
        result = run_shape_cegar('''
import torch.nn as nn
class GRUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(128, 256)
        self.fc = nn.Linear(256, 10)
    def forward(self, x):
        out = self.gru(x)
        return self.fc(out)
''', input_shapes={"x": ("seq_len", "batch", "features")})
        assert result.has_real_bugs


class TestRNNHiddenStateExtraction:
    """Test LSTM/GRU hidden state tuple unpacking patterns."""

    def test_lstm_nested_tuple_hidden_mismatch(self):
        """_, (h, _) = self.lstm(x) — hidden_size=256 but fc expects 128."""
        result = verify_model('''
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(100, 256, batch_first=True)
        self.fc = nn.Linear(128, 10)
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h.squeeze(0))
''', input_shapes={"x": ("batch", "seq", 100)})
        assert not result.safe, "Should detect hidden_size mismatch"

    def test_lstm_nested_tuple_hidden_correct(self):
        """_, (h, _) = self.lstm(x) — hidden_size=256 matches fc in_features."""
        result = verify_model('''
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(100, 256, batch_first=True)
        self.fc = nn.Linear(256, 10)
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h.squeeze(0))
''', input_shapes={"x": ("batch", "seq", 100)})
        assert result.safe, f"Should be safe: {result.pretty()}"

    def test_bilstm_hidden_state_not_doubled(self):
        """BiLSTM h_n last dim is hidden_size, not hidden_size*2."""
        result = verify_model('''
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(100, 256, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(256, 10)
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h.squeeze(0))
''', input_shapes={"x": ("batch", "seq", 100)})
        assert result.safe, f"h_n last dim is hidden_size, not *D: {result.pretty()}"

    def test_bilstm_hidden_mismatch(self):
        """BiLSTM hidden_size=256 but fc expects 128."""
        result = verify_model('''
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(100, 256, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(128, 10)
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h.squeeze(0))
''', input_shapes={"x": ("batch", "seq", 100)})
        assert not result.safe, "Should detect 128 != 256"

    def test_gru_flat_tuple_hidden_mismatch(self):
        """_, h = self.gru(x) — hidden_size=100 but fc expects 200."""
        result = verify_model('''
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(50, 100, batch_first=True)
        self.fc = nn.Linear(200, 10)
    def forward(self, x):
        _, h = self.gru(x)
        return self.fc(h.squeeze(0))
''', input_shapes={"x": ("batch", "seq", 50)})
        assert not result.safe, "Should detect 200 != 100"

    def test_gru_flat_tuple_hidden_correct(self):
        """_, h = self.gru(x) — hidden_size=100 matches fc in_features."""
        result = verify_model('''
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(50, 100, batch_first=True)
        self.fc = nn.Linear(100, 10)
    def forward(self, x):
        _, h = self.gru(x)
        return self.fc(h.squeeze(0))
''', input_shapes={"x": ("batch", "seq", 50)})
        assert result.safe, f"Should be safe: {result.pretty()}"

    def test_lstm_named_output_and_hidden(self):
        """output, (h_n, c_n) = self.lstm(x) — named variables."""
        result = verify_model('''
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(100, 256, batch_first=True)
        self.fc = nn.Linear(128, 10)
    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        return self.fc(h_n.squeeze(0))
''', input_shapes={"x": ("batch", "seq", 100)})
        assert not result.safe, "Should detect 128 != 256"

    def test_lstm_output_use_not_hidden(self):
        """output, _ = self.lstm(x) — using output, not hidden."""
        result = verify_model('''
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(100, 256, batch_first=True)
        self.fc = nn.Linear(256, 10)
    def forward(self, x):
        output, _ = self.lstm(x)
        return self.fc(output[:, -1, :])
''', input_shapes={"x": ("batch", "seq", 100)})
        assert result.safe, f"Should be safe: {result.pretty()}"
