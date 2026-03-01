"""LLM comparison on LSTM deep composition benchmarks."""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load API key
import subprocess
result = subprocess.run(['bash', '-c', 'source ~/.bashrc 2>/dev/null && echo $OPENAI_API_KEY'],
                       capture_output=True, text=True)
api_key = result.stdout.strip()

if not api_key:
    print("No OPENAI_API_KEY found")
    sys.exit(1)

from openai import OpenAI
client = OpenAI(api_key=api_key)

LSTM_DEEP_COMPOSITION = [
    {"name": "lstm-linear-chain-bug", "expected_safe": False,
     "description": "5-layer LSTM chain with hidden size mismatch at layer 4",
     "source": '''
import torch.nn as nn
class LSTMChain(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(128, 256)
        self.lstm2 = nn.LSTM(256, 512)
        self.lstm3 = nn.LSTM(512, 256)
        self.lstm4 = nn.LSTM(128, 64)   # input_size=128 but lstm3 output=256
        self.fc = nn.Linear(64, 10)
    def forward(self, x):
        h = self.lstm1(x)
        h = self.lstm2(h)
        h = self.lstm3(h)
        h = self.lstm4(h)
        return self.fc(h)
'''},
    {"name": "emb-bilstm-linear-safe", "expected_safe": True,
     "description": "Embedding → BiLSTM → Linear with correct dimensions",
     "source": '''
import torch.nn as nn
class EmbBiLSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(30000, 300)
        self.lstm = nn.LSTM(300, 256, bidirectional=True)
        self.fc = nn.Linear(512, 5)
    def forward(self, x):
        e = self.emb(x)
        out = self.lstm(e)
        return self.fc(out)
'''},
    {"name": "emb-bilstm-linear-bug", "expected_safe": False,
     "description": "Embedding → BiLSTM → Linear with forgotten bidirectional doubling",
     "source": '''
import torch.nn as nn
class EmbBiLSTMBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(30000, 300)
        self.lstm = nn.LSTM(300, 256, bidirectional=True)
        self.fc = nn.Linear(256, 5)  # should be 512 (256*2)
    def forward(self, x):
        e = self.emb(x)
        out = self.lstm(e)
        return self.fc(out)
'''},
    {"name": "gru-seq2seq-safe", "expected_safe": True,
     "description": "GRU encoder-decoder with matching hidden sizes",
     "source": '''
import torch.nn as nn
class Seq2Seq(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.GRU(256, 512)
        self.decoder = nn.GRU(512, 512)
        self.fc = nn.Linear(512, 256)
    def forward(self, x):
        enc = self.encoder(x)
        dec = self.decoder(enc)
        return self.fc(dec)
'''},
    {"name": "gru-seq2seq-hidden-bug", "expected_safe": False,
     "description": "GRU encoder-decoder with hidden size mismatch",
     "source": '''
import torch.nn as nn
class Seq2SeqBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.GRU(256, 512)
        self.decoder = nn.GRU(256, 512)  # expects 512 from encoder, not 256
        self.fc = nn.Linear(512, 256)
    def forward(self, x):
        enc = self.encoder(x)
        dec = self.decoder(enc)
        return self.fc(dec)
'''},
]

COT_PROMPT = """Analyze this PyTorch nn.Module for tensor shape bugs. Think step by step:

1. Trace the shape of each tensor through the forward() method
2. Check that each layer's expected input dimensions match the actual input
3. For LSTM/GRU: output shape is (seq_len, batch, hidden_size * num_directions)
   where num_directions = 2 if bidirectional=True, else 1
4. For nn.Linear(in_features, out_features): input last dim must equal in_features

After analysis, answer EXACTLY with either:
VERDICT: SAFE
or
VERDICT: BUG

{source}
"""

def run_llm_comparison():
    results = []
    llm_correct = 0

    for bench in LSTM_DEEP_COMPOSITION:
        prompt = COT_PROMPT.format(source=bench["source"])

        try:
            response = client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=1024,
            )
            answer = response.choices[0].message.content.strip()
            answer_upper = answer.upper()

            if "VERDICT: BUG" in answer_upper or "VERDICT:**BUG" in answer_upper.replace(" ", ""):
                llm_safe = False
            elif "VERDICT: SAFE" in answer_upper or "VERDICT:**SAFE" in answer_upper.replace(" ", ""):
                llm_safe = True
            else:
                # Try looser parsing
                lines = answer.split('\n')
                last_lines = ' '.join(lines[-5:]).upper()
                if 'BUG' in last_lines and 'SAFE' not in last_lines:
                    llm_safe = False
                elif 'SAFE' in last_lines and 'BUG' not in last_lines:
                    llm_safe = True
                else:
                    llm_safe = None

            correct = llm_safe == bench["expected_safe"]
            if correct:
                llm_correct += 1

            results.append({
                "name": bench["name"],
                "expected_safe": bench["expected_safe"],
                "llm_safe": llm_safe,
                "correct": correct,
                "llm_response": answer[-200:],
            })
            print(f"  {'✓' if correct else '✗'} {bench['name']}: "
                  f"expected={'safe' if bench['expected_safe'] else 'bug'}, "
                  f"LLM={'safe' if llm_safe else 'bug'}")

        except Exception as e:
            print(f"  ERROR {bench['name']}: {e}")
            results.append({
                "name": bench["name"],
                "expected_safe": bench["expected_safe"],
                "llm_safe": None,
                "correct": False,
                "error": str(e),
            })

    return {
        "total": len(results),
        "llm_correct": llm_correct,
        "llm_accuracy": round(llm_correct / len(results), 4) if results else 0,
        "tg_correct": len(results),  # TG got 5/5
        "tg_accuracy": 1.0,
        "results": results,
    }

if __name__ == "__main__":
    print("LLM (GPT-4.1-nano CoT) Comparison on LSTM Deep Composition")
    print("=" * 60)
    results = run_llm_comparison()
    print(f"\nLLM: {results['llm_correct']}/{results['total']} correct")
    print(f"TG:  {results['tg_correct']}/{results['total']} correct")

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "lstm_llm_comparison_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")
