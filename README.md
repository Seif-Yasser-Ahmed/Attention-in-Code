# Attention-in-Code

This repository provides clean and modular PyTorch implementations of core attention mechanisms used in Transformer architectures. It's built to serve as an educational tool for understanding and experimenting with attention models.

---

## Features

Implemented attention types:

- **Self-Attention** (Encoder-style)
- **Masked Self-Attention** (Decoder-style)
- **Cross (Encoder-Decoder) Attention**
- **Multi-Head Attention**

Each attention module is written from scratch with readability and clarity in mind.

---

## Project Structure

```
Attention-in-Code/
├── README.md
├── LICENSE
├── main.ipynb                   # Notebook demonstrating all attention types
└── src/
    ├── self_attention.py        # Self-Attention (Encoder)
    ├── masked_self_attention.py # Masked Self-Attention (Decoder)
    ├── attention.py             # Cross Attention (Encoder-Decoder)
    └── multi_head_attention.py  # Multi-Head Attention
```

---

## Getting Started

### Requirements

- Python 3.7+
- PyTorch ≥ 1.10

Install PyTorch from: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

or use pip:
```bash
pip install torch
```

### Usage

Run the `main.ipynb` notebook to explore and test the attention implementations.

#### Example: Self-Attention

```python
from src.self_attention import SelfAttention
import torch

x = torch.tensor([[1.16, 0.23], [0.57, 1.36], [4.41, -2.16]])
self_attention = SelfAttention(d_model=2)
output = self_attention(x)
print(output)
```

---

## Attention Types

### Self-Attention

Used in Transformer encoders, where each token attends to all others in the same sequence.

### Masked Self-Attention

Used in Transformer decoders, applying a causal mask to block access to future tokens during training.

### Cross Attention

Used in encoder-decoder architectures where the decoder attends to the encoder's output.

### Multi-Head Attention

Splits queries, keys, and values into multiple heads and combines their outputs to capture richer relationships.

---

## Acknowledgments

This project was greatly inspired by the course [Attention in Transformers: Concepts and Code in PyTorch](https://learn.deeplearning.ai/courses/attention-in-transformers-concepts-and-code-in-pytorch) on DeepLearning.AI. The course provided clear conceptual explanations and hands-on coding insights that guided the development of all the code in this repository.

---

## Future Improvements

Planned enhancements to this repository include:

- [ ] Support for batching and padding masks
- [ ] Positional encoding integration
- [ ] Full Transformer block (including feedforward and normalization layers)
- [ ] Attention visualization tools
- [ ] Unit tests and performance benchmarking

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Seif Yasser** (2025)

Feel free to open issues or contribute pull requests. Suggestions and improvements are always welcome!
