# 🔐 ECDSA Signature GAN – Synthetic Data Generator (PyTorch)

This project demonstrates how a **Generative Adversarial Network (GAN)** can be trained to generate
synthetic ECDSA-like signature data (`r`, `s`, `z`) using **PyTorch**.  
It’s an experimental, educational example showing how neural networks can learn distributions
of cryptographic-style numerical data — **not** a real cryptanalysis or attack tool.

──────────────────────────────────────────────────────────────
⚙️  OVERVIEW

The script implements a minimal GAN framework:
- **Generator (G)** creates artificial signature triplets (`r`, `s`, `z`) from random noise.
- **Discriminator (D)** learns to distinguish between real and fake signatures.
- Both models are trained together in an adversarial setup until G learns to mimic real data.

The dataset consists of a few pre-defined ECDSA-like numeric triplets,
each representing realistic (but arbitrary) signature components.

──────────────────────────────────────────────────────────────
🧠  MODEL ARCHITECTURE

**Generator**
Input: latent vector (size 16)
→ Linear(16 → 64)
→ ReLU
→ Linear(64 → 128)
→ ReLU
→ Linear(128 → 3) → Output: [r, s, z]


**Discriminator**


Input: [r, s, z]
→ Linear(3 → 128)
→ ReLU
→ Linear(128 → 64)
→ ReLU
→ Linear(64 → 1)
→ Sigmoid (real/fake probability)


──────────────────────────────────────────────────────────────
📊  TRAINING LOOP

1. **Train the Discriminator (D):**
   - Feed it both real and fake samples.
   - Maximize its ability to classify them correctly.

2. **Train the Generator (G):**
   - Generate fake samples and feed them to D.
   - Update G to “fool” D into thinking they are real.

3. **Loss Function:**
   - Binary Cross-Entropy (BCE Loss) used for both D and G.

4. **Optimizer:**
   - Adam optimizer with learning rate `0.0002`.

5. **Training Parameters:**


LATENT_DIM = 16
SIGNATURE_DIM = 3
BATCH_SIZE = 32
EPOCHS = 5000


──────────────────────────────────────────────────────────────
🚀  USAGE

Run the script directly:

```bash
python ecdsa_signature_gan.py


During training, the console prints progress:

Epoch 0 | Loss D: 1.39 | Loss G: 0.68
Epoch 100 | Loss D: 1.12 | Loss G: 0.89
...


After training, the generator produces synthetic data:

🔄 Generated synthetic ECDSA-like signatures (r, s, z):
[[1.24e+08, 3.19e+07, 9.99e+07],
 [2.33e+08, 4.56e+08, 1.12e+09],
 ...]


──────────────────────────────────────────────────────────────
📦 DEPENDENCIES

Install via pip:

pip install torch numpy


──────────────────────────────────────────────────────────────
🧩 PURPOSE

This example shows how GANs can learn numeric relationships between elements of structured data,
such as digital signatures.
It can be used for:

Data augmentation in research simulations.

Educational purposes in machine learning.

Demonstrating adversarial training in PyTorch.

──────────────────────────────────────────────────────────────
⚠️ DISCLAIMER

This project does not perform any real cryptographic attacks and does not recover private keys.
The signatures used are arbitrary numerical examples for training purposes only.
Use this code for learning and research only, and never for unauthorized cryptographic analysis.

──────────────────────────────────────────────────────────────
👨‍💻 AUTHOR

ECDSA Synthetic Data GAN (PyTorch Implementation)
Created as an educational experiment to explore adversarial learning for numeric data generation.

BTC donation address: bc1q4nyq7kr4nwq6zw35pg0zl0k9jmdmtmadlfvqhr
