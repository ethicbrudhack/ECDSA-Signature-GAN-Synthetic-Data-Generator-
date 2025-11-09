import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Parametry GAN
LATENT_DIM = 16  # Wymiar ukrytej przestrzeni
SIGNATURE_DIM = 3  # r, s, z
BATCH_SIZE = 32
EPOCHS = 5000
LR = 0.0002

# Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(LATENT_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, SIGNATURE_DIM)  # r, s, z
        )

    def forward(self, z):
        return self.model(z)

# Dyskryminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(SIGNATURE_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Inicjalizacja modeli
generator = Generator()
discriminator = Discriminator()

# Optymalizatory
optim_G = optim.Adam(generator.parameters(), lr=LR)
optim_D = optim.Adam(discriminator.parameters(), lr=LR)

# Funkcja straty
criterion = nn.BCELoss()

# Dane treningowe (pełne wartości r, s, z)
real_signatures = torch.tensor([
    [0xd8e2d92d3fca2a3293ed2e57c80a8db40069da2229225756b77de2f967baa1fb,
     0x6f2dc5ce39475b4c98ae27285a36939aadf19e38b3845c57400ef08326d24d23,
     0xcc5260cf9f0c439f2847dae4560a63f62da6fb6682ed77df872076f0f0aafd34],

    [0x5ebecec888b158797ded9ebc1421b4797d4077c2e16945f45361ac33f6abf41b,
     0x340050758fd9de606d45383f63f1b236a7a47318c595e99c910f4b943a88a364,
     0x5429e50aa800fe787d59bc03594476c704c86ce7b58060025ffe9ee6c2658273],

    [0xc1c83fb6cf745bf4eb518b4683dadb2e6eeab031fde8f7f27ff0da49a182d317,
     0x044812973948efef2db516c93f7eb4ee8d224ccc0181d3794fc3704ae3324a8b,
     0xac6ede455f205ac41a75ce9f1a88cc625a11a3e6b377531096074cbcdbf97a67],
], dtype=torch.float32)

# Trening GAN
for epoch in range(EPOCHS):
    # === TRENING DYSKRYMINATORA ===
    real_labels = torch.ones(BATCH_SIZE, 1)
    fake_labels = torch.zeros(BATCH_SIZE, 1)

    real_samples = real_signatures[torch.randint(0, len(real_signatures), (BATCH_SIZE,))]
    real_pred = discriminator(real_samples)
    loss_real = criterion(real_pred, real_labels)

    z_noise = torch.randn(BATCH_SIZE, LATENT_DIM)
    fake_samples = generator(z_noise)
    fake_pred = discriminator(fake_samples.detach())
    loss_fake = criterion(fake_pred, fake_labels)

    loss_D = loss_real + loss_fake
    optim_D.zero_grad()
    loss_D.backward()
    optim_D.step()

    # === TRENING GENERATORA ===
    z_noise = torch.randn(BATCH_SIZE, LATENT_DIM)
    fake_samples = generator(z_noise)
    fake_pred = discriminator(fake_samples)
    loss_G = criterion(fake_pred, real_labels)

    optim_G.zero_grad()
    loss_G.backward()
    optim_G.step()

    # Co 100 epok raportujemy postęp
    if epoch % 100 == 0:
        print(f"Epoch {epoch} | Loss D: {loss_D.item()} | Loss G: {loss_G.item()}")

# Generowanie nowych podatnych podpisów
z_noise = torch.randn(10, LATENT_DIM)
generated_signatures = generator(z_noise).detach().numpy()
print("\n🔄 Wygenerowane nowe podatne podpisy (r, s, z):")
print(generated_signatures)