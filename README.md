# ResNet-based CVAE
This is an implementation of custom `ResNet-based CVAE` to generate synthetic images.

```d
Input x ∈ R^{B×1×28×28}                                  Class label c ∈ {0..9}
        │                                                          │
        │                                                          └─ one-hot(10) → Linear(10→16) → SiLU → e ∈ R^{B×16}
        │
        ├────────────── ResNet Feature Extractor (Encoder CNN) ───────────────┐
        │                                                                      │
        │  Stem: Conv 3×3, s1, p1, c=32 → GroupNorm(8) → SiLU                  │
        │    └─ out: (B,32,28,28)                                              │
        │
        │  ResBlock↓ (32→32):                                                  │
        │    Main: Conv 3×3, s2, p1, c=32 → GN → SiLU → Conv 3×3, s1, p1, c=32 │
        │    Skip: Conv 1×1, s2, p0, c=32                                      │
        │    Add → SiLU                                                        │
        │    └─ out: (B,32,14,14)                                              │
        │
        │  ResBlock↓ (32→64):                                                  │
        │    Main: Conv 3×3, s2, p1, c=64 → GN → SiLU → Conv 3×3, s1, p1, c=64 │
        │    Skip: Conv 1×1, s2, p0, c=64                                      │
        │    Add → SiLU                                                        │
        │    └─ out: (B,64,7,7)                                                │
        │
        │  ResBlock  (64→64):                                                  │
        │    Main: Conv 3×3, s1, p1, c=64 → GN → SiLU → Conv 3×3, s1, p1, c=64 │
        │    Skip: Identity                                                    │
        │    Add → SiLU                                                        │
        │    └─ out: (B,64,7,7)                                                │
        │
        │  GlobalAvgPool (7×7 → 1×1): out (B,64)                               │
        │
        ├─[concat]→ h = concat( (B,64), e:(B,16) ) → (B,80)                    │  ← (class-aware encoder head)
        │
        ├─ μ = Linear(80→z)                                                    │
        ├─ logσ² = Linear(80→z)                                                │
        │
        └─ Reparameterization: z = μ + exp(0.5·logσ²) ⊙ ε,  ε~N(0,I)  (B,z) ───┘

                                  ▼

                 ┌─────────────── Decoder CNN (ResNet Upsampling) ────────────────┐
                 │                                                                │
                 │  z’ = concat( z:(B,z), e:(B,16) ) → Linear((z+16) → 7×7×64)    │
                 │  Reshape → (B,64,7,7)                                          │
                 │
                 │  ResBlock↑ (64→64):                                            │
                 │    Up×2 → Conv 3×3, s1, p1, c=64 → GN → SiLU → Conv 3×3, s1,p1 │
                 │    Skip: Up×2 → Conv 1×1, s1, p0, c=64                         │
                 │    Add → SiLU                                                  │
                 │    └─ out: (B,64,14,14)                                        │
                 │
                 │  ResBlock↑ (64→32):                                            │
                 │    Up×2 → Conv 3×3, s1, p1, c=32 → GN → SiLU → Conv 3×3, s1,p1 │
                 │    Skip: Up×2 → Conv 1×1, s1, p0, c=32                         │
                 │    Add → SiLU                                                  │
                 │    └─ out: (B,32,28,28)                                        │
                 │
                 │  Conv 1×1, s1, p0, c=1 →                                      │
                 │     • Sigmoid  (if Bernoulli likelihood, x ∈ [0,1])            │
                 │     • Identity (if Gaussian with fixed σ, x ∈ [-1,1])          │
                 │
                 └─ Reconstruction ŷ ∈ R^{B×1×28×28} ─────────────────────────────┘

Loss:  L =  E_{q(z|x,c)}[ −log p( x | z,c ) ]  +  β · KL( q(z|x,c) || N(0,I) )
       (Use β=1 with KL warm-up; or β≈2–4 for stronger regularization)

Defaults / Hyperparams:
  • z-dim: 16 (try 32 as ablation)       • Activation: SiLU (ReLU also fine)
  • Norm: GroupNorm(8) (BatchNorm OK if large batch)
  • Optim: AdamW (lr=2e-3, wd=1e-4)      • Grad clip: 1.0
  • Batch: 128                           • Epochs: 50–100 (MNIST)
```

