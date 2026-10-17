# Method summary

The full derivations are in [`paper/main.pdf`](../paper/main.pdf), and the proofs are in
[`paper/supplement.pdf`](../paper/supplement.pdf).

## Stage 1: semi-relaxed unbalanced OT aggregation ([`src/aggregators.py`](../src/aggregators.py))

- **Inputs.** Patch tokens tᵢ (n = 300) and a class token come from DINOv2 ViT-B/14. Two MLPs give
  cluster scores sᵢ ∈ ℝᵐ (m = 64) and features fᵢ ∈ ℝˡ (l = 128). A learnable dust-bin score z completes the
  score matrix S̄ = [S | z].
- **Problem.** Tokens carry unit mass, enforced hard. Cluster and dust-bin capacities b = (1ₘ, n − m) are
  only softly enforced, through a KL penalty τ:

  P^τ = argmax_{P ≥ 0, P1 = 1} ⟨S̄, P⟩ + εH(P) − τ·KL(Pᵀ1 ‖ b)

- **Algorithm** (`ot_assign`). This is log-domain Sinkhorn with the cluster-side update scaled by
  λ = τ/(τ + ε). λ = 1 gives SALAD (balanced OT) and λ = 0 gives the NetVLAD softmax with a dust-bin.
  Galatea learns λ through `mode='learn'`.
- **Aggregation.** V_{jk} = Σᵢ P_{ij} f_{ik}. Each cluster row is intra-normalised, the result is
  concatenated with the 256-d scene token, and the whole vector is L2-normalised (8448-d in total).

| Proposition | Statement | Practical consequence |
|---|---|---|
| 1. Structure, limits | The solution is P = exp((S̄ + f ⊕ g)/ε). λ → 0 gives NetVLAD; λ → 1 gives balanced OT. | One parameter covers both families. |
| 2. Linear convergence | The truncated iterations contract at rate λ in ‖·‖∞, independent of ε, n and m. | A few iterations (L = 3) suffice. |
| 3. Dust-bin gating | The dust-bin share of token i is a sigmoid of z + g_{m+1} − ε·LSE_j((s_ij + g_j)/ε). | Low-affinity tokens are suppressed. |
| 4. Column-scale invariance | After intra-normalisation, the descriptor depends on P only through per-token weights wᵢ = e^{fᵢ/ε}. | Explains why λ barely matters empirically. |
| 5. FGW invariances | The structural term is invariant to similarity transforms and permutations. | No camera model is needed for re-ranking. |

## Stage 2: partial fused Gromov–Wasserstein re-ranking ([`src/rerank.py`](../src/rerank.py))

- **Inputs.** SuperPoint keypoints (up to 384) for the query and for each of the top-K = 10 candidates.
- **Costs.** The appearance cost is M = 1 − ⟨d, e⟩. The structure matrices are mean-normalised intra-image
  distance matrices.
- **Solver.** Entropic mirror descent: 5 outer steps of dust-bin Sinkhorn on (1 − α)M + α·L(C_q, C_c) ⊗ T,
  with α = 0.3 and ε = 0.05.
- **Score.** The local score s_ℓ is the number of mutual arg-max matches with confidence above 0.2. MAGSAC++
  verification is optional.
- **Fusion** ([`src/analyze_rerank.py`](../src/analyze_rerank.py)). The final score is
  s = s_g + β·s_ℓ/(s_ℓ + κ), with β and κ tuned on the validation scene.

## Training ([`src/train_eval.py`](../src/train_eval.py))

- **Backbone and loss.** The DINOv2 backbone is frozen (SALAD's GSV-Cities fine-tuned weights). Training
  uses the multi-similarity loss with its miner, and different places closer than 15 m are masked out of the
  negatives.
- **Optimiser.** AdamW at learning rate 4e-4 with a one-cycle schedule, for 500 steps of 24 places × 4 views.
