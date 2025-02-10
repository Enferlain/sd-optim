## Merge method docs

## multi_domain_alignment:
Merges tensors A and B using multi-domain alignment with anchor guidance.

This method combines spatial and frequency domain information to create more 
robust merges, using the anchor tensor C as a reference point. 

Key Steps:
1. Frequency-Selective Alignment: Aligns the frequency distributions of A and B, guided
   by the anchor C, to preserve global features.
2. Cross-Attention (Optional): Calculates feature importance weights using 
   cross-attention between A, B, and C to emphasize consistently important features.
3. Dissimilarity Calculation: Measures the spatial dissimilarity between A and B, 
   guided by the anchor C.
4. Adaptive Interpolation: Merges A and B using slerp interpolation, with alpha values
   adaptively adjusted based on feature importance and dissimilarity.
5. Anchor Adjustment: Fine-tunes the merged tensor towards the anchor C to enhance 
   consistency and preserve anchor characteristics.

### Args:
    a (Tensor): The first tensor to merge.
    b (Tensor): The second tensor to merge.
    c (Tensor): The anchor tensor, used as a reference for alignment and adjustment.
    alpha (float): The base interpolation factor for slerp (0 <= alpha <= 1).
    beta (float): The strength of the anchor adjustment (0 <= beta <= 1).
    kernel_size (int): Size of the Gaussian kernel for smoothing the dissimilarity map (must be odd).
    centroid_margin_factor (float): Controls the width of the transition zone between 
                                 frequency bands during alignment.
    frequency_weight (float): Weight given to the frequency-domain contribution when 
                             combining aligned and original tensors (0 <= weight <= 1).
    use_cross_attention (bool): Whether to use cross-attention to calculate feature 
                                importance weights.

    Returns:
        Tensor: The merged tensor, combining the characteristics of A and B while guided
                by the anchor C.

Recommendations:

#### For stable, conservative merges:
    kernel_size=3
    centroid_margin_factor=0.1
    frequency_weight=0.3

#### For detail preservation:
    kernel_size=3
    centroid_margin_factor=0.08
    frequency_weight=0.4

#### For smoother blending:
    kernel_size=5
    centroid_margin_factor=0.15
    frequency_weight=0.25

basically project A onto B, optionally using C as the foundational base to create deltas from.

1. Build low rank version of model, then amplify the original model by using that low rank version A_diff = A - A_lowrank
2. Project the A diff onto the orthogonal base of model B
3. Add onto model B

## ties_sum_with_dropout

#### For Similar Models (e.g., models from same family/fine-tuning):
    probability: [0.85, 0.95]  # High keep rate since weights are likely meaningful
    della_eps: [0.05, 0.15]    # Smaller adjustments needed since models are similar
    rescale: [0.9, 1.1]        # Gentle rescaling since we expect coherent results
    k: 0.2                     # Standard TIES threshold
    vote_sgn: 0.0              # Use actual values for voting
    apply_stock: 0.0           # Stock averaging not critical for similar models
    apply_median: 1.0          # Geometric median helps with outliers
    eps: 1e-6                  # Standard numerical stability

#### For Different Models (e.g., different architectures/training):
    probability: [0.7, 0.85]   # Lower keep rate to be more selective
    della_eps: [0.15, 0.25]    # Larger magnitude-based adjustments to prefer strong weights
    rescale: [0.8, 1.2]        # Wider rescaling range to handle varying distributions
    k: 0.25                    # Slightly higher threshold to be more selective
    vote_sgn: 0.0              # Still use actual values
    apply_stock: 1.0           # Enable stock averaging to handle different distributions
    apply_median: 1.0          # Definitely use geometric median for different models
    eps: 1e-5                  # Slightly more permissive for numerical stability

#### Regarding `della_eps`:
- Positive values (recommended most cases):
  - Increases keep probability for high-magnitude weights
  - Decreases keep probability for low-magnitude weights
  - Good for preserving model structure and important features
  - Example: della_eps = 0.1 means up to ±10% adjustment to keep probabilities


- Negative values (specialized cases):
  - Reverses the magnitude-based selection
  - Could be useful when you want to:
    1. Encourage exploration of alternative pathways in the network
    2. Reduce dominance of very strong weights that might be overfitted
    3. Balance models where one has systematically larger magnitudes
  - I'd suggest small negative values if used: [-0.1, -0.05]
  - Use with higher base probability to maintain stability

#### For practical use:
1. Start with similar model settings
2. If results are too noisy: increase probability, decrease della_eps
3. If results are too conservative: decrease probability, increase della_eps

## ties_sum_extended
Memory-efficient TIES (Top-K Importance-based Ensemble Selection) implementation with optimized chunking.

#### Parameters:

    models : Tensor | LiftFlag[MergeSpace.DELTA]
        Input model tensors or delta flags to be merged. Must provide at least one model.
    
    k : Hyper, default=0.218
        The proportion of parameters to keep (1-k is the actual filter threshold).
        Range: [0, 1]. Higher values retain more parameters.
    
    vote_sgn : Hyper, default=0.0
        Controls voting mechanism:
        - If <= 0.0: Uses actual parameter values for voting
        - If > 0.0: Uses only parameter signs for voting
        Best used when parameter magnitudes vary significantly between models.
    
    apply_stock : Hyper, default=0.0
        Controls model stock computation:
        - If <= 0.0: Disabled
        - If > 0.0: Enables model stock calculation using cosine similarity
        Useful when dealing with potentially contradictory updates.
    
    cos_eps : Hyper, default=1e-6
        Epsilon value for cosine similarity calculation stability.
        Only relevant when apply_stock > 0.0.
    
    apply_median : Hyper, default=1.0
        Controls merge strategy:
        - If <= 0.0: Uses weighted average
        - If > 0.0: Uses geometric median
        Geometric median is more robust to outliers but computationally intensive.
    
    eps : Hyper, default=1e-6
        Small constant for numerical stability in various calculations.
    
    maxiter : Hyper, default=150
        Maximum number of iterations for geometric median computation.
        Only relevant when apply_median > 0.0.
    
    ftol : Hyper, default=1e-22
        Convergence tolerance for geometric median iteration.
        Only relevant when apply_median > 0.0.
    
    weight_decay : Hyper, default=0.0218
        Parameter decay factor applied to filtered values.
        Range: [0, 1]. Higher values cause stronger decay.
    
    min_agreement : Hyper, default=0.3
        Minimum proportion of models that must agree on parameter sign.
        Range: [0, 1]. Higher values require stronger consensus.
    
    chunk_size : int, default=16
        Number of models to process simultaneously for memory efficiency.
        Adjust based on available GPU memory.

#### Regarding the key parameters usage:

1. `vote_sgn`:
- Use > 0.0 when:
  - Models come from different training conditions/architectures
  - Parameter magnitudes vary significantly
  - You want to focus on directional agreement rather than magnitude
- Keep at 0.0 when:
  - Models are from similar conditions
  - Parameter magnitudes are comparable
  - You want to preserve magnitude information

2. `apply_stock`:
- Use > 0.0 when:
  - Dealing with potentially contradictory updates
  - Models might have conflicting parameter changes
  - You want to detect and handle model disagreement
- Keep at 0.0 when:
  - Models are expected to be mostly aligned
  - You want faster computation
  - Memory is constrained

3. `apply_median`:
- Use 1.0 (default) when:
  - There might be outlier models
  - Models could be corrupted or adversarial
  - Robustness is more important than speed
- Set to 0.0 when:
  - All models are trusted
  - Speed is crucial
  - Models are known to be well-behaved
