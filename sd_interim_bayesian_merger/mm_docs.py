## Merge method docs

MERGE_METHOD_DOCS = {
    "multi_domain_alignment": """
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

    Args:
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
    """,

    "clyb_merge": """
    clyb_merge
    
    basically project A onto B, optionally using C as the foundational base to create deltas from.
    
    1. Build low rank version of model, then amplify the original model by using that low rank version A_diff = A - A_lowrank
    2. Project the A diff onto the orthogonal base of model B
    3. Add onto model B
    """
}