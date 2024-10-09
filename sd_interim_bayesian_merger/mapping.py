# for methods that require selective optimization, ie contains on/off hypers, learning rate hypers, and such

OPTIMIZABLE_HYPERPARAMETERS = {
    "lu_merge": ["alpha", "theta"],
    "orth_pro": ["alpha"],
    "clyb_merge": ["alpha"],
    "ties_sum_extended": ["k"],
    "ties_sum_with_dropout": ["probability", "della_eps"],
    "slerp_norm_sign": ["alpha"],
    # ... other methods and their optimizable hyperparameters
}