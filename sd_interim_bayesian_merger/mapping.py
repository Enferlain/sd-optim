# for methods that require selective optimization, ie contains on/off hypers, learning rate hypers, and such

OPTIMIZABLE_HYPERPARAMETERS = {
    "lu_merge": ["alpha", "theta"],
    "orth_pro": ["alpha"],
    "clyb_merge": ["alpha"],
    "ties_sum_extended": ["k"],
    "ties_sum_with_dropout": ["probability", "della_eps"]
    # ... other methods and their optimizable hyperparameters
}

CUSTOM_BOUNDS = {
    "ties_sum_with_dropout": ["della_eps", "(-1.0, 1.0)"]


}