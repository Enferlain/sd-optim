## Merge method docs

clyb_merge

basically project A onto B, optionally using C as the foundational base to create deltas from.

1. Build low rank version of model, then amplify the original model by using that low rank version A_diff = A - A_lowrank
2. Project the A diff onto the orthogonal base of model B
3. Add onto model B