# sd-interim-bayesian-merger

Even more opinionated fork of the original repo by s1dlx currently undergoing a transformation hence the name

Since he became absent, I started updating the project with things that were missing at the time, which eventually built into getting my own ideas, which I'll be gradually introducing into this fork. 

Don't expect it to always be in a working state, but I generally try to only push when it is. Might still miss stuff tho

###### everything was done with Gemini and some advice/ideas from ljleb

### Stuff that's new:
- entirely different merge backend [mecha](https://github.com/ljleb/sd-mecha) (everyone say thank you to ljleb for boosting the merge game)
- pick what components to optimize
- group/select components/blocks in various ways, customize their bounds
- new juicer scorer
- both a1111 and forge support (comfy and swarm eventually)
- ability to skip during manual scoring

### Planned:
- more (and better) visualizations
- ~~ability to define custom bounds/behavior to a hyper of choice (optimizing false/true behaving hypers)~~
- switching between manual and automatic scoring with hotkeys
- more hotkeys for more behavior like early stopping and other qol
- adjusting batch size and payload selection during optimization
- scoring rethinking, categories, character objective, perceptual similarity metrics(lpips)
- gphedge in bayes (?) multi-objective optimization
- possibily pivot to hyperactive and others for way more optimization options
- more that I can't remember or will randomly come up with

wip text

-----------

## What is this?

An opinionated take on stable-diffusion models-merging automatic-optimisation.

The main idea is to treat models-merging procedure as a black-box model with 26 parameters: one for each block plus `base_alpha`.
We can then try to apply black-box optimisation techniques, in particular we focus on [Bayesian optimisation](https://en.wikipedia.org/wiki/Bayesian_optimization) with a [Gaussian Process](https://en.wikipedia.org/wiki/Gaussian_process) emulator.
Read more [here](https://github.com/fmfn/BayesianOptimization), [here](http://gaussianprocess.org) and [here](https://optimization.cbe.cornell.edu/index.php?title=Bayesian_optimization).

The optimisation process is split in two phases:
1. __exploration__: here we sample (at random for now, with some heuristic in the future) the 26-parameter hyperspace, our block-weights. The number of samples is set by the
`--init_points` argument. We use each set of weights to merge the two models we use the merged model to generate `batch_size * number of payloads` images which are then scored.
2. __exploitation__: based on the exploratory phase, the optimiser makes an idea of where (i.e. which set of weights) the optimal merge is.
This information is used to sample more set of weights `--n_iters` number of times. This time we don't sample all of them in one go. Instead, we sample once, merge the models,
generate and score the images and update the optimiser knowledge about the merging space. This way the optimiser can adapt the strategy step-by-step.

At the end of the exploitation phase, the set of weights scoring the highest score are deemed to be the optimal ones.

## OK, How Do I Use It In Practice?

Head to the [wiki](https://github.com/s1dlx/sd-webui-bayesian-merger/wiki/Home) for all the instructions to get you started.

## With the help of

- [sdweb-merge-block-weighted-gui](https://github.com/bbc-mc/sdweb-merge-block-weighted-gui)
- [sd-webui-supermerger](https://github.com/hako-mikan/sd-webui-supermerger)
- [sdweb-auto-MBW](https://github.com/Xerxemi/sdweb-auto-MBW)
- [SD-Chad](https://github.com/grexzen/SD-Chad.git)
