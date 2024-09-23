from hyperopt import STATUS_OK, Trials, atpe, fmin, hp

from sd_interim_bayesian_merger.optimizer import Optimizer

class ATPEOptimizer(Optimizer):
    def _target_function(self, params):
        res = self.sd_target_function(**params)
        return {
            "loss": -res + 10,
            "status": STATUS_OK,
            "params": params,
        }

    def optimize(self) -> None:
        bounds = self.init_params()
        space = {p: hp.uniform(p, *b) for p, b in bounds.items()}

        self.trials = Trials()
        # tpe._default_n_startup_jobs = self.cfg.init_points
        # algo = partial(tpe.suggest, n_startup_jobs=self.cfg.init_points)
        fmin(
            self._target_function,
            space=space,
            algo=atpe.suggest,
            trials=self.trials,
            max_evals=self.cfg.init_points + self.cfg.n_iters,
        )

    def postprocess(self) -> None:
        print("\nRecap!")
        scores = []
        for i, res in enumerate(self.trials.losses()):
            print(f"Iteration {i} loss: \n\t{res}")
            scores.append(res)

        # Remove unnecessary assignments:
        # best = self.trials.best_trial
        # best_weights = best["result"]["params"]
        # best_bases = best["result"]["params"]

        self.artist.visualize_optimization()  # Call the Artist's visualize_optimization method