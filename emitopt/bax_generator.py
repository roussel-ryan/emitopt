import logging
import pickle
from copy import deepcopy
from typing import Dict

import pandas as pd
from pydantic import Field

from emitopt.acquisition import ModelListExpectedInformationGain
from xopt.generators.bayesian.bax.algorithms import Algorithm
from xopt.generators.bayesian.bayesian_generator import BayesianGenerator

logger = logging.getLogger()


class BaxGenerator(BayesianGenerator):
    alias = "BAX"
    algorithm: Algorithm = Field(description="algorithm evaluated in the BAX process")
    algorithm_results: Dict = Field(
        None, description="dictionary results from algorithm", exclude=True
    )
    algorithm_results_file: str = Field(
        None, description="file name to save algorithm results at every step"
    )

    _n_calls: int = 0

    class Config:
        underscore_attrs_are_private = True

    def generate(self, n_candidates: int) -> pd.DataFrame:
        self._n_calls += 1
        return super().generate(n_candidates)

    def _get_acquisition(self, model):
        bax_model_ids = [
            self.vocs.output_names.index(name)
            for name in self.algorithm.model_names_ordered
        ]
        bax_model = model.subset_output(bax_model_ids)
        eig = ModelListExpectedInformationGain(
            bax_model, self.algorithm, self._get_optimization_bounds()
        )
        self.algorithm_results = eig.algorithm_results
        if self.algorithm_results_file is not None:
            results = deepcopy(self.algorithm_results)

            with open(
                f"{self.algorithm_results_file}_{self._n_calls}.pkl", "wb"
            ) as outfile:
                pickle.dump(results, outfile, protocol=pickle.HIGHEST_PROTOCOL)

        return eig
