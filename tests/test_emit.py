from copy import deepcopy

import torch
from emitopt.utils import compute_emit_bmag_thick_quad

test_emit_data = {'k_x': torch.tensor([-9.0134, -6.4382, -3.8629, -1.2876,  1.2876,  3.8629,  6.4382,  9.0134]),
                 'xrms': torch.tensor([9.75170701e-04, 7.83466195e-04, 5.94470216e-04, 4.08506756e-04,
                                       2.26977920e-04, 6.80165811e-05, 1.53504080e-04, 3.25192128e-04]),
                 'yrms': torch.tensor([0.00041552, 0.00016268, 0.00014606, 0.00040372, 0.00067622,
                                       0.00095402, 0.00123602, 0.00152192]),
                  'rmat_x': torch.tensor([[1.0000, 2.2000],
                                        [0.0000, 1.0000]]),
                  'rmat_y': torch.tensor([[1.0000, 2.2000],
                                        [0.0000, 1.0000]]),
                  'q_len': 0.1244,
                  'beta0_x': 5.01,
                  'beta0_y': 5.01,
                  'alpha0_x': 0.049,
                  'alpha0_y': 0.049,
                  'emit_x': torch.tensor(6.3876171836832266e-09), 
                  'bmag_x': torch.tensor(1.1310103128120694),
                  'emit_y': torch.tensor(1.2775234367366453e-08),
                  'bmag_y': torch.tensor(1.531975088602168),
                 }


class TestEmitCalc:
    # single fit calc (non-bayesian)
    def test_emit_fit(self):
        input_dicts = [{'k': test_emit_data['k_x'],
         'y_batch': test_emit_data['xrms']**2,
         'q_len': test_emit_data['q_len'],
         'rmat_quad_to_screen': test_emit_data['rmat_x'],
         'beta0': test_emit_data['beta0_x'],
         'alpha0': test_emit_data['alpha0_x']},
         {'k': -1*test_emit_data['k_x'],
         'y_batch': test_emit_data['yrms']**2,
         'q_len': test_emit_data['q_len'],
         'rmat_quad_to_screen': test_emit_data['rmat_y'],
         'beta0': test_emit_data['beta0_y'],
         'alpha0': test_emit_data['alpha0_y']}
        ]
        # proper outputs taken from Pyemittance results
        proper_outputs = [{'emit': test_emit_data['emit_x'],
                          'bmag': test_emit_data['bmag_x']},
                         {'emit': test_emit_data['emit_x'],
                          'bmag': test_emit_data['bmag_x']}]
        for input_dict, proper_output in zip(input_dicts, proper_outputs):
            emit, bmag, sig, is_valid = compute_emit_bmag_thick_quad(k=test_emit_data['k_x'], 
                                                                     y_batch=test_emit_data['xrms']**2, 
                                                                     q_len=test_emit_data['q_len'], 
                                                                     rmat_quad_to_screen=test_emit_data['rmat_x'],
                                                                     beta0=test_emit_data['beta0_x'], 
                                                                     alpha0=test_emit_data['alpha0_x']
                                                                    )
            assert torch.isclose(emit, proper_output['emit'].double(), rtol=1e-02)
            assert torch.isclose(bmag, proper_output['bmag'].double(), rtol=1e-02)
