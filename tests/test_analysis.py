from copy import deepcopy

import torch
import yaml
from emitopt.analysis import compute_emit_bmag, compute_emit_bayesian
from emitopt.resources.testing import TEST_EMIT_YAML


class TestAnalysis:
    # single fit calc (non-bayesian)
    def test_compute_emit_bmag(self):
        test_emit_data = yaml.safe_load(TEST_EMIT_YAML)['values']
        # generate input args for x and y emit calc
        input_dicts = [{'k': torch.tensor(test_emit_data['k_x']),
         'beamsize_squared': torch.tensor(test_emit_data['xrms'])**2,
         'q_len': torch.tensor(test_emit_data['q_len']),
         'rmat': torch.tensor(test_emit_data['rmat_x']),
         'beta0': torch.tensor(test_emit_data['beta0_x']),
         'alpha0': torch.tensor(test_emit_data['alpha0_x'])},
         {'k': torch.tensor(test_emit_data['k_y']),
         'beamsize_squared': torch.tensor(test_emit_data['yrms'])**2,
         'q_len': torch.tensor(test_emit_data['q_len']),
         'rmat': torch.tensor(test_emit_data['rmat_y']),
         'beta0': torch.tensor(test_emit_data['beta0_y']),
         'alpha0': torch.tensor(test_emit_data['alpha0_y'])}
        ]
        # "proper" outputs taken from PyEmittance results
        ground_truth_outputs = [{'emit': torch.tensor(test_emit_data['emit_x']),
                          'bmag': torch.tensor(test_emit_data['bmag_x'])},
                         {'emit': torch.tensor(test_emit_data['emit_y']),
                          'bmag': torch.tensor(test_emit_data['bmag_y'])}]
        for input_dict, ground_truth_output in zip(input_dicts, ground_truth_outputs):
            emit, bmag, sig, is_valid = compute_emit_bmag(**input_dict)
            
            assert torch.isclose(emit, ground_truth_output['emit'].double(), rtol=1e-02)
            assert torch.isclose(bmag, ground_truth_output['bmag'].double(), rtol=1e-02)
    
    def test_compute_emit_bmag_batched(self):
        test_emit_data = yaml.safe_load(TEST_EMIT_YAML)['values']

        q_len = test_emit_data['q_len']

        rmat_x = torch.tensor(test_emit_data['rmat_x'])
        rmat_y = torch.tensor(test_emit_data['rmat_y'])
        rmat = torch.stack((rmat_x, rmat_y))
        
        k_x = torch.tensor(test_emit_data['k_x'])
        k_y = torch.tensor(test_emit_data['k_y'])
        k = torch.stack((k_x, k_y))

        y_x = torch.tensor(test_emit_data['xrms'])
        y_y = torch.tensor(test_emit_data['yrms'])
        y = torch.stack((y_x, y_y))
        
        beta0_x = torch.tensor([test_emit_data['beta0_x']])
        beta0_y = torch.tensor([test_emit_data['beta0_y']])
        beta0 = torch.stack((beta0_x, beta0_y))
        
        alpha0_x = torch.tensor([test_emit_data['alpha0_x']])
        alpha0_y = torch.tensor([test_emit_data['alpha0_y']])
        alpha0 = torch.stack((alpha0_x, alpha0_y))
        
        emit, bmag, sig, is_valid = compute_emit_bmag(k,
                                                      y**2,
                                                      q_len,
                                                      rmat,
                                                      beta0=beta0,
                                                      alpha0=alpha0)
        
        emit_x_true = torch.tensor(test_emit_data['emit_x'])
        emit_y_true = torch.tensor(test_emit_data['emit_y'])
        emit_true = torch.stack((emit_x_true, emit_y_true))
        
        bmag_x_true = torch.tensor(test_emit_data['bmag_x'])
        bmag_y_true = torch.tensor(test_emit_data['bmag_y'])
        bmag_true = torch.stack((bmag_x_true, bmag_y_true))
        
        assert torch.allclose(emit, emit_true.double(), rtol=1e-02)
        assert torch.allclose(bmag, bmag_true.double(), rtol=1e-02)
        
    def test_compute_emit_bayesian_with_bmag(self):
        test_emit_data = yaml.safe_load(TEST_EMIT_YAML)['values']

        q_len = test_emit_data['q_len']

        rmat_x = torch.tensor(test_emit_data['rmat_x'])

        k_x = torch.tensor(test_emit_data['k_x'])

        y_x = torch.tensor(test_emit_data['xrms'])

        beta0_x = torch.tensor([test_emit_data['beta0_x']])

        alpha0_x = torch.tensor([test_emit_data['alpha0_x']])

        emit, bmag, sig, is_valid = compute_emit_bayesian(k_x,
                                                      y_x,
                                                      q_len,
                                                      rmat_x,
                                                      beta0=beta0_x,
                                                      alpha0=alpha0_x)

        emit_x_true = torch.tensor(test_emit_data['emit_x'])

        bmag_x_true = torch.tensor(test_emit_data['bmag_x'])

        assert (bmag.quantile(0.025) < bmag_x_true) and (bmag.quantile(0.975) > bmag_x_true)
        assert (emit.quantile(0.025) < emit_x_true) and (emit.quantile(0.975) > emit_x_true)
        
    def test_compute_emit_bayesian_without_bmag(self):
        test_emit_data = yaml.safe_load(TEST_EMIT_YAML)['values']

        q_len = test_emit_data['q_len']

        rmat_x = torch.tensor(test_emit_data['rmat_x'])

        k_x = torch.tensor(test_emit_data['k_x'])

        y_x = torch.tensor(test_emit_data['xrms'])

        beta0_x = torch.tensor([test_emit_data['beta0_x']])

        alpha0_x = torch.tensor([test_emit_data['alpha0_x']])

        emit_x_true = torch.tensor(test_emit_data['emit_x'])

        emit, bmag, sig, is_valid = compute_emit_bayesian(k_x,
                                                      y_x,
                                                      q_len,
                                                      rmat_x,
                                                      beta0=None,
                                                      alpha0=alpha0_x)

        assert bmag is None
        assert (emit.quantile(0.025) < emit_x_true) and (emit.quantile(0.975) > emit_x_true)
        
        emit, bmag, sig, is_valid = compute_emit_bayesian(k_x,
                                                      y_x,
                                                      q_len,
                                                      rmat_x,
                                                      beta0=beta0_x,
                                                      alpha0=None)
        
        assert bmag is None
        assert (emit.quantile(0.025) < emit_x_true) and (emit.quantile(0.975) > emit_x_true)
        
        emit, bmag, sig, is_valid = compute_emit_bayesian(k_x,
                                                      y_x,
                                                      q_len,
                                                      rmat_x,
                                                      beta0=None,
                                                      alpha0=None)
        
        assert bmag is None
        assert (emit.quantile(0.025) < emit_x_true) and (emit.quantile(0.975) > emit_x_true)