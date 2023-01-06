from collections import deque

import numpy as np
import pytest
import torch

from Starfish.models import (
    find_residual_peaks,
    optimize_residual_peaks,
    covariance_debugger,
)


class TestUtils:
    @pytest.fixture
    def model(self, mock_model):
        fake_residual = torch.DoubleTensor(np.random.randn(100, *mock_model.data.wave.shape))
        fake_residual[:, 10] = 1000
        mock_model.residuals = deque(fake_residual.tolist())
        yield mock_model

    def test_find_residuals_peaks(self, model):
        peaks = find_residual_peaks(model, num_residuals=50)
        assert model.data.wave[10] == peaks[0]
        assert len(peaks) < len(model.data.wave) - 2

    def test_optimize_residual_peaks(self, model):
        peaks = find_residual_peaks(model, num_residuals=50)
        params = optimize_residual_peaks(model, mus=peaks, num_residuals=50)
        assert len(params) == len(peaks)
        model.params["local_cov"].update(params)
        assert tuple(sorted([(k, v.item()) for k, v in model.params["local_cov"][0].items()])) == tuple(sorted(params[0].items()))

    def test_covariance_debugger(self):
        # Just test if it runs
        cov = np.eye(100)
        covariance_debugger(cov)
