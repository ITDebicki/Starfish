import os
from datetime import datetime
import textwrap

from flatdict import FlatterDict
import pytest
import numpy as np
import scipy.stats as st

import torch

from Starfish.models import SpectrumModel


class TestSpectrumModel:

    GP = [6000, 4.0, 0]

    def test_param_dict(self, mock_model):
        assert mock_model["T"].item() == self.GP[0]
        assert mock_model["logg"].item() == self.GP[1]
        assert mock_model["Z"].item() == self.GP[2]
        assert mock_model["vz"].item() == 0
        assert mock_model["Av"].item() == 0
        assert mock_model["log_scale"].item() == -10
        assert mock_model["vsini"].item() == 30
        assert torch.all(torch.cat(mock_model["cheb"]) == torch.DoubleTensor([0.1, -0.2]))

    def test_create_from_strings(self, mock_spectrum, mock_trained_emulator, tmpdir):
        tmp_emu = os.path.join(tmpdir, "emu.hdf5")
        mock_trained_emulator.save(tmp_emu)
        tmp_data = os.path.join(tmpdir, "data.hdf5")
        mock_spectrum.name = "test"
        mock_spectrum.save(tmp_data)

        model = SpectrumModel(tmp_emu, grid_params=[6000, 4.0, 0.0], data=tmp_data)
        for key in mock_trained_emulator.hyperparams.keys(True):
            assert np.isclose(
                mock_trained_emulator.hyperparams[key], model.emulator.hyperparams[key]
            )
        assert model.data_name == mock_spectrum.name

    def test_cheb_coeffs_index(self, mock_model):
        cs = list(sorted(filter(lambda k: k.startswith("cheb"), mock_model.params)))
        assert cs[0] == "cheb:1"
        assert cs[1] == "cheb:2"

    def test_cheb_coeffs_setindex(self, mock_model):
        mock_model["cheb"] = [-0.2, 0.1]
        assert mock_model["cheb:1"] == -0.2
        assert mock_model["cheb:2"] == 0.1

        with pytest.raises(KeyError):
            mock_model["cheb:0"] = 1

    def test_global_cov_param_dict(self, mock_model):
        assert "log_amp" in mock_model["global_cov"]
        assert "log_ls" in mock_model["global_cov"]
        assert "global_cov:log_amp" in mock_model.params

    def test_local_cov_param_dict(self, mock_model):
        assert len(mock_model.params["local_cov"]) == 2
        assert mock_model["local_cov:0:mu"] == 1e4
        assert "log_sigma" in mock_model["local_cov"]["1"]
        assert "local_cov:0:log_amp" in mock_model.params
        assert "local_cov:1:mu" in mock_model.params

    @pytest.mark.parametrize("param", ["global_cov:log_amp", "local_cov:0:log_amp"])
    def test_cov_freeze(self, mock_model, param):
        assert param in mock_model.labels
        mock_model.freeze(param)
        assert param not in mock_model.labels
        mock_model.thaw(param)
        assert param in mock_model.labels

    @pytest.mark.parametrize(
        "param",
        ["garbage", "global_cov:not quite", "global_cov:garbage", "local_cov:garbage"],
    )
    def test_add_bad_param(self, mock_model, param):
        with pytest.raises(KeyError):
            mock_model[param] = -4

    def test_labels(self, mock_model):
        assert sorted(mock_model.labels) == sorted(
            tuple(mock_model.params.keys(True, False))
        )

    def test_grid_params(self, mock_model):
        assert torch.all(mock_model.grid_params == torch.DoubleTensor(self.GP))

    def test_transform(self, mock_model):
        flux, cov = mock_model()
        assert cov.shape == (len(flux), len(flux))
        assert flux.shape == mock_model.data.wave.shape

    def test_freeze_vsini(self, mock_model):
        mock_model.freeze("vsini")
        assert "vsini" not in mock_model.labels

    def test_freeze_grid_param(self, mock_model):
        mock_model.freeze("logg")
        assert "T" in mock_model.labels
        assert "Z" in mock_model.labels
        assert "logg" not in mock_model.labels

    def test_freeze_thaw(self, mock_model):
        pre = mock_model["logg"]
        mock_model.freeze("logg")
        assert "logg" not in mock_model.labels
        mock_model.thaw("logg")
        assert "logg" in mock_model.labels
        assert mock_model.grid_params[1] == pre

    def test_freeze_thaw_many(self, mock_model):
        labels = ["global_cov:log_amp", "global_cov:log_ls"]
        mock_model.freeze(labels)
        assert all([x not in mock_model.labels for x in labels])
        mock_model.thaw(labels)
        assert all([x in mock_model.labels for x in labels])

    def test_setitem(self, mock_model):
        # Clear params
        original, mock_model.params = mock_model.params, FlatterDict()

        for key, value in original.items():
            mock_model[key] = value

        assert mock_model.params.values() == original.values()

    def test_cheb_skip_idx(self, mock_model):
        # add coeff out of order
        mock_model["cheb:4"] = 0.05

        assert list(mock_model.cheb) == [0.1, -0.2, 0, 0.05]
        assert mock_model["cheb:3"] == 0
        assert mock_model["cheb:4"] == 0.05

    def test_save_load(self, mock_model, tmpdir):
        path = os.path.join(tmpdir, "model.toml")
        P0 = mock_model.params
        mock_model.save(path)
        mock_model.load(path)
        assert sorted(P0.items()) == sorted(mock_model.params.items())

    def test_save_load_frozen(self, mock_model, tmpdir):
        path = os.path.join(tmpdir, "model.toml")
        to_freeze = ["logg", "vsini", "global_cov"]
        mock_model.freeze(to_freeze)
        P0 = mock_model.params
        f_0 = mock_model.params.frozen()
        mock_model.save(path)
        mock_model.load(path)
        assert P0 == mock_model.params
        assert f_0 == mock_model.params.frozen()

    def test_save_load_numpy(self, mock_model, tmpdir):
        """
        In TOML library numpy.float64(32/16) do not get saved as floats but as strings. This checks that
        it is correctly handled.
        """
        path = os.path.join(tmpdir, "model.toml")
        P0 = mock_model.params
        f_0 = mock_model.params.frozen()

        mock_model.save(path)
        mock_model.load(path)
        assert P0 == mock_model.params
        assert f_0 == mock_model.params.frozen()

    def test_save_load_meta(self, mock_model, tmpdir):
        path = os.path.join(tmpdir, "model.toml")
        P0 = mock_model.params
        metadata = {"name": "Test Model", "date": datetime.today()}
        mock_model.save(path, metadata=metadata)
        # Check metadata was written
        with open(path, "r") as fh:
            lines = fh.readlines()
        assert "[metadata]\n" in lines
        assert 'name = "Test Model"\n' in lines
        mock_model.load(path)
        assert P0 == mock_model.params

    def test_log_likelihood(self, mock_model):
        lnprob = mock_model.log_likelihood()
        assert np.isfinite(lnprob)
        flux, cov = mock_model()
        mock_model.data._flux = flux
        exact_lnprob = mock_model.log_likelihood()
        assert lnprob < exact_lnprob

    def test_str(self, mock_model):
        mock_model.freeze("logg")
        expected = textwrap.dedent(
            f"""
            SpectrumModel
            -------------
            Data: {mock_model.data_name}
            Emulator: {mock_model.emulator.name}
            Log Likelihood: {mock_model.log_likelihood().item()}

            Parameters
              cheb: [0.1, -0.2]
              local_cov:
                0: log_sigma: 2.0, log_amp: 2.0, mu: 10000.0
                1: log_sigma: 2.0, log_amp: 1.5, mu: 13000.0
              global_cov:
                log_ls: 1.0
                log_amp: 1.0
              vsini: 30.0
              log_scale: -10.0
              Av: 0.0
              vz: 0.0
              T: 6000.0
              Z: 0.0
            
            Frozen Parameters
              logg: 4.0
            """
        ).strip()
        assert str(mock_model) == expected

    def test_freeze_thaw_all(self, mock_model):
        params = mock_model.labels
        mock_model.freeze("all")
        assert set(params) == set(
            mock_model.params.frozen(True)
        )
        mock_model.thaw("all")
        assert set(params) == set(mock_model.labels)

    def test_freeze_thaw_global(self, mock_model):
        global_labels = [l for l in mock_model.labels if l.startswith("global_cov")]
        mock_model.freeze("global_cov")
        assert "global_cov:log_amp" in mock_model.params.frozen()
        assert "global_cov:log_ls" in mock_model.params.frozen()
        assert all([l in mock_model.params.frozen() for l in global_labels])
        mock_model.thaw("global_cov")
        assert "global_cov:log_amp" not in mock_model.params.frozen()
        assert "global_cov:log_ls" not in mock_model.params.frozen()
        assert all([l not in mock_model.params.frozen() for l in global_labels])

    def test_freeze_thaw_local(self, mock_model):
        local_labels = [l for l in mock_model.labels if l.startswith("local_cov")]
        mock_model.freeze("local_cov")
        assert "local_cov:0:log_amp" in mock_model.params.frozen()
        assert "local_cov:0:log_sigma" in mock_model.params.frozen()
        assert "local_cov:0:mu" in mock_model.params.frozen()
        assert "local_cov:1:log_amp" in mock_model.params.frozen()
        assert "local_cov:1:log_sigma" in mock_model.params.frozen()
        assert "local_cov:1:mu" in mock_model.params.frozen()
        assert all([l in mock_model.params.frozen() for l in local_labels])
        mock_model.thaw("local_cov")
        assert "local_cov:0:log_amp" not in mock_model.params.frozen()
        assert "local_cov:0:log_sigma" not in mock_model.params.frozen()
        assert "local_cov:0:mu" not in mock_model.params.frozen()
        assert "local_cov:1:log_amp" not in mock_model.params.frozen()
        assert "local_cov:1:log_sigma" not in mock_model.params.frozen()
        assert "local_cov:1:mu" not in mock_model.params.frozen()
        assert all([l not in mock_model.params.frozen() for l in local_labels])

    def test_freeze_thaw_cheb(self, mock_model):
        cheb_labels = [l for l in mock_model.labels if l.startswith("cheb")]
        mock_model.freeze("cheb")
        assert "cheb:1" in mock_model.params.frozen()
        assert "cheb:2" in mock_model.params.frozen()
        assert all([l in mock_model.params.frozen() for l in cheb_labels])
        mock_model.thaw("cheb")
        assert "cheb:1" not in mock_model.params.frozen()
        assert "cheb:2" not in mock_model.params.frozen()
        assert all([l not in mock_model.params.frozen() for l in cheb_labels])

    def test_cov_caching(self, mock_model):
        assert mock_model._glob_cov is None
        assert mock_model._loc_cov is None
        mock_model()
        assert mock_model._glob_cov.shape == mock_model._loc_cov.shape

    def test_cov_caching_frozen(self, mock_model):
        mock_model()
        glob = mock_model._glob_cov
        loc = mock_model._loc_cov
        mock_model.freeze("local_cov")
        assert mock_model._loc_cov is None
        mock_model()
        assert np.allclose(mock_model._loc_cov, loc)
        assert np.allclose(mock_model._glob_cov, glob)
        mock_model.freeze("global_cov")
        assert np.allclose(mock_model._loc_cov, loc)
        assert mock_model._glob_cov is None
        mock_model()
        assert np.allclose(mock_model._loc_cov, loc)
        assert np.allclose(mock_model._glob_cov, glob)

    def test_fails_with_multiple_orders(self, mock_spectrum, mock_emulator):
        two_order_spectrum = mock_spectrum.reshape((2, -1))
        with pytest.raises(ValueError):
            SpectrumModel(
                emulator=mock_emulator,
                data=two_order_spectrum,
                grid_params=[6000, 4.0, 0],
            )

    def test_delete(self, mock_model):
        mock_model.freeze("global_cov")
        mock_model()
        assert mock_model._glob_cov is not None
        del mock_model["global_cov"]
        assert "global_cov" not in mock_model.params
        assert "global_cov" not in mock_model.params.frozen()
        assert mock_model._glob_cov is None

    @pytest.mark.skip
    def test_train_no_priors(self, mock_model):
        soln = mock_model.train(options={"maxiter": 1})
        assert not soln.success

    @pytest.mark.skip
    def test_train_priors(self, mock_model):
        priors = {"T": st.uniform(5900, 6700)}
        soln = mock_model.train(priors, options={"maxiter": 1})
        assert not soln.success

    @pytest.mark.skip
    def test_train_custom_prior(self, mock_model):
        class Prior:
            @staticmethod
            def logpdf(x):
                return 1 / x**2

        priors = {"T": Prior}
        soln = mock_model.train(priors, options={"maxiter": 1})
        assert not soln.success

    def test_bad_prior_key(self, mock_model):
        priors = {"penguin": st.uniform(5900, 6700)}
        with pytest.raises(ValueError):
            mock_model.train(priors, options={"maxiter": 1})

    def test_bad_prior_value(self, mock_model):
        priors = {"penguin": lambda x: 1 / x}
        with pytest.raises(ValueError):
            mock_model.train(priors, options={"maxiter": 1})

    def test_freeze_bad_param(self, mock_model):
        fr = mock_model.params.frozen()
        mock_model.freeze("pinguino")
        assert all([old == new for old, new in zip(fr, mock_model.params.frozen())])

    def test_thaw_bad_param(self, mock_model):
        fr = mock_model.params.frozen()
        mock_model.thaw("pinguino")
        assert all([old == new for old, new in zip(fr, mock_model.params.frozen())])

    def test_normalize(self, mock_model):
        F1, _ = mock_model()
        mock_model.norm = True
        F2, _ = mock_model()
        factor = mock_model.emulator.norm_factor(mock_model.grid_params)
        assert np.allclose(F1 * factor, F2)

    def test_str_logscale_cheat(self, mock_model):
        mock_model.freeze("logg")
        del mock_model["log_scale"]
        expected = textwrap.dedent(
            f"""
            SpectrumModel
            -------------
            Data: {mock_model.data_name}
            Emulator: {mock_model.emulator.name}
            Log Likelihood: {mock_model.log_likelihood().item()}

            Parameters
              cheb: [0.1, -0.2]
              local_cov:
                0: log_sigma: 2.0, log_amp: 2.0, mu: 10000.0
                1: log_sigma: 2.0, log_amp: 1.5, mu: 13000.0
              global_cov:
                log_ls: 1.0
                log_amp: 1.0
              vsini: 30.0
              Av: 0.0
              vz: 0.0
              T: 6000.0
              Z: 0.0
              log_scale: {mock_model._log_scale} (fit)
            
            Frozen Parameters
              logg: 4.0
            """
        ).strip()
        assert str(mock_model) == expected
