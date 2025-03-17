import builtins
import time
from threading import Thread

import numpy as np
import pytest
from scipy.stats import norm

import aadc
import aadc.overrides
from aadc import Functions
from aadc.ndarray import AADCArray
from aadc.overrides import float, int, isinstance


@pytest.fixture
def problematic_code():
    def func(log_returns, bigt, assets):
        funcs = Functions()
        funcs.start_recording()
        log_returns.mark_as_input()

        try:
            asset_price_movements = np.ones((bigt, assets))
            for t in range(bigt):
                asset_price_movements[t, :] = asset_price_movements[t - 1, :] * np.exp(log_returns)
        finally:
            funcs.stop_recording()

        return asset_price_movements

    return func


def test_call_float_on_float() -> None:
    value = 3.14
    regular_float = float(value)
    assert builtins.isinstance(regular_float, builtins.float), "Regular float should be an instance of builtins.float"


def test_call_float_on_idouble() -> None:
    value = aadc.idouble(3.14)
    aadc_float = float(value)
    assert builtins.isinstance(aadc_float, aadc.idouble), "idouble should pass through float with noop"


def test_call_isinstace_on_float() -> None:
    assert isinstance(3.14, float), "regular float should be treated as the new float"
    assert isinstance(3.14, builtins.float), "regular float should be treated as a builtins.float"


def test_call_isinstace_on_idouble() -> None:
    value = aadc.idouble(10.0)
    assert isinstance(value, float), "aadc.idouble should be treated as the new float"
    assert isinstance(value, builtins.float), "aadc.idouble should be treated as a builtins.float"


def test_call_int_on_int() -> None:
    value = 10
    regular_int = int(value)
    assert isinstance(regular_int, builtins.int), "Regular int should be an instance of builtins.int"


def test_call_int_on_iint() -> None:
    value = aadc.iint(20)
    aadc_int = int(value)
    assert isinstance(aadc_int, aadc.iint), "iint should pass through int with noop"


def test_call_isinstace_on_int() -> None:
    assert isinstance(5, int), "Regular int should be treated as the new int"
    assert isinstance(5, builtins.int), "Regular int should be treated as a builtins.int"


def test_call_isinstace_on_iint() -> None:
    value = aadc.iint(30)
    assert isinstance(value, int), "aadc.iint should be treated as the new int"
    assert isinstance(value, builtins.int), "aadc.iint should be treated as a builtins.int"


def test_scipy_norm_cdf() -> None:
    rng = np.random.default_rng(1234)
    np_randoms = rng.standard_normal(10)

    funcs = Functions()
    funcs.start_recording()
    randoms = AADCArray(np_randoms)
    randoms.mark_as_input()

    with aadc.overrides.aadc_overrides():
        out_aadc = norm.cdf(randoms)
    out_np = norm.cdf(np_randoms)

    funcs.stop_recording()

    assert isinstance(out_aadc, AADCArray)
    assert np.allclose(out_aadc, out_np)


def test_full_like() -> None:
    funcs = Functions()
    funcs.start_recording()
    value = aadc.idouble(42.0)
    value.mark_as_input()

    test_array = np.array([1.0, 2.0, 3.0])

    with aadc.overrides.aadc_overrides():
        output = np.full_like(test_array, value)

    funcs.stop_recording()

    assert isinstance(output, AADCArray)
    assert np.allclose(output, 42.0)


def test_numpy_monkeypatch(problematic_code) -> None:
    rng = np.random.default_rng(1234)

    assets = 10
    bigt = 100

    log_returns = AADCArray(rng.standard_normal(assets))

    def without_overrides():
        time.sleep(0.05)
        with pytest.raises(ValueError):
            problematic_code(log_returns, bigt, assets)

    thread = Thread(target=without_overrides)
    thread.start()

    with aadc.overrides.aadc_overrides():  # Could specify a subset here
        time.sleep(0.1)
        problematic_code(log_returns, bigt, assets)

    thread.join()


def test_numpy_monkeypatch_gbm(problematic_code) -> None:
    s0 = 100
    rfr = 0.01
    sigma = 0.2
    final_time = 1
    num_tsteps = 100
    seed = 42
    rng = np.random.default_rng(seed)
    dt = final_time / num_tsteps
    num_paths = 1

    time = np.linspace(start=0.0, stop=final_time, num=num_tsteps)

    funcs = aadc.Functions()
    funcs.start_recording()

    normal_samples_numpy = rng.standard_normal((num_paths, num_tsteps - 1))
    normal_samples = AADCArray(normal_samples_numpy)
    normal_samples.mark_as_input()

    with aadc.overrides.aadc_overrides():
        bm = np.concatenate((np.atleast_2d(np.zeros(num_paths)).T, np.cumsum(normal_samples * dt, axis=1)), axis=1)

    s0 * np.exp((rfr - sigma**2 / 2) * time + sigma * bm)
    funcs.stop_recording()
