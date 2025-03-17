import numpy as np

import aadc
from aadc import Functions, idouble
from aadc.ndarray import AADCArray


def test_cumsum() -> None:
    func = Functions()
    func.start_recording()
    val = AADCArray([1.0, 2.0, 3.0])
    val.mark_as_input()
    val = np.cumsum(val)
    func.stop_recording()
    assert isinstance(val, AADCArray)
    assert val == np.array([1.0, 3.0, 6.0])


def test_max() -> None:
    func = Functions()
    func.start_recording()
    val = AADCArray([1.0, 2.0, 3.0])
    val.mark_as_input()
    val = np.max(val)
    func.stop_recording()
    assert isinstance(val, idouble)
    assert val == 3.0


def test_mean() -> None:
    func = Functions()
    func.start_recording()
    val = AADCArray([1.0, 2.0, 3.0])
    val.mark_as_input()
    val = np.mean(val)
    func.stop_recording()
    assert isinstance(val, idouble)
    assert val == 2.0


def test_min() -> None:
    func = Functions()
    func.start_recording()
    val = AADCArray([1.0, 2.0, 3.0])
    val.mark_as_input()
    val = np.min(val)
    func.stop_recording()
    assert isinstance(val, idouble)
    assert val == 1.0


def test_prod() -> None:
    func = Functions()
    func.start_recording()
    val = AADCArray([1.0, 2.0, 3.0])
    val.mark_as_input()
    val = np.prod(val)
    func.stop_recording()
    assert isinstance(val, idouble)
    assert val == 6.0


def test_std() -> None:
    func = Functions()
    func.start_recording()
    val = AADCArray([1.0, 2.0, 3.0])
    val.mark_as_input()
    val = np.std(val)
    func.stop_recording()
    assert isinstance(val, idouble)
    assert round(val, 2) == 0.82


def test_sum() -> None:
    func = Functions()
    func.start_recording()
    val = AADCArray([1.0, 2.0, 3.0])
    val.mark_as_input()
    val = np.sum(val)
    func.stop_recording()
    assert isinstance(val, idouble)
    assert val == 6.0


def test_sum_2d() -> None:
    func = Functions()
    func.start_recording()
    val = AADCArray([[1.0, 2.0], [3.0, 4.0]])
    val.mark_as_input()
    val = np.sum(val, axis=0)
    func.stop_recording()
    assert isinstance(val, AADCArray)
    assert np.all(val == [4.0, 6.0])


def test_sum_3d() -> None:
    func = Functions()
    func.start_recording()
    val = AADCArray([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    val.mark_as_input()
    val = np.sum(val, axis=2)
    func.stop_recording()
    assert isinstance(val, AADCArray)
    assert np.all(val == np.array([[3.0, 7.0], [11.0, 15.0]]))


def test_var() -> None:
    func = Functions()
    func.start_recording()
    val = AADCArray([1.0, 2.0, 3.0])
    val.mark_as_input()
    val = np.var(val)
    func.stop_recording()
    assert isinstance(val, idouble)
    assert round(val, 2) == 0.67


def test_diff() -> None:
    func = Functions()
    func.start_recording()
    val = AADCArray([2.0, 1.0, 2.0, 3.0])
    val.mark_as_input()
    val = np.diff(val)
    func.stop_recording()
    assert isinstance(val, AADCArray)
    assert val == np.array([1.0, -1.0, -4.0])


def test_average_without_weights() -> None:
    func = Functions()
    func.start_recording()
    val = AADCArray([1.0, 2.0, 3.0])
    val.mark_as_input()
    val = np.average(val)
    func.stop_recording()
    assert isinstance(val, idouble)
    assert val == 2.0


def test_average_without_weights_2d() -> None:
    func = Functions()
    func.start_recording()
    val = AADCArray([[1.0, 2.0], [3.0, 4.0]])
    val.mark_as_input()
    val = np.average(val, axis=1)
    func.stop_recording()
    assert isinstance(val, AADCArray)
    assert np.allclose(val, [1.5, 3.5])


def test_average_with_weights() -> None:
    func = Functions()
    func.start_recording()
    val = AADCArray([1.0, 2.0, 3.0])
    weights = AADCArray([1.0, 1.0, 1.0])
    val.mark_as_input()
    weights.mark_as_input()
    val = np.average(val, weights=weights)
    func.stop_recording()
    assert isinstance(val, idouble)
    assert val == 2.0


def test_average_with_weights_2d() -> None:
    func = Functions()
    func.start_recording()
    val_np = AADCArray([[1.0, 2.0], [3.0, 4.0]])
    weights_np = AADCArray([[1.0, 2.0], [3.0, 4.0]])
    val = AADCArray(val_np)
    weights = AADCArray(weights_np)
    val.mark_as_input()
    weights.mark_as_input()
    val = np.average(val, weights=weights, axis=1)
    func.stop_recording()
    assert isinstance(val, AADCArray)
    assert np.allclose(val, np.average(val_np, weights=weights_np, axis=1))


def test_cov() -> None:
    funcs = aadc.Functions()
    funcs.start_recording()
    xnp = np.array([1.0, 2.0])
    x = aadc.array(xnp)
    x.mark_as_input()
    z = np.cov(x, x)
    funcs.stop_recording()
    assert np.allclose(z, np.cov(xnp, xnp))


def test_cov_with_y() -> None:
    funcs = aadc.Functions()
    funcs.start_recording()
    xnp = np.array([1.0, 2.0])
    ynp = np.array([3.0, 4.0])
    x = aadc.array(xnp)
    y = aadc.array(ynp)
    x.mark_as_input()
    y.mark_as_input()
    z = np.cov(x, y)
    funcs.stop_recording()
    assert np.allclose(z, np.cov(xnp, ynp))


def test_cov_rowvar_false() -> None:
    funcs = aadc.Functions()
    funcs.start_recording()
    xnp = np.array([[1.0, 2.0], [3.0, 4.0]])
    x = aadc.array(xnp)
    x.mark_as_input()
    z = np.cov(x, rowvar=False)
    funcs.stop_recording()
    assert np.allclose(z, np.cov(xnp, rowvar=False))


def test_cov_with_bias() -> None:
    funcs = aadc.Functions()
    funcs.start_recording()
    xnp = np.array([1.0, 2.0, 3.0])
    x = aadc.array(xnp)
    x.mark_as_input()
    z = np.cov(x, bias=True)
    funcs.stop_recording()
    assert np.allclose(z, np.cov(xnp, bias=True))


def test_cov_with_fweights() -> None:
    funcs = aadc.Functions()
    funcs.start_recording()
    xnp = np.array([1.0, 2.0, 3.0])
    fweights_np = np.array([1.0, 2.0, 3.0])
    x = aadc.array(xnp)
    x.mark_as_input()
    z = np.cov(x, fweights=fweights_np)
    funcs.stop_recording()
    assert np.allclose(z, np.cov(xnp, fweights=fweights_np))
