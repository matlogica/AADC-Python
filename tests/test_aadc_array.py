import aadc
import numpy as np
import pytest
import scipy
from aadc import Functions
from aadc.ndarray import AADCArray

# For now enough is to see if the below pass without exception


@pytest.fixture()
def test_array():
    return AADCArray(np.array([[2.0, 9.0], [8.0, 4.0]]))


def test_slicing(test_array):
    assert isinstance(test_array[1, :2], AADCArray)
    assert isinstance(test_array[:2][1, 0], float)


def test_binary_comparisons(test_array):
    test_array > 4.0
    test_array >= 4.0
    test_array < 4.0
    test_array <= 4.0


def test_np_where(test_array):
    np.where((test_array <= 4.0), test_array, 0.0)


def test_np_where_scalars(test_array):
    res = np.where(aadc.ibool(True), aadc.array(np.ones(3)), np.zeros(3))
    assert np.array_equal(res, np.ones(3))


def test_isfinite(test_array):
    np.isfinite(test_array)


def test_scipy_functions(test_array):
    funcs = Functions()
    funcs.start_recording()
    test_array.mark_as_input()
    scipy.special.erf(test_array)
    scipy.special.erfc(test_array)
    funcs.stop_recording()


def test_normal_cdf(test_array):
    unrecorded_cdf = scipy.special.ndtr(test_array)
    funcs = Functions()
    funcs.start_recording()
    test_array.mark_as_input()
    recorded_cdf = scipy.special.ndtr(test_array)
    funcs.stop_recording()

    assert np.array_equal(unrecorded_cdf, recorded_cdf)


def test_matmul(test_array):
    unrecorded_matmul = np.matmul(test_array, test_array)
    funcs = Functions()
    funcs.start_recording()
    test_array.mark_as_input()
    recorded_matmul = np.matmul(test_array, test_array)
    funcs.stop_recording()

    assert np.array_equal(unrecorded_matmul, recorded_matmul)


def test_dot(test_array):
    unrecorded_dot = np.dot(test_array, test_array)
    funcs = Functions()
    funcs.start_recording()
    test_array.mark_as_input()
    recorded_dot = np.dot(test_array, test_array)
    funcs.stop_recording()

    assert np.array_equal(unrecorded_dot, recorded_dot)


def test_recording(test_array):
    funcs = Functions()
    funcs.start_recording()
    inputs = test_array.mark_as_input()

    def f(x):
        return np.exp(x).sum()

    f = f(test_array)
    output = f.mark_as_output()

    funcs.stop_recording()

    aadc_inputs = {func_input: 10 for func_input in inputs.ravel()}
    request = {output: inputs.ravel().tolist()}

    workers = aadc.ThreadPool(4)
    aadc.evaluate_old(funcs, request, aadc_inputs, workers)


def test_array_of_arrays(test_array):
    funcs = Functions()
    funcs.start_recording()
    test_array.mark_as_input()
    new_array = aadc.array([test_array, test_array])
    funcs.stop_recording()
    assert np.array_equal(new_array, np.stack((test_array, test_array)))


@pytest.mark.parametrize("convertion_function", [np.asarray, np.array])
def test_convert_active_array_to_numpy(test_array, convertion_function):
    funcs = Functions()
    funcs.start_recording()
    test_array.mark_as_input()
    with pytest.raises(ValueError) as exc_info:
        convertion_function(test_array)
    funcs.stop_recording()

    assert exc_info.type is ValueError
    assert exc_info.value.args[0] == "Cannot convert an active AADCArray to a numpy array"
