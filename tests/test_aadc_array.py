import numpy as np
import pytest
import scipy

import aadc
import aadc.overrides
from aadc import Functions, ibool, idouble
from aadc.ndarray import AADCArray
from aadc.recording_ctx import record_kernel


@pytest.fixture()
def test_array() -> None:
    return AADCArray(np.array([[2.0, 9.0], [8.0, 4.0]]))


def test_slicing(test_array) -> None:
    assert isinstance(test_array[1, :2], AADCArray)
    assert isinstance(test_array[:2][1, 0], float)


def test_binary_comparisons(test_array) -> None:
    test_array > 4.0
    test_array >= 4.0
    test_array < 4.0
    test_array <= 4.0


def test_np_where(test_array) -> None:
    np.where((test_array <= 4.0), test_array, 0.0)


def test_np_where_scalars(test_array) -> None:
    res = np.where(aadc.ibool(True), aadc.array(np.ones(3)), np.zeros(3))
    assert np.array_equal(res, np.ones(3))


def test_isfinite(test_array) -> None:
    np.isfinite(test_array)


def test_scipy_functions(test_array) -> None:
    funcs = Functions()
    funcs.start_recording()
    test_array.mark_as_input()
    scipy.special.erf(test_array)
    scipy.special.erfc(test_array)
    funcs.stop_recording()


def test_normal_cdf(test_array) -> None:
    unrecorded_cdf = scipy.special.ndtr(test_array)
    funcs = Functions()
    funcs.start_recording()
    test_array.mark_as_input()
    recorded_cdf = scipy.special.ndtr(test_array)
    funcs.stop_recording()

    assert np.array_equal(unrecorded_cdf, recorded_cdf)


def test_matmul(test_array) -> None:
    unrecorded_matmul = np.matmul(test_array, test_array)
    funcs = Functions()
    funcs.start_recording()
    test_array.mark_as_input()
    recorded_matmul = np.matmul(test_array, test_array)
    funcs.stop_recording()

    assert np.array_equal(unrecorded_matmul, recorded_matmul)


def test_dot(test_array) -> None:
    unrecorded_dot = np.dot(test_array, test_array)
    funcs = Functions()
    funcs.start_recording()
    test_array.mark_as_input()
    recorded_dot = np.dot(test_array, test_array)
    funcs.stop_recording()

    assert np.array_equal(unrecorded_dot, recorded_dot)


def test_recording(test_array) -> None:
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
    aadc.evaluate(funcs, request, aadc_inputs, workers)


def test_array_of_arrays(test_array) -> None:
    funcs = Functions()
    funcs.start_recording()
    test_array.mark_as_input()
    new_array = aadc.array([test_array, test_array])
    funcs.stop_recording()
    assert np.array_equal(new_array, np.stack((test_array, test_array)))


@pytest.mark.parametrize("convertion_function", [np.asarray, np.array])
def test_convert_active_array_to_numpy(test_array, convertion_function) -> None:
    funcs = Functions()
    funcs.start_recording()
    test_array.mark_as_input()
    with pytest.raises(ValueError) as exc_info:
        convertion_function(test_array)
    funcs.stop_recording()

    assert exc_info.type is ValueError
    assert exc_info.value.args[0] == "Cannot convert an active AADCArray to a numpy array"


@pytest.mark.parametrize("convertion_function", [np.asarray, np.array])
def test_convert_active_array_to_numpy_without_recording(test_array, convertion_function) -> None:
    numpy_test_array = np.asarray(test_array)
    funcs = Functions()
    funcs.start_recording()
    test_array.mark_as_input()
    funcs.stop_recording()
    assert np.allclose(convertion_function(test_array), numpy_test_array)


def test_scalar_mul_with_1el_array() -> None:
    kernel = aadc.Functions()
    kernel.start_recording()

    s0_1 = aadc.idouble(100.0)
    s0_1.mark_as_input()

    s1 = s0_1 * np.ones(1)
    print(s1[:])  # should not fail

    with aadc.overrides.aadc_overrides():
        s1 = s0_1 * np.ones(1)

    print(s1[:])  # should not fail
    kernel.stop_recording()


def test_scalar_mul_with_nel_array() -> None:
    kernel = aadc.Functions()
    kernel.start_recording()

    s0_1 = aadc.idouble(100.0)
    s0_1.mark_as_input()

    s1 = s0_1 * np.ones(5)
    print(s1[:])  # should not fail

    with aadc.overrides.aadc_overrides():
        s1 = s0_1 * np.ones(5)

    print(s1[:])  # should not fail
    kernel.stop_recording()


def test_iall_with_extra_args() -> None:
    with record_kernel():
        a = aadc.array([ibool(True), ibool(False), ibool(True)])
        result = not np.all(a)
    assert result


def test_iall_axis_1() -> None:
    with record_kernel():
        a = aadc.array([[ibool(True), ibool(False)], [ibool(True), ibool(True)]])
        result = np.all(a, axis=1)
    assert np.array_equal(result, np.array([False, True]))


def test_iall_out() -> None:
    with record_kernel():
        a = aadc.array([ibool(True), ibool(False), ibool(True)])
        result = np.array(ibool(False))
        np.all(a, out=result)
    assert not result


def test_iall_keepdims() -> None:
    with record_kernel():
        a = aadc.array([[ibool(True), ibool(False)], [ibool(True), ibool(True)]])
        result = np.all(a, axis=1, keepdims=True)
    assert np.array_equal(result, np.array([[False], [True]]))


def test_iall_where() -> None:
    with record_kernel():
        a = aadc.array([[ibool(True), ibool(False)], [ibool(True), ibool(False)]])
        cond = np.array([[True, False], [False, True]])
        result = (not np.all(a, where=cond),)
    assert result


def test_iand_with_extra_args() -> None:
    with record_kernel():
        a = aadc.array([ibool(True), ibool(False), ibool(True)])
        result = np.any(a)
    assert result


def test_iand_axis_1() -> None:
    with record_kernel():
        a = aadc.array([[ibool(True), ibool(False)], [ibool(True), ibool(True)]])
        result = np.any(a, axis=1)
    assert np.array_equal(result, np.array([True, True]))


def test_iand_out() -> None:
    with record_kernel():
        a = aadc.array([ibool(True), ibool(False), ibool(True)])
        result = np.array(ibool(True))
        np.any(a, out=result)
    assert result


def test_iand_keepdims() -> None:
    with record_kernel():
        a = aadc.array([[ibool(True), ibool(False)], [ibool(True), ibool(True)]])
        result = np.any(a, axis=1, keepdims=True)
    assert np.array_equal(result, np.array([[True], [True]]))


def test_iand_where() -> None:
    with record_kernel():
        a = aadc.array([[ibool(True), ibool(False)], [ibool(True), ibool(False)]])
        cond = np.array([[True, False], [False, True]])
        result = np.any(a, where=cond)
    assert result


def test_searchsorted_scalar() -> None:
    with record_kernel():
        x_args = np.array([1.0, 2.0, 3.0, 4.0])
        x0_val = idouble(2.5)
        x0_val.mark_as_input()
        idx = np.searchsorted(x_args, x0_val)
    assert idx == 2


def test_searchsorted_array() -> None:
    with record_kernel():
        x_args = np.array([1.0, 2.0, 3.0, 4.0])
        x0_vals = aadc.array([0.5, 2.5, 3.0, 4.5])
        x0_vals.mark_as_input()
        idx = np.searchsorted(x_args, x0_vals)
    expected = np.array([0, 2, 2, 4])
    assert np.array_equal(idx, expected)


def test_cholesky_factorization_single_matrix() -> None:
    with record_kernel():
        a = aadc.array([[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]])
        a.mark_as_input()

        l_aadc = np.linalg.cholesky(a)

    l_np = np.linalg.cholesky(a)

    assert np.allclose(l_aadc, l_np)
    assert np.allclose(l_aadc @ l_aadc.T, a)


def test_cholesky_factorization_vectorized() -> None:
    with record_kernel():
        a = aadc.array(
            [
                [
                    [[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]],
                    [[1.0, 2.0, 3.0], [2.0, 5.0, 6.0], [3.0, 6.0, 9.0]],
                ],
                [
                    [[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]],
                    [[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]],
                ],
            ]
        )
        a.mark_as_input()

        l_aadc = np.linalg.cholesky(a)

    l_np = np.linalg.cholesky(a)

    assert np.allclose(l_aadc, l_np)
    assert np.allclose(l_aadc @ np.transpose(l_aadc, (0, 1, 3, 2)), a)


def test_aadc_array_from_empty_array() -> None:
    output = aadc.array(np.full(0, 1, dtype=float)).mark_as_input()
    assert output.size == 0
    assert output.dtype == "O"
