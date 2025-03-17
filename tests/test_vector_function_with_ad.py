import numpy as np

import aadc
import aadc.ndarray
from aadc.ndarray import AADCArray
from aadc.recording_ctx import record_kernel


def test_vector_function_with_ad() -> None:
    mat = np.array([[1.0, 2.0], [-3.0, -4.0], [5.0, 6.0]])

    def analytics(x: AADCArray | np.ndarray, a: aadc.idouble | float) -> AADCArray | np.ndarray:
        if x.ndim == 1:
            return np.exp(-a * np.dot(mat, x))
        else:
            # For batched inputs
            return np.exp(-a * np.dot(mat, x.T)).T

    # Record the function
    with record_kernel() as kernel:
        x = aadc.array(np.ones(2))
        x_args = x.mark_as_input()

        a = aadc.idouble(1.0)
        a_arg = a.mark_as_input_no_diff()

        f = analytics(x, a)
        assert isinstance(f, AADCArray)  # Keep mypy happy
        f_res = f.mark_as_output()

    # Create VectorFunctionWithAD instance
    vec_func = aadc.VectorFunctionWithAD(kernel, x_args, f_res, param_args=[a_arg])

    # Test single input case
    test_set = [
        (np.array([1.0, 2.0]), 0.0),
        (np.array([1.0, 3.0]), np.log(2.0)),
        (np.array([2.0, -1.0]), np.log(3.0)),
    ]

    for x_it, a_it in test_set:
        r = analytics(x_it, a_it)
        vec_func.set_params([a_it])

        # Get both function value and jacobian in one call
        f_val, jac = vec_func.evaluate(x_it)

        # Test function value
        assert np.allclose(f_val.squeeze(), r)

        # Test jacobian (gradient)
        expected_jac = -a_it * (mat.T * r).T
        assert np.allclose(jac.squeeze(), expected_jac)

    # Test batched input case
    x_batch = np.array([[1.0, 2.0], [1.0, 3.0], [2.0, -1.0]])
    a_value = 0.5

    vec_func.set_params([a_value])

    # Expected results for batch
    r_batch = analytics(x_batch, a_value)
    expected_jac_batch = np.array([-a_value * (mat.T * r).T for r in r_batch])

    # Get both function values and jacobians for batch
    f_batch, jac_batch = vec_func.evaluate(x_batch)

    assert np.allclose(f_batch, r_batch)
    assert np.allclose(jac_batch, expected_jac_batch)


def test_vector_function_with_ad_threading() -> None:
    # Test with multiple threads
    n = 1000  # Large batch size to make threading worthwhile
    dim = 5

    # Create a simple quadratic function
    mat = np.random.randn(3, dim)

    def analytics(x: AADCArray | np.ndarray, a: aadc.idouble | float) -> AADCArray | np.ndarray:
        if x.ndim == 1:
            return np.exp(-a * np.dot(mat, x))
        else:
            return np.exp(-a * np.dot(mat, x.T)).T

    with record_kernel() as kernel:
        x = aadc.array(np.zeros(dim))
        x_args = x.mark_as_input_no_diff()

        a = aadc.idouble(0.0)
        a_arg = a.mark_as_input_no_diff()

        f = analytics(x, a)

        assert isinstance(f, AADCArray)  # Keep mypy happy
        f_res = f.mark_as_output()

    # Create instances with different thread counts
    vec_func_single = aadc.VectorFunctionWithAD(kernel, x_args, f_res, param_args=[a_arg], num_threads=1)
    vec_func_multi = aadc.VectorFunctionWithAD(kernel, x_args, f_res, param_args=[a_arg], num_threads=4)

    # Test with large batch
    x_batch = np.random.randn(n, dim)
    a_value = 0.5

    vec_func_single.set_params([a_value])
    vec_func_multi.set_params([a_value])

    # Results should be the same regardless of thread count
    f_single, jac_single = vec_func_single.evaluate(x_batch)
    f_multi, jac_multi = vec_func_multi.evaluate(x_batch)

    assert np.allclose(f_single, f_multi)
    assert np.allclose(jac_single, jac_multi)


def test_vector_function_with_ad_threading_batch_size_equal_1() -> None:
    # Test with multiple threads
    n = 1  # Batch size equal to 1
    dim = 5

    # Create a simple quadratic function
    mat = np.random.randn(3, dim)

    def analytics(x: AADCArray | np.ndarray, a: aadc.idouble | float) -> AADCArray | np.ndarray:
        if x.ndim == 1:
            return np.exp(-a * np.dot(mat, x))
        else:
            return np.exp(-a * np.dot(mat, x.T)).T

    # Record the function
    with record_kernel() as kernel:
        x = aadc.array(np.zeros(dim))
        x_args = x.mark_as_input_no_diff()

        a = aadc.idouble(0.0)
        a_arg = a.mark_as_input_no_diff()

        f = analytics(x, a)

        assert isinstance(f, AADCArray)  # Keep mypy happy
        f_res = f.mark_as_output()

    # Create instances with different thread counts
    vec_func_single = aadc.VectorFunctionWithAD(kernel, x_args, f_res, param_args=[a_arg], num_threads=1)
    vec_func_multi = aadc.VectorFunctionWithAD(kernel, x_args, f_res, param_args=[a_arg], num_threads=4)

    x_batch = np.random.randn(n, dim)
    a_value = 0.5

    vec_func_single.set_params([a_value])
    vec_func_multi.set_params([a_value])

    # Results should be the same regardless of thread count
    f_single, jac_single = vec_func_single.evaluate(x_batch)
    f_multi, jac_multi = vec_func_multi.evaluate(x_batch)

    assert np.allclose(f_single, f_multi)
    assert np.allclose(jac_single, jac_multi)
