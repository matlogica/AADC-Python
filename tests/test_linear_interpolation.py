import numpy as np
import pytest

import aadc
from aadc import idouble, record_kernel


def test_interp1d_scalar() -> None:
    with record_kernel():
        x_args = aadc.array([1.0, 2.0, 3.0, 4.0])
        x0_val = idouble(2.5)
        x0_val.mark_as_input()
        y_args = aadc.array([1.0, 4.0, 9.0, 16.0])
        y_args.mark_as_input()
        y0_val = np.interp(x0_val, x_args, y_args)
    assert np.isclose(y0_val, 6.5)


def test_interp1d_inactive_scalar() -> None:
    with record_kernel():
        x_args = np.array([1.0, 2.0, 3.0, 4.0])
        x0_val = 2.5
        y_args = aadc.array([1.0, 4.0, 9.0, 16.0])
        y_args.mark_as_input()
        y0_val = np.interp(x0_val, x_args, y_args)
    assert np.isclose(y0_val, 6.5)


def test_interp1d_vector() -> None:
    with record_kernel():
        x_args = aadc.array([1.0, 2.0, 3.0, 4.0])
        x0_vals = aadc.array([0.5, 2.5, 3.0, 4.5])
        x0_vals.mark_as_input()
        y_args = aadc.array([1.0, 4.0, 9.0, 16.0])
        y_args.mark_as_input()
        y0_vals = np.interp(x0_vals, x_args, y_args)
    expected = np.array([1.0, 6.5, 9.0, 16.0])
    assert np.allclose(y0_vals, expected)


def test_interp1d_vector_left_right() -> None:
    with record_kernel():
        x_args = aadc.array([1.0, 2.0, 3.0, 4.0])
        x0_vals = aadc.array([0.5, 2.5, 3.0, 4.5])
        x0_vals.mark_as_input()
        y_args = aadc.array([1.0, 4.0, 9.0, 16.0])
        y_args.mark_as_input()

        # Test 'left' behavior
        y0_vals_left = np.interp(x0_vals, x_args, y_args, left=0.0)
        expected_left = np.array([0.0, 6.5, 9.0, 16.0])

        y0_vals_right = np.interp(x0_vals, x_args, y_args, right=20.0)
        expected_right = np.array([1.0, 6.5, 9.0, 20.0])

    assert np.allclose(y0_vals_left, expected_left)
    assert np.allclose(y0_vals_right, expected_right)


def test_interp1d_inactive_vector() -> None:
    with record_kernel():
        x_args = aadc.array([1.0, 2.0, 3.0, 4.0])
        x0_vals = aadc.array([0.5, 2.5, 3.0, 4.5])
        y_args = aadc.array([1.0, 4.0, 9.0, 16.0])
        y_args.mark_as_input()
        y0_vals = np.interp(x0_vals, x_args, y_args)
    expected = np.array([1.0, 6.5, 9.0, 16.0])
    assert np.allclose(y0_vals, expected)


def test_interp1d_nonaadc_xargs() -> None:
    with record_kernel():
        times = np.array([1.0, 4.0, 9.0, 16.0])

        kernel = aadc.Kernel()
        kernel.start_recording()

        zero_rates = aadc.array([0.0025 + 0.005 * 0.02 * i for i in range(len(times))])
        zero_rates.mark_as_input()

        dfs = np.exp(-zero_rates * times)
        t = aadc.idouble(5.0)
        t.mark_as_input()
        np.interp(t, times, dfs)


@pytest.mark.parametrize("seed", np.arange(20))
def test_interpolation_vector_function(seed) -> None:
    np.random.seed(seed)
    num_pts = 10
    xs_np = np.linspace(0, 1, num_pts)
    ys_np = np.random.randn(num_pts)

    def func(xs, ys, bump=0.0):
        xs_np = xs + bump
        ys_np = ys
        t = xs_np.mean()
        yt = np.interp(t, xs_np, ys_np)
        t2 = 0.25
        yt2 = np.interp(t2, xs_np, ys_np)
        return yt + yt2

    with aadc.record_kernel() as kernel:
        xs = aadc.array(xs_np)
        ys = aadc.array(ys_np)
        xs_args = xs.mark_as_input()
        ys_args = ys.mark_as_input()
        yout = func(xs, ys)
        result = yout.mark_as_output()

    f = aadc.VectorFunctionWithAD(kernel, args=xs_args, res=[result], param_args=ys_args)
    f.set_params(np.asarray(ys))

    _, ad_jacobian = f.evaluate(np.asarray(xs))
    ad_jacobian = ad_jacobian[0]  # Extract the jacobian from the tuple

    bump_size = 1e-5
    fd_jacobian = np.zeros(xs.shape)

    for i in range(xs.shape[0]):
        bump = np.zeros(xs.shape)
        bump[i] = bump_size
        fd_jacobian[i] = (func(xs_np, ys_np, bump) - func(xs_np, ys_np, -bump)) / (2 * bump_size)

    assert np.allclose(ad_jacobian.flatten(), fd_jacobian, rtol=1e-6, atol=1e-6)


@pytest.mark.skip("Limitation: cannot have xs different from those used at recording time as input")
def test_interpolation_vector_function_different_xs() -> None:
    np.random.seed(42)
    num_pts = 10
    xs_np = np.linspace(0, 1, num_pts)
    ys_np = np.random.randn(num_pts)
    # Create different evaluation points
    xs_eval = np.linspace(-0.5, 1.5, num_pts)  # Wider range than original xs

    def func(xs, ys, bump=0.0):
        xs_np = xs + bump
        ys_np = ys
        t = xs_np.mean()
        yt = np.interp(t, xs_np, ys_np)
        t2 = 0.25
        yt2 = np.interp(t2, xs_np, ys_np)
        return yt + yt2

    with aadc.record_kernel() as kernel:
        xs = aadc.array(xs_np)
        ys = aadc.array(ys_np)
        xs_args = xs.mark_as_input()
        ys_args = ys.mark_as_input()
        yout = func(xs, ys)
        result = yout.mark_as_output()

    f = aadc.VectorFunctionWithAD(kernel, args=xs_args, res=[result], param_args=ys_args)
    f.set_params(np.asarray(ys))

    # Evaluate at different x values
    _, ad_jacobian = f.evaluate(np.asarray(xs_eval))
    ad_jacobian = ad_jacobian[0]  # Extract the jacobian from the tuple

    bump_size = 1e-5
    fd_jacobian = np.zeros(xs_eval.shape)

    for i in range(xs_eval.shape[0]):
        bump = np.zeros(xs_eval.shape)
        bump[i] = bump_size
        fd_jacobian[i] = (func(xs_eval, ys_np, bump) - func(xs_eval, ys_np, -bump)) / (2 * bump_size)

    assert np.allclose(ad_jacobian.flatten(), fd_jacobian, rtol=1e-6, atol=1e-6)
