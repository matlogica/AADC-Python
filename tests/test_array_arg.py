import numpy as np

import aadc
from aadc.evaluate_wrappers import evaluate_matrix_inputs
from aadc.recording_ctx import record_kernel


def test_array_arg_all_derivatives() -> None:
    batch_size = 2
    rng = np.random.default_rng(1234)
    a_np = rng.standard_normal((3, 3))

    def func(a, x):
        return (a @ x).mean()

    with record_kernel() as kernel:
        a = aadc.array(a_np)
        a_arg = a.mark_as_input()

        x = aadc.array(np.ones(3))
        x_arg = x.mark_as_input()

        y = func(a, x)
        y_arg = y.mark_as_output()

    request = [(y_arg, [a_arg, x_arg])]
    inputs = [(x_arg, rng.standard_normal((batch_size, 3))), (a_arg, np.tile(a_np, (batch_size, 1, 1)))]

    output_values, output_grads = evaluate_matrix_inputs(kernel, request, inputs, 4)

    assert output_values.shape == (batch_size, 1), "Forward pass shape correct"

    expected_results = [
        # result, grad a, grad x
        (
            0.22142861151869342,
            np.array([[0.1146, -0.1708, 0.4413], [0.1146, -0.1708, 0.4413], [0.1146, -0.1708, 0.4413]]),
            np.array([-0.9767, 0.6244, 0.6626]),
        ),
        (
            0.32630306664349257,
            np.array([[-0.2868, 0.1732, -0.4217], [-0.2868, 0.1732, -0.4217], [-0.2868, 0.1732, -0.4217]]),
            np.array([-0.9767, 0.6244, 0.6626]),
        ),
    ]

    for i, (res, grad_a, grad_x) in enumerate(expected_results):
        assert np.isclose(res, output_values[i]), f"For {i}-th batch element, forward pass results correct"
        assert np.allclose(grad_a, output_grads[y_arg][0][i], atol=1e-4), f"Grads correct for a, {i}-th batch element"
        assert np.allclose(grad_x, output_grads[y_arg][1][i], atol=1e-4), f"Grads correct for x, {i}-th batch element"


def test_array_arg_params_constant_across_batch() -> None:
    rng = np.random.default_rng(1234)
    batch_size = 2
    a_np = rng.standard_normal((3, 3))

    def func(a, x):
        return (a @ x).mean()

    with record_kernel() as kernel:
        a = aadc.array(a_np)
        a_arg = a.mark_as_input()

        x = aadc.array(np.ones(3))
        x_arg = x.mark_as_input()

        y = func(a, x)
        y_arg = y.mark_as_output()

    request = [(y_arg, [a_arg, x_arg])]
    inputs = [(x_arg, rng.standard_normal((batch_size, 3))), (a_arg, a_np[np.newaxis, :])]

    output_values, output_grads = evaluate_matrix_inputs(kernel, request, inputs, 4)

    assert output_values.shape == (batch_size, 1), "Forward pass shape correct"

    expected_results = [
        # result, grad a, grad x
        (
            0.22142861151869342,
            np.array([[0.1146, -0.1708, 0.4413], [0.1146, -0.1708, 0.4413], [0.1146, -0.1708, 0.4413]]),
            np.array([-0.9767, 0.6244, 0.6626]),
        ),
        (
            0.32630306664349257,
            np.array([[-0.2868, 0.1732, -0.4217], [-0.2868, 0.1732, -0.4217], [-0.2868, 0.1732, -0.4217]]),
            np.array([-0.9767, 0.6244, 0.6626]),
        ),
    ]

    for i, (res, grad_a, grad_x) in enumerate(expected_results):
        assert np.isclose(res, output_values[i]), f"For {i}-th batch element, forward pass results correct"
        assert np.allclose(grad_a, output_grads[y_arg][0][i], atol=1e-4), f"Grads correct for a, {i}-th batch element"
        assert np.allclose(grad_x, output_grads[y_arg][1][i], atol=1e-4), f"Grads correct for x, {i}-th batch element"


def test_array_arg_params_only_in_request() -> None:
    rng = np.random.default_rng(1234)
    batch_size = 2
    a_np = rng.standard_normal((3, 3))

    def func(a, x):
        return (a @ x).mean()

    with record_kernel() as kernel:
        a = aadc.array(a_np)
        a_arg = a.mark_as_input()

        x = aadc.array(np.ones(3))
        x_arg = x.mark_as_input_no_diff()

        y = func(a, x)
        y_arg = y.mark_as_output()

    request = [(y_arg, [a_arg])]
    inputs = [(x_arg, rng.standard_normal((batch_size, 3))), (a_arg, a_np[np.newaxis, :])]

    output_values, output_grads = evaluate_matrix_inputs(kernel, request, inputs, 1)

    assert output_values.shape == (batch_size, 1), "Forward pass shape correct"

    expected_results = [
        # result, grad a
        (
            0.22142861151869342,
            np.array([[0.1146, -0.1708, 0.4413], [0.1146, -0.1708, 0.4413], [0.1146, -0.1708, 0.4413]]),
        ),
        (
            0.32630306664349257,
            np.array([[-0.2868, 0.1732, -0.4217], [-0.2868, 0.1732, -0.4217], [-0.2868, 0.1732, -0.4217]]),
        ),
    ]

    for i, (res, grad_a) in enumerate(expected_results):
        assert np.isclose(res, output_values[i]), f"For {i}-th batch element, forward pass results correct"
        assert np.allclose(grad_a, output_grads[y_arg][0][i], atol=1e-4), f"Grads correct for a, {i}-th batch element"
