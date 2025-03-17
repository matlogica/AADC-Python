import math
import time

import numpy as np
import pytest

import aadc
import aadc.record_function
import aadc.vectorize


@pytest.fixture()
def monkeypatch_mathexp():
    orig = math.exp
    math.exp = aadc.math.exp
    yield
    math.exp = orig


def test_external_function(monkeypatch_mathexp) -> None:
    class MyFunc(aadc.VectorFunction):
        def func(self, x):
            print("func in python")
            return x + 100.0

    mf = MyFunc()

    x = np.ones(10)

    print(mf.func(x))
    print(x)

    print(np.array(aadc.ExternalFunction(mf, x)) + 1.0)

    funcs = aadc.Functions()

    funcs.start_recording()
    inputs = [aadc.idouble(i) for i in range(10)]
    inputs_args = [i.mark_as_input() for i in inputs]

    inputs = np.array(inputs) + 0.001

    outputs = aadc.ExternalFunction(mf, inputs)

    outputs = np.array(outputs) + 10000

    outputs_arg = [outputs[i].mark_as_output() for i in range(10)]

    funcs.stop_recording()
    funcs.print_passive_extract_locations()

    request = {}  #   outputsArg[0] : [] }

    for i in range(len(outputs_arg)):
        request[outputs_arg[i]] = [inputs_args[i]]  # inputsArgs

    workers = aadc.ThreadPool(1)

    # inputs = [{ inputsArgs[i] : np.array([ float(i) ]) } for i in range(10)]
    inputs = {}
    for i in range(10):
        inputs[inputs_args[i]] = [float(i)]

    print(inputs)

    res = aadc.evaluate(funcs, request, inputs, workers)

    for i in range(len(outputs_arg)):
        print(res[0][outputs_arg[i]])
    #    print(res[1][outputsArg[i]])


def test_vector_function_with_jac() -> None:
    rng = np.random.default_rng(1234)

    a = rng.standard_normal((10, 10))

    def func_a(x):
        return a @ x

    x0 = np.ones(10)

    vf = aadc.record_function.record(func_a, x0, bump_size=1e-3)

    res = vf.func(x0)
    jac = vf.jac(x0)

    assert np.allclose(res, func_a(x0))
    assert np.allclose(jac, a)


def test_vectorize() -> None:
    rng = np.random.default_rng(1234)

    def func(x):
        return np.exp(x[0] * x[1] + x[2])

    def func_np(x):
        return np.exp(x[:, 0] * x[:, 1] + x[:, 2])

    x0 = rng.standard_normal((1_000_000, 3))
    vf = aadc.vectorize.VectorizedFunction(func, num_threads=12)

    start = time.time()
    y_np = func_np(x0)
    end = time.time()
    print(f"Elapsed time pure np: {end - start:.6f} seconds")

    eps = 1e-8
    fd0_np = (func_np(x0 + np.array([eps, 0.0, 0.0])) - func_np(x0 - np.array([eps, 0.0, 0.0]))) / (2 * eps)
    fd1_np = (func_np(x0 + np.array([0.0, eps, 0.0])) - func_np(x0 - np.array([0.0, eps, 0.0]))) / (2 * eps)
    fd2_np = (func_np(x0 + np.array([0.0, 0.0, eps])) - func_np(x0 - np.array([0.0, 0.0, eps]))) / (2 * eps)
    fd_grad = np.stack((fd0_np, fd1_np, fd2_np))

    start = time.time()
    y, y_grads = vf(x0)
    end = time.time()
    print(f"Elapsed time aadc: {end - start:.6f} seconds")

    start = time.time()
    y, y_grads = vf(x0)
    end = time.time()
    print(f"Elapsed time aadc no recording: {end - start:.6f} seconds")

    assert np.allclose(y, y_np)
    assert np.allclose(y_grads, fd_grad, atol=1e-5)
