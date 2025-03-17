import numpy as np

import aadc
from aadc.evaluate_wrappers import evaluate_kernel
from aadc.recording_ctx import record_kernel


def test_recording_with_scalar_eval() -> None:
    funcs = aadc.Functions()
    funcs.start_recording()
    x = aadc.idouble(1.0)
    y = aadc.idouble(2.0)
    z = aadc.idouble(3.0)
    xin = x.mark_as_input()
    f = aadc.math.exp(x / y + z) + x
    fout = f.mark_as_output()
    funcs.stop_recording()

    inputs = {xin: 1.0}
    request = {fout: [xin]}

    workers = aadc.ThreadPool(1)
    aadc.evaluate(funcs, request, inputs, workers)


def test_recording_with_scalar_eval_ctx_and_evaluate_args() -> None:
    with record_kernel() as kernel:
        x = aadc.idouble(1.0)
        y = aadc.idouble(2.0)
        z = aadc.idouble(3.0)
        xin = x.mark_as_input()
        f = aadc.math.exp(x / y + z) + x
        fout = f.mark_as_output()

    output = evaluate_kernel(kernel, request={fout: [xin]}, inputs={xin: 1.0}, num_threads=1)

    assert np.isclose(output.values[fout].item(), 34.11545196)
    assert np.isclose(output.derivs[fout][xin].item(), 17.55772598)
