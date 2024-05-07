import aadc


def test_recording_with_scalar_eval():
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
