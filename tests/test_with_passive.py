import aadc


def test_with_passive() -> None:
    funcs = aadc.Functions()
    funcs.start_recording()
    x = aadc.idouble(1.0)
    y = aadc.idouble(2.0)
    z = aadc.idouble(3.0)
    x.mark_as_input()
    y.mark_as_input()
    z.mark_as_input()
    f = aadc.math.exp(x / y + z) + x
    with aadc.passive():
        print(float(f))
    f.mark_as_output()
    funcs.stop_recording()
    funcs.print_passive_extract_locations()
