import numpy as np

import aadc


def test_iand_ibool_with_bool() -> None:
    funcs = aadc.Functions()

    funcs.start_recording()

    x = aadc.array([1.0, aadc.idouble(2.0)])
    x[1].mark_as_input()

    np.all(x > 0.0)


def test_ior_ibool_with_bool() -> None:
    funcs = aadc.Functions()

    funcs.start_recording()

    x = aadc.array([1.0, aadc.idouble(2.0)])
    x[1].mark_as_input()

    np.any(x > 0.0)
