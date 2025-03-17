import math

import numpy as np

import aadc

#math.exp = aadc.math.exp


def f(x, y, z):
    return np.exp(x / y + z)


funcs = aadc.Functions()
x = aadc.idouble(1.0)
y = aadc.idouble(2.0)
z = aadc.idouble(3.0)

funcs.start_recording()
x_arg = x.mark_as_input()
y_arg = y.mark_as_input()
z_arg = z.mark_as_input()

f = f(x, y, z) + x

f_res = f.mark_as_output()
funcs.stop_recording()
funcs.print_passive_extract_locations()

print("f=", f)

inputs = {
    x_arg: (1.0 * np.ones((120))),
    y_arg: (2.0),
    z_arg: (3.0),
}

request = {f_res: [x_arg, y_arg, z_arg]}  ## key: what output, value: what gradients are needed

workers = aadc.ThreadPool(4)
res = aadc.evaluate(funcs, request, inputs, workers)

print("Result", res[0][f_res])
print("df/dx", res[1][f_res][x_arg])
print("df/dy", res[1][f_res][y_arg])
print("df/dz", res[1][f_res][z_arg])

if False:
    res = aadc.evaluate(funcs, request, inputs)
    print("Result", res[0][f_res])
    print("df/dx", res[1][f_res][x_arg])
    print("df/dy", res[1][f_res][y_arg])
    print("df/dz", res[1][f_res][z_arg])

print("Done!")
