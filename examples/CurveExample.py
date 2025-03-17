# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Cash-flow analysis
#
# Copyright (&copy;) 2020 StatPro Italia srl
#
# This file is part of QuantLib, a free-software/open-source library
# for financial quantitative analysts and developers - https://www.quantlib.org/
#
# QuantLib is free software: you can redistribute it and/or modify it under the
# terms of the QuantLib license.  You should have received a copy of the
# license along with this program; if not, please email
# <quantlib-dev@lists.sf.net>. The license is also available online at
# <https://www.quantlib.org/license.shtml>.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the license for more details.

# %%
import numpy as np

import aadc

dates = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
rates = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]  # aadc.idouble(0.1)]
print(type(rates))
irates = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, aadc.idouble(0.09), aadc.idouble(0.1)]
print(type(irates))
iirates = [
    aadc.idouble(0.01),
    aadc.idouble(0.02),
    aadc.idouble(0.03),
    aadc.idouble(0.04),
    aadc.idouble(0.05),
    aadc.idouble(0.06),
    aadc.idouble(0.07),
    aadc.idouble(0.08),
    aadc.idouble(0.09),
    aadc.idouble(0.1),
]

nrates = np.ones(10)

forecast_curve = aadc.MyDemoCurve(nrates, nrates)
nrates[9] = aadc.idouble(0.1)  # automatically converts to double and this is OK
forecast_curve = aadc.MyDemoCurve(nrates, nrates)


forecast_curve = aadc.MyDemoCurve(dates, rates)
forecast_curve_ii = aadc.MyDemoCurve(dates, iirates)
forecast_curve = aadc.MyDemoCurve(dates, irates)

print("Start recording")
funcs = aadc.Functions()
funcs.start_recording()
arg1 = irates[9].mark_as_input()

# nRates[9] = iRates[9]   # Exception - Can we store idouble in numpy array?
# nn_forecast_curve = aadc.MyDemoCurve(nRates, nRates)

forecast_curve = aadc.MyDemoCurve(dates, irates)
forecast_curve_passive = aadc.MyDemoCurve(dates, rates)

print(forecast_curve(1))  # returns active double
print(forecast_curve_passive(1))  # returns double
print(forecast_curve_passive(aadc.idouble(1)))  # returns double
print(forecast_curve_passive(irates[9]))  # returns active double

exp1 = irates[9] + irates[9]
print(exp1)  # returns active idouble

print(irates[8] + irates[8])  # returns native double?
print(float(irates[8] + irates[8]))  # returns native double

try:
    print(float(exp1))  # returns exception
except Exception as e:
    print(e)

funcs.stop_recording()

print("Stop recording")


# forecast_curve = aadc.MyDemoCurve(aadc.idouble(0.03))

print(forecast_curve(1))
