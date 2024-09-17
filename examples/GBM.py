import numpy as np
import aadc
import math
import time

class Params:
    def __init__(self):
        self.NumberOfScenarios = 10000
        self.NumberOfTimesteps = 252

class Trade:
    def __init__(self):
        self.time_to_maturity = 1.0
        self.risk_free_rate = 0.02
        self.volatility = 0.2
        self.stock_price = 100.0

class AsianOption:
    # simulate risk factors using GBM stochastic differential equation
    def SimulateRiskFactor(self, trade):
        prices = []
        timestep = trade.time_to_maturity / self.Configuration.NumberOfTimesteps
        for scenarioNumber in range(self.Configuration.NumberOfScenarios):
            running_sum_price = 0.0
            stock_price = trade.stock_price
            for timestepNumber in range(self.Configuration.NumberOfTimesteps):
                drift = (trade.risk_free_rate - 0.5 * (trade.volatility ** 2)) * timestep
                uncertainty = trade.volatility * np.sqrt(timestep) * self.random_normals[scenarioNumber][timestepNumber] # np.random.normal(0, 1)
                stock_price = stock_price * np.exp(drift + uncertainty)
                running_sum_price += stock_price
            prices.append(running_sum_price/self.Configuration.NumberOfTimesteps)
        return prices




option = AsianOption()
option.Configuration = Params()
trade = Trade()
# fix seed
np.random.seed(17)

# simulate risk factors
timer_start = time.time()
option.random_normals = np.random.normal(0, 1, (option.Configuration.NumberOfScenarios, option.Configuration.NumberOfTimesteps))
print("Time to simulate risk factors:", time.time() - timer_start)

timer_start = time.time()
prices = option.SimulateRiskFactor(trade)
print(np.average(prices))
print("Time to price:", time.time() - timer_start)
# bump-and-revalue risk

h = 1e-6
trade.stock_price += h
print("dP/dS:", (np.average(option.SimulateRiskFactor(trade)) - np.average(prices)) / h)
trade.stock_price -= h
trade.risk_free_rate += h
print("dP/dR:", (np.average(option.SimulateRiskFactor(trade)) - np.average(prices)) / h)
trade.risk_free_rate -= h
trade.volatility += h
print("dP/dV:", (np.average(option.SimulateRiskFactor(trade)) - np.average(prices)) / h)
trade.volatility -= h
print("Time to bump-and-revalue:", time.time() - timer_start)

# Now lets do AADC way

option.Configuration.NumberOfScenarios_copy = option.Configuration.NumberOfScenarios
option.Configuration.NumberOfScenarios = 1

option.random_normals_copy = option.random_normals

option.random_normals = aadc.array(np.random.normal(0, 1, (option.Configuration.NumberOfScenarios, option.Configuration.NumberOfTimesteps)))

timer_start = time.time()
kernel = aadc.Functions()
kernel.start_recording()

random_normals_arg = option.random_normals.mark_as_input_no_diff()   # we can set new values, but we don't want to differentiate it

trade.stock_price = aadc.idouble(trade.stock_price)
stock_price_arg = trade.stock_price.mark_as_input() # we can set new values and differentiate it

trade.risk_free_rate = aadc.idouble(trade.risk_free_rate)
risk_free_rate_arg = trade.risk_free_rate.mark_as_input()

trade.volatility = aadc.idouble(trade.volatility)
volatility_arg = trade.volatility.mark_as_input()

one_path_prices = option.SimulateRiskFactor(trade)
#print(random_normals_arg)
print(one_path_prices[0])

one_path_prices_res = one_path_prices[0].mark_as_output()

kernel.stop_recording()
#print(kernel.recording_stats("GBM"))
print("Time to do AADC recording:", time.time() - timer_start)

# restore original values
trade.stock_price = 100.0
trade.risk_free_rate = 0.02
trade.volatility = 0.2
option.random_normals = option.random_normals_copy
option.Configuration.NumberOfScenarios = option.Configuration.NumberOfScenarios_copy


inputs = {}

inputs[stock_price_arg] = trade.stock_price
inputs[risk_free_rate_arg] = trade.risk_free_rate
inputs[volatility_arg] = trade.volatility

for random_i in range(random_normals_arg[0].size):
    inputs[random_normals_arg[0][random_i]] = option.random_normals_copy[:, random_i].copy()

# AADC pricing only:
timer_start = time.time()
request = {
    one_path_prices_res : []
}

results = aadc.evaluate(kernel, request, inputs, aadc.ThreadPool(1))
print("AADC time to price:", time.time() - timer_start)

print("price:", np.average(results[0][one_path_prices_res]))

# AADC price + AAD first order risk
timer_start = time.time()
request = {
    one_path_prices_res : [stock_price_arg, risk_free_rate_arg, volatility_arg]
}

results = aadc.evaluate(kernel, request, inputs, aadc.ThreadPool(1))
print("AADC time to price + AAD:", time.time() - timer_start)

print("price:", np.average(results[0][one_path_prices_res]))
print("dP/dS:", np.average(results[1][one_path_prices_res][stock_price_arg]))
print("dP/dR:", np.average(results[1][one_path_prices_res][risk_free_rate_arg]))
print("dP/dV:", np.average(results[1][one_path_prices_res][volatility_arg]))

# Use bump-and-revalue of AAD derivatives for second order

inputs[stock_price_arg] = trade.stock_price + h
results_up = aadc.evaluate(kernel, request, inputs, aadc.ThreadPool(1))
inputs[stock_price_arg] = trade.stock_price - h
results_down = aadc.evaluate(kernel, request, inputs, aadc.ThreadPool(1))

print("d2P/dS2:", (np.average(results_up[0][one_path_prices_res]) - 2 * np.average(results[0][one_path_prices_res]) + np.average(results_down[0][one_path_prices_res])) / (h ** 2))


# kernel can be pickled and saved for later use. For example to do intra-day pricing

