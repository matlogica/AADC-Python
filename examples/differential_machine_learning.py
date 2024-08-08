import numpy as np
import aadc

# Example of AADC interface use for Differential Machine Learning. Here we need intermediate state variables with values and adjoints
# See https://github.com/differential-machine-learning

#  Regular python code to calculate payoff of a simple Monte Carlo simulation on just one path

state = np.array([ 100.0, 10.0])  # initial state of the model. Here we assume we have 2 "assets"

T = 5  # number of time steps

payoff = 0.0
for t in range(1, T):
    new_state = state + np.random.normal(0, 0.1, state.__len__())
    state = new_state
#    if t == T - 1:
#        payoff = np.maximum(0, 100 - state)
    payoff += np.sum(state)

print(f"Payoff: {payoff}")

# AADC version
state = aadc.array([aadc.idouble(100.0), aadc.idouble(10.0)])

funcs = aadc.Functions()  # object to hold valuation graph as compiled kernel

funcs.start_recording()  # Record 1 MC path

t0_state_arg = [val.mark_as_input() for val in state]  # here we define t0 state as input. I.e. we can change asset levels and we calc adjoints

randoms_arg = []  # array to collect arguments for random numbers that drive the path

print(t0_state_arg)

payoff = 0.0
int_state_arg = [] # array to collect arguments(handles) for state variables that are calculated in the path. We use them to access adjoints
int_state_res = [] # array to collect results(handles) for state variables that are calculated in the path. We use them to access values

for t in range(1, T):
    random = aadc.array(np.random.normal(0, 0.1, len(state)))
    randoms_arg.append(random.mark_as_input_no_diff()) # we do not need adjoints for random numbers

    new_state = state + random
    int_state_arg = int_state_arg + [val.mark_as_diff() for val in new_state] # collect variable handles
    int_state_res = int_state_res + [val.mark_as_output() for val in new_state] # collect variable handles. Actually same ids, just different type

    state = new_state

#    if t == T - 1:
#        payoff = np.sum(np.maximum(0, 100.0 - state))
    payoff = payoff + np.sum(state)

payoff_res = payoff.mark_as_output()

funcs.stop_recording()

#print(randoms_arg)

# Prepare inputs for the kernel

NumMC = 10
inputs = {}
for args in randoms_arg:
    for arg in args:
        inputs.update({arg: np.random.normal(0, 0.1, NumMC)})

for arg in t0_state_arg:
    # we can reuse aadc kernel for arbitrary state values. I.e. do scenarios etc.
    inputs.update({arg: 100.0})

#print(inputs)

# return payoff and gradients w.r.t. state variables
request = {payoff_res: t0_state_arg + int_state_arg}  
for res in int_state_res:
    # return intermediate state variables    
    request.update({res: []})

workers = aadc.ThreadPool(4)
res = aadc.evaluate(funcs, request, inputs, workers)   # run the kernel

print("Result", res[0][payoff_res])
for state_i in range(len(state)):
    print("dPayoff/dState[0][", state_i, "]", res[1][payoff_res][int_state_arg[state_i]])

for t in range(1, T):
    for state_i in range(len(state)):
        print("State[", t, "][", state_i, "]", res[0][int_state_res[state_i + (t-1)*len(state)]])
        print("dPayoff/dState[", t, "][", state_i, "]", res[1][payoff_res][int_state_arg[state_i + (t-1)*len(state)]])

