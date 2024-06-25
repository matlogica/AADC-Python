# AADC-Python

## Introduction

This library provides high-performance components leveraging the hardware acceleration support and automatic differentiation. It uses MatLogica's specialised run-time compiler, AADC (AAD Compiler) to generate efficient binary kernels for execution of the original and the adjoint functions on the fly.

The solution has two main aspects (that can be used separately):

Accelerating simulations, such as Monte-Carlo simulations, historical analysis, and "what if" scenario analysis, bump & revalue, etc.
Automatic Differentiation: AADC can speed up the AAD method itself, and deliver pricing and scenario analysis simply and effectively, in a way that is unattainable with competing products.
AADC uses Code Generation AAD™ approach that combines Code Transformation and Operator Overloading to efficiently extract the valuation graph and, at runtime, generate efficient binary kernels replicating the original program, and where required, it's adjoint. AADC utilizes native CPU vectorization and multi-threading, delivering performance comparable to a GPU. This results in faster computing of the model and its first and higher-order derivatives. The approach is particularly useful for models with many parameters that require frequent gradient updates during training.

The performance of the runtime graph compilation is crucial, because it's now part of the overall model execution. This is why a specialised graph compiler needs to be used - any off-the-shelf compiler would introduce substantial delay, making the approach not practically viable. This is where TensorFlow for finance project failed.

Please join our discord: https://discord.gg/YqYDfWj6

## Use Cases

The solution can be used for greenfield or existing projects. It allows developers to focus on modeling, rather than performance optimisations, greatly improving time-to-market for new features, simplifying IT architecture and infrastructure.

## In Finance

The solution is used for speeding up and computing derivatives for various financial models, including pricing exotic derivatives, payoff languages, curve building, XVA, EPE, Loss-given-default, and others.

It enables transitioning to Live Risk from Batch processing by applying the Automated Implicit Function Theorem.

Stress-Testing, Back-Testing, What-if analysis, VaR can be accelerated with the solution.

## Neural Networks

The solution can be used to develop new Neural Network Architectures. Refer to research:

https://arxiv.org/abs/2207.03577



## Other applications

Life science, physics, drug research, disease diagnosis (https://elib.uni-stuttgart.de/bitstream/11682/13787/7/PhD_Thesis_Ivan.pdf) benefit from simplifying development, automatic differentiation and improving performance of simulations.

* Topology Optimization 101: How to Use Algorithmic Models to Create Lightweight Design:  https://formlabs.com/blog/topology-optimization/

* AuTO: a framework for Automatic differentiation in Topology Optimization:  https://link.springer.com/article/10.1007/s00158-021-03025-8

* A set of examples that use AD for several purposes with simulation:  https://www.dolfin-adjoint.org/en/stable/documentation/examples.html.


## Package contents

The package includes 2 projects: basic examples and QuantLib(https://www.quantlib.org/) examples.

Please refer to Manual.pdf on the functionality and uses.

## Installation and API reference

To install the `aadc` package, use pip:

```sh
pip install aadc
```

## Usage

You can use most of your existing code without any modifications. Only the inputs to the evaluation graph you want to record using AADC have to be explicitly initialized as active floating point types.

A stand-alone `aadc.idboule` type stores a single active double, while `aadc.array()` function returns a multi-dimensional `AADCArray` of `idboule`s and can be used as a drop-in replacement of NumPy's `np.array()`.

---

### `evaluate`

```python
aadc.evaluate(funcs: Kernel, request: dict, inputs: dict, workers: ThreadPool) -> list
```

**Description:**
The `evaluate` function executes the recorded kernel `funcs` for multiple inputs, and returns the evaluated results and sensitivities according to `request`.

**Parameters:**
- `funcs` (Kernel): The recorded kernel object.
- `request` (dict): A dictionary specifying the outputs and the gradients required. The key (`AADCResult`) is the workspace index of an output (see below), and the value is a list of `AADCArgument` - indices of inputs whose adjoints are requested,
- `inputs` (dict): A dictionary of inputs for the evaluation. The key (`AADCArgument`) identifies the input node, and the value is an `np.array` of all scenarios for this input.
- `workers` (ThreadPool): A thread pool to manage parallel execution.

**Returns:**
- `list`: [{
        outputArg -> outputValues
    },
    {
        outputArg -> {
            inputArg -> adjoints
        }
    }]

**Example:**

```python
results = aadc.evaluate(funcs, request, inputs, workers)
```

---

## Classes and Methods

### `Kernel`

```python
class aadc.Kernel:
    def start_recording(self) -> None
    def stop_recording(self) -> None
```

**Description:**
The `Kernel` class provides methods to start and stop recording the calculation graph and stores the JIT-compiled AADC machine code kernels for forward and reverse (adjoint) passes

**Methods:**

#### `start_recording`

```python
def start_recording() -> None
```

**Description:**
Starts recording.

**Returns:**
- `None`

**Example:**

```python
funcs = aadc.Kernel()
funcs.start_recording()
```

#### `stop_recording`

```python
def stop_recording() -> None
```

**Description:**
Stops recording the operations.

**Returns:**
- `None`

**Example:**

```python
funcs.stop_recording()
```

---

### `ThreadPool`

```python
class aadc.ThreadPool:
    def __init__(self, num_threads: int)
```

**Description:**
The `ThreadPool` class manages a pool of threads for parallel execution.

**Parameters:**
- `num_threads` (int): The number of threads to include in the pool.

**Example:**

```python
workers = aadc.ThreadPool(4)
```

---

## Methods

### `mark_as_input`

```python
def mark_as_input() -> AADCArgument
```

**Description:**
Marks the variable as an input of the computation being recorded. Both `idouble` and `AADCArray` implement this method

**Returns:**
- `AADCArgument`: An index into the kernel workspace where the input is stored before the forward pass / adjoints can be read after the reverse pass; or a list of `AADCArgument` in the array case.

**Example:**

```python
stock_arg = stock_price.mark_as_input()
```

### `mark_as_output`

```python
def mark_as_output() -> AADCResult
```

**Description:**
Marks the variable as an output of the computation being recorded.

**Returns:**
- `AADCResult`: An index into the kernel workspace where the output can be read after the forward pass / adjoint should be set to 1.0 before the reverse pass

**Example:**

```python
price_res = price.mark_as_output()
```

---

### record

```python
aadc.record(computation: Callable, x0: NDArray, params: tuple, bump_size: float) -> CurreidRecordedFunction
```

**Description:**
Given `computation` - a function `f(x, *params)` `record` records the computation passing `x0` and `params`

**Returns:**
- `CurreidRecordedFunction`: A structure with 3 class methods
    - `func`: the JIT-compiled version of `computation` closed over `params` with the signature `f(x)`
    - `jac`: a function returning `computation`s Jacobian w.r.t. `x`, calculated using finite differences with `bump_size`
    - `set_params`: use this method to change the `params` in `func`s closure

**Example:**

```python
rec = aadc.record(f, np.zeros(shape), params=[0.0, 0.0], bump_size=1e-10)
x = np.ones(shape)
rec.set_params(a)

assert np.allclose(rec.func(x), f(x, a))
```
