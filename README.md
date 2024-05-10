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
