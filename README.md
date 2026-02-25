## This is Fork of Mase
  Source: [Mase](https://github.com/DeepWok/mase)

This work is to document and backup work done for the module Advanced Deep Learning Systems instructed by Aaron Zhao [Teaching](https://aaron-zhao123.github.io/teaching/adls/)

### NanoVLLM incompatibility with MASE (fx.symbolic_trace)
torch.fx.symbolic_trace does a symbolic execution of a model's `forward()` method. It passes a proxy object and records every operation into a compute graph. Compatibility fails when operations are not determinable or upon control flow. 

NanoVLLM's implementation of the Qwen3-0.6B model modifies the original pytorch model with custom modules that allow for run time optimisations such as Tensor Parallelism. Two notables are: `ColumnParallelLinear` and `RowParallelLinear`, to which FX can not decompose into graph IRs. Moreover, control flow such as  `if residual is None:` causes excessive fx graph breaks. Additionally, NanoVLLM's implementation of Qwen3 includes split passes. Instead of a single forward pass, the model computes multiple passes sequentially to stich an output. MASE only attempts to trace the `forward()` method. 

Hence this project will abandon using MASE for software level optimisations. Instead layer-wise or graph free frameworks will be prefered for this project. We plan to build an optimisation pipline based on trace-free approaches. Or fix the compatibilty of the Qwen3 implementation but risk degraded accuracy, and major debugging.

Available frameworks:
- AutoAWQ - Layer-wise Activtation aware Weight Quantisation (Qwen and Gemma Support), has been adopted by vLLM however, probably not on NanoVLLM
- LLM Compressor - Standard, Attention, KV Cache, AWQ, AutoRound, MXFP4 Quantisation <-- My Choice
  - SparseGPT for Pruning
  - GPTQ for quantisation





