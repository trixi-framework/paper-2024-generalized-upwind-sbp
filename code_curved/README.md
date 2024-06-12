# Curvilinear numerical experiments

This directory contains all source code and mesh files required to reproduce
the numerical experiments for the curvilinear upwind SBP solver presented
in the paper. It is developed for Julia v1.10.0.

To reproduce the numerical experiments, start Julia in this directory and
execute the following commands in the Julia REPL. We recommend starting
Julia with several threads to improve the runtime performance.

```julia
julia> include("code.jl")

julia> free_stream_preservation_generate_data()
```