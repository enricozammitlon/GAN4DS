#!/bin/bash
mkdir out/
export SINGULARITY_DISABLE_CACHE=True
singularity pull docker://enricozl/gan4ds:argan-runner
nohup singularity run --nv --bind $(pwd)/out:out/ gan4ds_argan-runner.sif > gan4ds.out 2> gan4ds.err < /dev/null &
