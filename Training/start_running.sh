#!/bin/bash
SINGULARITY_DISABLE_CACHE=True singularity pull docker://enricozl/gan4ds:argan-runner
nohup singularity run --nv gan4ds_argan-runner.sif > gan4ds.out 2> gan4ds.err < /dev/null &