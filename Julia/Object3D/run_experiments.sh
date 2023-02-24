#!/bin/bash

julia --projec=. -t 4 src/experiments.jl | tee data/exp-$(date +%m%d%y%H%M)-$1.dat
