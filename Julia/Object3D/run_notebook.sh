#!/bin/bash

julia --projec=. -e 'ENV["JULIA_NUM_THREADS"] = 4; using IJulia; notebook()'
