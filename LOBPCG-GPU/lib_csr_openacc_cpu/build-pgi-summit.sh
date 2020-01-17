#!/bin/bash
module list
module load pgi/19.5
module list
set -x
set -e
make -f Makefile.summit clean
make -f Makefile.summit
