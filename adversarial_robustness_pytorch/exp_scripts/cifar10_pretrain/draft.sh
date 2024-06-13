#!/bin/sh

sbatch run_1_proj_gradnorm.sb
sbatch run_1_proj.sb

sbatch run_1_proj_resnet34_gradnorm.sb
sbatch run_1_proj_resnet34.sb