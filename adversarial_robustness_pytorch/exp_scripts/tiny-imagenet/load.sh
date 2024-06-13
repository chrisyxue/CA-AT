#!/bin/sh

sbatch run_1_proj_layer.sb
sbatch run_1_proj_layer_resnet34.sb
sbatch run_1_proj_layer_resnet50.sb


sbatch run_1_proj.sb
sbatch run_1_proj_resnet34.sb
sbatch run_1_proj_resnet50.sb


sbatch run_1.sb
sbatch run_1_resnet34.sb
sbatch run_1_resnet50.sb