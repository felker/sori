// make -f makefile_dprf
// /usr/local/cuda/bin/nvprof ./dprf.exe forest_245_15_dan.h5 
// gcc dprf.c -o dprf -lhdf5 -std=c99
#ifndef DPRF_H
#define DPRF_H

#include "dprf_kernel.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <hdf5.h>
#include <cuda.h>

#ifdef __cplusplus
  extern "C" {
#endif
int read_model_sizes(char *fname, int *n_trees, int *n_classes, int *n_nodes);

int read_model_data(char *fname, int *feature, int *children_left, int *children_right, int *tree_start, float *threshold, float *value);

int read_input_sizes(char *fname, int *n_features, int *n_times);

int read_input_data(char *fname, float *X, float *time);

int eval_tree(int tree_index, float *X_eval, int *feature, int*children_left, int *children_right, int *tree_start, float *threshold, float *value, int n_classes, float *result, float *feature_contributions);
#ifdef __cplusplus
  };
#endif

#endif
