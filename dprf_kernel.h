#ifndef DPRF_KERNEL_H
#define DPRF_KERNEL_H

#include "cuda.h"
#include "cuda_runtime.h"

#define TPB 640
#define MAX_TREES 300
#define NFEATURES 8
#define NPOINTS 200

#ifdef __cplusplus
  extern "C" {
#endif
void eval_forest_launcher(int cpu_n_trees, int *n_points, int *n_trees, float *X_eval, int *feature, int*children_left, int *children_right, int *tree_start, float *threshold, float *value, int *n_classes, float *result, float *feature_contributions, float *feature_contributions_forest, float *total_result, int *calculate_contributions);
#ifdef __cplusplus
  };
#endif
#endif
