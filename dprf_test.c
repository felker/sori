// make -f makefile_dprf
// /usr/local/cuda/bin/nvprof ./dprf.exe forest_245_15_dan.h5
// gcc dprf.c -o dprf -lhdf5 -std=c99
#include "dprf.h"
//#include "dprf_kernel.h"
//#include <stdio.h>
//#include <stdlib.h>
//#include <string.h>
//#include <hdf5.h>
//#include <cuda.h>

int main(int argc, char* argv[]) {

    FILE *fp;
    fp = fopen("dprf_test.out","w");

    int n_points = NPOINTS;
    int n_trees;
    int n_classes;
    int n_nodes;

    int n_features;
    int n_times;
    float *X_input;
    float *time_input;

    int *feature;
    int *children_left;
    int *children_right;
    int *tree_start;
    float *threshold;
    float *value;
    float *tree_result;
    float input_data[NFEATURES*NPOINTS] = {0};
    float feature_contributions[NFEATURES*NPOINTS] = {0};
    float forest_feature_contributions[NFEATURES*NPOINTS] = {0};
    float *total_result;
    int calculate_contributions = 0;
    int k = 0;
    int i = 0;
    int j = 0;

    int* gpu_n_points;
    int* gpu_n_trees;
    int* gpu_n_classes;
    int* gpu_n_nodes;
    int* gpu_feature;
    int* gpu_children_left;
    int* gpu_children_right;
    int* gpu_tree_start;
    float* gpu_threshold;
    float* gpu_value;
    float* gpu_tree_result;
    float* gpu_input_data;
    float* gpu_feature_contributions;
    float* gpu_feature_contributions_forest;
    float* gpu_total_result;
    int* gpu_calculate_contributions;



    char input_fname[80];
    sprintf(input_fname,"%s", argv[2]);
    read_input_sizes(input_fname, &n_features, &n_times);
    printf("no of features: %d\n", n_features);
    printf("no of times: %d\n", n_times);
    X_input = (float *) malloc(n_features * n_times * sizeof(float));
    time_input = (float *) malloc(n_times * sizeof(float));

    read_input_data(input_fname, X_input, time_input); //X_input[n_features*time_index+feature_index]

    int stride = 50;
    for (k=0;k<NPOINTS;k++){
        for (j=0;j<NFEATURES;j++){
            input_data[k*NFEATURES+j] = X_input[stride*k*NFEATURES+j];
        }
    }

    char fname[80];
    sprintf(fname, "%s", argv[1]);

    read_model_sizes(fname, &n_trees, &n_classes, &n_nodes);
    printf("no of trees: %d\n", n_trees);
    printf("no of classes: %d\n", n_classes);
    printf("no of nodes: %d\n", n_nodes);

    feature = (int *) malloc(n_nodes * sizeof(int));
    children_left = (int *) malloc(n_nodes * sizeof(int));
    children_right = (int *) malloc(n_nodes * sizeof(int));
    tree_start = (int *) malloc(n_nodes * sizeof(int));
    threshold = (float *) malloc(n_nodes * sizeof(float));
    value = (float *) malloc(n_nodes * n_classes * sizeof(float));
    tree_result = (float *) malloc(NPOINTS * n_trees * sizeof(float));
    total_result = (float *) malloc(NPOINTS * sizeof(float));

  cudaMalloc((void**) &gpu_n_points, sizeof(int));
    cudaMalloc((void**) &gpu_n_trees, sizeof(int));
    cudaMalloc((void**) &gpu_n_classes, sizeof(int));
    cudaMalloc((void**) &gpu_n_nodes, sizeof(int));
    cudaMalloc((void**) &gpu_feature, n_nodes * sizeof(int));
    cudaMalloc((void**) &gpu_children_left, n_nodes * sizeof(int));
    cudaMalloc((void**) &gpu_children_right, n_nodes * sizeof(int));
    cudaMalloc((void**) &gpu_tree_start, n_nodes * sizeof(int));
    cudaMalloc((void**) &gpu_threshold, n_nodes * sizeof(float));
    cudaMalloc((void**) &gpu_value, n_nodes * n_classes * sizeof(float));
    cudaMalloc((void**) &gpu_tree_result, NPOINTS * n_trees * sizeof(float));
    cudaMalloc((void**) &gpu_input_data, NPOINTS * NFEATURES * sizeof(float));
    cudaMalloc((void**) &gpu_feature_contributions, NPOINTS * NFEATURES * sizeof(float));
    cudaMalloc((void**) &gpu_feature_contributions_forest, NPOINTS * NFEATURES * n_trees * sizeof(float));
    cudaMalloc((void**) &gpu_total_result, NPOINTS * sizeof(float));
    cudaMalloc((void**) &gpu_calculate_contributions, sizeof(int));

    read_model_data(fname, feature, children_left, children_right, tree_start, threshold, value);
    printf("feature[0]: %d\n", feature[0]);
    printf("children_left[0]: %d\n", children_left[0]);
    printf("children_right[0]: %d\n", children_right[0]);
    printf("tree_start[0]: %d\n", tree_start[0]);
    printf("threshold[0]: %f\n", threshold[0]);
    printf("value[node0,class0]: %f\n", value[0]);
  printf("value[node0,class1]: %f\n", value[1]);
  printf("Evaluating tree...\n");

    cudaMemcpy(gpu_n_points, &n_points, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_n_trees, &n_trees, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_n_classes, &n_classes, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_n_nodes, &n_nodes, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_feature,feature,n_nodes * sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_children_left,children_left,n_nodes * sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_children_right,children_right,n_nodes * sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_tree_start,tree_start,n_nodes * sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_threshold,threshold,n_nodes * sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_value,value,n_nodes * n_classes * sizeof(float),cudaMemcpyHostToDevice);
    //cudaMemcpy(gpu_tree_result,tree_result,NPOINTS * n_trees * sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_input_data,input_data,NPOINTS * NFEATURES * sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_calculate_contributions,&calculate_contributions,sizeof(int),cudaMemcpyHostToDevice);


  // Evaluate tree for each point
    float sum_proba[NPOINTS] = {0};

    float result = 0;
    float tree_feature_contributions[NFEATURES] = {0};
    for (k=0; k<NPOINTS; k++){
        for (i=0; i<n_trees; i++){

            eval_tree(i,&input_data[k*NFEATURES], feature, children_left, children_right, tree_start, threshold, value, n_classes, &result, tree_feature_contributions);
            tree_result[k*n_trees+i] = result;
            //printf("result: %f\n", result);
            sum_proba[k]    += result;
            //printf("tree: %d\n",i);
            for (j=0; j<NFEATURES; j++){
                forest_feature_contributions[k*NFEATURES+j] += tree_feature_contributions[j];
                //printf("Feature contribution[%d,%d]: %f\n", k, j, tree_feature_contributions[j]);
            }
        }
        sum_proba[k] = sum_proba[k]/((float)n_trees);
        //printf("Result[%d]: %f\n", k,sum_proba[k]);

        for (j=0; j<NFEATURES; j++){
            forest_feature_contributions[k*NFEATURES+j] = forest_feature_contributions[k*NFEATURES+j]/((float)n_trees);
            //printf("Feature contribution[%d]: %f\n", j, forest_feature_contributions[k*NFEATURES+j]);
        }
    }
    //printf("Result: %f\n", sum_proba);
    //Repeat with gpu kernel

    eval_forest_launcher(n_trees, gpu_n_points, gpu_n_trees, gpu_input_data, gpu_feature, gpu_children_left, gpu_children_right, gpu_tree_start, gpu_threshold, gpu_value, gpu_n_classes, gpu_tree_result, gpu_feature_contributions, gpu_feature_contributions_forest, gpu_total_result,gpu_calculate_contributions);

    cudaMemcpy(tree_result,gpu_tree_result,NPOINTS * n_trees * sizeof(float),cudaMemcpyDeviceToHost);
  cudaMemcpy(total_result,gpu_total_result,NPOINTS * sizeof(float),cudaMemcpyDeviceToHost);

  cudaMemcpy(feature_contributions,gpu_feature_contributions, NPOINTS * NFEATURES * sizeof(float),cudaMemcpyDeviceToHost);

    //sum_proba = 0.0;
    //for (i=0; i<n_trees; i++){
        //printf("result: %f\n", result);
    //  sum_proba   += tree_result[i];
    //}
    //sum_proba = sum_proba/((float)n_trees);

    //printf("Result (gpu): %f\n", sum_proba);
    for (k=0; k<NPOINTS; k++){
        fprintf(fp,"Result[%d] (gpu): %f\n", k,total_result[k]/(float)n_trees);
        for (j=0; j<NFEATURES; j++){
            feature_contributions[k*NFEATURES+j] = feature_contributions[k*NFEATURES+j]/((float)n_trees);
            fprintf(fp,"Feature contribution[%d,%d]: %f\n", k,j, feature_contributions[k*NFEATURES+j]);
        //sum_proba = sum_proba + forest_feature_contributions[j];
        }
    }

    cudaFree(gpu_n_points);
    cudaFree(gpu_n_trees);
    cudaFree(gpu_n_classes);
    cudaFree(gpu_n_nodes);
    cudaFree(gpu_feature);
    cudaFree(gpu_children_left);
    cudaFree(gpu_children_right);
    cudaFree(gpu_tree_start);
    cudaFree(gpu_threshold);
    cudaFree(gpu_value);
    cudaFree(gpu_tree_result);
    cudaFree(gpu_input_data);
    cudaFree(gpu_total_result);
    cudaFree(gpu_feature_contributions);
    cudaFree(gpu_feature_contributions_forest);



    free(feature);
    free(children_left);
    free(children_right);
    free(tree_start);
    free(threshold);
    free(value);
    free(tree_result);
    free(X_input);
    free(time_input);

    fclose(fp);
    //cudaDeviceReset();
    exit(0);
}
