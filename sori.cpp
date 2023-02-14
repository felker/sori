#include "sori_kernel.h"
#include "dprf.h"
#include "mytime.h"
#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
#include <cstdlib>
#include <ctime>


//Print matrix A(nr_rows_A, nr_cols_A) storage in column-major format
void print_matrix(const float *A, int nr_rows_A, int nr_cols_A) {
    for(int i = 0; i < nr_rows_A; ++i){
        for(int j = 0; j < nr_cols_A; ++j){
            std::cout << A[j * nr_rows_A + i] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void fprint_matrix(FILE *fp, const float *A, int nr_rows_A, int nr_cols_A) {
    for(int i = 0; i < nr_rows_A; ++i){
        for(int j = 0; j < nr_cols_A; ++j){
            fprintf(fp,"%g",A[j * nr_rows_A + i]);
      fprintf(fp," ");
        }
        fprintf(fp,"\n");
    }
    fprintf(fp,"\n");
}

void print_matrix_bool(const bool *A, int nr_rows_A, int nr_cols_A) {
    for(int i = 0; i < nr_rows_A; ++i){
        for(int j = 0; j < nr_cols_A; ++j){
            std::cout << A[j * nr_rows_A + i] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void fprint_matrix_bool(FILE *fp, const bool *A, int nr_rows_A, int nr_cols_A) {
    for(int i = 0; i < nr_rows_A; ++i){
        for(int j = 0; j < nr_cols_A; ++j){
            fprintf(fp,"%d",A[j * nr_rows_A + i]);
      fprintf(fp," ");
        }
        fprintf(fp,"\n");
    }
    fprintf(fp,"\n");
}

void print_matrix_int(const int *A, int nr_rows_A, int nr_cols_A) {
    for(int i = 0; i < nr_rows_A; ++i){
        for(int j = 0; j < nr_cols_A; ++j){
            std::cout << A[j * nr_rows_A + i] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void fprint_matrix_int(FILE *fp, const int *A, int nr_rows_A, int nr_cols_A) {
    for(int i = 0; i < nr_rows_A; ++i){
        for(int j = 0; j < nr_cols_A; ++j){
            fprintf(fp,"%d",A[j * nr_rows_A + i]);
      fprintf(fp," ");
        }
        fprintf(fp,"\n");
    }
    fprintf(fp,"\n");
}

int argmax(float *array, int n){
    int out = 0;
    for(int i = 0; i < n; ++i){
        // printf("array[%d] = %f. array[%d] = %f.",i,array[i],out,array[out]);
        if (array[i]    > array[out]){
            out = i;
        }
    }
    return out;
}

int main(int argc, char* argv[]) {
  //Set up streams
  const int NSTREAMS = 3;

  //sizes
  const int NPTS = 900;
  const int NLMI = 3;
  const int NPOP = 400;
  const int NCON = NLMI * NPOP;
  const int NDIM = 2;
  const int TPB = 32;
  const int TPB_DPRF = 640;
  const int NGEN = 25;
  const int NTOURN = 4;
  const int NELITE = 10;
  int best = -1;

  cudaStream_t* streams;
  streams = new cudaStream_t[NSTREAMS];
  for(int i=0;i<NSTREAMS; i++)
    cudaStreamCreate ( &(streams[i]));

  //Load dprf model from hdf5
  char fname[80];
  sprintf(fname, "%s", argv[1]);
  printf("Model filename: %s\n",fname);

    int n_trees;
    int n_classes;
    int n_nodes;


    int *feature;
    int *children_left;
    int *children_right;
    int *tree_start;
    float *threshold;
    float *value;


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





    read_model_data(fname, feature, children_left, children_right, tree_start, threshold, value);
    printf("feature[0]: %d\n", feature[0]);
    printf("children_left[0]: %d\n", children_left[0]);
    printf("children_right[0]: %d\n", children_right[0]);
    printf("tree_start[0]: %d\n", tree_start[0]);
    printf("threshold[0]: %f\n", threshold[0]);
    printf("value[node0,class0]: %f\n", value[0]);
  printf("value[node0,class1]: %f\n", value[1]);

    int* gpu_n_trees;
    int* gpu_n_classes;
    int* gpu_n_nodes;
    int* gpu_feature;
    int* gpu_children_left;
    int* gpu_children_right;
    int* gpu_tree_start;
    float* gpu_threshold;
    float* gpu_value;

    cudaMalloc((void**) &gpu_n_trees, sizeof(int));
    cudaMalloc((void**) &gpu_n_classes, sizeof(int));
    cudaMalloc((void**) &gpu_n_nodes, sizeof(int));
    cudaMalloc((void**) &gpu_feature, n_nodes * sizeof(int));
    cudaMalloc((void**) &gpu_children_left, n_nodes * sizeof(int));
    cudaMalloc((void**) &gpu_children_right, n_nodes * sizeof(int));
    cudaMalloc((void**) &gpu_tree_start, n_nodes * sizeof(int));
    cudaMalloc((void**) &gpu_threshold, n_nodes * sizeof(float));
    cudaMalloc((void**) &gpu_value, n_nodes * n_classes * sizeof(float));

    cudaMemcpy(gpu_n_trees, &n_trees, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_n_classes, &n_classes, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_n_nodes, &n_nodes, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_feature,feature,n_nodes * sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_children_left,children_left,n_nodes * sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_children_right,children_right,n_nodes * sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_tree_start,tree_start,n_nodes * sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_threshold,threshold,n_nodes * sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_value,value,n_nodes * n_classes * sizeof(float),cudaMemcpyHostToDevice);


    float scan_ranges[2][8] = {
      {0.1e6, 1e5, 1e19, 0.05, 0.1, 0.02, 0.1, 0.2},
      {0.02e6, 2e4, 4e18, 0.005, 0.01, 0.0025, 0.01, 0.02}
    };

   //load dprf input data
    int n_features;
    int n_times;
    float *X_input;
    float *time_input;


  char input_fname[80];
    sprintf(input_fname,"%s", argv[2]);
    read_input_sizes(input_fname, &n_features, &n_times);
    printf("no of features: %d\n", n_features);
    printf("no of times: %d\n", n_times);
    X_input = (float *) malloc(n_features * n_times * sizeof(float));
    time_input = (float *) malloc(n_times * sizeof(float));

    read_input_data(input_fname, X_input, time_input); //X_input[n_features*time_index+feature_index]

    // Set up variables for evaluating dprf at single operating point

    int n_operating_points = 1;
    int* gpu_n_operating_points;
    cudaMalloc((void**) &gpu_n_operating_points, sizeof(int));
    cudaMemcpy(gpu_n_operating_points, &n_operating_points, sizeof(int), cudaMemcpyHostToDevice);


    float *tree_result_op;
    float current_operating_point[NFEATURES] = {0};
    float feature_contributions_op[NFEATURES] = {0};
    float forest_feature_contributions_op[NFEATURES] = {0};
    float *total_result_op;
    int calculate_contributions_op = 1;

    tree_result_op = (float *) malloc(n_operating_points * n_trees * sizeof(float));
    total_result_op = (float *) malloc(n_operating_points * sizeof(float));


    float* gpu_tree_result_op;
    float* gpu_current_operating_point;
    float* gpu_feature_contributions_op;
    float* gpu_feature_contributions_forest_op;
    float* gpu_total_result_op;
    int* gpu_calculate_contributions_op;
    cudaMalloc((void**) &gpu_tree_result_op, n_operating_points * n_trees * sizeof(float));

    cudaMalloc((void**) &gpu_current_operating_point, NFEATURES * sizeof(float));
    cudaMalloc((void**) &gpu_feature_contributions_op, n_operating_points * NFEATURES * sizeof(float));
    cudaMalloc((void**) &gpu_feature_contributions_forest_op, n_operating_points * NFEATURES * n_trees * sizeof(float));
    cudaMalloc((void**) &gpu_total_result_op, n_operating_points * sizeof(float));
    cudaMalloc((void**) &gpu_calculate_contributions_op, sizeof(int));
    cudaMemcpy(gpu_calculate_contributions_op,&calculate_contributions_op,sizeof(int),cudaMemcpyHostToDevice);

    //set up variables for evaluating dprf at NPTS
    int n_scan_points = NPTS;
    int* gpu_n_scan_points;
    cudaMalloc((void**) &gpu_n_scan_points, sizeof(int));
    cudaMemcpy(gpu_n_scan_points, &n_scan_points, sizeof(int), cudaMemcpyHostToDevice);

    float scan_input_data[NFEATURES*NPTS] = {0};
     float* gpu_scan_input_data;
    cudaMalloc((void**) &gpu_scan_input_data, NPTS * NFEATURES * sizeof(float));

    float *tree_result_scan;
    float feature_contributions_scan[NFEATURES] = {0};
    float forest_feature_contributions_scan[NFEATURES] = {0};
    float *total_result_scan;
    int calculate_contributions_scan = 1;

    tree_result_scan = (float *) malloc(n_scan_points * n_trees * sizeof(float));
    total_result_scan = (float *) malloc(n_scan_points * sizeof(float));


    float* gpu_tree_result_scan;
    float* gpu_feature_contributions_scan;
    float* gpu_feature_contributions_forest_scan;
    float* gpu_total_result_scan;
    int* gpu_calculate_contributions_scan;
    cudaMalloc((void**) &gpu_tree_result_scan, n_scan_points * n_trees * sizeof(float));

    cudaMalloc((void**) &gpu_scan_input_data, n_scan_points * NFEATURES * sizeof(float));
    cudaMalloc((void**) &gpu_feature_contributions_scan, n_scan_points * NFEATURES * sizeof(float));
    cudaMalloc((void**) &gpu_feature_contributions_forest_scan, n_scan_points * NFEATURES * n_trees * sizeof(float));
    cudaMalloc((void**) &gpu_total_result_scan, n_scan_points * sizeof(float));
    cudaMalloc((void**) &gpu_calculate_contributions_scan, sizeof(int));
    cudaMemcpy(gpu_calculate_contributions_scan,&calculate_contributions_scan,sizeof(int),cudaMemcpyHostToDevice);

     // setup SORI variables
    init_launcher(1, NPTS, NCON, TPB);
    FILE *fp;
    fp = fopen("sori_test.out","w");

  // allocate an array of floats representing test points on the CPU and GPU
  float *cpu_points = (float *)malloc(NPTS * NDIM * sizeof(float));
  float* gpu_points;

  float* gpu_disruptivity;
  float cpu_disruptivity[NPTS];

  float *cpu_constraints = (float *)malloc(NCON * NDIM * sizeof(float));
  float* gpu_constraints;

  float *cpu_constraints_prev = (float *)malloc(NCON * NDIM * sizeof(float));
  float* gpu_constraints_prev;

  float* gpu_constraint_sos;
  float cpu_constraint_sos[NCON];

  float *cpu_dot_products = (float *)malloc(NCON * NPTS * sizeof(float));
  float* gpu_dot_products;

  bool *cpu_pt_satisfies_constraint = (bool *)malloc(NCON * NPTS * sizeof(bool));
  bool* gpu_pt_satisfies_constraint;

  bool *cpu_pt_satisfies_all_constraints = (bool *)malloc(NPOP * NPTS * sizeof(bool));
  bool* gpu_pt_satisfies_all_constraints;

  int cpu_n_safe_inside[NPOP];
  int cpu_n_unsafe_inside[NPOP];
  int* gpu_n_safe_inside;
  int* gpu_n_unsafe_inside;

  int* gpu_winners;
  int cpu_winners[NPOP];
  int* gpu_tournament;

  float* gpu_J;
  float* gpu_J_elite;
  float cpu_J[NPOP];
  float cpu_J_elite[NELITE];

  float *cpu_result = (float *)malloc(NLMI * NDIM * sizeof(float));
  float* gpu_result;
  //float cpu_result[NLMI*NDIM];

  cudaMalloc((void**) &gpu_points, NPTS * NDIM * sizeof(float));
  cudaMalloc((void**) &gpu_disruptivity, NPTS * sizeof(float));
  cudaMalloc((void**) &gpu_constraints, NCON * NDIM * sizeof(float));
  cudaMalloc((void**) &gpu_constraints_prev, NCON * NDIM * sizeof(float));
  cudaMalloc((void**) &gpu_constraint_sos, NCON * sizeof(float));
  cudaMalloc((void**) &gpu_dot_products, NCON * NPTS * sizeof(float));
  cudaMalloc((void**) &gpu_pt_satisfies_constraint, NCON * NPTS * sizeof(bool));
  cudaMalloc((void**) &gpu_pt_satisfies_all_constraints, NPOP * NPTS * sizeof(bool));
  cudaMalloc((void**) &gpu_n_unsafe_inside,NPOP * sizeof(int));
  cudaMalloc((void**) &gpu_n_safe_inside, NPOP * sizeof(int));
  cudaMalloc((void**) &gpu_J, NPOP * sizeof(float));
  cudaMalloc((void**) &gpu_J_elite, NELITE * sizeof(float));
  cudaMalloc((void**) &gpu_winners, NPOP * sizeof(int));
  cudaMalloc((void**) &gpu_tournament, NPOP * NTOURN * sizeof(int));
  cudaMalloc((void**) &gpu_result, NLMI * NDIM * sizeof(float));

  struct timespec ts0, te0;
  struct timespec ts1, te1;

    float feature_points_out[NFEATURES*NPTS] = {0};
    float* gpu_feature_points_out;
    cudaMalloc((void**) &gpu_feature_points_out, NPTS * NFEATURES * sizeof(float));

    int* gpu_n_dim;
    int n_dim = NDIM;
    cudaMalloc((void**) &gpu_n_dim, sizeof(int));
    cudaMemcpy(gpu_n_dim, &n_dim, sizeof(int), cudaMemcpyHostToDevice);

    float* gpu_scale;
    cudaMalloc((void**) &gpu_scale, 2*n_features*sizeof(float));
    cudaMemcpy(gpu_scale, &scan_ranges, 2*n_features*sizeof(float), cudaMemcpyHostToDevice);

    int important_features[8] = {1,1,0,0,0,0,0,0};

    int* gpu_important_features;
    cudaMalloc((void**) &gpu_important_features, n_features*sizeof(int));
    cudaMemcpy(gpu_important_features, &important_features, n_features*sizeof(int), cudaMemcpyHostToDevice);

    TIME(ts1);
    // Evaluate dprf at operating point
    for (int sample_index = 0; sample_index < NSAMPLES; sample_index++){
        for (int j=0;j<NFEATURES;j++){
            current_operating_point[j] = X_input[sample_index*NFEATURES+j];
        }


        cudaMemcpy(gpu_current_operating_point,current_operating_point, NFEATURES * sizeof(float),cudaMemcpyHostToDevice);


        generate_test_points_launcher(gpu_feature_points_out, gpu_points, NPTS, NFEATURES, NDIM, gpu_scale, gpu_current_operating_point, gpu_important_features, TPB, streams);


      random_points_launcher(gpu_constraints, NDIM, NCON, 0.5, 0.25,best,TPB, streams);

      eval_forest_launcher(n_trees, gpu_n_scan_points, gpu_n_trees, gpu_feature_points_out, gpu_feature, gpu_children_left, gpu_children_right, gpu_tree_start, gpu_threshold, gpu_value, gpu_n_classes, gpu_tree_result_scan, gpu_feature_contributions_scan, gpu_feature_contributions_forest_scan, gpu_disruptivity,gpu_calculate_contributions_scan);

      // genetic algorithm loop
      for (int i = 0; i < NGEN; i++) {
        evaluate_constraint_sos_launcher(gpu_constraints, gpu_constraint_sos, NDIM, NCON, TPB, streams);
        gpu_mmul_ABT_launcher(gpu_constraints, gpu_points, gpu_dot_products, NCON, NDIM, NPTS,TPB,streams);

        //Copy constraints to prev_gen
        GpuCopy_launcher(gpu_constraints_prev, gpu_constraints, NCON, NDIM, TPB, streams);
        // Determine which constraints are satisfied by each point (dot_products < sos)
        evaluate_constraints_satisfied_launcher(gpu_dot_products, gpu_constraint_sos, gpu_pt_satisfies_constraint, NCON, NPTS,TPB);

        // Determine which points satisfy all constraints (eventually there will be sets of constraints representing individuals in genetic alg.)

        float weights = 20.0f;
        evaluate_all_constraints_satisfied_launcher(gpu_pt_satisfies_constraint, gpu_pt_satisfies_all_constraints, gpu_n_safe_inside, gpu_n_unsafe_inside, gpu_disruptivity, 0.5, weights, gpu_J,NPTS, NPOP, NLMI, NCON, TPB);

        //calculate nsafe inside and nunsafe inside
        genetic_select_launcher(gpu_J, gpu_tournament, gpu_winners, gpu_constraints, gpu_constraints_prev, NPOP, NTOURN, NCON, NLMI, NDIM, TPB, streams);

        //Based on probability, mate two individuals (two in, two out, modify in place).
        float crossover_prob = 0.5f;
        genetic_mate_launcher(gpu_constraints, crossover_prob, NPOP, NLMI, NDIM, NCON,TPB,NPOP,streams);//TODO: could split over more threads

        //Based on probability, mutate individual traits
        float mutate_prob = 0.5f;
        float mutate_stdev = 0.3f;
        genetic_mutate_launcher(gpu_constraints, mutate_prob, mutate_stdev,NCON*NDIM, TPB,streams);

        //Carry over elite individual(s) from previous generation

        carry_over_elite_launcher(gpu_J, gpu_J_elite, gpu_constraints, gpu_constraints_prev, gpu_result,NELITE,NPOP%NELITE,NPOP/NELITE,NLMI,NDIM,NCON,streams);

      }//end loop over generations

    TIME(te1);
    TIME(ts0);
    cudaMemcpy(cpu_constraints,gpu_constraints,NCON * NDIM * sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_J_elite, gpu_J_elite, NELITE * sizeof(float), cudaMemcpyDeviceToHost);
    TIME(te0);
    std::cout << "cpu_constraints =" << std::endl;
    print_matrix(cpu_constraints, NCON, NDIM);
    std::cout << "cpu_J_elite =" << std::endl;
    print_matrix(cpu_J_elite, NELITE, 1);
    int best = argmax(cpu_J_elite, NELITE);
    printf("best= %d\n",best);
    for (int i=0;i<(NLMI*NDIM);i++){
        cpu_result[i] = cpu_constraints[best*(int(floor(NPOP/NELITE)))*(NLMI*NDIM)+i];
    }
    print_matrix(cpu_result, NLMI, NDIM);
    //fprint_matrix(fp,cpu_result, NLMI, NDIM);
    //printf("Time taken (1): %f seconds\n",tdiff(ts0,te0));
    //printf("Time taken (2): %f seconds\n",tdiff(ts1,te1));
    }//end loop over samples



  // free allocated memory

  cudaFree(gpu_feature_points_out);
  cudaFree(gpu_n_dim);
  cudaFree(gpu_scale);
  cudaFree(gpu_important_features);

  free_random_states();

  cudaFree(gpu_points);
  cudaFree(gpu_disruptivity);
  cudaFree(gpu_constraints);
  cudaFree(gpu_constraints_prev);
  cudaFree(gpu_constraint_sos);
  cudaFree(gpu_dot_products);
  cudaFree(gpu_pt_satisfies_constraint);
  cudaFree(gpu_pt_satisfies_all_constraints);
  cudaFree(gpu_n_safe_inside);
  cudaFree(gpu_n_unsafe_inside);
  cudaFree(gpu_winners);
  cudaFree(gpu_tournament);
  cudaFree(gpu_result);
  cudaFree(gpu_J);
  cudaFree(gpu_J_elite);


  free(cpu_points);
  free(cpu_constraints);
  free(cpu_constraints_prev);
  free(cpu_dot_products);
  free(cpu_pt_satisfies_constraint);
  free(cpu_pt_satisfies_all_constraints);
  free(cpu_result);
  fclose(fp);


  cudaFree(gpu_n_scan_points);
  cudaFree(gpu_n_trees);
  cudaFree(gpu_n_classes);
  cudaFree(gpu_n_nodes);
  cudaFree(gpu_feature);
  cudaFree(gpu_children_left);
  cudaFree(gpu_children_right);
  cudaFree(gpu_tree_start);
  cudaFree(gpu_threshold);
  cudaFree(gpu_value);


  cudaFree(gpu_tree_result_op);
  cudaFree(gpu_total_result_op);
  cudaFree(gpu_feature_contributions_op);
  cudaFree(gpu_feature_contributions_forest_op);
  cudaFree(gpu_current_operating_point);

  cudaFree(gpu_tree_result_scan);
  cudaFree(gpu_scan_input_data);
  cudaFree(gpu_total_result_scan);
  cudaFree(gpu_feature_contributions_scan);
  cudaFree(gpu_feature_contributions_forest_scan);

  free(feature);
  free(children_left);
  free(children_right);
  free(tree_start);
  free(threshold);
  free(value);

  free(tree_result_op);
  free(X_input);
  free(time_input);

  for(int i=0;i<NSTREAMS; i++)
      cudaStreamDestroy ( streams[i]);

  delete[] streams;

  //cudaDeviceReset();
  exit(0);
  return 0;
}
