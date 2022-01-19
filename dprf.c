// make -f makefile_dprf
// /usr/local/cuda/bin/nvprof ./dprf.exe forest_245_15_dan.h5 
// gcc dprf.c -o dprf -lhdf5 -std=c99
//#include "dprf_kernel.h"
#include "dprf.h"
//#include <stdio.h>
//#include <stdlib.h>
//#include <string.h>
//#include <hdf5.h>
//#include <cuda.h>

int read_model_sizes(char *fname, int *n_trees, int *n_classes, int *n_nodes) {
	int tmp_int;

	hid_t file_id, ds_id;
	herr_t status;
	file_id = H5Fopen(fname, H5F_ACC_RDWR, H5P_DEFAULT);
	//n_trees
	ds_id = H5Dopen2(file_id, "/n_trees", H5P_DEFAULT);
	status = H5Dread(ds_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &tmp_int);
	status = H5Dclose(ds_id);
	*n_trees = tmp_int;

  //n_classes
	ds_id = H5Dopen2(file_id, "/n_classes", H5P_DEFAULT);
	status = H5Dread(ds_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &tmp_int);
	status = H5Dclose(ds_id);
	*n_classes = tmp_int;

	//n_nodes
	ds_id = H5Dopen2(file_id, "/n_nodes", H5P_DEFAULT);
	status = H5Dread(ds_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &tmp_int);
	status = H5Dclose(ds_id);
	*n_nodes = tmp_int;

	status = H5Fclose(file_id);

	return status;
}

int read_model_data(char *fname, int *feature, int *children_left, int *children_right, int *tree_start, float *threshold, float *value) {

	hid_t file_id, ds_id;
	herr_t status;
	file_id = H5Fopen(fname, H5F_ACC_RDWR, H5P_DEFAULT);
	//n_trees
	ds_id = H5Dopen2(file_id, "/feature", H5P_DEFAULT);
	status = H5Dread(ds_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, feature);
	status = H5Dclose(ds_id);

	ds_id = H5Dopen2(file_id, "/children_left", H5P_DEFAULT);
	status = H5Dread(ds_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, children_left);
	status = H5Dclose(ds_id);

	ds_id = H5Dopen2(file_id, "/children_right", H5P_DEFAULT);
	status = H5Dread(ds_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, children_right);
	status = H5Dclose(ds_id);

	ds_id = H5Dopen2(file_id, "/tree_start", H5P_DEFAULT);
	status = H5Dread(ds_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, tree_start);
	status = H5Dclose(ds_id);

	ds_id = H5Dopen2(file_id, "/threshold", H5P_DEFAULT);
	status = H5Dread(ds_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, threshold);
	status = H5Dclose(ds_id);

	ds_id = H5Dopen2(file_id, "/value", H5P_DEFAULT);
	status = H5Dread(ds_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, value);
	status = H5Dclose(ds_id);
	//*features = tmp_int;

	status = H5Fclose(file_id);

	return status;
}

int read_input_sizes(char *fname, int *n_features, int *n_times) {
	int tmp_int;

	hid_t file_id, ds_id;
	herr_t status;
	file_id = H5Fopen(fname, H5F_ACC_RDWR, H5P_DEFAULT);
	//n_features
	ds_id = H5Dopen2(file_id, "/n_features", H5P_DEFAULT);
	status = H5Dread(ds_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &tmp_int);
	status = H5Dclose(ds_id);
	*n_features = tmp_int;

  	//n_times
	ds_id = H5Dopen2(file_id, "/n_times", H5P_DEFAULT);
	status = H5Dread(ds_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &tmp_int);
	status = H5Dclose(ds_id);
	*n_times = tmp_int;

	status = H5Fclose(file_id);

	return status;
}

int read_input_data(char *fname, float *X, float *time) {

	hid_t file_id, ds_id;
	herr_t status;
	file_id = H5Fopen(fname, H5F_ACC_RDWR, H5P_DEFAULT);
	//X
	ds_id = H5Dopen2(file_id, "/X", H5P_DEFAULT);
	status = H5Dread(ds_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, X);
	status = H5Dclose(ds_id);
	//time
	ds_id = H5Dopen2(file_id, "/time", H5P_DEFAULT);
	status = H5Dread(ds_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, time);
	status = H5Dclose(ds_id);

	status = H5Fclose(file_id);

	return status;
}

int eval_tree(int tree_index, float *X_eval, int *feature, int*children_left, int *children_right, int *tree_start, float *threshold, float *value, int n_classes, float *result, float *feature_contributions){
	int current_node;
	int current_index;
	int current_feature;
	float current_value;
	float last_value;
	current_node = 0;
	current_index = tree_start[tree_index]+current_node;
	current_value = value[current_index * n_classes + 1];
	int i=0;
	float feature_contributions_tmp[NFEATURES] = {0};
	while (children_left[current_index] != children_right[current_index]){
		//printf("i: %d\n",i);
		//i=i+1;
		last_value = current_value;
		current_feature = feature[current_index];
		if (X_eval[current_feature] <= threshold[current_index]){
				current_node = children_left[current_index];
		}
		else {
				current_node = children_right[current_index];
		}
		current_index = tree_start[tree_index]+current_node;
		current_value = value[current_index * n_classes + 1];
		feature_contributions_tmp[current_feature] += (current_value - last_value);
		//printf("current_value: %f\n", current_value);
		//printf("last_value: %f\n", last_value);
	}
   //printf("current_index: %d\n", current_index);
	//float sum_values = 0.0;
	//int i = 0;
	//for (i = 0; i < n_classes; i++){
	//	sum_values += value[current_index * n_classes + i];
	//	//printf("value[%d]: %f\n", i, value[current_index * n_classes + i]);
	//}
	//printf("sum_values: %f\n", sum_values);
	*result = current_value;///sum_values; // find class 1 probability
	for (i=0;i<NFEATURES;i++){
		feature_contributions[i] = feature_contributions_tmp[i];
	}
   //printf("result: %f\n", *result);
	return 0;
}
