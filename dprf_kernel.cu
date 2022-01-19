extern "C" {
#include "dprf_kernel.h"
}
#include <unistd.h>
#include <stdio.h>
#include <time.h>

/* this GPU kernel function is used to evaluate a forest of trees */
extern "C" {
__global__ void eval_forest(int *n_points_in, int *n_trees_in, float *X_eval, int *feature, int*children_left, int *children_right, int *tree_start, float *threshold, float *value, int *n_classes_in, float *result, float *feature_contributions, float *feature_contributions_forest, float *total_result, int *calculate_contributions) {
  __shared__ float sX_eval[20];
  __shared__ float stotal_result;
  __shared__ float sresult[MAX_TREES];
  __shared__ float sfeature_contributions_forest[NFEATURES*MAX_TREES];
  __shared__ float sfeature_contributions[NFEATURES];
  
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  int i;
  int j;
  unsigned int k;
  int tree4thread;
  int point4thread;
  int n_trees = *(n_trees_in);
  int n_points = *(n_points_in);
  int n_classes = *(n_classes_in);
  if (threadIdx.x < NFEATURES){
      sX_eval[threadIdx.x] = X_eval[blockIdx.x*NFEATURES+threadIdx.x];
      //printf("blockId = %d, threadId = %d, X_eval = %f\n", blockIdx.x, threadIdx.x, X_eval[blockIdx.x*NFEATURES+threadIdx.x]);
  }
  //__threadfence();
  __syncthreads();
  if (threadIdx.x < n_trees){//(idx < n_trees *n_points){
  		point4thread = blockIdx.x;//idx/(n_trees);
		tree4thread = threadIdx.x;//idx%(n_trees);
		int current_node;
		int current_index;
		int child_left;
		int child_right;
		float current_value;
		float last_value;
		current_node = 0;
		int current_feature;
		int start_idx = tree_start[tree4thread];
		current_index = start_idx+current_node;
		current_value = value[current_index * n_classes + 1];
		child_left = children_left[current_index];
		child_right = children_right[current_index];
		float feature_contributions_tmp[NFEATURES] = {0};
		while (child_left != child_right){
			last_value = current_value;
			current_feature = feature[current_index];
			if (sX_eval[current_feature] <= threshold[current_index]){
					current_index = start_idx + child_left;
			}
			else {
					current_index = start_idx + child_right;
			}
			current_value = value[current_index * n_classes + 1];
			child_left = children_left[current_index];
			child_right = children_right[current_index];
			feature_contributions_tmp[current_feature] += (current_value - last_value);
		}
		sresult[threadIdx.x] = current_value;	
		if (*calculate_contributions != 0){	
			for (i=0;i<NFEATURES;i++){
				sfeature_contributions_forest[NFEATURES * threadIdx.x + i] = feature_contributions_tmp[i]; 
			}
		}
  }
  //__threadfence();
  __syncthreads();
    //if (tree4thread == 0){
    //    sresult[n_trees+1] = 0.0;
    //    sresult[n_trees] = 0.0;
    //}
    
	for(k = 1; k < n_trees; k <<= 1) {
		idx = threadIdx.x;
		if ((idx & ((k << 1) - 1)) == 0){
		    //if(blockIdx.x == 0){
		    //    printf("idx=%d, k=%d, idx+k=%d\n",idx,k,idx+k);
		    //}
		    if (idx+k < n_trees){ //TODO: change previous conditionals to remove need for this one?
				sresult[idx] += sresult[idx + k];
				for (j=0;j<NFEATURES;j++){
					sfeature_contributions_forest[NFEATURES * idx+j] += sfeature_contributions_forest[NFEATURES * (idx + k)+j];
				}
			}
		//__threadfence();
		__syncthreads();
		}
	}

	//__threadfence();
	//__syncthreads();
	//printf("tree4thread=%d,point4thread=%d,n_points=%d\n",tree4thread,point4thread,n_points);
	if ((tree4thread == 0) && (point4thread<n_points)){
	    //printf("blockIdx.x=%d\n",blockIdx.x);
	    //for (j=0;j<n_trees;j++){
	    //    total_result[point4thread] += sresult[j];
		//}
		total_result[point4thread] = sresult[0]/float(n_trees);
	    //total_result[point4thread] = total_result[point4thread]/n_trees;	//printf("tree4thread=%d,point4thread=%d,result=%f,n_trees=%d,sresult[0]=%f\n",tree4thread,point4thread,total_result[point4thread],n_trees,sresult[0]);
		//printf("calc contributions = %d\n",*calculate_contributions);
		if ((*calculate_contributions != 0.0)) {
			for (j=0;j<NFEATURES;j++){
				feature_contributions[point4thread*NFEATURES+j] = sfeature_contributions_forest[j];
			}
		}
	}				
}

void eval_forest_launcher(int cpu_n_trees, int *n_points, int *n_trees, float *X_eval, int *feature, int*children_left, int *children_right, int *tree_start, float *threshold, float *value, int *n_classes, float *result, float *feature_contributions, float *feature_contributions_forest, float *total_result, int *calculate_contributions) {
	printf("n_trees = %d\n",cpu_n_trees);
	eval_forest<<<(NPOINTS), (cpu_n_trees)>>>(n_points, n_trees, X_eval, feature, children_left, children_right, tree_start, threshold, value, n_classes, result, feature_contributions, feature_contributions_forest, total_result, calculate_contributions);
}
}
