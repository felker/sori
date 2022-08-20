#include "rtgpu_dprf_kernel.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <unistd.h>
#include <stdio.h>

/* this GPU kernel function is used to evaluate a forest of trees */
__global__ void eval_forest(int n_points_in, int n_trees_in, int nFeatures, float *X_eval, int *feature, int*children_left, int *children_right, int *tree_start, float *threshold, float *value, int n_classes_in, float *result, float *feature_contributions, float *feature_contributions_forest, float *total_result, int *calculate_contributions) {
  __shared__ float sX_eval[20];
  __shared__ float stotal_result;
  __shared__ float sresult[MAX_TREES];
  __shared__ float sfeature_contributions_forest[MAX_FEATURES*MAX_TREES];
  __shared__ float sfeature_contributions[MAX_FEATURES];
  
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  int i;
  int j;
  unsigned int k;
  int tree4thread;
  int point4thread;
  int n_trees = n_trees_in;
  int n_points = n_points_in;
  int n_classes = n_classes_in;
  if (threadIdx.x < nFeatures) {
      sX_eval[threadIdx.x] = X_eval[blockIdx.x*nFeatures+threadIdx.x];
      //printf("blockId = %d, threadId = %d, X_eval = %f\n", blockIdx.x, threadIdx.x, X_eval[blockIdx.x*nFeatures+threadIdx.x]);
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
		float feature_contributions_tmp[MAX_FEATURES] = {0};
		while (child_left != child_right){
			last_value = current_value;
			current_feature = feature[current_index];
			if (sX_eval[current_feature] <= threshold[current_index]){
				current_index = start_idx + child_left;
			} else {
				current_index = start_idx + child_right;
			}
			current_value = value[current_index * n_classes + 1];
			child_left = children_left[current_index];
			child_right = children_right[current_index];
			feature_contributions_tmp[current_feature] += (current_value - last_value);
		}
		sresult[threadIdx.x] = current_value;	
		if (*calculate_contributions != 0){	
			for (i=0;i<nFeatures;i++){
				sfeature_contributions_forest[nFeatures * threadIdx.x + i] = feature_contributions_tmp[i]; 
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
				for (j=0;j<nFeatures;j++){
					sfeature_contributions_forest[nFeatures * idx+j] += sfeature_contributions_forest[nFeatures * (idx + k)+j];
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
	    //total_result[point4thread] = total_result[point4thread]/n_trees;
		if ((*calculate_contributions != 0)) {
			for (j=0;j<nFeatures ;j++){
				//feature_contributions[point4thread*nFeatures+j] = sfeature_contributions_forest[j];
			}
		}
	}
	//printf("tree4thread=%d,point4thread=%d,result=%f,n_trees=%d,sresult[0]=%f %f %f\n",tree4thread,point4thread,total_result[point4thread],n_trees,sresult[0], sresult[1], sresult[2]);
}

void dprf_forestLauncher(DprfIn in, float *result, float *feature_contributions, float *feature_contributions_forest, float *total_result, int *calculate_contributions) {
	eval_forest<<<in.nPoints, in.nTrees>>>(in.nPoints, in.nTrees, in.nFeatures, in.xEval, in.feature, in.chLeft, in.chRight, in.treeStart, in.threshold, in.value, in.nClasses, result, feature_contributions, feature_contributions_forest, total_result, calculate_contributions);
	cudaDeviceSynchronize();
}
