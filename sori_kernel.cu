#include "sori_kernel.h"
#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

extern "C" {
typedef cudaStream_t cudaStream_t;

__device__ curandState_t states[MAXNPTS*MAXNCON];

__device__ float sum_of_squares(float* points, int jrow, int ncol, int nrow)
{
  float sos = 0.0;
  for(int i = 0; i < ncol; ++i){
    //sos = sos + powf(points[i * nrow + jrow],2.0);
    sos = sos + powf(points[jrow*ncol + i],2.0);
  }
  //printf("inside function: jrow=%d, ncol=%d, nrow=%d, x=%f, y=%f, d=%f\n", jrow, ncol, nrow, points[0 * nrow + jrow], points[1 * nrow + jrow], sos);
  //printf("inside function: jrow=%d, ncol=%d, nrow=%d, x=%f, y=%f, d=%f\n", jrow, ncol, nrow, points[jrow*ncol + 0], points[jrow*ncol + 1], sos);
  return sos;
}

__device__ float cost_function(int nsafe_inside, int nunsafe_inside, float weights){
	float J = (nsafe_inside) - weights*(nunsafe_inside);
	return J;
}

__device__ float norm_of_row(float* points, int jrow, int ncol, int nrow)
{
  return sqrt(sum_of_squares(points, jrow, ncol, nrow));
}

__device__ int genetic_tournament(float* gpu_J, int* individuals, int n_individuals){
	int winner = individuals[0];
	float best_score = gpu_J[winner];
	//printf("Initial winner: %d has score %f.\n", winner, best_score);
	for(int i = 1; i < n_individuals; ++i){
	   //printf("Contender: %d has score %f.\n", individuals[i], gpu_J[individuals[i]]);
	   float score = gpu_J[individuals[i]];
		if (score > best_score){
			//printf("New winner\n");
			winner = individuals[i];
			best_score = score;
		}
	}
	return winner;
}

__device__ void random_ints(curandState_t state, int* ints_out, int n_ints_out, int max_int)
{
  for(int i = 0; i < n_ints_out; ++i){
  	ints_out[i] = curand(&state)%max_int;
  }
}

__device__ int find_elite(float* gpu_J, int NPOP)
{
	int elite = 0;
	float best_score = gpu_J[0];
	//printf("Initial winner: %d has score %f.\n", winner, best_score);

	for(int i = 1; i < NPOP; ++i){
	   //printf("Contender: %d has score %f.\n", individuals[i], gpu_J[individuals[i]]);
		if (gpu_J[i] > best_score){
			//printf("New winner\n");
			elite = i;
			best_score = gpu_J[i];
		}
	}
	return elite;
}

void free_random_states()
{
	cudaFree(states);
}

/* this GPU kernel function is used to initialize the random states */
__global__ void init(unsigned int seed, int NPTS, int NCON) {
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (idx < NPTS*NCON){
  		curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
              idx, /* the sequence number should be different for each core (unless you want all cores to get the same sequence of numbers for some reason - use thread id! */
              0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
              &states[idx]);
  }
}

void init_launcher(unsigned int seed, int NPTS, int NCON, int TPB) {
	init<<<(NPTS*NCON + TPB -1)/TPB, TPB>>>(seed, NPTS, NCON);
}


/* this GPU kernel takes an array of states, and an array of floats, and puts a random float into each */
__global__ void randoms(float* numbers) {
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  /* curand works like rand - except that it takes a state as a parameter */
  numbers[idx] = curand_uniform(&states[idx]);
}

/* this GPU kerenel calculates the sum of squares for each of the constraints */
__global__ void evaluate_constraint_sos(float* constraints, float* constraint_sos, int ncols, int nrows) {
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (idx<nrows){
  	constraint_sos[idx] = sum_of_squares(constraints, idx, ncols, nrows);
  }
}

void evaluate_constraint_sos_launcher(float* constraints, float* constraint_sos, int NDIM, int NCON, int TPB, cudaStream_t* stream) {
  evaluate_constraint_sos<<<(NCON+TPB-1)/TPB, TPB, 0, stream[0]>>>(constraints, constraint_sos, NDIM, NCON);
}

/* this GPU kernel takes an array of states, and an array of testpoints, and puts random data into each */
// __global__ void random_points(curandState_t* states, float* points_out, int ncols)
__global__ void random_points(float* points_out, int ncols, int nrows, float scale, float offset, int skip_idx)
{
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if ((idx<nrows) && (idx != skip_idx)){
  	for(int i = 0; i < ncols; ++i){
  		points_out[idx+nrows*i] = scale*curand_uniform(&states[idx])-offset;
		//printf("random_points[idx=%d]=%f\n",idx,points_out[idx+nrows*i]);
  	}
  }
}

void random_points_launcher(float* points_out, int ncols, int nrows, float scale, float offset, int skip_idx, int TPB, cudaStream_t* stream)
{
  random_points<<<(nrows+TPB-1)/TPB, TPB, 0, stream[0]>>>(points_out, ncols, nrows, scale, offset, skip_idx);
}

/* This GPU kernel generates random points in feature space to evaluate disruptivity at */
__global__ void generate_test_points(float* feature_points_out, float* dim_points_out, int gn_points, int gn_features, int gn_dim, float* scale, float* offset, int* important_features)
{
	int point4thread = blockIdx.x*blockDim.x + threadIdx.x;
 	float random_val;
	int j_dim = 0;
	int n_points = gn_points;
	int n_features = gn_features;
	int n_dim = gn_dim;
	float scale2use;
	if (point4thread<n_points){
		for(int i = 0; i < n_features; ++i){
			random_val = curand_uniform(&states[point4thread]);
			//printf("point[%d] feature[%d]: random_val=%f\n",point4thread,i,random_val);
			if (important_features[i] == 1){
				//dim_points_out[point4thread+n_points*j_dim] = 2.0*random_val-1.0;
				dim_points_out[point4thread*n_dim+j_dim] = 2.0*random_val-1.0;
				scale2use = scale[i];
				//printf("point[%d] feature[%d]: important, index = %d, dim_points_out = %f, scale=%f\n",point4thread,i,point4thread+n_points*j_dim,2.0*random_val-1.0,scale2use);
				j_dim++;


			}
			else{
				scale2use = scale[n_features+ i];
			}
  	    	//feature_points_out[point4thread+n_points*i] = scale2use*(random_val-0.5)+offset[i];
  	    	feature_points_out[point4thread*n_features+i] = scale2use*(random_val-0.5)+offset[i];
  	    	
  		}
  	}
}

void generate_test_points_launcher(float* feature_points_out, float* dim_points_out, int gn_points, int gn_features, int gn_dim, float* scale, float* offset, int* important_features, int TPB, cudaStream_t* streams)
{
	generate_test_points<<<(gn_points+TPB-1)/TPB, TPB, 0, streams[1]>>>(feature_points_out, dim_points_out, gn_points, gn_features, gn_dim, scale, offset, important_features);
}

/* This GPU kernel tests whether a point satisfies each of the constraints (individually)*/
__global__ void evaluate_constraints_satisfied(float* gpu_dot_products, float* gpu_constraint_sos, bool* gpu_pt_satisfies_constraint,int nconstraints, int npoints) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx < (nconstraints*npoints)){
		gpu_pt_satisfies_constraint[idx] = gpu_dot_products[idx] < gpu_constraint_sos[idx/npoints];
		//printf("Thread %d: NCON = %d, dot product = %f, idx_div_NPTS = %d, constraint_sos = %f, satisfied=%d\n",idx, NCON, gpu_dot_products[idx], idx/npoints, gpu_constraint_sos[idx/npoints], gpu_pt_satisfies_constraint[idx]);
	}
}

void evaluate_constraints_satisfied_launcher(float* gpu_dot_products, float* gpu_constraint_sos, bool* gpu_pt_satisfies_constraint, int nconstraints, int npoints, int TPB) {
	evaluate_constraints_satisfied<<<(nconstraints*npoints+TPB-1)/TPB, TPB>>>(gpu_dot_products, gpu_constraint_sos, gpu_pt_satisfies_constraint, nconstraints, npoints);
}

/* This GPU kernel tests whether a point satisifies all constraints of a constraint set, updating the number of safe and unsafe points, as well as the cost function value */
__global__ void evaluate_all_constraints_satisfied(bool* gpu_pt_satisfies_constraint, bool* gpu_pt_satisfies_all_constraints, int* nsafe_inside, int* nunsafe_inside, float* disruptivity, float disruptivity_threshold, float weights, float* gpu_J, int npoints, int npopulation, int nlmis, int nconstraints) {
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx<(npopulation)){
		nsafe_inside[idx] = 0;
		nunsafe_inside[idx] = 0;
	}
	__syncthreads();

  //printf("idx=%d,npoints=%d,npopulation=%d\n",idx,npoints,npopulation);
  if (idx<(npoints*npopulation)){
  	 int pop_index = idx/npoints;
  	 int pt_index = idx%npoints;

	 bool pt_satisfies_all_constraints = 1;

  	 for(int i = 0; i < nlmis; ++i){
     	pt_satisfies_all_constraints = pt_satisfies_all_constraints && gpu_pt_satisfies_constraint[pop_index*(npoints*nlmis)+pt_index+i*npoints];
  	 }
  	 gpu_pt_satisfies_all_constraints[idx] = pt_satisfies_all_constraints;
       //printf("idx=%d,pt_sat_all_con=%d,pt_index=%d,disruptivity=%f,disruptivity_threshold=%f\n",idx,pt_satisfies_all_constraints,pt_index,disruptivity[pt_index],disruptivity_threshold);
  	 if (pt_satisfies_all_constraints){
  	 	if (disruptivity[pt_index] < disruptivity_threshold){
  	 		atomicAdd(&nsafe_inside[pop_index],1);
  	 	}
  	 	else{
  	 		atomicAdd(&nunsafe_inside[pop_index],1);
  	 	}
  	 }
  }
  __threadfence();
  __syncthreads();
  if (idx<(npopulation)){
  	//printf("New. idx=%d, nsafe=%d, nunsafe=%d, cost=%f\n", idx, nsafe_inside[idx], nunsafe_inside[idx], gpu_J[idx]);
  	gpu_J[idx] = cost_function(nsafe_inside[idx], nunsafe_inside[idx], weights);
  	//printf("New. idx=%d, nsafe=%d, nunsafe=%d, cost=%f\n", idx, nsafe_inside[idx], nunsafe_inside[idx], gpu_J[idx]);
  }
}

void evaluate_all_constraints_satisfied_launcher(bool* gpu_pt_satisfies_constraint, bool* gpu_pt_satisfies_all_constraints, int* nsafe_inside, int* nunsafe_inside, float* disruptivity, float disruptivity_threshold, float weights, float* gpu_J, int npoints, int npopulation, int nlmis, int nconstraints, int TPB) {
	evaluate_all_constraints_satisfied<<<(npoints*npopulation+TPB-1)/TPB, TPB>>>(gpu_pt_satisfies_constraint, gpu_pt_satisfies_all_constraints, nsafe_inside, nunsafe_inside, disruptivity, disruptivity_threshold, weights, gpu_J, npoints, npopulation, nlmis, nconstraints);
}

/* Copy array source to destination */
__global__ void GpuCopy( float* des , float* __restrict__ sour ,const int M , const int N )
{
    int tx=blockIdx.x*blockDim.x+threadIdx.x;
    if(tx<N*M)
        des[tx]=sour[tx];
}

void GpuCopy_launcher( float* des , float* __restrict__ sour ,const int M , const int N, int TPB, cudaStream_t* stream)
{
	GpuCopy<<<(M*N+TPB-1)/TPB,TPB, 0, stream[2]>>>(des, sour, M, N);
}

/* This gpu kernel performs the tournament selection step of the genetic algorithm */
__global__ void genetic_select(float* gpu_J, int* tournament_members, int* winners, float* gpu_constraints, float* gpu_constraints_prev, int npopulation, int ntournament, int nconstraints, int nlmis, int ndimensions)
{
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
        //float l_gpu_J[NTOURN];
	//int individuals[NTOURN];
	if (idx<npopulation*ntournament){
		tournament_members[idx] = curand(&states[idx])%npopulation;
	}
	if (idx<npopulation) {
		//random_ints(states[idx], individuals, NTOURN, NPOP);
		//for(int i = 0; i < n_individuals; ++i){
		//	printf("Thread %d individual %d: %d\n",blockIdx.x,i,individuals[i]);
   	//}
                //for(int i = 0; i < NTOURN; ++i){
                //    l_gpu_J[i] = gpu_J[i];
                //}
		winners[idx] = genetic_tournament(gpu_J, &tournament_members[idx*ntournament], ntournament);

		for(int i = 0; i < nlmis; ++i){
			for(int j = 0; j < ndimensions; ++j){
				gpu_constraints[idx*(nlmis*ndimensions) + j + ndimensions*i] = gpu_constraints_prev[winners[idx]*(nlmis*ndimensions) + j + ndimensions*i];
			}
		}
	}
	//printf("Thread %d Winner: %d\n",blockIdx.x, winners[blockIdx.x]);
}

void genetic_select_launcher(float* J, int* tournament_members, int* winners, float* constraints, float* constraints_prev, int npopulation, int ntournament, int nconstraints, int nlmis, int ndimensions, int TPB, cudaStream_t* stream)
{
	genetic_select<<<(npopulation*ntournament+TPB-1)/TPB,TPB,0,stream[1]>>>(J, tournament_members, winners, constraints, constraints_prev, npopulation, ntournament, nconstraints, nlmis, ndimensions);
}

/* This GPU algorithm performs the crossover/mate step of the genetic algorithm */
__global__ void genetic_mate(float* gpu_constraints, float crossover_prob, int npopulation, int nlmis, int ndimensions, int nconstraints)
{
// loop over traits of neigboring individuals and swap values with a probability of crossover_prob
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	if (idx<npopulation/2){
		int idx1 = 2 * idx;
		int idx2 = idx1 + 1;
		for(int i = 0; i < nlmis; ++i){
			float crossover;
			crossover = curand_uniform(&states[idx]);
			for(int j = 0; j < ndimensions; ++j){
				if (crossover < crossover_prob) {
					float tmp;
					int a1 = nlmis*ndimensions;
					int a2 = j+ndimensions*i;
					int idxA = idx1*a1 + a2;
					int idxB = idx2*a1 + a2;
					tmp = gpu_constraints[idxA];
					gpu_constraints[idxA] = gpu_constraints[idxB];
					gpu_constraints[idxB] = tmp;
				}
			}
		}
	}
}

void genetic_mate_launcher(float* constraints, float crossover_prob, int npopulation, int nlmis, int ndimensions, int nconstraints, int TPB, int NPOP, cudaStream_t* stream)
{
	genetic_mate<<<(NPOP/2+TPB-1)/TPB,TPB,0,stream[1]>>>(constraints, crossover_prob, npopulation, nlmis, ndimensions, nconstraints);
}

/* This gpu kernel mutates the population with small perturbations */
__global__ void genetic_mutate(float* gpu_constraints, float mutate_prob, float mutate_stdev, int ncon_m_ndim)
{
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	if (idx<(ncon_m_ndim)){
		float mutate;
		mutate = curand_uniform(&states[idx]);
		if (mutate < mutate_prob) {
			float mutation;
			mutation = curand_normal(&states[idx]) * mutate_stdev;
			gpu_constraints[idx] = max(min(gpu_constraints[idx] + mutation,1.0),-1.0);
		}
	}
}

void genetic_mutate_launcher(float* constraints, float mutate_prob, float mutate_stdev, int ncon_m_ndim, int TPB, cudaStream_t* stream)
{
	genetic_mutate<<<(ncon_m_ndim+TPB-1)/TPB,TPB,0,stream[1]>>>(constraints, mutate_prob, mutate_stdev, ncon_m_ndim);
}

/* This gpu kernel ensures the best performing inviduals survive to the next generation */
__global__ void carry_over_elite(float* gpu_J, float* gpu_J_elite, float* gpu_constraints, float* gpu_constraints_prev, float* gpu_result, int n_elite, int npopulation_mod_n_elite, int npopulation_div_n_elite, int nlmis, int ndimensions, int nconstraints)
{
	//int idx=blockIdx.x*blockDim.x+threadIdx.x;
	int idx = blockIdx.x;

	__shared__ int t_elite;
	__shared__ float t_best_score;

	if (idx<n_elite){
		if (threadIdx.x == 0){
			t_elite = idx*(npopulation_div_n_elite);
			t_best_score = gpu_J[t_elite];
			float score;
			int remainder = 0;
			if (idx==n_elite-1){
				remainder = npopulation_mod_n_elite;
			}

			for(int i = 0; i < npopulation_div_n_elite+remainder; ++i){
				int inner_idx = i+idx*(npopulation_div_n_elite);
				score = gpu_J[inner_idx];
				if (score > t_best_score){
					t_elite = inner_idx;
					t_best_score = score;
				}
			}
		}
		gpu_J_elite[idx] = t_best_score;
                //printf("block=%d,threadIdx.x=%d,t_elite=%d\n",idx,threadIdx.x,t_elite);
		gpu_constraints[idx*nlmis*ndimensions*npopulation_div_n_elite+threadIdx.x] = gpu_constraints_prev[t_elite*nlmis*ndimensions+threadIdx.x];
		//for(int i = 0; i < nlmis; ++i){
		//	for(int j = 0; j < ndimensions; ++j){
		//		int a1 = nlmis*ndimensions;
		//	        int a2 = j+ndimensions*i;
		//		//int idxA = idx*a1 + a2;
		//		gpu_constraints[idx*a1*npopulation_div_n_elite+a2] = gpu_constraints_prev[t_elite*a1+a2];
		//	}
		//}
	}
}

void carry_over_elite_launcher(float* J, float* J_elite, float* constraints, float* constraints_prev, float* gpu_result, int n_elite, int npopulation_mod_n_elite, int npopulation_div_n_elite,  int nlmis, int ndimensions, int nconstraints, cudaStream_t* stream)
{
	carry_over_elite<<<n_elite,nlmis*ndimensions,0,stream[0]>>>(J, J_elite, constraints, constraints_prev, gpu_result, n_elite, npopulation_mod_n_elite, npopulation_div_n_elite, nlmis, ndimensions, nconstraints);
}


// Multiply the arrays A and B on GPU and save the result in C
// C(m,n) = A(m,k) * transpose(B(n,k))
__global__ void gpu_mmul_ABT(const float *A, const float *B, float *C, const int m, const int k, const int n) {
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	int row = idx/n;
	int col = idx%n;
	if (idx<m*n){
		float tmp = 0.0;
		//C[idx] = 0.0;
		for(int i = 0; i < k; ++i){
			//C[idx] = C[idx] + A[row+m*i]*B[col+n*i];
			//tmp = tmp + A[row+m*i]*B[col+n*i];
                        tmp = tmp + A[k*row+i]*B[k*col+i];
                        //printf("idx=%d,row=%d,col=%d,i=%d,k*row+i=%d,k*col+i=%d,A[k*col+i]=%f,B[k*col+i]=%f\n",idx,row,col,i,k*row+i,k*col+i,A[k*row+i],B[k*col+i]);
		}
		C[idx] = tmp;
	}
}

void gpu_mmul_ABT_launcher(const float *A, const float *B, float *C, const int m, const int k, const int n, int TPB, cudaStream_t* stream) {
	gpu_mmul_ABT<<<(m*n+TPB-1)/TPB,TPB,0,stream[1]>>>(A, B, C, m, k, n);
}

} // close extern "C"
