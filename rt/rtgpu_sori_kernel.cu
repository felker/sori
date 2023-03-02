#include "soriRt.hh"

#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <stdio.h>

__device__ curandState_t sori_states[SORI_MAXNPTS * SORI_MAXNCON];

__device__ float sori_sumOfSquares(float * points, int jrow, int ncol, int nrow) {
    float sos = 0.0f;
    for (int i = 0; i < ncol; ++i)
        sos += powf(points[jrow*ncol + i], 2.0f);
    return sos;
}

__device__ float sori_costFunction(int nsafe_inside, int nunsafe_inside, float weights) {
    return nsafe_inside - weights * nunsafe_inside;
}

__device__ int sori_geneticTournament(float * gpu_J, int * individuals, int n_individuals) {
    int winner = individuals[0];
    float best_score = gpu_J[winner];
    for (int i = 1; i < n_individuals; ++i) {
        float score = gpu_J[individuals[i]];
        if (score > best_score) {
            winner = individuals[i];
            best_score = score;
        }
    }
    return winner;
}

__device__ void sori_randomInts(curandState_t state, int * ints_out, int n_ints_out, int max_int) {
    for (int i = 0; i < n_ints_out; ++i)
        ints_out[i] = curand(&state) % max_int;
}

// Initialize the random states
__global__ void sori_init(unsigned int seed, int NPTS, int NCON) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < NPTS * NCON) {
        curand_init(seed,  // the seed can be the same for each core, here we pass the time in from the CPU
            idx,       // the sequence number should be different for each core (unless you want all cores to get the same sequence of numbers for some reason - use thread id!)
            0,         // the offset is how much extra we advance in the sequence for each call, can be 0
            &sori_states[idx]);
    }
}

void Sori::gpuInit(unsigned int seed, SoriCfg const & cfg) {
    NFEATURES = cfg.dprf.nFeatures;
    NCON = cfg.nConstraints;
    NDIM = cfg.nDimensions;
    NPOP = cfg.nPopulation;
    NPTS = cfg.nPoints;
    NTOURN = cfg.nTournaments;
    NELITE = cfg.nElite;
    NLMI = cfg.nLmi;
    //TPB = cfg. FIXME Get from PCS
    TPB=32;

    points = cfg.points;
    disruptivity = cfg.disruptivity;
    constraints = cfg.constraints;
    constraintsPrev = cfg.constraintsPrev;
    constraintSos = cfg.constraintSos;
    dotProducts = cfg.dotProducts;
    J = cfg.J;
    JElite = cfg.JElite;
    ptSatisfiesConstraint = cfg.ptSatisfiesConstraint;
    ptSatisfiesAllConstraints = cfg.ptSatisfiesAllConstraints;
    nUnsafeInside = cfg.nUnsafeInside;
    nSafeInside = cfg.nSafeInside;
    tournament = cfg.tournament;
    winners = cfg.winners;

    cudaDeviceSynchronize();
    sori_init<<<(NPTS*NCON + TPB -1)/TPB, TPB>>>(seed, NPTS, NCON);
    cudaDeviceSynchronize();
}

// Fill array with random numbers
__global__ void randoms(float * numbers) {
    // curand works like rand, except that it takes a state as a parameter
    int const idx = blockIdx.x * blockDim.x + threadIdx.x;
    numbers[idx] = curand_uniform(&sori_states[idx]);
}

// Calculate the sum of squares for each constraint
__global__ void evaluate_constraint_sos(float* constraints, float* constraint_sos, int ncols, int nrows) {
    int const idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nrows)
        constraint_sos[idx] = sori_sumOfSquares(constraints, idx, ncols, nrows);
}

void Sori::gpuEvaluateConstraintSos(cudaStream_t* stream) {
    evaluate_constraint_sos<<<(NCON+TPB-1)/TPB, TPB, 0, stream[0]>>>(constraints, constraintSos, NDIM, NCON);
}

// Fill array of testpoints with random numbers
__global__ void random_points(float* points_out, int ncols, int nrows, float scale, float offset, int skip_idx) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if ((idx<nrows) && (idx != skip_idx))
        for(int i = 0; i < ncols; ++i)
            points_out[idx+nrows*i] = scale*curand_uniform(&sori_states[idx])-offset;
}

void Sori::gpuRandomPoints(float* points_out, int ncols, int nrows, float scale, float offset, int skip_idx, cudaStream_t* stream) {
    random_points<<<(nrows+TPB-1)/TPB, TPB, 0, stream[0]>>>(points_out, ncols, nrows, scale, offset, skip_idx);
}

// Generate random points in feature space to evaluate disruptivity at
__global__ void generate_test_points(float* feature_points_out, float* dim_points_out, int gn_points, int gn_features, int gn_dim, float* scale, float* offset, int* important_features)
{
    int point4thread = blockIdx.x*blockDim.x + threadIdx.x;
    float random_val;
    int j_dim = 0;
    int n_points = gn_points;
    int n_features = gn_features;
    int n_dim = gn_dim;
    float scale2use;
    if (point4thread < n_points)
        for(int i = 0; i < n_features; ++i) {
            random_val = curand_uniform(&sori_states[point4thread]);
            if (important_features[i] == 1) {
                dim_points_out[point4thread*n_dim+j_dim] = 2.0f * random_val - 1.0f;
                // KGF: if it is important, use "scale 1"
                scale2use = scale[i];
                ++j_dim;
            } else {
                // KGF: signal is not important, use scale 2
                // TODO(KGF): check paper for referneces to 2x scales. What is "important"
                // vs. active???
                scale2use = scale[n_features+ i];
            }
            feature_points_out[point4thread*n_features+i] = scale2use*(random_val-0.5)+offset[i];
        }
}

void Sori::gpuGenerateTestPoints(float* feature_points_out, float* scale, float* offset, int* important_features, cudaStream_t* streams)
{
    cudaDeviceSynchronize();
    generate_test_points<<<(NPTS+TPB-1)/TPB, TPB, 0, streams[1]>>>(feature_points_out, points, NPTS, NFEATURES, NDIM, scale, offset, important_features);
    cudaDeviceSynchronize();
}

/* This GPU kernel tests whether a point satisfies each of the constraints (individually)*/
__global__ void evaluate_constraints_satisfied(float* gpu_dot_products, float* gpu_constraint_sos, bool* gpu_pt_satisfies_constraint,int nconstraints, int npoints) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < (nconstraints*npoints)){
        gpu_pt_satisfies_constraint[idx] = gpu_dot_products[idx] < gpu_constraint_sos[idx/npoints];
        //if (idx >= nconstraints*npoints - 10){
            //printf("Thread %d: NCON = %d, dot product = %f, idx_div_NPTS = %d, constraint_sos = %f, satisfied=%d\n",idx, nconstraints, gpu_dot_products[idx], idx/npoints, gpu_constraint_sos[idx/npoints], gpu_pt_satisfies_constraint[idx]);
        //}
    }
}

void Sori::gpuEvaluateConstraintsSatisfied() {
    evaluate_constraints_satisfied<<<(NCON * NPTS +TPB-1)/TPB, TPB>>>(dotProducts, constraintSos, ptSatisfiesConstraint, NCON, NPTS);
}

/* This GPU kernel tests whether a point satisifies all constraints of a constraint set, updating the number of safe and unsafe points, as well as the cost function value */
__global__ void evaluate_all_constraints_satisfied(bool* gpu_pt_satisfies_constraint, bool* gpu_pt_satisfies_all_constraints, int* nsafe_inside, int* nunsafe_inside, float* disruptivity, float disruptivity_threshold, float weights, float* gpu_J, int npoints, int npopulation, int nlmis, int nconstraints) {
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
    /*if (idx<(npopulation)){
        nsafe_inside[idx] = 0;
        nunsafe_inside[idx] = 0;
    }
    __syncthreads();
    */
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
  /*__threadfence_block();
  __threadfence();
  __syncthreads();
  if (idx<(npopulation)){
    //printf("0 idx=%d, nsafe=%d, nunsafe=%d, cost=%f\n", idx, nsafe_inside[idx], nunsafe_inside[idx], gpu_J[idx]);
    gpu_J[idx] = cost_function(nsafe_inside[idx], nunsafe_inside[idx], weights);
    printf("1 idx=%d, nsafe=%d, nunsafe=%d, cost=%f\n", idx, nsafe_inside[idx], nunsafe_inside[idx], gpu_J[idx]);
  }
  */
}

__global__ void reset_nsafe_nunsafe(int* nsafe_inside, int* nunsafe_inside, int npopulation) {
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx<(npopulation)){
        nsafe_inside[idx] = 0;
        nunsafe_inside[idx] = 0;
    }
}
__global__ void calculate_cost(int* nsafe_inside, int* nunsafe_inside, float weights, float* gpu_J, int npopulation) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < npopulation) {
        //printf("0 idx=%d, nsafe=%d, nunsafe=%d, cost=%f\n", idx, nsafe_inside[idx], nunsafe_inside[idx], gpu_J[idx]);
        gpu_J[idx] = sori_costFunction(nsafe_inside[idx], nunsafe_inside[idx], weights);
        //printf("1 idx=%d, nsafe=%d, nunsafe=%d, cost=%f\n", idx, nsafe_inside[idx], nunsafe_inside[idx], gpu_J[idx]);
    }
}

void Sori::gpuEvaluateAllConstraintsSatisfied(float disruptivity_threshold, float weights) {
    reset_nsafe_nunsafe<<<(NPOP + TPB-1)/TPB, TPB>>>(nSafeInside, nUnsafeInside, NPOP);
    evaluate_all_constraints_satisfied<<<(NPTS * NPOP + TPB-1)/TPB, TPB>>>(ptSatisfiesConstraint, ptSatisfiesAllConstraints, nSafeInside, nUnsafeInside, disruptivity, disruptivity_threshold, weights, J, NPTS, NPOP, NLMI, NCON);
    calculate_cost<<<(NPOP + TPB-1)/TPB, TPB>>>(nSafeInside, nUnsafeInside, weights, J, NPOP);
}

/* Copy array source to destination */
__global__ void GpuCopy(float* des , float* __restrict__ sour ,const int M , const int N ) {
    int tx = blockIdx.x*blockDim.x+threadIdx.x;
    if (tx < N * M)
        des[tx] = sour[tx];
}

void Sori::gpuCopy(float* des, float* __restrict__ src, int const M, int const N, cudaStream_t* stream) {
    GpuCopy<<<(M*N+TPB-1)/TPB,TPB, 0, stream[2]>>>(des, src, M, N);
}

/* This gpu kernel performs the tournament selection step of the genetic algorithm */
__global__ void genetic_select(float* gpu_J, int* tournament_members, int* winners, float* gpu_constraints, float* gpu_constraints_prev, int npopulation, int ntournament, int nconstraints, int nlmis, int ndimensions)
{
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if (idx<npopulation) {
        for (int i=0; i < ntournament; ++i){
            tournament_members[idx*ntournament+i] = curand(&sori_states[idx])%npopulation;
            }
        winners[idx] = sori_geneticTournament(gpu_J, &tournament_members[idx*ntournament], ntournament);
        for(int i = 0; i < nlmis; ++i){
            for(int j = 0; j < ndimensions; ++j){
                gpu_constraints[idx*(nlmis*ndimensions) + j + ndimensions*i] = gpu_constraints_prev[winners[idx]*(nlmis*ndimensions) + j + ndimensions*i];
            }
        }
    }
}

void Sori::gpuGeneticSelect(cudaStream_t* stream)
{
    genetic_select<<<(NPOP * NTOURN +TPB-1)/TPB,TPB,0,stream[1]>>>(J, tournament, winners, constraints, constraintsPrev, NPOP, NTOURN, NCON, NLMI, NDIM);
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
            crossover = curand_uniform(&sori_states[idx]);
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

void Sori::gpuGeneticMate(float crossover_prob, cudaStream_t* stream) {
    genetic_mate<<<(NPOP/2+TPB-1)/TPB,TPB,0,stream[1]>>>(constraints, crossover_prob, NPOP, NLMI, NDIM, NCON);
}

/* This gpu kernel mutates the population with small perturbations */
__global__ void genetic_mutate(float* gpu_constraints, float mutate_prob, float mutate_stdev, int ncon_m_ndim)
{
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if (idx<(ncon_m_ndim)){
        float mutate;
        mutate = curand_uniform(&sori_states[idx]);
        float mutation;
        mutation = curand_normal(&sori_states[idx]) * mutate_stdev;
        if (mutate < mutate_prob) {
            gpu_constraints[idx] = max(min(gpu_constraints[idx] + mutation,1.0),-1.0);
        }
    }
}

void Sori::gpuGeneticMutate(float mutate_prob, float mutate_stdev, cudaStream_t* stream) {
    genetic_mutate<<<(NCON * NDIM + TPB - 1)/TPB,TPB,0,stream[1]>>>(constraints, mutate_prob, mutate_stdev, NCON * NDIM);
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
        __threadfence();
        __syncthreads();
        gpu_J_elite[idx] = t_best_score;
                //printf("block=%d,threadIdx.x=%d,t_elite=%d,J_elite=%f\n",idx,threadIdx.x,t_elite,gpu_J_elite[idx]);
        gpu_constraints[idx*nlmis*ndimensions*npopulation_div_n_elite+threadIdx.x] = gpu_constraints_prev[t_elite*nlmis*ndimensions+threadIdx.x];
        //for(int i = 0; i < nlmis; ++i){
        //  for(int j = 0; j < ndimensions; ++j){
        //      int a1 = nlmis*ndimensions;
        //          int a2 = j+ndimensions*i;
        //      //int idxA = idx*a1 + a2;
        //      gpu_constraints[idx*a1*npopulation_div_n_elite+a2] = gpu_constraints_prev[t_elite*a1+a2];
        //  }
        //}
    }
}

void Sori::gpuCarryOverElite(float* gpu_result, cudaStream_t* stream)
{
    carry_over_elite<<<NELITE, NLMI*NDIM, 0, stream[0]>>>(J, JElite, constraints, constraintsPrev, gpu_result, NELITE, NPOP % NELITE, NPOP / NELITE, NLMI, NDIM, NCON);
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

void Sori::gpuMmulConPts(const float * cons, const float * pts, float * dps, cudaStream_t* stream) {
    gpu_mmul_ABT<<<(NCON * NPTS + TPB - 1)/TPB,TPB,0,stream[1]>>>(cons, pts, dps, NCON, NDIM, NPTS);
}
