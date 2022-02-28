#ifndef KERNEL_H
#define KERNEL_H

#include <cuda_runtime.h>

#define NSAMPLES 50
#define NPTS 200
//#define MAXNPTS 1000
//#define MAXNCON 400
#define NDIM 2
#define TPB 32
#define NLMI 4
#define NGEN 10
#define NPOP 60
#define NCON (NLMI*NPOP)
#define NTOURN 4
#define NELITE 8
#define NELITEMAX 8

extern "C" {

void init_launcher(unsigned int seed);
void random_points_launcher(float* points_out, int ncols, int nrows, float scale, float offset, int skip_idx, cudaStream_t* stream);
void evaluate_disruptivity_launcher(float* points, float* disruptivity);
void evaluate_constraint_sos_launcher(float* constraints, float* constraint_sos, cudaStream_t* stream);
void evaluate_constraints_satisfied_launcher(float* gpu_dot_products, float* gpu_constraint_sos, bool* gpu_pt_satisfies_constraint, int nconstraints, int npoints);
void evaluate_all_constraints_satisfied_launcher(bool* gpu_pt_satisfies_constraint, bool* gpu_pt_satisfies_all_constraints, int* nsafe_inside, int* nunsafe_inside, float* disruptivity, float disruptivity_threshold, float weights, float* gpu_J, int npoints, int npopulation, int nlmis, int nconstraints);
void nsafe_nunsafe_inside_launcher(int* nsafe_inside, int* nunsafe_inside, bool* pt_satisfies_all_constraints, float* disruptivity, float disruptivity_threshold, float weights, float* J);
void GpuCopy_launcher( float* des , float* __restrict__ sour ,const int M , const int N, cudaStream_t* stream );
void genetic_select_launcher(float* J, int* tournament_members, int* winners, float* constraints, float* constraints_prev, int npopulation, int ntournament, int nconstraints, int nlmis, int ndimensions, cudaStream_t* stream);
void genetic_mate_launcher(float* constraints, float crossover_prob,  int npopulation, int nlmis, int ndimensions, int nconstraints, cudaStream_t* stream);
void genetic_mutate_launcher(float* constraints, float mutate_prob, float mutate_stdev, int ncon_m_ndim, cudaStream_t* stream);
void carry_over_elite_launcher(float* J, float* J_elite, float* constraints, float* constraints_prev, float* result,  int n_elite, int npopulation_mod_n_elite, int npopulation_div_n_elite, int nlmis, int ndimensions, int nconstraints, cudaStream_t* stream);
void gpu_mmul_ABT_launcher(const float *A, const float *B, float *C, const int m, const int k, const int n, cudaStream_t* stream);
void free_random_states();
void generate_test_points_launcher(float* feature_points_out, float* dim_points_out, int gn_points, int gn_features, int gn_dim, float* scale, float* offset, int* important_features, cudaStream_t* stream);
}
#endif
