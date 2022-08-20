#pragma once

#include <cuda_runtime.h>

#define SORI_MAXNPTS 1000
#define SORI_MAXNCON 2000

/*
template<typename T>
class GpuMem {
public:
	GpuMem(std::size_t nElements):
		n(nElements),
		sz(nElements * sizeof(T)),
		dataC(nElements),
		dataG(api(cudaMalloc, sz))
	{ }

	~GpuMem() {
		delete[] dataC;
		api(cudaFree, dataG);
	}

	void rezero() {
		api(cudaMemset, dataG, 0, sz);
	}

	void push(T * in) {
		api(cudaMemcpy, dataG, in, sz, cudaMemcpyHostToDevice);
	}

	T const * pull() {
		api(cudaMemcpy, dataC.data(), dataG, sz, cudaMemcpyDeviceToHost);
		return dataC.data();
	}

	T * get() {
		return dataG;
	}

private:
	std::size_t n;
	std::size_t sz;
	std::vector<T> dataC;
	T * dataG = nullptr;
};
*/

struct SoriCfg {
	struct {
		// Sizes
		int nClasses;
		int nFeatures;
		int nNodes;
		int nTrees;

		// Model
		int * feature;
		int * chLeft;
		int * chRight;
		int * treeStart;
		float * threshold;
		float * value;

		// Inputs
		float * xEval;

		// Outputs
		float * treeResult;
		float * featureContributions;
		float * featureContributionsForest;
		float * totalResult;
		int * calculateContributions;
	} dprf;

	// Sizes
	int nStreams;
	int nDimensions;
	int nGenerations;
	int nPoints;
	int nLmi;
	int nPopulation;
	int nConstraints;
	int nTournaments;
	int nElite;

	// Scratch space
	float * operatingPoint;
	float * points;
	// FIXME: Add gpu_feature_points_out
	float * disruptivity;
	float * constraints;
	float * constraintsPrev;
	float * constraintSos;
	float * dotProducts;
	bool * ptSatisfiesConstraint;
	bool * ptSatisfiesAllConstraints;
	int * nUnsafeInside;
	int * nSafeInside;
	float * J;
	float * JElite;
	int * winners;
	int * tournament;
	float * result;

	// Inputs
	float * scales;
	int * important;
};

class Sori {
public:
	void gpuInit(unsigned int seed, SoriCfg const & cfg);
	void gpuEvaluateConstraintSos(cudaStream_t* stream);
	void gpuRandomPoints(float* points_out, int ncols, int nrows, float scale, float offset, int skip_idx, cudaStream_t* stream);
	void gpuGenerateTestPoints(float* feature_points_out, float* scale, float* offset, int* important_features, cudaStream_t* streams);
	void gpuEvaluateConstraintsSatisfied();
	void gpuEvaluateAllConstraintsSatisfied(float disruptivity_threshold, float weights);
	void gpuCopy(float* des, float* __restrict__ src, int const M, int const N, cudaStream_t* stream);
	void gpuGeneticSelect(cudaStream_t* stream);
	void gpuGeneticMate(float crossover_prob, cudaStream_t* stream);
	void gpuGeneticMutate(float mutate_prob, float mutate_stdev, cudaStream_t* stream);
	void gpuCarryOverElite(float* gpu_result, cudaStream_t* stream);
	void gpuMmulConPts(const float * cons, const float * pts, float * dps, cudaStream_t* stream);

private:
	int NFEATURES;
	int NCON;
	int NDIM;
	int NPOP;
	int NPTS;
	int NTOURN;
	int NELITE;
	int NLMI;
	int TPB;

	float * points;
	float * disruptivity;
	float * constraints;
	float * constraintsPrev;
	float * constraintSos;
	float * dotProducts;
	float * J;
	float * JElite;
	bool * ptSatisfiesConstraint;
	bool * ptSatisfiesAllConstraints;
	int * nUnsafeInside;
	int * nSafeInside;
	int * tournament;
	int * winners;
};
