#include "rtgpu_sori.hh"
#include "soriRt.hh"

#include "rth5reader.hh"

#include "rtgpu_dprf_kernel.h"

#include "rtmlShm.h"
#include "os/ppplAtomic.h"
#include "util/logger.hh"

#include "rtcommon.hh"

#include <cstddef>
#include <memory>
#include <type_traits>

#include <cuda_runtime.h>

SoriCfg soriCfg;
Sori sori;

static std::vector<cudaStream_t> streams;

template <typename F, typename... Args> void api(F f, Args&&... args) {
	cudaError_t e = f(std::forward<Args>(args)...);
	if (__builtin_expect(e != cudaSuccess, false)) {
		Logger log("cuda");
		//log(Logger::Level::ERROR) << "cuda failed (" << f.target_type().name() << ": " << cudaGetErrorString(e) << std::endl;
		log(Logger::Level::ERROR) << "cuda failed: " << cudaGetErrorString(e) << std::endl;
		std::abort();
	}
}

rtgpu_Sori::rtgpu_Sori(SoriIoCfg const & cfg) {
	Logger log("rtgpu_Sori()");

	std::size_t const nFeatures = soriCfg.dprf.nFeatures = cfg.sizes.nfeatures;
	soriCfg.nStreams = 3;
	std::size_t const nDimensions = soriCfg.nDimensions = cfg.sizes.ndim;
	soriCfg.nGenerations = cfg.sizes.ngen;
	std::size_t const nPoints = soriCfg.nPoints = cfg.sizes.npts;
	std::size_t const nLmi = soriCfg.nLmi = cfg.sizes.nlmi;
	std::size_t const nPopulation = soriCfg.nPopulation = cfg.sizes.npop;
	std::size_t const nConstraints = soriCfg.nConstraints = cfg.sizes.nlmi * cfg.sizes.npop;
	std::size_t const nTournaments = soriCfg.nTournaments = cfg.sizes.ntourn;
	std::size_t const nElite = soriCfg.nElite = cfg.sizes.nelite;

	char const * fileForest = "forest.h5";
	RtH5Reader model(fileForest);

	std::size_t const nClasses = soriCfg.dprf.nClasses = model.getData<int>("/n_classes");
	std::size_t const nNodes = soriCfg.dprf.nNodes = model.getData<int>("/n_nodes");
	std::size_t const nTrees = soriCfg.dprf.nTrees = model.getData<int>("/n_trees");

	streams.resize(soriCfg.nStreams);
	for (auto & i: streams)
		api(cudaStreamCreate, &i);

	auto initCudaMalloc = [&]<typename T, typename U = void *>(T & ptr, std::size_t sz, U src = nullptr) {
		sz *= sizeof(*ptr);
		cudaMalloc(&ptr, sz);
		ptrs.emplace_back(ptr);

		if (src == nullptr)
			api(cudaMemset, ptr, 0, sz);
		else
			api(cudaMemcpy, ptr, src, sz, cudaMemcpyHostToDevice);
	};

	auto getData = [&]<typename T>(char const * name, T & gpu, size_t sz) {
		std::vector<std::remove_pointer_t<T>> data(sz);
		model.getData(name, data.data());
		initCudaMalloc(gpu, sz, data.data());
	};
	getData("/feature", soriCfg.dprf.feature, nNodes);
	getData("/children_left", soriCfg.dprf.chLeft, nNodes);
	getData("/children_right", soriCfg.dprf.chRight, nNodes);
	getData("/tree_start", soriCfg.dprf.treeStart, nNodes);
	getData("/threshold", soriCfg.dprf.threshold, nNodes);
	getData("/value", soriCfg.dprf.value, nNodes * nClasses);

	initCudaMalloc(soriCfg.dprf.xEval, nPoints * nFeatures);
	initCudaMalloc(soriCfg.dprf.treeResult, nPoints * nTrees);
	initCudaMalloc(soriCfg.dprf.featureContributions, nPoints * nFeatures);
	initCudaMalloc(soriCfg.dprf.featureContributionsForest, nPoints * nFeatures * nTrees);
	initCudaMalloc(soriCfg.dprf.calculateContributions, 1);

	initCudaMalloc(soriCfg.operatingPoint, nFeatures);
	initCudaMalloc(soriCfg.points, nPoints * nDimensions);
	initCudaMalloc(soriCfg.disruptivity, nPoints);
	initCudaMalloc(soriCfg.constraints, nConstraints * nDimensions);
	initCudaMalloc(soriCfg.constraintsPrev, nConstraints * nDimensions);
	initCudaMalloc(soriCfg.constraintSos, nConstraints);
	initCudaMalloc(soriCfg.dotProducts, nConstraints * nPoints);
	initCudaMalloc(soriCfg.ptSatisfiesConstraint, nConstraints * nPoints);
	initCudaMalloc(soriCfg.ptSatisfiesAllConstraints, nPopulation * nPoints);
	initCudaMalloc(soriCfg.nUnsafeInside,nPopulation);
	initCudaMalloc(soriCfg.nSafeInside, nPopulation);
	initCudaMalloc(soriCfg.J, nPopulation);
	initCudaMalloc(soriCfg.JElite, nElite);
	initCudaMalloc(soriCfg.winners, nPopulation);
	initCudaMalloc(soriCfg.tournament, nPopulation * nTournaments);
	initCudaMalloc(soriCfg.result, nLmi * nDimensions);

	// FIXME what is this? cudaMalloc(&gpu_feature_points_out, nPoints * nFeatures * sizeof(float));

	initCudaMalloc(soriCfg.scales, nFeatures * 2, cfg.scaling);
	initCudaMalloc(soriCfg.important, nFeatures, cfg.important);

	sori.gpuInit(1, soriCfg);
}

void rtgpu_Sori::run(SoriIoIn in, SoriIoOut & out) {
	Logger log("sorirun");
	bool const dbg = (cycle > 10 && cycle < 14);

	api(cudaMemcpy, soriCfg.operatingPoint, in.features, soriCfg.dprf.nFeatures * sizeof(float), cudaMemcpyHostToDevice); // this cycle's input
	sori.gpuGenerateTestPoints(soriCfg.dprf.xEval, soriCfg.scales, soriCfg.operatingPoint, soriCfg.important, streams.data());

	DprfIn dprfIn {
		.nTrees = soriCfg.dprf.nTrees,
		.nPoints = soriCfg.nPoints,
		.nClasses = soriCfg.dprf.nClasses,
		.nFeatures = soriCfg.dprf.nFeatures,
		.xEval = soriCfg.dprf.xEval,
		.feature = soriCfg.dprf.feature,
		.chLeft = soriCfg.dprf.chLeft,
		.chRight = soriCfg.dprf.chRight,
		.treeStart = soriCfg.dprf.treeStart,
		.threshold = soriCfg.dprf.threshold,
		.value = soriCfg.dprf.value
	};

	dprf_forestLauncher(dprfIn,
			soriCfg.dprf.treeResult,
			soriCfg.dprf.featureContributions,
			soriCfg.dprf.featureContributionsForest,
			soriCfg.disruptivity,
			soriCfg.dprf.calculateContributions);
	api(cudaDeviceSynchronize);

	// FIXME: What is 'best'? best needs to be stored and kept between each call to this function. Store here or in PCS?
	sori.gpuRandomPoints(soriCfg.constraints, soriCfg.nDimensions, soriCfg.nConstraints, 0.5f, 0.25f, -1 /*best*/, streams.data());
	api(cudaDeviceSynchronize);

	float constraints[soriCfg.nConstraints * soriCfg.nDimensions];
	for (int i = 0; i < soriCfg.nGenerations; ++i) {
		//if (dbg) {
			//log(Logger::Level::INFO) << "Start of generation: " << i << '\n';
			//cudaMemcpy(constraints, soriCfg.constraints, sizeof(float) * soriCfg.nConstraints * soriCfg.nDimensions, cudaMemcpyDeviceToHost);
			//api(cudaDeviceSynchronize);
			//for (int i = 0; i < 20; ++i)
				//log(Logger::Level::INFO) << "1 constraints: " << constraints[i] << '\n';
		//}

		sori.gpuEvaluateConstraintSos(streams.data());
		api(cudaDeviceSynchronize);

		sori.gpuMmulConPts(soriCfg.constraints, soriCfg.points, soriCfg.dotProducts, streams.data());
		api(cudaDeviceSynchronize);

		sori.gpuCopy(soriCfg.constraintsPrev, soriCfg.constraints, soriCfg.nConstraints, soriCfg.nDimensions, streams.data());
		api(cudaDeviceSynchronize);
		
		sori.gpuEvaluateConstraintsSatisfied();
		api(cudaDeviceSynchronize);

	  	float const weights = 40.0f;
	  	sori.gpuEvaluateAllConstraintsSatisfied(0.4f, weights);
		api(cudaDeviceSynchronize);
 
	  	sori.gpuGeneticSelect(streams.data());
		api(cudaDeviceSynchronize);

	  	//Based on probability, mate two individuals (two in, two out, modify in place).
	  	float const crossoverProb = 0.5f;
	  	sori.gpuGeneticMate(crossoverProb, streams.data());
		api(cudaDeviceSynchronize);

	  	//Based on probability, mutate individual traits
	  	float const mutateProb = 0.5f;
	  	float const mutateStdev = 0.3f;
	  	sori.gpuGeneticMutate(mutateProb, mutateStdev, streams.data());
		api(cudaDeviceSynchronize);

	  	//Carry over elite individual(s) from previous generation
	  	sori.gpuCarryOverElite(soriCfg.result, streams.data());
		api(cudaDeviceSynchronize);
	}

	float jElite[soriCfg.nElite];
	cudaMemcpy(jElite, soriCfg.JElite, sizeof(float) * soriCfg.nElite, cudaMemcpyDeviceToHost);
	api(cudaDeviceSynchronize);
	//Find index of maximum value of jElite
	int best_elite = 0;
	float best_J = jElite[0];
	for (int i = 0; i < soriCfg.nElite; ++i) {
		if (jElite[i] > best_J) {
			best_elite = i;
			best_J = jElite[best_elite];
		}
	}
	//Convert to index into soriCfg.constraints
	int best = best_elite * soriCfg.nLmi * soriCfg.nDimensions * (soriCfg.nPopulation / soriCfg.nElite);

	cudaMemcpy(constraints, soriCfg.constraints, sizeof(float) * soriCfg.nConstraints * soriCfg.nDimensions, cudaMemcpyDeviceToHost);
	api(cudaDeviceSynchronize);
	for (int i = 0; i < soriCfg.nLmi * soriCfg.nDimensions; ++i) {
		out.best[i] = constraints[best+i];
		// if (dbg) log(Logger::Level::INFO) << "best constraints: " << out.best[i] << '\n';
	}

	++cycle;
}

rtgpu_Sori::~rtgpu_Sori() {
	for (auto & i: streams)
		api(cudaStreamDestroy, i);
	streams.clear();
	for (auto & i: ptrs)
		api(cudaFree, i);
}
