#pragma once

#define MLGPU_SORI_NFEATURES 8
#define MLGPU_SORI_NLMI 3
#define MLGPU_SORI_NDIM 2
#define MLGPU_SORI_NBEST (MLGPU_SORI_NLMI * MLGPU_SORI_NDIM)

#ifndef __cplusplus
#include <stdbool.h>
#endif

struct SoriIoCfg {
	struct {
		int tpb;
		int ndim;
		int ngen;
		int npts;
		int nlmi;
		int npop;
		int ncon;
		int ntourn;
		int nelite;
		int nfeatures;
	} sizes;
	float scaling[2][MLGPU_SORI_NFEATURES];
	int important[1][MLGPU_SORI_NFEATURES];
};

struct SoriIoIn {
	bool valid;
	float features[MLGPU_SORI_NFEATURES];
};

struct SoriIoOut {
	float best[MLGPU_SORI_NBEST];
};
