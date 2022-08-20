#pragma once

#ifndef SORI
#define TPB 640
#define MAX_TREES 300
#define MAX_FEATURES 8
//#define NPOINTS 200
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct DprfIn {
	int nTrees;
	int nPoints;
	int nClasses;
	int nFeatures;
	float * xEval;
	int * feature;
	int * chLeft;
	int * chRight;
	int * treeStart;
	float * threshold;
	float * value;
};

void dprf_forestLauncher(DprfIn,
		float *result,
		float *feature_contributions,
		float *feature_contributions_forest,
		float *total_result,
		int *calculate_contributions);

#ifdef __cplusplus
};
#endif
