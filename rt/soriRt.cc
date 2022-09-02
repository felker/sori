#include <vector>

#include "util/logger.hh"
#include "rth5reader.hh"
#include "rtgpu_sori.hh"
#include <iostream>
#include <fstream>
#include<cmath>
using namespace std;

int main() {
	Logger log("soriRt");
	log(Logger::Level::INFO) << "Loading first cycle from shot file:" << std::endl;

	char const * shotFile = "180808_new.h5";
	RtH5Reader shot(shotFile);

	int nFeatures = shot.getData<int>("/n_features");
	int nTimes = shot.getData<int>("/n_times");
	std::vector<float> data(nFeatures * nTimes);
	shot.getData("/X", data.data());
	float sample[8] = {1.10563e+06, 187400, 1.93632e+19, 0.600951, 0.43912, 0.326558, 1.77533, 0.868978};//{382907, 34999, 1.17261e+19, 0.622526, 0.00270697, 0.414342, 1.41309, 0.631528};
    //int sample_index = 19200;//18892;
	//for (int i = 0; i < nFeatures; ++i)
	//	log << "value " << i << ' ' << data[i+nFeatures*sample_index] << '\n';

    int nlmi = 3;
    int npop = 300;
	SoriIoCfg cfg {
		.sizes = {
			.tpb = 32,
			.ndim = 2,
			.ngen = 25,
			.npts = 900,
			.nlmi = nlmi,
			.npop = npop,
			.ncon = nlmi*npop,
			.ntourn = 4,
			.nelite = 10,
			.nfeatures = nFeatures
		},
		.scaling = {
			{ 0.5e6f,  6.0e5f, 5.0e19f, 0.05f,  0.4f,  0.02f,   0.4f,   0.5f  },
			{ 0.05e6f, 5.0e4f, 1.0e19f, 0.02f, 0.04f, 0.01f, 0.05f, 0.1f }
		},
		.important = {
			{ 1, 1, 0, 0, 0, 0, 0, 0 }
		}
	};
	for (int i = 0; i < nFeatures; ++i){
        log << "important: " << cfg.important[0][i] << '\n';
        }
	rtgpu_Sori sori(cfg);
	SoriIoIn in {
		.valid = true,
		.features {}
	};
	for (int i = 0; i < nFeatures; ++i){
		in.features[i] = sample[i];//data[i+nFeatures*sample_index];
		log << "features[" << i << "]:" << in.features[i] << '\n';
	}
	SoriIoOut out;

	sori.run(in, out);

	log(Logger::Level::INFO) << "Output: \n";
	ofstream constraint_file;
    constraint_file.open ("./constraint.txt");
	for (int i = 0; i < MLGPU_SORI_NBEST; ++i)
		log << "best[" << i << "]: " << out.best[i] << '\n';
    
    for (int i=0; i < nlmi; ++i)
        constraint_file << out.best[i*2] << ' ' << out.best[i*2+1] << '\n';
    
    constraint_file.close();
    
    int active_feature_index = 0;
    int feature_index = 0;
    int proximity_normalization[2]; //FIXME: 2 should be ndim, but only planning to use 2 in experiment
    //FIXME: this could be done in pre-shot setup
    while ((active_feature_index < cfg.sizes.ndim) && (feature_index < cfg.sizes.nfeatures)) {
        //log << "feature_index: " << feature_index << '\n';
        //log << "active_feature_index: " << active_feature_index << '\n';
        //log << "important: " << cfg.important[0][feature_index] << '\n';
        if (cfg.important[0][feature_index] == 1){
            //log << "important\n";
            proximity_normalization[active_feature_index] = cfg.scaling[0][feature_index]/cfg.scaling[1][feature_index];
            ++active_feature_index;
        }
        ++feature_index;
    }
    
    //Find the row in out.best with the smallest normalized distance to disruption
    float norm_dist_arr[MLGPU_SORI_NLMI];
    int min_norm_dist_index = 0;
    float min_norm_dist = 1e6f;
    for (int i=0; i < nlmi; ++i){
        norm_dist_arr[i] = sqrt(pow(out.best[i*2]*proximity_normalization[0],2) + pow(out.best[i*2+1]*proximity_normalization[1],2));
        if (norm_dist_arr[i] < min_norm_dist){
            min_norm_dist = norm_dist_arr[i];
            min_norm_dist_index = i;
        }
        //log << "norm_dist[" << i << "]: " << norm_dist_arr[i] << "\n";
    }
    //log << "min_norm_dist_index: " << min_norm_dist_index << " min_norm_dist: " << min_norm_dist << "\n";
    
    //Find the direction to change active features to increase distance from disruption by 1 (in normalized distance)
    float opt_dir_scaled[MLGPU_SORI_NFEATURES];
    
    active_feature_index = 0;
    for (feature_index=0; feature_index< cfg.sizes.nfeatures; ++feature_index){
        if (cfg.important[0][feature_index] == 1){
            opt_dir_scaled[feature_index] = -out.best[min_norm_dist_index*2+active_feature_index]*cfg.scaling[0][feature_index]/min_norm_dist;
            ++active_feature_index;
        }
        else opt_dir_scaled[feature_index] = 0.0;
    }
    
    //FIXME: min_norm_dist to be sent to Proximity Control
    //FIXME: opt_dir_scaled to be sent to Proximity Control (requires mapping vector elements to the order and units Jayson expects).
    
    for (int i=0; i < cfg.sizes.nfeatures; ++i){
        log << "opt_dir_scaled[" << i << "]: " << opt_dir_scaled[i] << '\n';
    }
    //log << "prox norm: " << proximity_normalization[0] << ' ' << proximity_normalization[1] << '\n'; 
	return 0;
}
