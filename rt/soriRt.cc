#include <vector>

#include "util/logger.hh"
#include "rth5reader.hh"
#include "rtgpu_sori.hh"

int main() {
	Logger log("soriRt");
	log(Logger::Level::INFO) << "Loading first cycle from shot file:" << std::endl;

	char const * shotFile = "180808_new.h5";
	RtH5Reader shot(shotFile);

	int nFeatures = shot.getData<int>("/n_features");
	int nTimes = shot.getData<int>("/n_times");
	std::vector<float> data(nFeatures * nTimes);
	shot.getData("/X", data.data());

	for (int i = 0; i < nFeatures; ++i)
		log << "value " << i << ' ' << data[i] << '\n';

	SoriIoCfg cfg {
		.sizes = {
			.tpb = 32,
			.ndim = 2,
			.ngen = 25,
			.npts = 900,
			.nlmi = 3,
			.npop = 400,
			.ncon = 1200,
			.ntourn = 4,
			.nelite = 10,
			.nfeatures = nFeatures
		},
		.scaling = {
			{ 0.5e6f,  6.0e5f, 5.0e19f, 0.05f,  0.4f,  0.02f,   0.4f,   0.5f  },
			{ 0.02e6f, 2.0e4f, 4.0e18f, 0.005f, 0.01f, 0.0025f, 0.005f, 0.05f }
		},
		.important = {
			{ 1, 1, 0, 0, 0, 0, 0, 0 }
		}
	};
	rtgpu_Sori sori(cfg);

	SoriIoIn in {
		.valid = true,
		.features {}
	};
	for (int i = 0; i < nFeatures; ++i)
		in.features[i] = data[i];
	SoriIoOut out;

	sori.run(in, out);

	log(Logger::Level::INFO) << "Output: \n";
	for (int i = 0; i < MLGPU_SORI_NBEST; ++i)
		log << "best[" << i << "]: " << out.best[i] << '\n';

	return 0;
}
