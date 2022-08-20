#pragma once

#include "rtgpu_soriIo.h"

#include <stdbool.h>
#include <stddef.h>

#define MLGPUSHM_NCYCLES 100
#define MLGPUSHM_NDATA_OUT 16
#define MLGPUSHM_NDATA_IN 16
#define MLGPUSHM_SORI_NRING 1024

enum MlgpuShmKey {
	MLGPU_SHM_KEY_SORICFG = 42,
	MLGPU_SHM_KEY_SORIIN,
	MLGPU_SHM_KEY_SORIOUT,
};

struct MlgpuShmPcsOut {
	size_t index;
	struct {
		float data[MLGPUSHM_NDATA_OUT];
	} output[MLGPUSHM_NCYCLES];
};

struct MlgpuShmPcsIn {
	size_t index;
	struct {
		float data[MLGPUSHM_NDATA_IN];
	} input[MLGPUSHM_NCYCLES];
};

struct MlgpuShmSoriCfg {
	size_t index;
	struct SoriIoCfg ring[MLGPUSHM_SORI_NRING];
};

struct MlgpuShmSoriIn {
	size_t index;
	struct SoriIoIn ring[MLGPUSHM_SORI_NRING];
};

struct MlgpuShmSoriOut {
	size_t index;
	struct SoriIoOut ring[MLGPUSHM_SORI_NRING];
};

#ifdef __cplusplus
extern "C" {
#endif

// FIXME: This should return a const pointer to guarantee it can be passed back to release
void * rtMlgpuShmGet(enum MlgpuShmKey);

#ifdef __cplusplus
}

#include "os/ppplShm.h"

#include <type_traits>
template <typename T, typename U = std::enable_if_t<!std::is_void<T>::value>>
T * rtMlgpuShmGet(MlgpuShmKey key) {
	return static_cast<T *>(rtMlgpuShmGet(key));
}

template <class T> class MlgpuShm {
public:
	//MlgpuShm(): addr(rteceShmGet<T>(key) {}
	MlgpuShm() {}
	~MlgpuShm() {
		ppplShmRelease(static_cast<void *>(addr));
	}

	T * operator->() {
		return addr;
	}

private:
	auto key() {
		if (std::is_same<T, MlgpuShmSoriCfg>::value) return MLGPU_SHM_KEY_SORICFG;
		else if (std::is_same<T, MlgpuShmSoriOut>::value) return MLGPU_SHM_KEY_SORIOUT;
		else if (std::is_same<T, MlgpuShmSoriIn>::value) return MLGPU_SHM_KEY_SORIIN;
		else
			throw;
	}

	T * const addr = rtMlgpuShmGet<T>(key());
};

#endif
