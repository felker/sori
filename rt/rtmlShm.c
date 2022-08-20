#include "rtmlShm.h"

#include "os/ppplShm.h"

#include <stddef.h>
#include <sys/ipc.h>	// key_t

void * rtMlgpuShmGet(enum MlgpuShmKey keyid) {
	key_t const key = (key_t) keyid;
	size_t size;

	switch (keyid) {
	case MLGPU_SHM_KEY_SORICFG:  size = sizeof(struct MlgpuShmSoriCfg); break;
	case MLGPU_SHM_KEY_SORIOUT:  size = sizeof(struct MlgpuShmSoriOut); break;
	case MLGPU_SHM_KEY_SORIIN:  size = sizeof(struct MlgpuShmSoriIn); break;
	default:
		return NULL;
		break;
	}

	return ppplShmGet(key, size);
}

