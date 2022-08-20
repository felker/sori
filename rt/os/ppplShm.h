#pragma once

#include <stddef.h>
#include <sys/ipc.h>	// key_t

#ifdef __cplusplus
extern "C" {
#endif

void * ppplShmGet(key_t key, size_t size);
void ppplShmRelease(void const * const addr);

#ifdef __cplusplus
}
#endif
