#include "ppplShm.h"

#include <errno.h>
#include <error.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/ipc.h>	// key_t
#include <sys/mman.h>
#include <sys/shm.h>

void * ppplShmGet(key_t key, size_t size) {
	int const mask = IPC_CREAT | 0666;

	errno = 0;
	int shmid = shmget(key, size, mask);

	if (shmid == -1) {
		switch (errno) {
		case EACCES: error(1, errno, "EACCES: Permission denied"); break;
		case EEXIST: error(1, errno, "EEXIST: Shm id already exists while trying to exclusively create"); break;
		case EINVAL: error(1, errno, "EINVAL: Invalid size"); break;
		case ENFILE: error(1, errno, "ENFILE: Too many shm segments and files in use"); break;
		case ENOENT: error(1, errno, "ENOENT: Shm id doesn't exist, and IPC_CREAT is not set"); break;
		case ENOMEM: error(1, errno, "ENOMEM: Not enough shm space available"); break;
		case ENOSPC: error(1, errno, "ENOSPC: Not enough shm space or shm ids available"); break;
		default: error(1, errno, "Unknown"); break;
		}
	}

	errno = 0;
	void * addr = shmat(shmid, NULL, 0);
	if ((intptr_t)addr == -1) {
		switch (errno) {
		case EACCES: error(1, errno, "EACCES: Permission denied"); break;
		case EINVAL: error(1, errno, "EINVAL: Invalid shm id or invalid address"); break;
		case EMFILE: error(1, errno, "EMFILE: Too many shm segments in use"); break;
		case ENOMEM: error(1, errno, "ENOMEM: Not enough shm space available"); break;
		default: error(1, errno, "Unknown"); break;
		}
	}

	return addr;
}

void ppplShmRelease(void const * const addr) {
	errno = 0;
	if (shmdt(addr) == -1) {
		switch(errno) {
		case EINVAL: error(1, errno, "EINVAL: Supplied address was not attached"); break;
		default:
			return;
		}
	}
}

