#pragma once

#include "rtgpu_soriIo.h"

#include <vector>

class rtgpu_Sori {
public:
	rtgpu_Sori(SoriIoCfg const & cfg);
	~rtgpu_Sori();

	void run(SoriIoIn, SoriIoOut &);

private:
	std::vector<void *> ptrs;
	unsigned int cycle = 0;
};
