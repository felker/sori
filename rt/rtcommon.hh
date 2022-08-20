#pragma once

#include <atomic>

namespace rt {

static std::atomic<bool> shutdown { false };

}
