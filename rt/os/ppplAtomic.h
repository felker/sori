#pragma once

#ifdef __cplusplus

template<typename T>
inline void ppplAtomicStore(T * dest, T src) {
	__atomic_store_n(dest, src, __ATOMIC_RELEASE);
}

template <typename T>
inline T ppplAtomicLoad(T * src) {
	return __atomic_load_n(src, __ATOMIC_ACQUIRE);
}

#else

#define ppplAtomicStore(dest, src) __atomic_store_n(dest, src, __ATOMIC_RELEASE)
#define ppplAtomicLoad(src) __atomic_load_n(src, __ATOMIC_ACQUIRE)

#endif

