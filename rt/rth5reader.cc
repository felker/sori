#include "rth5reader.hh"

#include <type_traits>

#include <hdf5.h>

static hid_t getHid(void * in) {
	return *((hid_t *)in);
}

RtH5Reader::RtH5Reader(std::string const & fname): fname(fname) {
	file_id = new hid_t;
	*((hid_t *)file_id) = H5Fopen(fname.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
}

RtH5Reader::~RtH5Reader() {
	hid_t f = getHid(file_id);
	H5Fclose(f);
	delete (hid_t *)file_id;
}

template<typename T>
void RtH5Reader::getData(char const * name, T * data) {
	hid_t ds_id = H5Dopen2(getHid(file_id), name, H5P_DEFAULT);
	if constexpr (std::is_same_v<T, int>)
	/*herr_t status =*/ H5Dread(ds_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
	else if constexpr (std::is_same_v<T, float>)
	/*herr_t status =*/ H5Dread(ds_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
	/*status =*/ H5Dclose(ds_id);
}
template void RtH5Reader::getData(char const *, int *);
template void RtH5Reader::getData(char const *, float *);

template<typename T>
T RtH5Reader::getData(char const * name) {
	T tmp;
	getData(name, &tmp);
	return tmp;
}
template int RtH5Reader::getData(char const *);
template float RtH5Reader::getData(char const *);
