#include <string>

class RtH5Reader {
public:
	RtH5Reader(std::string const & fname);
	~RtH5Reader();
	template<typename T> T getData(char const * name);
	template<typename T> void getData(char const * name, T * data);

private:
	std::string fname;
	void * file_id;
};

//extern template int RtH5Reader::getData(char const *);
//extern template float RtH5Reader::getData(char const *);
