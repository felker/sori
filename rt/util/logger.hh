#pragma once

#include <ctime>
#include <iostream>
#include <string>
#include <vector>

class Logger {
public:
	enum class Level {
		SILENT,
		ERROR,
		WARN,
		INFO,
		TIME,
		DEBUG,
		DEFAULT
	};
	void setLogLevel(Level l);

	Logger(std::ostream &, std::string const & name, Level l = Level::DEFAULT);
	Logger(std::string const & name, Level l = Level::DEFAULT);

	template<typename T> friend Logger & operator<<(Logger & l, T const & s);
	Logger & operator<<(std::ostream & (*f)(std::ostream&));
	Logger & operator()(Level l);
	Logger & operator()();
	Logger & log(Level l);

	void flush();

	void timeSinceStart();
	void timeSinceLastSnap();
	void timeSinceSnap(std::string);
	void add_snapshot(std::string n, bool quiet = true);

private:
	std::ostream & os;
	std::string name;
	Level lvlLog = Level::DEFAULT;
	Level lvlMsg;

  std::time_t _now;
  std::time_t _start;
  std::vector<std::time_t> _snaps;
  std::vector<std::string> _snap_ns;
};

template<typename T> Logger & operator<<(Logger & l, T const & s) {
	if (l.lvlMsg <= l.lvlLog )
		l.os << s;
	return l;
}

// Allow logging of a void volatile * (and subsequently std::cout in general)
inline std::ostream & operator<<(std::ostream & os, void const volatile * p) {
	return os << const_cast<void const *>(p);
}
