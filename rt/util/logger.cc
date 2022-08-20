#include "logger.hh"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <iomanip>

Logger::Logger(std::ostream& os, std::string const & n, Level l) :
	os(os),
	lvlLog(l),
	lvlMsg(Level::SILENT)
{
	name = "[" + n + "]";
	std::time(&_now);
	std::time(&_start);
}

Logger::Logger(std::string const & n, Level l): Logger(std::cout, n, l) {}

//#define BSLOG_TIME    "\033[0;35m[ TIME    ]\033[0;0m"
//#define BSLOG_DEBUG   "[ DEBUG   ]"
//#define BSLOG_ERROR   "\033[0;31m[ ERROR   ]\033[0;0m"
//#define BSLOG_WARNING "\033[0;33m[ WARNING ]\033[0;0m"
//#define BSLOG_INFO    "\033[0;34m[ INFO    ]\033[0;0m"
static constexpr char const * getLevel(Logger::Level l) {
	switch (l) {
		case Logger::Level::ERROR: return "[ERROR]";
		case Logger::Level::WARN:  return "[WARN ]";
		case Logger::Level::INFO:  return "[INFO ]";
		case Logger::Level::TIME:  return "[TIME ]";
		case Logger::Level::DEBUG: return "[DEBUG]";
		default: return "";
	}
}

using Timestamp = char[30];

static char * getTime(Timestamp time) {
	using Clock = std::chrono::high_resolution_clock;
	auto t = Clock::now();
	std::time_t tp = Clock::to_time_t(t);
	std::strftime(time, 21, "%F %T." , std::localtime(&tp));
	int us = std::chrono::duration_cast<std::chrono::microseconds>(t.time_since_epoch()).count() % 1000000;
	std::sprintf(time+20, "%06d: ", us);
	return time;
}

Logger & Logger::operator()() {
	return *this;
}

Logger & Logger::operator()(Level l) {
	lvlMsg = l;
	Timestamp t;
	if (lvlMsg <= lvlLog)
		os << getTime(t) << getLevel(l) << name << ": ";
	return *this;
}

Logger & Logger::log(Level l) {
	return this->operator()(l);
}

// Allow logging of a std::endl, which is a function template
Logger & Logger::operator<<(std::ostream & (*f)(std::ostream&)) {
	f(os);
	return *this;
}

void Logger::timeSinceStart() {
	if (lvlLog >= Level::TIME) {
		std::time(&_now);
		log(Level::TIME) << std::difftime(_now, _start) << "s since instantiation\n";
	}
}

void Logger::timeSinceLastSnap() {
	if (lvlLog >= Level::TIME && _snap_ns.size() > 0) {
		std::time(&_now);
		log(Level::TIME) << std::difftime(_now, _snaps.back()) << "s since snap '" << _snap_ns.back() << "'\n";
	}
}

void Logger::timeSinceSnap(std::string s) {
	if (lvlLog >= Level::TIME) {
		std::time(&_now);
		auto it = find(_snap_ns.begin(), _snap_ns.end(), s);
		if (it == _snap_ns.end()) {
			log(Level::WARN) << "Could not find snapshot " << s << '\n';
			return;
		}

		auto const dist = std::distance(_snap_ns.begin(), it);
		log(Level::TIME) << std::difftime(_now, _snaps[dist]) << "s since snap '" << _snap_ns[dist] << "'\n";
	}
}

void Logger::add_snapshot(std::string n, bool quiet) {
	std::time_t now;
	std::time(&now);
	_snaps.push_back(now);
	_snap_ns.push_back(n);
	if (lvlLog >= Level::TIME && ! quiet)
		log(Level::TIME) << ": Added snap '" << n << "'\n";
}

void Logger::setLogLevel(Level l) {
	lvlLog = l;
}

void Logger::flush() {
	os.flush();
}

