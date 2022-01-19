#include <time.h>
#define TIME(t) clock_gettime(CLOCK_REALTIME, &t)
double tdiff(struct timespec tbeg, struct timespec tend) {
        double t1 = (double)tbeg.tv_sec + (double)(tbeg.tv_nsec * 1e-9);
        double t2 = (double)tend.tv_sec + (double)(tend.tv_nsec * 1e-9);
        return t2-t1;
}

