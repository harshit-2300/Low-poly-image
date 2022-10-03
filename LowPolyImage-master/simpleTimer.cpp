#include "time.h"
#include "simpleTimer.h"
#include <iostream>
#include <string>

simpleTimer::simpleTimer(std::string name)
{
    my_name = name;
    clock_gettime(CLOCK_MONOTONIC, &start);
}

void simpleTimer::GetDuration()
{
    double time_used;

    clock_gettime(CLOCK_MONOTONIC, &end);
    if ((end.tv_nsec - start.tv_nsec) < 0) {
        temp.tv_sec = end.tv_sec-start.tv_sec-1;
        temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
    } else {
        temp.tv_sec = end.tv_sec - start.tv_sec;
        temp.tv_nsec = end.tv_nsec - start.tv_nsec;
    }
    time_used = temp.tv_sec + (double) temp.tv_nsec / 1000000000.0;
    
    std::cout << my_name << ": " << time_used << "seconds" << std::endl;
}