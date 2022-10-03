#pragma once

#include "time.h"
#include "stdio.h"
#include <string>

class simpleTimer
{
    private:
        struct timespec start;
        struct timespec end;
        struct timespec temp;
        std::string my_name;
    public:
        simpleTimer(std::string name);
        void GetDuration();
};