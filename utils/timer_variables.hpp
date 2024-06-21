#ifndef __TIMER_VARIABLE_HPP__
#define __TIMER_VARIABLE_HPP__
#pragma once
#include "rdtsc.h"

//Detailed Timing Variables
extern uint64_t rootPredictCycle;
extern uint64_t rootCorrectCycle;
extern uint64_t segmentPredictCycle;
extern uint64_t segmentCorrectCycle;

#endif