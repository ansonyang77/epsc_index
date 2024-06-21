#ifndef __DEBUG_UTIL_HPP__
#define __DEBUG_UTIL_HPP__

#pragma once
#include <stdio.h>
#include <cstdio>

#define PRINT_MESSAGE(x) printf("%s \n",x)

#define ASSERT_MESSAGE(exp, msg) assert(((void)msg, exp))

#ifdef DEBUG
    #define LOG_DEBUG(format,...) printf("\033[33mDEBUG : %s.%d.%s.: \033[0m" format" \n", __FILE__, __LINE__, __PRETTY_FUNCTION__,##__VA_ARGS__);
    #define LOG_DEBUG_SHOW(format,...) printf("\033[33mDEBUG : %s.%d.%s.: \033[0m" format" \n", __FILE__, __LINE__, __PRETTY_FUNCTION__,##__VA_ARGS__);
#else
    #define LOG_DEBUG(format,...) do {} while(0);
    #define LOG_DEBUG_SHOW(format,...) printf("\033[33mDEBUG : %s.%d.%s.: \033[0m" format" \n", __FILE__, __LINE__, __PRETTY_FUNCTION__,##__VA_ARGS__);
#endif

#ifdef LOG_ON
    #define LOG_ERROR(format,...) printf("\033[31mERROR : %s.%d.%s.: \033[0m" format" \n", __FILE__, __LINE__, __PRETTY_FUNCTION__,##__VA_ARGS__); 
    #define LOG_INFO(format,...) printf("\033[32mINFO : %s.%d.%s.: \033[0m" format" \n", __FILE__, __LINE__, __PRETTY_FUNCTION__,##__VA_ARGS__); 
    #define LOG_WARNING(format,...) printf("\033[33mWARNING : %s.%d.%s.: \033[0m" format" \n", __FILE__, __LINE__, __PRETTY_FUNCTION__,##__VA_ARGS__);
#else
    #define LOG_ERROR(format,...) printf("\033[31mERROR : %s.%d.%s.: \033[0m" format" \n", __FILE__, __LINE__, __PRETTY_FUNCTION__,##__VA_ARGS__); 
    #define LOG_INFO(format,...) do {} while(0);
    #define LOG_WARNING(format,...) do {} while(0);
#endif

#endif