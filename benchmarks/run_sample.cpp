#pragma once
#include<iostream>
#include<algorithm>
#include<iterator>
#include<vector>
#include<array>
#include<numeric>
#include<cassert>

#include <x86intrin.h>
#include <iomanip>
#include <type_traits>

#include "../src/rmi.hpp"
#include "../src/rmi_equal_split_key.hpp"
#include "../src/rmi_equal_split_size.hpp"

#include "../utils/load.hpp"
#include "../utils/rdtsc.h"

//Experiment Configurations
#ifndef NUM_TRAILS
#define NUM_TRAILS 30
#endif

#ifndef NUM_SAMPLES
#define NUM_SAMPLES 10000000
#endif

enum class RunType {NONE = -1, RMI = 0, EQUAL_KEY = 1, EQUAL_SIZE = 2, EQUAL_SIZE_DISTR = 3};
RunType RUN_FLAG = RunType::NONE;

//Index Configuration (Confirguration details in src/helper_functions/helper_rmi.hpp)
rmi::ModelType ROOT_TYPE = rmi::ModelType::NONE;
rmi::ModelType LEAF_TYPE = rmi::ModelType::NONE;
rmi::SearchType ROOT_SEARCH_TYPE = rmi::SearchType::NONE;
rmi::SearchType LEAF_SEARCH_TYPE = rmi::SearchType::NONE;

//Result Tracking Structures
int trailCounter = 0;
#ifndef DETAILED_TIME
vector<double> searchCycleTrails(NUM_TRAILS);
#else
vector<double> rootPredictCycleTrails(NUM_TRAILS);
vector<double> rootCorrectCycleTrails(NUM_TRAILS);
vector<double> segmentPredictCycleTrails(NUM_TRAILS);
vector<double> segmentCorrectCycleTrails(NUM_TRAILS);
#endif
vector<double> memoryUsageTrails(NUM_TRAILS);
vector<double> searchRangeTrails(NUM_TRAILS);
vector<double> dataErrorTrails(NUM_TRAILS);

//Function Headers
void praser(int argc, char** argv);
uint64_t run_rmi(vector<uint64_t> & sampledData);
uint64_t run_equal_split_key(vector<uint64_t> & sampledData);
uint64_t run_equal_split_data(vector<uint64_t> & sampledData);
uint64_t run_equal_split_data_distribution(vector<uint64_t> & sampledData, vector<uint64_t> & underlyingData);

int main(int argc, char** argv)
{
    ASSERT_MESSAGE (NUM_SAMPLES < NUM_DATA, "NUM_DATA must be larger than NUM_SAMPLES");

    praser(argc, argv);

    vector<uint64_t> data = load_data(NUM_DATA);
    sort(data.begin(), data.end());
	data.erase( unique( data.begin(), data.end() ), data.end());

    vector<uint64_t> samples(NUM_SAMPLES);
    volatile uint64_t count = 0;
    for (int trail = 0; trail < NUM_TRAILS; ++trail)
    {
        // sample_from_data(NUM_SAMPLES,data,samples);
        // sort(samples.begin(), samples.end());
        // samples.erase( unique( samples.begin(), samples.end() ), samples.end());

        sample_from_data_without_replacement(NUM_SAMPLES,data,samples);
        sort(samples.begin(), samples.end());

        switch(RUN_FLAG)
        {
            case RunType::RMI:
                count += run_rmi(samples);
                break;
            
            case RunType::EQUAL_KEY:
                count += run_equal_split_key(samples);
                break;
            
            case RunType::EQUAL_SIZE:
                count += run_equal_split_data(samples);
                break;
            
            case RunType::EQUAL_SIZE_DISTR:
                count += run_equal_split_data_distribution(samples,data);
                break;
            
            default:
                break;
        }
        ++trailCounter;
    }

    //Mean
    #ifndef DETAILED_TIME
    double searchCycleAvg = 0;
    #else
    double rootPredictCycleAvg = 0;
    double rootCorrectCycleAvg = 0;
    double segmentPredictCycleAvg = 0;
    double segmentCorrectCycleAvg = 0;
    #endif
    double searchRangeAvg = 0;
    double memoryUsageAvg = 0;
    double dataErrorAvg = 0;
    for (int i = 0; i < NUM_TRAILS; ++i)
    {
        #ifndef DETAILED_TIME
        searchCycleAvg += (double)searchCycleTrails[i];
        #else
        rootPredictCycleAvg = (double)rootPredictCycleTrails[i];
        rootCorrectCycleAvg = (double)rootCorrectCycleTrails[i];
        segmentPredictCycleAvg = (double)segmentPredictCycleTrails[i];
        segmentCorrectCycleAvg = (double)segmentCorrectCycleTrails[i];
        #endif
        searchRangeAvg += (double)searchRangeTrails[i];
        memoryUsageAvg += (double)memoryUsageTrails[i];
        dataErrorAvg += (double)dataErrorTrails[i];
    }
    #ifndef DETAILED_TIME
    searchCycleAvg /= (double)NUM_TRAILS;
    #else
    rootPredictCycleAvg /= (double)NUM_TRAILS;
    rootCorrectCycleAvg /= (double)NUM_TRAILS;
    segmentPredictCycleAvg /= (double)NUM_TRAILS;
    segmentCorrectCycleAvg /= (double)NUM_TRAILS;
    #endif
    searchRangeAvg /= (double)NUM_TRAILS;   
    memoryUsageAvg /= (double)NUM_TRAILS;
    dataErrorAvg /= (double)NUM_TRAILS;

    //Variance
    #ifndef DETAILED_TIME
    double searchCycleSumX = 0.0;
    #else
    double rootPredictCycleSumX = 0.0;
    double rootCorrectCycleSumX = 0.0;
    double segmentPredictCycleSumX = 0.0;
    double segmentCorrectCycleSumX = 0.0;
    #endif
    double searchRangeSumX = 0.0;
    double memoryUsageSumX = 0.0;
    double dataErrorSumX = 0.0;
    for (int i = 0; i < NUM_TRAILS; ++i)
    {
        #ifndef DETAILED_TIME
        searchCycleSumX += pow((double)searchCycleTrails[i] - searchCycleAvg,2);
        #else
        rootPredictCycleSumX += pow((double)rootPredictCycleTrails[i] - rootPredictCycleAvg,2);
        rootCorrectCycleSumX += pow((double)rootCorrectCycleTrails[i] - rootCorrectCycleAvg,2);
        segmentPredictCycleSumX += pow((double)segmentPredictCycleTrails[i] - segmentPredictCycleAvg,2);
        segmentCorrectCycleSumX += pow((double)segmentCorrectCycleTrails[i] - segmentCorrectCycleAvg,2);
        #endif
        searchRangeSumX += pow((double)searchRangeTrails[i] - searchRangeAvg,2);
        memoryUsageSumX += pow((double)memoryUsageTrails[i] - memoryUsageAvg,2);
        dataErrorSumX += pow((double)dataErrorTrails[i] - dataErrorAvg,2);
    }
    #ifndef DETAILED_TIME
    double searchCycleStd = sqrt(searchCycleSumX / (double)NUM_TRAILS);
    #else
    double rootPredictCycleStd = sqrt(rootPredictCycleSumX / (double)NUM_TRAILS);
    double rootCorrectCycleStd = sqrt(rootCorrectCycleSumX / (double)NUM_TRAILS);
    double segmentPredictCycleStd = sqrt(segmentPredictCycleSumX / (double)NUM_TRAILS);
    double segmentCorrectCycleStd = sqrt(segmentCorrectCycleSumX / (double)NUM_TRAILS);
    #endif
    double searchRangeStd = sqrt(searchRangeSumX / (double)NUM_TRAILS);
    double memoryUsageStd = sqrt(memoryUsageSumX / (double)NUM_TRAILS);
    double dataErrorStd = sqrt(dataErrorSumX / (double)NUM_TRAILS);

    cout << ";Data=" << FILE_NAME << ";NumData=" << data.size() << ";NumSamples=" << NUM_SAMPLES << ";NumTrails=" << NUM_TRAILS;
    cout << ";NumQueries=" << NUM_QUERIES << ";NumSegment=" << NUM_SEGMENT;
    #ifndef DETAILED_TIME
    cout << ";SearchTimeAvg=" << searchCycleAvg/(double)MACHINE_FREQUENCY << ";SearchTimeStd=" << searchCycleStd/(double)MACHINE_FREQUENCY;
    #else
    cout << ";RootPredictCycleAvg=" << rootPredictCycleAvg/(double)MACHINE_FREQUENCY << ";RootPredictCycleStd=" << rootPredictCycleStd/(double)MACHINE_FREQUENCY;
    cout << ";RootCorrectCycleAvg=" << rootCorrectCycleAvg/(double)MACHINE_FREQUENCY << ";RootCorrectCycleStd=" << rootCorrectCycleStd/(double)MACHINE_FREQUENCY;
    cout << ";SegmentPredictCycleAvg=" << segmentPredictCycleAvg/(double)MACHINE_FREQUENCY << ";SegmentPredictCycleStd=" << segmentPredictCycleStd/(double)MACHINE_FREQUENCY;
    cout << ";SegmentCorrectCycleAvg=" << segmentCorrectCycleAvg/(double)MACHINE_FREQUENCY << ";SegmentCorrectCycleStd=" << segmentCorrectCycleStd/(double)MACHINE_FREQUENCY;
    #endif
    cout << ";SearchRangeAvg=" << searchRangeAvg << ";SearchRangeStd=" << searchRangeStd;
    cout << ";DataErrorAvg=" << dataErrorAvg << ";DataErrorStd=" << dataErrorStd;
    cout << ";MemoryUsageAvg=" << memoryUsageAvg << ";MemoryUsageStd=" << memoryUsageStd;
    cout << ";SearchCount=" << (double)count/NUM_TRAILS;
    cout << ";" << endl;

    return 0;
}

void praser(int argc, char** argv)
{
    ASSERT_MESSAGE(argc == 6, "Usage: ./run_sample <run_flag:int> <root_type:int> <leaf_type:int> <root_search_type:int> <leaf_search_type:int>");

    //Run Flag
    switch (atoi(argv[1]))
    {
        case 0: //RMI
            RUN_FLAG = RunType::RMI;
            cout << "Index=RMI";
            break;
        case 1: //EQUAL_KEY
            RUN_FLAG = RunType::EQUAL_KEY;
            cout << "Index=EqualSplitKey";
            break;
        case 2: //EQUAL_SIZE
            RUN_FLAG = RunType::EQUAL_SIZE;
            cout << "Index=EqualSplitSize";
            break;
        case 3: //EQUAL_SIZE_DISTR
            RUN_FLAG = RunType::EQUAL_SIZE_DISTR;
            cout << "Index=EqualSplitSizeDistribution";
            break;
        default:
            ASSERT_MESSAGE(false, "Invalid Run Type: RMI (0), EQUAL_KEY (1), EQUAL_SIZE (2), or EQUAL_SIZE_DISTR (3)");
            break;
    }
    ASSERT_MESSAGE(RUN_FLAG != RunType::NONE, "Invalid Run Type: NONE");

    //Root Type
    if (RUN_FLAG == RunType::EQUAL_KEY)
    {
        ROOT_TYPE = rmi::ModelType::NONE;
        cout << ";RootType=NONE";
    }
    else
    {
        switch (atoi(argv[2]))
        {
            case 0: //CONSTANT
                ROOT_TYPE = rmi::ModelType::CONSTANT;
                cout << ";RootType=CONSTANT";
                break;
            case 1: //LINEAR_SPLINE
                ROOT_TYPE = rmi::ModelType::LINEAR_SPLINE;
                cout << ";RootType=LINEAR_SPLINE";
                break;
            case 2: //LINEAR_REGRESSION
                ROOT_TYPE = rmi::ModelType::LINEAR_REGRESSION;
                cout << ";RootType=LINEAR_REGRESSION";
                break;
            case 3: //CUBIC_SPLINE
                ROOT_TYPE = rmi::ModelType::CUBIC_SPLINE;
                cout << ";RootType=CUBIC_SPLINE";
                break;
            default:
                ASSERT_MESSAGE(false, "Invalid Root Type: CONSTANT (0), LINEAR_SPLINE (1), LINEAR_REGRESSION (2), or CUBIC_SPLINE (3)");
                break;
        }
        ASSERT_MESSAGE(ROOT_TYPE != rmi::ModelType::NONE, "Invalid Root Type: NONE");
    }

    //Leaf Type
    switch (atoi(argv[3]))
    {
        case 0: //CONSTANT
            LEAF_TYPE = rmi::ModelType::CONSTANT;
            cout << ";LeafType=CONSTANT";
            break;
        case 1: //LINEAR_SPLINE
            LEAF_TYPE = rmi::ModelType::LINEAR_SPLINE;
            cout << ";LeafType=LINEAR_SPLINE";
            break;
        case 2: //LINEAR_REGRESSION
            LEAF_TYPE = rmi::ModelType::LINEAR_REGRESSION;
            cout << ";LeafType=LINEAR_REGRESSION";
            break;
        case 3: //CUBIC_SPLINE
            LEAF_TYPE = rmi::ModelType::CUBIC_SPLINE;
            cout << ";LeafType=CUBIC_SPLINE";
            break;
        default:
            ASSERT_MESSAGE(false, "Invalid Leaf Type: CONSTANT (0), LINEAR_SPLINE (1), LINEAR_REGRESSION (2), or CUBIC_SPLINE (3)");
            break;
    }
    ASSERT_MESSAGE(LEAF_TYPE != rmi::ModelType::NONE, "Invalid Leaf Type: NONE");

    //Root Search Type
    switch (atoi(argv[4]))
    {
        case 0: //BINARY_MAX
            ROOT_SEARCH_TYPE = rmi::SearchType::BINARY_MAX;
            cout << ";SearchType=BINARY_MAX";
            break;
        case 1: //BINARY_MAX_SEG
            ROOT_SEARCH_TYPE = rmi::SearchType::BINARY_MAX_SEG;
            cout << ";SearchType=BINARY_MAX_SEG";
            break;
        case 2: //BINARY_ENTIRE
            ROOT_SEARCH_TYPE = rmi::SearchType::BINARY_ENTIRE;
            cout << ";SearchType=BINARY_ENTIRE";
            break;
        case 3: //EXPONENTIAL
            ROOT_SEARCH_TYPE = rmi::SearchType::EXPONENTIAL;
            cout << ";SearchType=EXPONENTIAL";
            break;
        case 4: //LINEAR
            ROOT_SEARCH_TYPE = rmi::SearchType::LINEAR;
            cout << ";SearchType=LINEAR";
            break;
        default:
            ASSERT_MESSAGE(false, "Invalid Root Search Type: BINARY_MAX (0), BINARY_MAX_SEG (1), BINARY_ENTIRE (2), EXPONENTIAL (3), or LINEAR (4");
            break;
    }
    ASSERT_MESSAGE(ROOT_SEARCH_TYPE != rmi::SearchType::NONE, "Invalid Root Search Type: NONE");

    //Leaf Search Type
    switch (atoi(argv[5]))
    {
        case 0: //BINARY_MAX
            LEAF_SEARCH_TYPE = rmi::SearchType::BINARY_MAX;
            cout << ";SearchType=BINARY_MAX";
            break;
        case 1: //BINARY_MAX_SEG
            LEAF_SEARCH_TYPE = rmi::SearchType::BINARY_MAX_SEG;
            cout << ";SearchType=BINARY_MAX_SEG";
            break;
        case 2: //BINARY_ENTIRE
            LEAF_SEARCH_TYPE = rmi::SearchType::BINARY_ENTIRE;
            cout << ";SearchType=BINARY_ENTIRE";
            break;
        case 3: //EXPONENTIAL
            LEAF_SEARCH_TYPE = rmi::SearchType::EXPONENTIAL;
            cout << ";SearchType=EXPONENTIAL";
            break;
        case 4: //LINEAR
            LEAF_SEARCH_TYPE = rmi::SearchType::LINEAR;
            cout << ";SearchType=LINEAR";
            break;
        default:
            ASSERT_MESSAGE(false, "Invalid Leaf Search Type: BINARY_MAX (0), BINARY_MAX_SEG (1), BINARY_ENTIRE (2), EXPONENTIAL (3), or LINEAR (4");
            break;
    }
    ASSERT_MESSAGE(LEAF_SEARCH_TYPE != rmi::SearchType::NONE, "Invalid Leaf Search Type: NONE");
}

uint64_t run_rmi(vector<uint64_t> & data)
{
    rmi::RMI<uint64_t> rmi(data,ROOT_TYPE,LEAF_TYPE,NUM_SEGMENT,LEAF_SEARCH_TYPE);
        
    //Setup Random Number Generator for Search
    random_device rd;
    mt19937 gen(1); //gen(rd());
    // uniform_int_distribution<uint64_t> distrib(data.front(), data.back());
    uniform_int_distribution<int> distrib(0, data.size()-1);
    
    //Search Time & Search Range
    volatile uint64_t count = 0;
    #ifndef DETAILED_TIME
    vector<uint64_t> searchCycle(NUM_QUERIES);
    #endif
    vector<size_t> searchRange(NUM_QUERIES);
    uint64_t searchCycleMax = 0;
    uint64_t searchRangeMax = 0;
    for (int i = 0; i < NUM_QUERIES; ++i)
    {
        #if NUM_QUERIES  == NUM_DATA
        uint64_t searchKey = data[i];
        #else
        // uint64_t searchKey = distrib(gen);
        uint64_t searchKey = data[distrib(gen)];
        #endif

        #ifndef DETAILED_TIME
        uint64_t tempSearchCycle = 0;
        startTimer(&tempSearchCycle);
        #endif

        bool found = rmi.search(searchKey);

        #ifndef DETAILED_TIME
        stopTimer(&tempSearchCycle);
        searchCycle[i] = tempSearchCycle;
        if (tempSearchCycle > searchCycleMax) {searchCycleMax = tempSearchCycle;}
        #endif
        
        if (found) {++count;}

        searchRange[i] = rmi.search_range(searchKey);
        if (searchRange[i] > searchRangeMax) {searchRangeMax = searchRange[i];}
    }

    //Latency
    #ifndef DETAILED_TIME
    double searchCycleAvg = 0;
    double searchRangeAvg = 0;
    for (int i = 0; i < NUM_QUERIES; ++i)
    {
        searchCycleAvg += (double)searchCycle[i];
        searchRangeAvg += (double)searchRange[i];
    }
    searchCycleTrails[trailCounter] = searchCycleAvg / (double)NUM_QUERIES;
    searchRangeTrails[trailCounter] = searchRangeAvg / (double)NUM_QUERIES;   
    #else
    rootPredictCycleTrails[trailCounter] = (double)rootPredictCycle/(double)NUM_QUERIES;
    rootCorrectCycleTrails[trailCounter] = (double)rootCorrectCycle/(double)NUM_QUERIES;
    segmentPredictCycleTrails[trailCounter] = (double)segmentPredictCycle/(double)NUM_QUERIES;
    segmentCorrectCycleTrails[trailCounter] = (double)segmentCorrectCycle/(double)NUM_QUERIES;
    double searchRangeAvg = 0;
    for (int i = 0; i < NUM_QUERIES; ++i)
    {
        searchRangeAvg += (double)searchRange[i];
    }
    searchRangeTrails[trailCounter] = searchRangeAvg / (double)NUM_QUERIES;
    rootPredictCycle = 0;
    rootCorrectCycle = 0;
    segmentPredictCycle = 0;
    segmentCorrectCycle = 0;
    #endif

    //Memory (bytes)
    memoryUsageTrails[trailCounter] = (double)rmi.size_of();

    //Data Error
    vector<int> dataError = rmi.get_data_error();
    double dataErrorAvg = 0;
    for (auto & error : dataError)
    {
        dataErrorAvg += (double)error;
    }
    dataErrorTrails[trailCounter] = dataErrorAvg / (double)dataError.size();

    return count;
}


uint64_t run_equal_split_key(vector<uint64_t> & data)
{
    rmi::EqualSplitKey<uint64_t> equalSplit(data,LEAF_TYPE,NUM_SEGMENT,LEAF_SEARCH_TYPE);
    
    //Setup Random Number Generator for Search
    random_device rd;
    mt19937 gen(1); //gen(rd());
    // uniform_int_distribution<uint64_t> distrib(data.front(), data.back());
    uniform_int_distribution<int> distrib(0, data.size()-1);
    
    //Search Time & Search Range
    volatile uint64_t count = 0;
    #ifndef DETAILED_TIME
    vector<uint64_t> searchCycle(NUM_QUERIES);
    #endif
    vector<size_t> searchRange(NUM_QUERIES);
    uint64_t searchCycleMax = 0;
    uint64_t searchRangeMax = 0;
    for (int i = 0; i < NUM_QUERIES; ++i)
    {
        #if NUM_QUERIES  == NUM_DATA
        uint64_t searchKey = data[i];
        #else
        // uint64_t searchKey = distrib(gen);
        uint64_t searchKey = data[distrib(gen)];
        #endif

        #ifndef DETAILED_TIME
        uint64_t tempSearchCycle = 0;
        startTimer(&tempSearchCycle);
        #endif

        bool found = equalSplit.search(searchKey);

        #ifndef DETAILED_TIME
        stopTimer(&tempSearchCycle);
        searchCycle[i] = tempSearchCycle;
        if (tempSearchCycle > searchCycleMax) {searchCycleMax = tempSearchCycle;}
        #endif

        if (found) {++count;}

        searchRange[i] = equalSplit.search_range(searchKey);
        if (searchRange[i] > searchRangeMax) {searchRangeMax = searchRange[i];}
    }
    //Latency
    #ifndef DETAILED_TIME
    double searchCycleAvg = 0;
    double searchRangeAvg = 0;
    for (int i = 0; i < NUM_QUERIES; ++i)
    {
        searchCycleAvg += (double)searchCycle[i];
        searchRangeAvg += (double)searchRange[i];
    }
    searchCycleTrails[trailCounter] = searchCycleAvg / (double)NUM_QUERIES;
    searchRangeTrails[trailCounter] = searchRangeAvg / (double)NUM_QUERIES;   
    #else
    rootPredictCycleTrails[trailCounter] = (double)rootPredictCycle/(double)NUM_QUERIES;
    rootCorrectCycleTrails[trailCounter] = (double)rootCorrectCycle/(double)NUM_QUERIES;
    segmentPredictCycleTrails[trailCounter] = (double)segmentPredictCycle/(double)NUM_QUERIES;
    segmentCorrectCycleTrails[trailCounter] = (double)segmentCorrectCycle/(double)NUM_QUERIES;
    double searchRangeAvg = 0;
    for (int i = 0; i < NUM_QUERIES; ++i)
    {
        searchRangeAvg += (double)searchRange[i];
    }
    searchRangeTrails[trailCounter] = searchRangeAvg / (double)NUM_QUERIES;
    rootPredictCycle = 0;
    rootCorrectCycle = 0;
    segmentPredictCycle = 0;
    segmentCorrectCycle = 0;
    #endif

    //Memory (bytes)
    memoryUsageTrails[trailCounter] = (double)equalSplit.size_of();

    //Data Error
    vector<int> dataError = equalSplit.get_data_error();
    double dataErrorAvg = 0;
    for (auto & error : dataError)
    {
        dataErrorAvg += (double)error;
    }
    dataErrorTrails[trailCounter] = dataErrorAvg / (double)dataError.size();
    
    return count;
}


uint64_t run_equal_split_data(vector<uint64_t> & data)
{
    rmi::EqualSplit<uint64_t> rmiEqualSplit(data,ROOT_TYPE,LEAF_TYPE,NUM_SEGMENT,
                                            ROOT_SEARCH_TYPE, LEAF_SEARCH_TYPE);

    //Setup Random Number Generator for Search
    random_device rd;
    mt19937 gen(1); //gen(rd());
    // uniform_int_distribution<uint64_t> distrib(data.front(), data.back());
    uniform_int_distribution<int> distrib(0, data.size()-1);
    
    //Search Time & Search Range
    volatile uint64_t count = 0;
    #ifndef DETAILED_TIME
    vector<uint64_t> searchCycle(NUM_QUERIES);
    #endif
    vector<size_t> searchRange(NUM_QUERIES);
    uint64_t searchCycleMax = 0;
    uint64_t searchRangeMax = 0;
    for (int i = 0; i < NUM_QUERIES; ++i)
    {
        #if NUM_QUERIES  == NUM_DATA
        uint64_t searchKey = data[i];
        #else
        // uint64_t searchKey = distrib(gen);
        uint64_t searchKey = data[distrib(gen)];
        #endif

        #ifndef DETAILED_TIME
        uint64_t tempSearchCycle = 0;
        startTimer(&tempSearchCycle);
        #endif

        bool found = rmiEqualSplit.search(searchKey);

        #ifndef DETAILED_TIME
        stopTimer(&tempSearchCycle);
        searchCycle[i] = tempSearchCycle;
        if (tempSearchCycle > searchCycleMax) {searchCycleMax = tempSearchCycle;}
        #endif
        
        if (found) {++count;}

        searchRange[i] = rmiEqualSplit.search_range(searchKey);
        if (searchRange[i] > searchRangeMax) {searchRangeMax = searchRange[i];}
    }
    //Latency
    #ifndef DETAILED_TIME
    double searchCycleAvg = 0;
    double searchRangeAvg = 0;
    for (int i = 0; i < NUM_QUERIES; ++i)
    {
        searchCycleAvg += (double)searchCycle[i];
        searchRangeAvg += (double)searchRange[i];
    }
    searchCycleTrails[trailCounter] = searchCycleAvg / (double)NUM_QUERIES;
    searchRangeTrails[trailCounter] = searchRangeAvg / (double)NUM_QUERIES;   
    #else
    rootPredictCycleTrails[trailCounter] = (double)rootPredictCycle/(double)NUM_QUERIES;
    rootCorrectCycleTrails[trailCounter] = (double)rootCorrectCycle/(double)NUM_QUERIES;
    segmentPredictCycleTrails[trailCounter] = (double)segmentPredictCycle/(double)NUM_QUERIES;
    segmentCorrectCycleTrails[trailCounter] = (double)segmentCorrectCycle/(double)NUM_QUERIES;
    double searchRangeAvg = 0;
    for (int i = 0; i < NUM_QUERIES; ++i)
    {
        searchRangeAvg += (double)searchRange[i];
    }
    searchRangeTrails[trailCounter] = searchRangeAvg / (double)NUM_QUERIES;
    rootPredictCycle = 0;
    rootCorrectCycle = 0;
    segmentPredictCycle = 0;
    segmentCorrectCycle = 0;
    #endif

    //Memory (bytes)
    memoryUsageTrails[trailCounter] = (double)rmiEqualSplit.size_of();

    //Data Error
    vector<int> dataError = rmiEqualSplit.get_data_error();
    double dataErrorAvg = 0;
    for (auto & error : dataError)
    {
        dataErrorAvg += (double)error;
    }
    dataErrorTrails[trailCounter] = dataErrorAvg / (double)dataError.size();
    
    return count;
}

uint64_t run_equal_split_data_distribution(vector<uint64_t> & data, vector<uint64_t> & distribution)
{
    rmi::EqualSplit<uint64_t> rmiEqualSplit(data,distribution,ROOT_TYPE,LEAF_TYPE,NUM_SEGMENT,
                                            ROOT_SEARCH_TYPE, LEAF_SEARCH_TYPE);

    //Setup Random Number Generator for Search
    random_device rd;
    mt19937 gen(1); //gen(rd());
    // uniform_int_distribution<uint64_t> distrib(data.front(), data.back());
    uniform_int_distribution<int> distrib(0, data.size()-1);
    
    //Search Time & Search Range
    volatile uint64_t count = 0;
    #ifndef DETAILED_TIME
    vector<uint64_t> searchCycle(NUM_QUERIES);
    #endif
    vector<size_t> searchRange(NUM_QUERIES);
    uint64_t searchCycleMax = 0;
    uint64_t searchRangeMax = 0;
    for (int i = 0; i < NUM_QUERIES; ++i)
    {
        #if NUM_QUERIES  == NUM_DATA
        uint64_t searchKey = data[i];
        #else
        // uint64_t searchKey = distrib(gen);
        uint64_t searchKey = data[distrib(gen)];
        #endif

        #ifndef DETAILED_TIME
        uint64_t tempSearchCycle = 0;
        startTimer(&tempSearchCycle);
        #endif

        bool found = rmiEqualSplit.search(searchKey);

        #ifndef DETAILED_TIME
        stopTimer(&tempSearchCycle);
        searchCycle[i] = tempSearchCycle;
        if (tempSearchCycle > searchCycleMax) {searchCycleMax = tempSearchCycle;}
        #endif
        
        if (found) {++count;}

        searchRange[i] = rmiEqualSplit.search_range(searchKey);
        if (searchRange[i] > searchRangeMax) {searchRangeMax = searchRange[i];}
    }
    //Latency
    #ifndef DETAILED_TIME
    double searchCycleAvg = 0;
    double searchRangeAvg = 0;
    for (int i = 0; i < NUM_QUERIES; ++i)
    {
        searchCycleAvg += (double)searchCycle[i];
        searchRangeAvg += (double)searchRange[i];
    }
    searchCycleTrails[trailCounter] = searchCycleAvg / (double)NUM_QUERIES;
    searchRangeTrails[trailCounter] = searchRangeAvg / (double)NUM_QUERIES;   
    #else
    rootPredictCycleTrails[trailCounter] = (double)rootPredictCycle/(double)NUM_QUERIES;
    rootCorrectCycleTrails[trailCounter] = (double)rootCorrectCycle/(double)NUM_QUERIES;
    segmentPredictCycleTrails[trailCounter] = (double)segmentPredictCycle/(double)NUM_QUERIES;
    segmentCorrectCycleTrails[trailCounter] = (double)segmentCorrectCycle/(double)NUM_QUERIES;
    double searchRangeAvg = 0;
    for (int i = 0; i < NUM_QUERIES; ++i)
    {
        searchRangeAvg += (double)searchRange[i];
    }
    searchRangeTrails[trailCounter] = searchRangeAvg / (double)NUM_QUERIES;
    rootPredictCycle = 0;
    rootCorrectCycle = 0;
    segmentPredictCycle = 0;
    segmentCorrectCycle = 0;
    #endif

    //Memory (bytes)
    memoryUsageTrails[trailCounter] = (double)rmiEqualSplit.size_of();

    //Data Error
    vector<int> dataError = rmiEqualSplit.get_data_error();
    double dataErrorAvg = 0;
    for (auto & error : dataError)
    {
        dataErrorAvg += (double)error;
    }
    dataErrorTrails[trailCounter] = dataErrorAvg / (double)dataError.size();
    
    return count;
}