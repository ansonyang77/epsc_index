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

#include "../src/rmi_equal_split_key.hpp"

#include "../utils/load.hpp"
#include "../utils/rdtsc.h"

using namespace std;

//Equal Split Configuration (Confirguration in src/helper_functions/helper_rmi.hpp)
rmi::ModelType EQUAL_SPLIT_LEAF_TYPE = rmi::ModelType::NONE;
rmi::SearchType EQUAL_SPLIT_SEARCH_TYPE = rmi::SearchType::NONE;

//Function Headers
void praser(int argc, char** argv);
void run_equal_split(vector<uint64_t> & data, rmi::EqualSplitKey<uint64_t> & equalSplit);

int main(int argc, char** argv)
{
    praser(argc, argv);

    vector<uint64_t> data = load_data(NUM_DATA);
    sort(data.begin(), data.end());
	data.erase( unique( data.begin(), data.end() ), data.end() );
    
    rmi::EqualSplitKey<uint64_t> equalSplit(data,EQUAL_SPLIT_LEAF_TYPE,NUM_SEGMENT,EQUAL_SPLIT_SEARCH_TYPE);
    run_equal_split(data, equalSplit);

    return 0;
}

void praser(int argc, char** argv)
{
    ASSERT_MESSAGE(argc == 3, "Usage: ./run_equal_split <leaf_type:int> <search_type:int>");

    cout << "Index=EqualSplitKey";
    cout << ";RootType=NONE";

    //Leaf Type
    switch (atoi(argv[1]))
    {
        case 0: //CONSTANT
            EQUAL_SPLIT_LEAF_TYPE = rmi::ModelType::CONSTANT;
            cout << ";LeafType=CONSTANT";
            break;
        case 1: //LINEAR_SPLINE
            EQUAL_SPLIT_LEAF_TYPE = rmi::ModelType::LINEAR_SPLINE;
            cout << ";LeafType=LINEAR_SPLINE";
            break;
        case 2: //LINEAR_REGRESSION
            EQUAL_SPLIT_LEAF_TYPE = rmi::ModelType::LINEAR_REGRESSION;
            cout << ";LeafType=LINEAR_REGRESSION";
            break;
        case 3: //CUBIC_SPLINE
            EQUAL_SPLIT_LEAF_TYPE = rmi::ModelType::CUBIC_SPLINE;
            cout << ";LeafType=CUBIC_SPLINE";
            break;
        default:
            ASSERT_MESSAGE(false, "Invalid EqualSplitKey Leaf Type: CONSTANT (0), LINEAR_SPLINE (1), LINEAR_REGRESSION (2), or CUBIC_SPLINE (3)");
            break;
    }
    ASSERT_MESSAGE(EQUAL_SPLIT_LEAF_TYPE != rmi::ModelType::NONE, "Invalid EqualSplitKey Leaf Type: NONE");

    //Search Type
    switch (atoi(argv[2]))
    {
        case 0: //BINARY_MAX
            EQUAL_SPLIT_SEARCH_TYPE = rmi::SearchType::BINARY_MAX;
            cout << ";SearchType=BINARY_MAX";
            break;
        case 1: //BINARY_MAX_SEG
            EQUAL_SPLIT_SEARCH_TYPE = rmi::SearchType::BINARY_MAX_SEG;
            cout << ";SearchType=BINARY_MAX_SEG";
            break;
        case 2: //BINARY_ENTIRE
            EQUAL_SPLIT_SEARCH_TYPE = rmi::SearchType::BINARY_ENTIRE;
            cout << ";SearchType=BINARY_ENTIRE";
            break;
        case 3: //EXPONENTIAL
            EQUAL_SPLIT_SEARCH_TYPE = rmi::SearchType::EXPONENTIAL;
            cout << ";SearchType=EXPONENTIAL";
            break;
        case 4: //LINEAR
            EQUAL_SPLIT_SEARCH_TYPE = rmi::SearchType::LINEAR;
            cout << ";SearchType=LINEAR";
            break;
        default:
            ASSERT_MESSAGE(false, "Invalid EqualSplitKey Search Type: BINARY_MAX (0), BINARY_MAX_SEG (1), BINARY_ENTIRE (2), EXPONENTIAL (3), or LINEAR (4");
            break;
    }
    ASSERT_MESSAGE(EQUAL_SPLIT_SEARCH_TYPE != rmi::SearchType::NONE, "Invalid EqualSplitKey Search Type: NONE");
}

void run_equal_split(vector<uint64_t> & data, rmi::EqualSplitKey<uint64_t> & equalSplit)
{
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
    #ifndef DETAILED_TIME
    double searchCycleAvg = 0;
    double searchRangeAvg = 0;
    for (int i = 0; i < NUM_QUERIES; ++i)
    {
        searchCycleAvg += (double)searchCycle[i];
        searchRangeAvg += (double)searchRange[i];
    }
    searchCycleAvg /= (double)NUM_QUERIES;
    searchRangeAvg /= (double)NUM_QUERIES;   
    double searchCycleSumX = 0.0;
    double searchRangeSumX = 0.0;
    for (int i = 0; i < NUM_QUERIES; ++i)
    {
        searchCycleSumX += pow((double)searchCycle[i] - searchCycleAvg,2);
        searchRangeSumX += pow((double)searchRange[i] - searchRangeAvg,2);
    }
    double searchCycleStd = sqrt(searchCycleSumX / (double)NUM_QUERIES);
    double searchRangeStd = sqrt(searchRangeSumX / (double)NUM_QUERIES);
    searchCycle.clear();
    searchRange.clear();
    #else
    double rootPredictCycleAvg = (double)rootPredictCycle/(double)NUM_QUERIES;
    double rootCorrectCycleAvg = (double)rootCorrectCycle/(double)NUM_QUERIES;
    double segmentPredictCycleAvg = (double)segmentPredictCycle/(double)NUM_QUERIES;;
    double segmentCorrectCycleAvg = (double)segmentCorrectCycle/(double)NUM_QUERIES;;
    double searchRangeAvg = 0;
    for (int i = 0; i < NUM_QUERIES; ++i)
    {
        searchRangeAvg += (double)searchRange[i];
    }
    searchRangeAvg /= (double)NUM_QUERIES;   
    double searchRangeSumX = 0.0;
    for (int i = 0; i < NUM_QUERIES; ++i)
    {
        searchRangeSumX += pow((double)searchRange[i] - searchRangeAvg,2);
    }
    double searchRangeStd = sqrt(searchRangeSumX / (double)NUM_QUERIES);
    #endif

    //Memory (bytes)
    uint64_t memoryUsage = equalSplit.size_of();

    //Data Error
    vector<int> dataError = equalSplit.get_data_error();
    double dataErrorAvg = 0;
    for (auto & error : dataError)
    {
        dataErrorAvg += (double)error;
    }
    dataErrorAvg /= (double)dataError.size();
    double dataErrorMax = 0;
    double sumX = 0.0;
    for (auto & error : dataError)
    {
        sumX += pow((double)error - dataErrorAvg,2);
        if (error > dataErrorMax) {dataErrorMax = error;}
    }
    double dataErrorStd = sqrt(sumX / (double)dataError.size());
    dataError.clear();

    //Print Results
    #ifndef DETAILED_TIME
    cout << ";Data=" << FILE_NAME << ";NumData=" << data.size() << ";NumQueries=" << NUM_QUERIES << ";NumSegment=" << NUM_SEGMENT;
    cout << ";SearchTimeAvg=" << searchCycleAvg/(double)MACHINE_FREQUENCY << ";SearchTimeStd=" << searchCycleStd/(double)MACHINE_FREQUENCY;
    cout << ";SearchTimeMax=" << searchCycleMax/(double)MACHINE_FREQUENCY << ";SearchCount=" << count;
    cout << ";MemoryUsage=" << memoryUsage;
    cout << ";SearchRangeAvg=" << searchRangeAvg << ";SearchRangeStd=" << searchRangeStd << ";SearchRangeMax=" << searchRangeMax; 
    cout << ";DataErrorAvg=" << dataErrorAvg << ";DataErrorStd=" << dataErrorStd << ";DataErrorMax=" << dataErrorMax;
    cout << ";" << endl;
    #else
    cout << ";Data=" << FILE_NAME << ";NumData=" << data.size() << ";NumQueries=" << NUM_QUERIES << ";NumSegment=" << NUM_SEGMENT;
    cout << ";RootPredictTimeAvg=" << rootPredictCycleAvg/(double)MACHINE_FREQUENCY;
    cout << ";RootCorrectTimeAvg=" << rootCorrectCycleAvg/(double)MACHINE_FREQUENCY;
    cout << ";SegmentPredictTimeAvg=" << segmentPredictCycleAvg/(double)MACHINE_FREQUENCY;
    cout << ";SegmentCorrectTimeMax=" << segmentCorrectCycle/(double)MACHINE_FREQUENCY << ";SearchCount=" << count;
    cout << ";MemoryUsage=" << memoryUsage;
    cout << ";SearchRangeAvg=" << searchRangeAvg << ";SearchRangeStd=" << searchRangeStd << ";SearchRangeMax=" << searchRangeMax; 
    cout << ";DataErrorAvg=" << dataErrorAvg << ";DataErrorStd=" << dataErrorStd << ";DataErrorMax=" << dataErrorMax;
    cout << ";" << endl;
    #endif
}