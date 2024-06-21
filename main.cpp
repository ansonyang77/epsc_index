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

#include "src/rmi.hpp"
#include "src/rmi_equal_split_key.hpp"
#include "src/rmi_equal_split_size.hpp"

#include "utils/load.hpp"
#include "utils/rdtsc.h"

using namespace std;

//main.cpp test configuration
#define NUM_TEST_DATA 1000
#define NUM_TEST_SEGMENTS 100

//Index Configuration (Confirguration details in src/helper_functions/helper_rmi.hpp)
rmi::ModelType ROOT_TYPE = rmi::ModelType::LINEAR_REGRESSION;
rmi::ModelType LEAF_TYPE = rmi::ModelType::LINEAR_REGRESSION;
rmi::SearchType ROOT_SEARCH_TYPE = rmi::SearchType::BINARY_MAX_SEG;
rmi::SearchType LEAF_SEARCH_TYPE = rmi::SearchType::BINARY_MAX_SEG;

//Function Headers
bool run_rmi(vector<uint64_t> & data, uint64_t searchKey);
bool run_espc_key(vector<uint64_t> & data, uint64_t searchKey);
bool run_espc_data(vector<uint64_t> & data, uint64_t searchKey);

int main()
{
    ASSERT_MESSAGE (NUM_TEST_DATA <= NUM_DATA, "NUM_DATA must be larger than NUM_TEST_DATA");
    ASSERT_MESSAGE (NUM_TEST_SEGMENTS <= NUM_SEGMENT, "NUM_TEST_SEGMENTS must be larger than NUM_SEGMENT");
    
    //load data
    vector<uint64_t> data = load_data(NUM_TEST_DATA);
    sort(data.begin(), data.end());
	data.erase( unique( data.begin(), data.end() ), data.end());

    //Generate Random Query
    random_device rd;
    mt19937 gen(1);
    uniform_int_distribution<int> distrib(0, data.size()-1);
    uint64_t searchKey = data[distrib(gen)];

    //Run Indexes
    run_rmi(data,searchKey);

    run_espc_key(data,searchKey);

    run_espc_data(data,searchKey);

    return 0;
}

bool run_rmi(vector<uint64_t> & data, uint64_t searchKey)
{
    //Setup
    rmi::RMI<uint64_t> rmi(data,ROOT_TYPE,LEAF_TYPE,NUM_TEST_SEGMENTS,LEAF_SEARCH_TYPE);
    
    //Lookup
    bool found = rmi.search(searchKey);

    //Return Search Range for key
    size_t searchRange = rmi.search_range(searchKey);

    //Return index size
    double memoryUsage = (double)rmi.size_of();

    //Return Average Data Error
    vector<int> dataError = rmi.get_data_error();
    double dataErrorAvg = 0;
    for (auto & error : dataError)
    {
        dataErrorAvg += (double)error;
    }
    dataErrorAvg /= (double)dataError.size();

    cout << "RMI searching " << searchKey << endl;
    cout << "key found: " << ((found)? "true":"false") << endl;
    cout << "key search range: " << searchRange << endl;
    cout << "index memory usage: " << memoryUsage << endl;
    cout << "index avg data error: " << dataErrorAvg << endl;
    cout << endl;

    return found;
}

bool run_espc_key(vector<uint64_t> & data, uint64_t searchKey)
{
    //Setup
    rmi::EqualSplitKey<uint64_t> equalSplit(data,LEAF_TYPE,NUM_TEST_SEGMENTS,LEAF_SEARCH_TYPE);
    
    //Lookup
    bool found = equalSplit.search(searchKey);

    //Return Search Range for key
    size_t searchRange = equalSplit.search_range(searchKey);

    //Return index size
    double memoryUsage = (double)equalSplit.size_of();

    //Return Average Data Error
    vector<int> dataError = equalSplit.get_data_error();
    double dataErrorAvg = 0;
    for (auto & error : dataError)
    {
        dataErrorAvg += (double)error;
    }
    dataErrorAvg /= (double)dataError.size();

    cout << "ESPC (keys) searching " << searchKey << endl;
    cout << "key found: " << ((found)? "true":"false") << endl;
    cout << "key search range: " << searchRange << endl;
    cout << "index memory usage: " << memoryUsage << endl;
    cout << "index avg data error: " << dataErrorAvg << endl;
    cout << endl;

    return found;
}

bool run_espc_data(vector<uint64_t> & data, uint64_t searchKey)
{
    //Setup
    rmi::EqualSplit<uint64_t> rmiEqualSplit(data,ROOT_TYPE,LEAF_TYPE,NUM_TEST_SEGMENTS,
                                            ROOT_SEARCH_TYPE, LEAF_SEARCH_TYPE);

    //Lookup
    bool found = rmiEqualSplit.search(searchKey);

    //Return Search Range for key
    size_t searchRange = rmiEqualSplit.search_range(searchKey);

    //Return index size
    double memoryUsage = (double)rmiEqualSplit.size_of();

    //Return Average Data Error
    vector<int> dataError = rmiEqualSplit.get_data_error();
    double dataErrorAvg = 0;
    for (auto & error : dataError)
    {
        dataErrorAvg += (double)error;
    }
    dataErrorAvg /= (double)dataError.size();

    cout << "ESPC (data) searching " << searchKey << endl;
    cout << "key found: " << ((found)? "true":"false") << endl;
    cout << "key search range: " << searchRange << endl;
    cout << "index memory usage: " << memoryUsage << endl;
    cout << "index avg data error: " << dataErrorAvg << endl;
    cout << endl;

    return found;
}