#ifndef __LOAD_HELPER_HPP__
#define __LOAD_HELPER_HPP__

#pragma once
#include <string>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <random>
#include <algorithm>

#include "debug.hpp"
#include "../parameters.hpp"

using namespace std;

template<typename K>
bool load_binary(string filename, vector<K> &v)
{
    ifstream ifs(filename, ios::in | ios::binary);
    assert(ifs);

    K size;
    ifs.read(reinterpret_cast<char*>(&size), sizeof(K));
    v.resize(size);
    ifs.read(reinterpret_cast<char*>(v.data()), size * sizeof(K));
    ifs.close();

    return ifs.good();
}


/*
uint64_t data loaders
*/

void load_data(vector<uint64_t> & data)
{
    string input_file = DATA_DIR FILE_NAME;
    load_binary(input_file, data);
    data.shrink_to_fit();

    sort(data.begin(), data.end());
	data.erase( unique( data.begin(), data.end() ), data.end() );
}

vector<uint64_t> load_data(int num_data)
{
    vector<uint64_t> temp_data;
    string input_file = DATA_DIR FILE_NAME;
    load_binary(input_file, temp_data);
    
    return vector<uint64_t>(temp_data.begin(), temp_data.begin() + num_data);
}

vector<uint64_t> load_data_sample(int no_samples, uint64_t seed = 0)
{
    vector<uint64_t> temp_data;
    string input_file = DATA_DIR FILE_NAME;
    load_binary(input_file, temp_data);

    sort(temp_data.begin(), temp_data.end());
	temp_data.erase( unique( temp_data.begin(), temp_data.end() ), temp_data.end() );

    random_device rd;
    if (seed == 0) {seed = rd();}
    mt19937 gen(seed);
    uniform_int_distribution<int> distrib(0, temp_data.size()-1);

    vector<uint64_t> data(no_samples);
    for (int i = 0; i < no_samples; ++i)
    {
        data[i] = temp_data[distrib(gen)];
    }
    return data;
}

void sample_from_data(int no_samples, vector<uint64_t> & data, vector<uint64_t> & output, uint64_t seed = 0)
{
    ASSERT_MESSAGE(output.size() == no_samples, "OUTPUT vector size must be same as number of samples");

    random_device rd;
    if (seed == 0) {seed = rd();}
    mt19937 gen(seed);
    uniform_int_distribution<int> distrib(0, data.size()-1);

    for (int i = 0; i < no_samples; ++i)
    {
        output[i] = data[distrib(gen)];
    }
}

void sample_from_data_without_replacement(int no_samples, vector<uint64_t> data, vector<uint64_t> & output, uint64_t seed = 0)
{
    ASSERT_MESSAGE(output.size() == no_samples, "OUTPUT vector size must be same as number of samples");

    random_device rd;
    if (seed == 0) {seed = rd();}
    mt19937 gen(seed);

    shuffle(data.begin(),data.end(),gen);
    for (int i = 0; i < no_samples; ++i)
    {
        output[i] = data[i];
    }
}

#endif