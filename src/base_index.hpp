#ifndef __BASE_INDEX_HPP__
#define __BASE_INDEX_HPP__

#pragma once
#include <iostream>
#include <vector>
#include <math.h>
#include <algorithm>

#include "../parameters.hpp"
#include "../utils/debug.hpp"


template<class Type_Key>
class LearnedIndex
{
protected:
    int m_noSegment;
    std::vector<Type_Key> m_data;
    std::vector<std::pair<int,int>> m_segmentIndexes;

public:
    LearnedIndex():m_noSegment(-1){};
    LearnedIndex(std::vector<Type_Key> & data, int noSegment);

    //child class must implement segmentation method.
    virtual void segment_data() = 0;
    virtual bool search(Type_Key key) = 0;
    void clear_segmentation();

    void display_data();
    void display_segment();

    void print_data();
    void print_segment();
    void print_segment_first_last(bool printIndex);
    void print_segment_predict_first_last(bool printIndex);
    
    void set_noSegment(int noSegment);
    void set_data(std::vector<Type_Key> & newData);

    int get_noSegment();
    void print_segment_info();

    uint64_t size_of();
};


template<class Type_Key>
LearnedIndex<Type_Key>::
LearnedIndex(std::vector<Type_Key> & data, int noSegment)
{
    LOG_DEBUG("%s", "Start");
    this->m_data = data;
    m_data.shrink_to_fit();
    this->m_noSegment = noSegment;
    LOG_DEBUG("%s", "End");
}

template<class Type_Key>
inline void LearnedIndex<Type_Key>::clear_segmentation()
{
    LOG_DEBUG("%s", "Start");
    m_segmentIndexes = std::vector<std::pair<int,int>>();
    LOG_DEBUG("%s", "End");
}

template<class Type_Key>
void LearnedIndex<Type_Key>::display_data()
{
    LOG_DEBUG("%s", "Start");       
    for(auto & it : this->m_data)
    {
        std::cout << it << " ";
    }
    std::cout << std::endl;
    LOG_DEBUG("%s", "End");
}

template<class Type_Key>
void LearnedIndex<Type_Key>::display_segment()
{
    LOG_DEBUG("%s", "Start");       
    for (auto & it: this->m_segmentIndexes)
    {
        std::cout << "[";
        for (int i = it.first; i != it.second+1; ++i)
        {
            std::cout << this->m_data[i] << ",";
        }
        std::cout << "]" << std::endl;
    }
    LOG_DEBUG("%s", "End");
}

template<class Type_Key>
void LearnedIndex<Type_Key>::print_data()
{       
    std::cout << this->m_data.front;
    for(auto it = this->m_data.begin()+1; 
        it != this->m_data.end(); ++it)
    {
        std::cout << "," << *it;
    }
}

template<class Type_Key>
void LearnedIndex<Type_Key>::print_segment()
{      
    for (auto & it: this->m_segmentIndexes)
    {
        std::cout << "[" << this->m_data[it.first];
        for (int i = it.first+1; i <= it.second; ++i)
        {
            std::cout << "," << this->m_data[i];
        }
        std::cout << "],";
    }
}

template<class Type_Key>
void LearnedIndex<Type_Key>::print_segment_first_last(bool printIndex)
{      
    for (auto & it: this->m_segmentIndexes)
    {
        if (printIndex)
        {
           std::cout << "[" << it.first << "," << it.second << "],"; 
        }
        else
        {
            std::cout << "[" << this->m_data[it.first] << "," 
                    << this->m_data[it.second] << "],";
        }
    }
}

template<class Type_Key>
void LearnedIndex<Type_Key>::print_segment_predict_first_last(bool printIndex)
//The default method is same as the print_segment_first_last
{      
    if (printIndex)
    {
        std::cout << "[" << this->m_segmentIndexes[0].first 
        << "," << this->m_segmentIndexes[0].second << "]"; 
    }
    else
    {
        std::cout << "[" << this->m_data[this->m_segmentIndexes[0].first] << "," 
                << this->m_data[this->m_segmentIndexes[0].second] << "]";
    }
    
    for (int i = 1; i < this->m_noSegment; ++i)
    {
        if (printIndex)
        {
           std::cout << ",[" << this->m_segmentIndexes[i].first 
           << "," << this->m_segmentIndexes[i].second << "]"; 
        }
        else
        {
            std::cout << ",[" << this->m_data[this->m_segmentIndexes[i].first] << "," 
                    << this->m_data[this->m_segmentIndexes[i].second] << "]";
        }
    }
}

template<class Type_Key>
inline void LearnedIndex<Type_Key>::set_noSegment(int noSegment)
{
    this->m_noSegment = noSegment;
}

template<class Type_Key>
inline void LearnedIndex<Type_Key>::set_data(std::vector<Type_Key> & newData)
{
    this->m_data = newData;
}

template<class Type_Key>
inline int LearnedIndex<Type_Key>::get_noSegment()
{
    return m_noSegment;
}

template<class Type_Key>
void LearnedIndex<Type_Key>::print_segment_info()
{
    if (this->m_segmentIndexes.size())
    {
        int avgSize = 0;
        int maxSize = 0;
        std::vector<int> segSize;
        segSize.reserve(this->m_segmentIndexes.size());
        for (auto & it: this->m_segmentIndexes)
        {
            int segmentSize = it.second - it.first + 1;
            segSize.push_back(segmentSize);
            avgSize += segmentSize;

            if (segmentSize > maxSize)
            {
                maxSize = segmentSize;
            }
        }

        std::sort(segSize.begin(),segSize.end());

        std::cout << "Data=" << FILE_NAME;
        std::cout << ";noSegment=" << segSize.size();
        std::cout << ";maxSize=" << maxSize << ";avgSize=" << avgSize/segSize.size();
        std::cout << ";25thPer=" << segSize[floor(segSize.size()*0.25)];
        std::cout << ";75thPer=" << segSize[floor(segSize.size()*0.75)];
        std::cout << std::endl;
    }
    else
    {
        std::cout << "Data=" << FILE_NAME;
        std::cout << ";noSegment=" << 0;
        std::cout << ";maxSize=" << 0 << ";avgSize=" << 0;
        std::cout << ";25thPer=" << 0;
        std::cout << ";75thPer=" << 0;
        std::cout << std::endl;
    }
}

template<class Type_Key>
uint64_t LearnedIndex<Type_Key>::size_of() //In terms of bytes
{
    return sizeof(int) + sizeof(std::vector<Type_Key>) + sizeof(std::vector<std::pair<int,int>>)
            + sizeof(Type_Key)*m_data.size() + sizeof(std::pair<int,int>)* m_segmentIndexes.size();
}
#endif