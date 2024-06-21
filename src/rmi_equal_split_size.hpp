#ifndef __LOAD_BALANCE_RMI_HPP__
#define __LOAD_BALANCE_RMI_HPP__

#pragma once
#include <iterator>
#include "base_index.hpp"
#include "helper_functions/helper_rmi.hpp"

using namespace std;

namespace rmi{

/*
Equal Split in size, the number of keys in each segment is the same.
*/

template<class Type_Key>
class EqualSplit: public LearnedIndex<Type_Key>
{
private:
    //Variables
    rmi::ModelType m_rootModel;
    rmi::ModelType m_leafModel;
    rmi::SearchType m_rootSearchMethod;
    rmi::SearchType m_leafSearchMethod;

    int* leafMaxError;
    modelRMI<Type_Key>* rootLevel;
    modelRMI<Type_Key>** leafLevel;

    int rootMaxError;
    Type_Key* rootStartingKeys;
    
public:
    //Constructor & Destructor
    EqualSplit();

    EqualSplit(vector<Type_Key> & data, 
        rmi::ModelType rootModelType,
        rmi::ModelType leafSegMethod, 
        int noSegment=1000,
        rmi::SearchType rootSearchMethod = rmi::SearchType::BINARY_MAX,
        rmi::SearchType leafSearchMethod = rmi::SearchType::BINARY_ENTIRE);

    EqualSplit(vector<Type_Key> & samples, vector<Type_Key> & distribution,
        rmi::ModelType rootModelType,
        rmi::ModelType leafSegMethod, 
        int noSegment=1000,
        rmi::SearchType rootSearchMethod = rmi::SearchType::BINARY_MAX,
        rmi::SearchType leafSearchMethod = rmi::SearchType::BINARY_ENTIRE);

    ~EqualSplit();

    //Segment data
    void segment_data();

    //New method that generates the segments (starting keys) based on another set of keys.
    //The set of keys should be a superset of the data and described the overall data distribution.
    //The starting keys are found based on equal split of the data_distribution dataset.
    void segment_data_using_distribution(vector<Type_Key> & data_distribution);

private:
    //Segmentation helper
    template <typename AnyIt> 
    void generate_new_segment(int segIndex, AnyIt first, AnyIt last);

    void find_max_error_single();
    void find_max_error_segment();

public:
    //Search
    bool search(Type_Key key);

    //Finds the search range for a given key 
    //(e.g., for binary search the performance would be log2(search_range(key)))
    size_t search_range(Type_Key key);

private:
    //Search helper
    size_t get_segment_id(const Type_Key key);
    size_t correct_segment_id(const Type_Key key, int predictedSegmentID);
    size_t exponential_search(Type_Key key, int predictedPos);
    size_t linear_search(Type_Key key, int predictedPos);

public:
    //Getters & Setters
    void set_root_model_type(rmi::ModelType modelType);
    void set_leaf_segmentation_method(rmi::ModelType segMethod);
    void set_root_search_method(rmi::SearchType searchMethod);
    void set_leaf_search_method(rmi::SearchType searchMethod);

    double get_entropy_of_leaf_level();
    vector<int> get_segment_size();
    int get_root_error();
    vector<int> get_data_error();

    //Print
    void print_segment_predict_first_last(bool printIndex);

    //Memory footprint
    uint64_t size_of();
};

/*
Constructor & Destructor
*/
template<class Type_Key>
EqualSplit<Type_Key>::EqualSplit()
{
    LOG_DEBUG("%s", "Start");
    this->m_rootModel = rmi::ModelType::NONE;
    this->m_leafModel = rmi::ModelType::NONE;
    this->m_rootSearchMethod = rmi::SearchType::NONE;
    this->m_leafSearchMethod = rmi::SearchType::NONE;
    rootLevel = nullptr;
    leafLevel = nullptr;
    leafMaxError = nullptr;
    rootMaxError = 0;
    rootStartingKeys = nullptr;
    LOG_DEBUG("%s", "End");
}

template<class Type_Key>
EqualSplit<Type_Key>::
EqualSplit(vector<Type_Key> & data, 
    rmi::ModelType rootModelType, rmi::ModelType leafSegMethod, 
    int noSegment,
    rmi::SearchType rootSearchMethod, 
    rmi::SearchType leafSearchMethod)
:LearnedIndex<Type_Key>(data, noSegment)
{
    LOG_DEBUG("%s", "Start");
    this->m_rootModel = rootModelType;
    this->m_leafModel = leafSegMethod;
    this->m_rootSearchMethod = rootSearchMethod;
    this->m_leafSearchMethod = leafSearchMethod;
    rootLevel = nullptr;
    leafLevel = new modelRMI<Type_Key>*[noSegment];
    fill_n(leafLevel, noSegment, nullptr);
    leafMaxError = nullptr;
    rootMaxError = 0;
    rootStartingKeys = new Type_Key[noSegment];
    segment_data();
    LOG_DEBUG("%s", "End");
}

template<class Type_Key>
EqualSplit<Type_Key>::
EqualSplit(vector<Type_Key> & samples, vector<Type_Key> & distribution,
    rmi::ModelType rootModelType, rmi::ModelType leafSegMethod, 
    int noSegment,
    rmi::SearchType rootSearchMethod, 
    rmi::SearchType leafSearchMethod)
:LearnedIndex<Type_Key>(samples, noSegment)
{
    LOG_DEBUG("%s", "Start");
    this->m_rootModel = rootModelType;
    this->m_leafModel = leafSegMethod;
    this->m_rootSearchMethod = rootSearchMethod;
    this->m_leafSearchMethod = leafSearchMethod;
    rootLevel = nullptr;
    leafLevel = new modelRMI<Type_Key>*[noSegment];
    fill_n(leafLevel, noSegment, nullptr);
    leafMaxError = nullptr;
    rootMaxError = 0;
    rootStartingKeys = new Type_Key[noSegment];
    segment_data_using_distribution(distribution);
    LOG_DEBUG("%s", "End");
}

template<class Type_Key>
EqualSplit<Type_Key>::~EqualSplit()
{
    if (rootLevel != nullptr)
    {
        delete rootLevel;
        rootLevel = nullptr;
    }
    
    if (leafLevel != nullptr)
    {
        for (int i = 0; i < this->m_noSegment; ++i)
        {
            if (leafLevel[i] != nullptr)
            {
                delete leafLevel[i];
                leafLevel[i] = nullptr;
            }
        }
        delete[] leafLevel;
        leafLevel = nullptr;
    }

    if (leafMaxError != nullptr)
    {
        delete[] leafMaxError;
        leafMaxError = nullptr;
    }

    if (rootStartingKeys != nullptr)
    {
        delete[] rootStartingKeys;
        rootStartingKeys = nullptr;
    }
}

/*
Segment Data
*/
template<class Type_Key>
void EqualSplit<Type_Key>::segment_data()
{
    if (this->m_segmentIndexes.size() > 0) {return;} //Already segmented

    int segSize = (double)this->m_data.size() / (double)this->m_noSegment; //Last segment may be slightly larger
    segSize = max(segSize, 1);

    int startIndex = 0;
    int endIndex = 0;
    for (int i = 0; i < this->m_noSegment-1; ++i)
    {
        startIndex = i*segSize;
        endIndex = (i+1)*segSize-1;
        this->m_segmentIndexes.push_back(make_pair(startIndex,endIndex));
        generate_new_segment(i,this->m_data.begin()+startIndex,this->m_data.begin()+endIndex);
        rootStartingKeys[i] = this->m_data[startIndex];
    }
    startIndex = (this->m_noSegment-1)*segSize;
    endIndex = this->m_data.size()-1;
    this->m_segmentIndexes.push_back(make_pair(startIndex,endIndex));
    generate_new_segment(this->m_noSegment-1,this->m_data.begin()+startIndex,this->m_data.end());
    rootStartingKeys[this->m_noSegment-1] = this->m_data[startIndex];

    switch (m_rootModel)
    {
        case rmi::ModelType::CONSTANT:
            rootLevel = new constantRMI<Type_Key>(rootStartingKeys,rootStartingKeys+this->m_noSegment);
            break;
        case rmi::ModelType::LINEAR_SPLINE:
            rootLevel = new LinearSplineRMI<Type_Key>(rootStartingKeys,rootStartingKeys+this->m_noSegment);
            break;
        case rmi::ModelType::LINEAR_REGRESSION:
            rootLevel = new LinearRegressionRMI<Type_Key>(rootStartingKeys,rootStartingKeys+this->m_noSegment);
            break;
        case rmi::ModelType::CUBIC_SPLINE:
            rootLevel = new CubicSplineRMI<Type_Key>(rootStartingKeys,rootStartingKeys+this->m_noSegment);
            break;
        default:
            LOG_ERROR("Invalid ModelType");
            abort();
            return;
    }

    //Compute root error for search
    switch (m_rootSearchMethod)
    {
        case rmi::SearchType::NONE:
            break;
        
        //BINARY_ENTIRE, EXPONENTIAL or LINEAR (No need error)
        case rmi::SearchType::EXPONENTIAL:
            break;
        case rmi::SearchType::LINEAR:
            break;
        case rmi::SearchType::BINARY_ENTIRE:
            break;

        //Other two binary search cases are the same as there is one model in the root.
        default:
            for (int i = 0; i < this->m_noSegment; ++i)
            {
                int predictedSegmentID = get_segment_id(rootStartingKeys[i]);
                int error = abs(predictedSegmentID - i);
                if (error > rootMaxError)
                {
                    rootMaxError = error;
                }
            }
            break;
    }

    //Compute leaf error for search
    switch (m_leafSearchMethod)
    {
        case rmi::SearchType::NONE: //No Search (skip entirely)
            break;
        case rmi::SearchType::BINARY_MAX: //Single Error
            find_max_error_single();
            break;
        case rmi::SearchType::BINARY_MAX_SEG: //Error for each segment
            find_max_error_segment();
            break;
        default: //BINARY_ENTIRE, EXPONENTIAL or LINEAR (No need error)
            break;
    }
}


template<class Type_Key>
void EqualSplit<Type_Key>::segment_data_using_distribution(vector<Type_Key> & data_distribution)
{
    if (this->m_segmentIndexes.size() > 0) {return;} //Already segmented

    //Find the starting keys based on the keys in data_distribution and build model
    int segSize = (double)data_distribution.size() / (double)this->m_noSegment;
    segSize = max(segSize, 1);

    int startIndex = 0;
    int endIndex = 0;
    for (int i = 0; i < this->m_noSegment-1; ++i)
    {
        startIndex = i*segSize;
        rootStartingKeys[i] = data_distribution[startIndex];
    }
    rootStartingKeys[this->m_noSegment-1] = data_distribution[(this->m_noSegment-1)*segSize];

    switch (m_rootModel)
    {
        case rmi::ModelType::CONSTANT:
            rootLevel = new constantRMI<Type_Key>(rootStartingKeys,rootStartingKeys+this->m_noSegment);
            break;
        case rmi::ModelType::LINEAR_SPLINE:
            rootLevel = new LinearSplineRMI<Type_Key>(rootStartingKeys,rootStartingKeys+this->m_noSegment);
            break;
        case rmi::ModelType::LINEAR_REGRESSION:
            rootLevel = new LinearRegressionRMI<Type_Key>(rootStartingKeys,rootStartingKeys+this->m_noSegment);
            break;
        case rmi::ModelType::CUBIC_SPLINE:
            rootLevel = new CubicSplineRMI<Type_Key>(rootStartingKeys,rootStartingKeys+this->m_noSegment);
            break;
        default:
            LOG_ERROR("Invalid ModelType");
            abort();
            return;
    }

    //Generate segments based on the starting keys
    startIndex = 0;
    endIndex = 0;
    for (int i = 1; i < this->m_noSegment; ++i)
    {
        if (startIndex < this->m_data.size()) //Within data range
        {
            while (endIndex < this->m_data.size() && this->m_data[endIndex] < rootStartingKeys[i])
            {
                ++endIndex;
            }

            if (startIndex != endIndex) //Generate Segments
            {
                this->m_segmentIndexes.push_back(make_pair(startIndex,endIndex));
                generate_new_segment(i-1,this->m_data.begin()+startIndex,this->m_data.begin()+endIndex);
                ++endIndex;
                startIndex = endIndex;
            }
            else //Empty Segment
            {
                this->m_segmentIndexes.push_back(make_pair(startIndex-1,startIndex-1));
                generate_new_segment(i-1,this->m_data.begin()+startIndex-1,this->m_data.begin()+startIndex);
            }
        }
        else //Outside Data Range, then generate all equal segments
        {
            this->m_segmentIndexes.push_back(make_pair(this->m_data.size()-1,this->m_data.size()-1));
            generate_new_segment(i-1,this->m_data.end()-1,this->m_data.end());
        }
    }

    if (startIndex < this->m_data.size())
    {
        this->m_segmentIndexes.push_back(make_pair(startIndex,this->m_data.size()-1));
        generate_new_segment(this->m_noSegment-1,this->m_data.begin()+startIndex,this->m_data.end());
    }
    else
    {
        this->m_segmentIndexes.push_back(make_pair(this->m_data.size()-1,this->m_data.size()-1));
        generate_new_segment(this->m_noSegment-1,this->m_data.end()-1,this->m_data.end());
    }
    
    //Compute root error for search
    switch (m_rootSearchMethod)
    {
        case rmi::SearchType::NONE:
            break;
        
        //BINARY_ENTIRE, EXPONENTIAL or LINEAR (No need error)
        case rmi::SearchType::EXPONENTIAL:
            break;
        case rmi::SearchType::LINEAR:
            break;
        case rmi::SearchType::BINARY_ENTIRE:
            break;

        //Other two binary search cases are the same as there is one model in the root.
        default:
            for (int i = 0; i < this->m_noSegment; ++i)
            {
                int predictedSegmentID = get_segment_id(rootStartingKeys[i]);
                int error = abs(predictedSegmentID - i);
                if (error > rootMaxError)
                {
                    rootMaxError = error;
                }
            }
            break;
    }

    //Compute leaf error for search
    switch (m_leafSearchMethod)
    {
        case rmi::SearchType::NONE: //No Search (skip entirely)
            break;
        case rmi::SearchType::BINARY_MAX: //Single Error
            find_max_error_single();
            break;
        case rmi::SearchType::BINARY_MAX_SEG: //Error for each segment
            find_max_error_segment();
            break;
        default: //BINARY_ENTIRE, EXPONENTIAL or LINEAR (No need error)
            break;
    }
}


/*
Generate Segment Helper
*/
template<class Type_Key>
template <typename AnyIt>
inline void EqualSplit<Type_Key>::generate_new_segment(int segIndex, AnyIt first, AnyIt last)
{
    if (leafLevel[segIndex] != nullptr)
    {
        delete leafLevel[segIndex];
        leafLevel[segIndex] = nullptr;
    }
    
    switch (m_leafModel)
    {
        case rmi::ModelType::CONSTANT:
            leafLevel[segIndex] = new constantRMI<Type_Key>(first,last);
            break;
        case rmi::ModelType::LINEAR_SPLINE:
            leafLevel[segIndex] = new LinearSplineRMI<Type_Key>(first,last);
            break;
        case rmi::ModelType::LINEAR_REGRESSION:
            leafLevel[segIndex] = new LinearRegressionRMI<Type_Key>(first,last);
            break;
        case rmi::ModelType::CUBIC_SPLINE:
            leafLevel[segIndex] = new CubicSplineRMI<Type_Key>(first,last);
            break;
        default:
            LOG_ERROR("Invalid ModelType");
            abort();
            return;
    }
}

template<class Type_Key>
void EqualSplit<Type_Key>::find_max_error_single()
{
    if (leafMaxError != nullptr)
    {
        delete[] leafMaxError;
    }
    leafMaxError = new int[1];

    int maxError = 0;
    for (int i = 0; i < this->m_noSegment; ++i)
    {
        if (this->m_segmentIndexes[i].first != this->m_segmentIndexes[i].second)
        {
            for (int ii = this->m_segmentIndexes[i].first; 
                ii <= this->m_segmentIndexes[i].second; ++ii)
            {
                int pos = clamp<int>(ii,0,this->m_data.size()-1);
                int predictedPos =  this->m_segmentIndexes[i].first +
                                    clamp<double>(leafLevel[i]->predict(this->m_data[pos]),
                                    0,this->m_segmentIndexes[i].second-this->m_segmentIndexes[i].first+1);
                int error = abs(predictedPos - pos);
                if (error > maxError)
                {
                    maxError = error;
                }
            }
        }
    }
    leafMaxError[0] = maxError;
}

template<class Type_Key>
void EqualSplit<Type_Key>::find_max_error_segment()
{
    if (leafMaxError != nullptr)
    {
        delete[] leafMaxError;
    }
    leafMaxError = new int[this->m_noSegment];
    
    int maxError;
    for (int i = 0; i < this->m_noSegment; ++i)
    {
        maxError = 0;
        if (this->m_segmentIndexes[i].first != this->m_segmentIndexes[i].second)
        {
            for (int ii = this->m_segmentIndexes[i].first; 
                ii <= this->m_segmentIndexes[i].second; ++ii)
            {
                int pos = clamp<int>(ii,0,this->m_data.size()-1);
                int predictedPos = this->m_segmentIndexes[i].first +
                                    clamp<double>(leafLevel[i]->predict(this->m_data[pos]),
                                    0,this->m_segmentIndexes[i].second-this->m_segmentIndexes[i].first+1);
                int error = abs(predictedPos - pos);
                if (error > maxError)
                {
                    maxError = error;
                }
            }
        }
        leafMaxError[i] = maxError;
    }
}

/*
Search
*/
template<class Type_Key>
bool EqualSplit<Type_Key>::search(Type_Key key)
{
    if (m_rootSearchMethod == rmi::SearchType::NONE 
            || m_leafSearchMethod == rmi::SearchType::NONE)
    {
        return false;
    }
    
    #ifdef DETAILED_TIME
    uint64_t temp = 0;
    startTimer(&temp);
    #endif

    size_t predictedSegmentID = get_segment_id(key);

    #ifdef DETAILED_TIME
    stopTimer(&temp);
    rootPredictCycle += temp;
    temp = 0;
    startTimer(&temp);
    #endif

    predictedSegmentID = correct_segment_id(key,predictedSegmentID);

    #ifdef DETAILED_TIME
    stopTimer(&temp);
    rootCorrectCycle += temp;
    temp = 0;
    startTimer(&temp);
    #endif

    int predictedPos = this->m_segmentIndexes[predictedSegmentID].first + 
                        clamp<int>(leafLevel[predictedSegmentID]->predict(key),
                        0,this->m_segmentIndexes[predictedSegmentID].second-this->m_segmentIndexes[predictedSegmentID].first+1);

    #ifdef DETAILED_TIME
    stopTimer(&temp);
    segmentPredictCycle += temp;
    #endif

    int lowerSearchPos = 0;
    int upperSearchPos = this->m_data.size();
    auto it = this->m_data.begin();
    size_t foundPos;
    switch (m_leafSearchMethod)
    {
        case rmi::SearchType::BINARY_MAX:
            #ifdef DETAILED_TIME
            temp = 0;
            startTimer(&temp);
            #endif

            lowerSearchPos = clamp<int>(predictedPos-leafMaxError[0],0,predictedPos);
            upperSearchPos = clamp<int>(predictedPos+leafMaxError[0]+1,predictedPos,this->m_data.size());
            it = lower_bound(this->m_data.begin()+ lowerSearchPos,
                            this->m_data.begin()+upperSearchPos,
                            key);

            #ifdef DETAILED_TIME
            stopTimer(&temp);
            segmentCorrectCycle += temp;
            #endif

            if (it != this->m_data.end() && *it == key)
            {
                return true;
            }
            return false;

        case rmi::SearchType::BINARY_MAX_SEG:
            #ifdef DETAILED_TIME
            temp = 0;
            startTimer(&temp);
            #endif

            lowerSearchPos = clamp<int>(predictedPos-leafMaxError[predictedSegmentID],0,predictedPos);
            upperSearchPos = clamp<int>(predictedPos+leafMaxError[predictedSegmentID]+1,predictedPos,this->m_data.size());
            it = lower_bound(this->m_data.begin()+ lowerSearchPos,
                            this->m_data.begin()+upperSearchPos,
                            key);

            #ifdef DETAILED_TIME
            stopTimer(&temp);
            segmentCorrectCycle += temp;
            #endif

            if (it != this->m_data.end() && *it == key)
            {
                return true;
            }
            return false;

        case rmi::SearchType::BINARY_ENTIRE:
            #ifdef DETAILED_TIME
            temp = 0;
            startTimer(&temp);
            #endif

            it = lower_bound(this->m_data.begin()+this->m_segmentIndexes[predictedSegmentID].first,
                            this->m_data.begin()+this->m_segmentIndexes[predictedSegmentID].second+1,
                            key);
            
            #ifdef DETAILED_TIME
            stopTimer(&temp);
            segmentCorrectCycle += temp;
            #endif

            if (it != this->m_data.begin()+this->m_segmentIndexes[predictedSegmentID].second+1 && *it == key)
            {
                return true;
            }
            return false;

        case rmi::SearchType::EXPONENTIAL:
            #ifdef DETAILED_TIME
            temp = 0;
            startTimer(&temp);
            #endif

            predictedPos = clamp<int>(predictedPos,0,this->m_data.size()-1);
            foundPos = exponential_search(key,predictedPos);
            
            #ifdef DETAILED_TIME
            stopTimer(&temp);
            segmentCorrectCycle += temp;
            #endif

            if (foundPos < this->m_data.size() && this->m_data[foundPos] == key)
            {
                return true;
            }
            return false;

        case rmi::SearchType::LINEAR:
            #ifdef DETAILED_TIME
            temp = 0;
            startTimer(&temp);
            #endif

            predictedPos = clamp<int>(predictedPos,0,this->m_data.size()-1);
            foundPos = linear_search(key,predictedPos);
            
            #ifdef DETAILED_TIME
            stopTimer(&temp);
            segmentCorrectCycle += temp;
            #endif

            if (foundPos < this->m_data.size() && this->m_data[foundPos] == key)
            {
                return true;
            }
            return false;
    }
}

template<class Type_Key>
size_t EqualSplit<Type_Key>::search_range(Type_Key key)
{
    if (m_rootSearchMethod == rmi::SearchType::NONE 
            || m_leafSearchMethod == rmi::SearchType::NONE)
    {
        return 0;
    }

    size_t predictedSegmentID = get_segment_id(key);
    predictedSegmentID = correct_segment_id(key,predictedSegmentID);

    int predictedPos = this->m_segmentIndexes[predictedSegmentID].first + 
                        clamp<int>(leafLevel[predictedSegmentID]->predict(key),
                        0,this->m_segmentIndexes[predictedSegmentID].second-this->m_segmentIndexes[predictedSegmentID].first+1);
    
    int lowerSearchPos = 0;
    int upperSearchPos = this->m_data.size();
    auto it = this->m_data.begin();
    size_t foundPos;
    switch (m_leafSearchMethod)
    {
        case rmi::SearchType::BINARY_MAX:
            lowerSearchPos = clamp<int>(predictedPos-leafMaxError[0],0,predictedPos);
            upperSearchPos = clamp<int>(predictedPos+leafMaxError[0]+1,predictedPos,this->m_data.size());
            return abs(upperSearchPos - lowerSearchPos);

        case rmi::SearchType::BINARY_MAX_SEG:
            lowerSearchPos = clamp<int>(predictedPos-leafMaxError[predictedSegmentID],0,predictedPos);
            upperSearchPos = clamp<int>(predictedPos+leafMaxError[predictedSegmentID]+1,predictedPos,this->m_data.size());
            return abs(upperSearchPos - lowerSearchPos);

        case rmi::SearchType::BINARY_ENTIRE:
            return abs(this->m_data.begin()+this->m_segmentIndexes[predictedSegmentID].second+1 
                        - this->m_data.begin()+this->m_segmentIndexes[predictedSegmentID].first);

        case rmi::SearchType::EXPONENTIAL:
            predictedPos = clamp<int>(predictedPos,0,this->m_data.size()-1);
            foundPos = exponential_search(key,predictedPos);
            return abs((int)foundPos - predictedPos);

        case rmi::SearchType::LINEAR:
            predictedPos = clamp<int>(predictedPos,0,this->m_data.size()-1);
            foundPos = linear_search(key,predictedPos);
            return abs((int)foundPos - predictedPos);
    }
}

/*
Search Helper
*/
template<class Type_Key>
inline size_t EqualSplit<Type_Key>::exponential_search(Type_Key key, int predictedPos)
{
    if (key < this->m_data[predictedPos])
    {          
        int index = 1;
        while (predictedPos - index >= 0 && this->m_data[predictedPos - index] >= key)
        {
            index *= 2;
        }
        
        auto startIt = this->m_data.begin() + (predictedPos - min(index,predictedPos));
        auto endIt = this->m_data.begin() + (predictedPos - static_cast<int>(index/2));
        auto lowerBoundIt = lower_bound(startIt,endIt,key);

        return ((*lowerBoundIt < key)?lowerBoundIt+1:lowerBoundIt) - this->m_data.begin();
    }
    else if (key > this->m_data[predictedPos])
    {
        int index = 1;
        while (predictedPos + index  < this->m_data.size() && this->m_data[predictedPos + index] < key)
        {
            index *= 2;
        }

        auto startIt = this->m_data.begin() + (predictedPos + static_cast<int>(index/2));
        auto endIt = this->m_data.begin() + min(predictedPos + index, (int)this->m_data.size()-1);
        auto lowerBoundIt = lower_bound(startIt,endIt,key);

        return ((*lowerBoundIt < key)?lowerBoundIt+1:lowerBoundIt) - this->m_data.begin();
    }
    else
    {
        return predictedPos;
    }
}

template<class Type_Key>
inline size_t EqualSplit<Type_Key>::linear_search(Type_Key key, int predictedPos)
{
    if (key < this->m_data[predictedPos])
    {
        while (predictedPos >= 0 && this->m_data[predictedPos] >= key)
        {
            --predictedPos;
        }
        return predictedPos+1;
    }
    else if (key > this->m_data[predictedPos])
    {
        while (predictedPos < this->m_data.size() && this->m_data[predictedPos] < key)
        {
            ++predictedPos;
        }
        return predictedPos;
    }
    else
    {
        return predictedPos;
    }
}

/*
Getters & Setters
*/
template<class Type_Key>
inline void EqualSplit<Type_Key>::set_root_model_type(rmi::ModelType modelType)
{
    this->m_rootModel = modelType;
}

template<class Type_Key>
inline void EqualSplit<Type_Key>::set_leaf_segmentation_method(rmi::ModelType segMethod)
{
    this->m_leafModel = segMethod;
}

template<class Type_Key>
inline void EqualSplit<Type_Key>::set_root_search_method(rmi::SearchType searchMethod)
{
    this->m_rootSearchMethod = searchMethod;
}

template<class Type_Key>
inline void EqualSplit<Type_Key>::set_leaf_search_method(rmi::SearchType searchMethod)
{
    this->m_leafSearchMethod = searchMethod;
}

template<class Type_Key>
inline size_t EqualSplit<Type_Key>::correct_segment_id(const Type_Key key, int predictedSegmentID)
{
    switch (m_leafSearchMethod)
    {
        case rmi::SearchType::LINEAR:
        {
            if (key < rootStartingKeys[predictedSegmentID])
            {
                while (predictedSegmentID >= 0 && key <= rootStartingKeys[predictedSegmentID])
                {
                    --predictedSegmentID;
                }
                ++predictedSegmentID;
            }
            else if (rootStartingKeys[predictedSegmentID] < key)
            {
                while (predictedSegmentID < this->m_noSegment-1 && rootStartingKeys[predictedSegmentID] < key)
                {
                    ++predictedSegmentID;
                }
            }
            return predictedSegmentID;
        }
        case rmi::SearchType::BINARY_ENTIRE:
        {
            Type_Key* itSegID = rootStartingKeys + predictedSegmentID;
            if (key > rootStartingKeys[predictedSegmentID])
            {
                itSegID = lower_bound(rootStartingKeys+predictedSegmentID, rootStartingKeys+this->m_noSegment, key);
            }
            else
            {
                itSegID = lower_bound(rootStartingKeys, rootStartingKeys+predictedSegmentID+1, key);
            }
            if (!(itSegID == rootStartingKeys || (itSegID != rootStartingKeys+this->m_noSegment && *itSegID == key)))
            {
                --itSegID;
            } 
            return itSegID - rootStartingKeys;
        }
        case rmi::SearchType::EXPONENTIAL:
        {
            Type_Key* itSegID = rootStartingKeys + predictedSegmentID;
            if (key < rootStartingKeys[predictedSegmentID])
            {          
                int index = 1;
                while (predictedSegmentID - index >= 0 && key <= rootStartingKeys[predictedSegmentID - index])
                {
                    index *= 2;
                }
                
                Type_Key* startIt = rootStartingKeys + (predictedSegmentID - min(index,predictedSegmentID));
                Type_Key* endIt = rootStartingKeys + (predictedSegmentID - static_cast<int>(index/2));
                itSegID = lower_bound(startIt,endIt,key);
            }
            else if (rootStartingKeys[predictedSegmentID] < key)
            {
                int index = 1;
                while (predictedSegmentID + index  < this->m_noSegment && rootStartingKeys[predictedSegmentID + index] < key)
                {
                    index *= 2;
                }

                Type_Key* startIt =rootStartingKeys + (predictedSegmentID + static_cast<int>(index/2));
                Type_Key* endIt = rootStartingKeys + min(predictedSegmentID + index, (int)this->m_noSegment-1);
                itSegID = lower_bound(startIt,endIt,key);
            }

            if (!(itSegID == rootStartingKeys || (itSegID != rootStartingKeys+this->m_noSegment && *itSegID == key)))
            {
                --itSegID;
            } 
            return itSegID - rootStartingKeys;
        }
        default: //BINARY_MAX or BINARY_MAX_SEG
        {
            int lowerSearchSegID = clamp<int>(predictedSegmentID-rootMaxError,0,predictedSegmentID);
            int upperSearchSegID = clamp<int>(predictedSegmentID+rootMaxError+1,predictedSegmentID,this->m_noSegment);
            Type_Key* itSegID = lower_bound(rootStartingKeys+ lowerSearchSegID,
                            rootStartingKeys + upperSearchSegID,
                            key);
            if (!(itSegID == rootStartingKeys || (itSegID != rootStartingKeys+this->m_noSegment && *itSegID == key)))
            {
                --itSegID;
            } 
            return itSegID - rootStartingKeys;
        }
    }
}

template<class Type_Key>
inline size_t EqualSplit<Type_Key>::get_segment_id(const Type_Key key)
{
    return clamp<double>(rootLevel->predict(key),0,this->m_noSegment-1);
}


template<class Type_Key>
double EqualSplit<Type_Key>::get_entropy_of_leaf_level()
{
    double entropy = 0;
    for (int i = 0; i < this->m_noSegment; ++i)
    {
        double pi = static_cast<double>(this->m_segmentIndexes[i].second - 
                    this->m_segmentIndexes[i].first + 1)/this->m_data.size();
        entropy += pi*log2(pi);
    }
    return -entropy;
}


template<class Type_Key>
vector<int> EqualSplit<Type_Key>::get_segment_size()
{
    vector<int> segmentSize(this->m_noSegment);
    for (int i = 0; i < this->m_noSegment; ++i)
    {
        segmentSize[i] = this->m_segmentIndexes[i].second - 
                            this->m_segmentIndexes[i].first + 1;
    }
    return segmentSize;
}

template<class Type_Key>
int EqualSplit<Type_Key>::get_root_error()
{
    return rootMaxError;
}

template<class Type_Key>
vector<int> EqualSplit<Type_Key>::get_data_error()
{
    vector<int> dataError;
    dataError.reserve(this->m_data.size());
    for (int i = 0; i < this->m_data.size(); ++ i)
    {
        size_t predictedSegmentID = get_segment_id(this->m_data[i]);
        predictedSegmentID = correct_segment_id(this->m_data[i],predictedSegmentID);
        
        int predictedPos = this->m_segmentIndexes[predictedSegmentID].first + 
                            clamp<int>(leafLevel[predictedSegmentID]->predict(this->m_data[i]),
                            0,this->m_segmentIndexes[predictedSegmentID].second-this->m_segmentIndexes[predictedSegmentID].first+1);

        dataError.push_back(abs(predictedPos - i));
    }
    return dataError;
}

/*
Print functions
*/
template<class Type_Key>
void EqualSplit<Type_Key>::print_segment_predict_first_last(bool printIndex)
{      
    if (printIndex)
    {
        cout << "[" << this->m_segmentIndexes[0].first << "," 
            << this->m_segmentIndexes[0].second << "]"; 
    }
    else
    {
        cout << "[";
        int predictedPos = clamp<double>(leafLevel[0]->predict(this->m_data[this->m_segmentIndexes[0].first]),
                                        0,this->m_segmentIndexes[0].second - this->m_segmentIndexes[0].first +1);
        cout << this->m_data[predictedPos + this->m_segmentIndexes[0].first] << ",";
        predictedPos = clamp<double>(leafLevel[0]->predict(this->m_data[this->m_segmentIndexes[0].second]),
                                    0,this->m_segmentIndexes[0].second - this->m_segmentIndexes[0].first +1);
        cout << this->m_data[predictedPos + this->m_segmentIndexes[0].second];
        cout << "]";
        
    } 
    
    for (int i = 1; i < this->m_noSegment; ++i)
    {
        if (printIndex)
        {
            cout << ",[" << this->m_segmentIndexes[i].first << "," 
                << this->m_segmentIndexes[i].second << "]"; 
        }
        else
        {
            cout << ",[";
            int predictedPos = clamp<double>(leafLevel[i]->predict(this->m_data[this->m_segmentIndexes[i].first]),
                                            0,this->m_segmentIndexes[i].second - this->m_segmentIndexes[i].first +1);
            cout << this->m_data[predictedPos + this->m_segmentIndexes[i].first] << ",";
            predictedPos = clamp<double>(leafLevel[i]->predict(this->m_data[this->m_segmentIndexes[i].second]),
                                        0,this->m_segmentIndexes[i].second - this->m_segmentIndexes[i].first +1);
            cout << this->m_data[predictedPos + this->m_segmentIndexes[i].second];
            cout << "]"; 
        } 
    }
}

/*
Memory Footprint (in terms of bytes)
*/
template<class Type_Key>
uint64_t EqualSplit<Type_Key>::size_of()
{
    //Constants
    uint64_t temp = sizeof(int)*2 + sizeof(vector<Type_Key>) 
                    + sizeof(vector<pair<int,int>>)
                    + sizeof(Type_Key)*this->m_data.size() 
                    + sizeof(pair<int,int>)*this->m_segmentIndexes.size()
                    + sizeof(rmi::ModelType)*2 + sizeof(rmi::SearchType)*2;

    //Leaf Error (for search)
    if (leafMaxError != nullptr)
    {
        if (m_leafSearchMethod == rmi::SearchType::BINARY_MAX)
        {
            temp += sizeof(int*) + sizeof(int);
        }
        else if (m_leafSearchMethod == rmi::SearchType::BINARY_MAX_SEG)
        {
            temp += sizeof(int*) + sizeof(int)*this->m_noSegment;
        }
    }

    //Root level
    if (rootLevel != nullptr)
    {
        temp += sizeof(modelRMI<Type_Key>*) + rootLevel->size_of();
    }

    //Leaf Leavel
    if (leafLevel != nullptr)
    {
        temp += sizeof(modelRMI<Type_Key>**);
        for (int i = 0; i < this->m_noSegment; ++i)
        {
            temp += sizeof(modelRMI<Type_Key>*) + leafLevel[i]->size_of();
        }
    }

    return temp;
}

} // namespace rmi
#endif