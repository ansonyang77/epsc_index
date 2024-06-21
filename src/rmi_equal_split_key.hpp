#ifndef __EQUAL_KEYS_RMI_HPP__
#define __EQUAL_KEYS_RMI_HPP__

#pragma once
#include "base_index.hpp"
#include "helper_functions/helper_rmi.hpp"

using namespace std;

namespace rmi{

/*
Equal Split in the key range (segments are not equal in size)
*/

template<class Type_Key>
class EqualSplitKey : public LearnedIndex<Type_Key>
{
private:
    //Variables
    rmi::ModelType m_leafModel;
    rmi::SearchType m_searchMethod;

    int* leafMaxError;
    Type_Key* startingKeys;
    modelRMI<Type_Key>** leafLevel;

public:
    //Constructor & Destructor
    EqualSplitKey();
    EqualSplitKey(vector<Type_Key> & data, 
                rmi::ModelType leafSegMethod, 
                int noSegment=1000, 
                rmi::SearchType searchMethod = rmi::SearchType::BINARY_ENTIRE);
    ~EqualSplitKey();

    //Segment data
    void segment_data();

    //New method that generates the segments (starting keys) based on another set of keys
    //The set of keys should be a superset of the data and described the overall data distribution.
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
    size_t exponential_search(Type_Key key, int predictedPos);
    size_t linear_search(Type_Key key, int predictedPos);

public:
    //Getters & Setters
    void set_leaf_segmentation_method(rmi::ModelType segMethod);
    void set_search_method(rmi::SearchType searchMethod);

    double get_entropy_of_leaf_level();
    vector<int> get_segment_size();
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
EqualSplitKey<Type_Key>::EqualSplitKey()
{
    LOG_DEBUG("%s", "Start");
    this->m_leafModel = rmi::ModelType::NONE;
    this->m_searchMethod = rmi::SearchType::NONE;
    startingKeys = nullptr;
    leafLevel = nullptr;
    leafMaxError = nullptr;
    LOG_DEBUG("%s", "End");
}

template<class Type_Key>
EqualSplitKey<Type_Key>::
EqualSplitKey(vector<Type_Key> & data, 
            rmi::ModelType leafSegMethod, 
            int noSegment, 
            rmi::SearchType searchMethod)
:LearnedIndex<Type_Key>(data,noSegment)
{
    LOG_DEBUG("%s", "Start");
    this->m_leafModel = leafSegMethod;
    this->m_searchMethod = searchMethod;
    startingKeys = new Type_Key[noSegment];
    leafLevel = new modelRMI<Type_Key>*[noSegment];
    fill_n(leafLevel, noSegment, nullptr);
    leafMaxError = nullptr;

    segment_data();
    LOG_DEBUG("%s", "End");
}

template<class Type_Key>
EqualSplitKey<Type_Key>::~EqualSplitKey()
{
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

    if (startingKeys != nullptr)
    {
        delete[] startingKeys;
        startingKeys = nullptr;
    }
}

/*
Segment Data
*/
template<class Type_Key>
void EqualSplitKey<Type_Key>::segment_data()
{
    if (this->m_segmentIndexes.size() > 0) {return;} //Already segmented
    
    ASSERT_MESSAGE(this->m_noSegment > 0, "ERROR: Segment Limit must be greater than 0");

    //Segment using equal key ranges (skewness will be an issue)
    Type_Key keySize = (double)(this->m_data.back() - this->m_data.front()) / (double)this->m_noSegment;
    keySize = max(keySize, (Type_Key)1); //Prevent keySize = 0 (when data is too small
    
    int segmentIndex = 0;
    int dataStartIndex = 0;
    int dataEndIndex = 0;
    while (segmentIndex < this->m_noSegment-1)
    {
        while(this->m_data[dataEndIndex] < this->m_data.front()+static_cast<Type_Key>((double)keySize*(segmentIndex+1)))
        {
            ++dataEndIndex;
        }
    
        generate_new_segment(segmentIndex, this->m_data.begin()+dataStartIndex, this->m_data.begin()+dataEndIndex);
        startingKeys[segmentIndex] = keySize*(segmentIndex);
        if (dataStartIndex != dataEndIndex)
        {
            this->m_segmentIndexes.push_back(make_pair(dataStartIndex, dataEndIndex-1));
        }
        else
        {
            int previousIndex = (dataEndIndex == 0)? 0 : dataEndIndex-1;
            this->m_segmentIndexes.push_back(make_pair(previousIndex, previousIndex));
        }
        dataStartIndex = dataEndIndex;
        ++segmentIndex;
    }
    generate_new_segment(segmentIndex, this->m_data.begin()+dataStartIndex, this->m_data.end());
    startingKeys[segmentIndex] =  keySize*(segmentIndex);
    this->m_segmentIndexes.push_back(make_pair(dataStartIndex, this->m_data.size()-1));
    this->m_segmentIndexes.shrink_to_fit();

    // //Segment using equal number of data
    // int segSize = this->m_data.size() / this->m_noSegment;

    // int segmentIndex = 0;
    // int dataStartIndex;
    // for( dataStartIndex = 0; dataStartIndex <= this->m_data.size(); dataStartIndex+=segSize)
    // {
    //     if(segmentIndex >= this->m_noSegment-1)
    //     {
    //         break;
    //     }

    //     generate_new_segment(segmentIndex, this->m_data.begin()+dataStartIndex, this->m_data.begin()+dataStartIndex+segSize);
    //     startingKeys[segmentIndex] = this->m_data[dataStartIndex];
    //     this->m_segmentIndexes.push_back(make_pair(dataStartIndex, dataStartIndex+segSize-1 ));
    //     ++segmentIndex;
    // }
    // generate_new_segment(segmentIndex, this->m_data.begin()+dataStartIndex, this->m_data.end());
    // startingKeys[segmentIndex] = this->m_data[dataStartIndex];
    // this->m_segmentIndexes.push_back(make_pair(dataStartIndex, this->m_data.size()-1));
    // this->m_segmentIndexes.shrink_to_fit();

    //Compute error for search
    switch (m_searchMethod)
    {
        case rmi::SearchType::NONE: //No Search (skip entirely)
            break;
        case rmi::SearchType::BINARY_MAX: //Single Error
            find_max_error_single();
            break;
        case rmi::SearchType::BINARY_MAX_SEG: //Error for each segment
            find_max_error_segment();
            break;
        default: //BINARY_ENTIRE or EXPONENTIAL or LINEAR (No need error)
            break;
    }
}

/*
Generate Segment Helper
*/
template<class Type_Key>
template <typename AnyIt>
inline void EqualSplitKey<Type_Key>::generate_new_segment(int segIndex, AnyIt first, AnyIt last)
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
void EqualSplitKey<Type_Key>::find_max_error_single()
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
    }
    leafMaxError[0] = maxError;
}

template<class Type_Key>
void EqualSplitKey<Type_Key>::find_max_error_segment()
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
bool EqualSplitKey<Type_Key>::search(Type_Key key)
{
    if (m_searchMethod == rmi::SearchType::NONE)
    {
        return false;
    }

    #ifdef DETAILED_TIME
    uint64_t temp = 0;
    startTimer(&temp);
    #endif

    //Find Segment (Segment using equal key ranges)
    Type_Key keySize = (double)(this->m_data.back() - this->m_data.front()) / this->m_noSegment;
    keySize = max(keySize, (Type_Key)1); //Prevent keySize = 0 (when data is too small)
    int segIndex = clamp<int>((double)(key - this->m_data.front()) / (double)keySize, 0, this->m_noSegment-1);

    #ifdef DETAILED_TIME
    stopTimer(&temp);
    rootPredictCycle += temp;
    temp = 0;
    startTimer(&temp);
    #endif

    //Predict with Segment
    int predictedPos = this->m_segmentIndexes[segIndex].first + 
                        clamp<int>(leafLevel[segIndex]->predict(key),
                        0,this->m_segmentIndexes[segIndex].second-this->m_segmentIndexes[segIndex].first+1);

    #ifdef DETAILED_TIME
    stopTimer(&temp);
    segmentPredictCycle += temp;
    #endif

    int lowerSearchPos = 0;
    int upperSearchPos = this->m_data.size();
    auto it = this->m_data.begin();
    size_t foundPos;
    switch (m_searchMethod)
    {
        case rmi::SearchType::BINARY_MAX:
            #ifdef DETAILED_TIME
            temp = 0;
            startTimer(&temp);
            #endif

            lowerSearchPos = clamp<int>(predictedPos-leafMaxError[0],0,predictedPos);
            upperSearchPos = clamp<int>(predictedPos+leafMaxError[0]+1,predictedPos,this->m_data.size());
            it = lower_bound(this->m_data.begin()+ lowerSearchPos,
                            this->m_data.begin() + upperSearchPos,
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

            lowerSearchPos = clamp<int>(predictedPos-leafMaxError[segIndex],0,predictedPos);
            upperSearchPos = clamp<int>(predictedPos+leafMaxError[segIndex]+1,predictedPos,this->m_data.size());
            it = lower_bound(this->m_data.begin()+ lowerSearchPos,
                            this->m_data.begin() + upperSearchPos,
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

            it = lower_bound(this->m_data.begin()+this->m_segmentIndexes[segIndex].first,
                            this->m_data.begin()+this->m_segmentIndexes[segIndex].second+1,
                            key);

            #ifdef DETAILED_TIME
            stopTimer(&temp);
            segmentCorrectCycle += temp;
            #endif

            if (it != this->m_data.begin()+this->m_segmentIndexes[segIndex].second+1 && *it == key)
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
size_t EqualSplitKey<Type_Key>::search_range(Type_Key key)
{
    if (m_searchMethod == rmi::SearchType::NONE)
    {
        return 0;
    }

    //Find Segment (Segment using equal key ranges)
    Type_Key keySize = (double)(this->m_data.back() - this->m_data.front()) / this->m_noSegment;
    keySize = max(keySize, (Type_Key)1); //Prevent keySize = 0 (when data is too small)
    int segIndex = clamp<int>((double)(key - this->m_data.front()) / (double)keySize, 0, this->m_noSegment-1);

    //Predict with Segment
    int predictedPos = this->m_segmentIndexes[segIndex].first + 
                        clamp<int>(leafLevel[segIndex]->predict(key),
                        0,this->m_segmentIndexes[segIndex].second-this->m_segmentIndexes[segIndex].first+1);
    
    int lowerSearchPos = 0;
    int upperSearchPos = this->m_data.size();
    auto it = this->m_data.begin();
    size_t foundPos;
    switch (m_searchMethod)
    {
        case rmi::SearchType::BINARY_MAX:
            lowerSearchPos = clamp<int>(predictedPos-leafMaxError[0],0,predictedPos);
            upperSearchPos = clamp<int>(predictedPos+leafMaxError[0]+1,predictedPos,this->m_data.size());
            return abs(upperSearchPos - lowerSearchPos);

        case rmi::SearchType::BINARY_MAX_SEG:
            lowerSearchPos = clamp<int>(predictedPos-leafMaxError[segIndex],0,predictedPos);
            upperSearchPos = clamp<int>(predictedPos+leafMaxError[segIndex]+1,predictedPos,this->m_data.size());
            return abs(upperSearchPos - lowerSearchPos);

        case rmi::SearchType::BINARY_ENTIRE:
            return abs(this->m_data.begin()+this->m_segmentIndexes[segIndex].second+1 
                        - this->m_data.begin()+this->m_segmentIndexes[segIndex].first);

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
inline size_t EqualSplitKey<Type_Key>::exponential_search(Type_Key key, int predictedPos)
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
inline size_t EqualSplitKey<Type_Key>::linear_search(Type_Key key, int predictedPos)
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
inline void EqualSplitKey<Type_Key>::set_leaf_segmentation_method(rmi::ModelType segMethod)
{
    this->m_leafModel = segMethod;
}

template<class Type_Key>
inline void EqualSplitKey<Type_Key>::set_search_method(rmi::SearchType searchMethod)
{
    this->m_searchMethod = searchMethod;
}

template<class Type_Key>
double EqualSplitKey<Type_Key>::get_entropy_of_leaf_level()
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
vector<int> EqualSplitKey<Type_Key>::get_segment_size()
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
vector<int> EqualSplitKey<Type_Key>::get_data_error()
{
    Type_Key keySize = (double)(this->m_data.back() - this->m_data.front()) / this->m_noSegment;
    keySize = max(keySize, (Type_Key)1); //Prevent keySize = 0 (when data is too small)

    vector<int> dataError;
    dataError.reserve(this->m_data.size());
    for (int i = 0; i < this->m_data.size(); ++i)
    {
        int segIndex = clamp<int>((double)(this->m_data[i] - this->m_data.front()) / (double)keySize, 0, this->m_noSegment-1);
        int predictedPos = this->m_segmentIndexes[segIndex].first + 
                                clamp<int>(leafLevel[segIndex]->predict(this->m_data[i]),
                                0,this->m_segmentIndexes[segIndex].second-this->m_segmentIndexes[segIndex].first+1);

        dataError.push_back(abs(predictedPos - i));
    }
    return dataError;
}

/*
Print functions
*/
template<class Type_Key>
void EqualSplitKey<Type_Key>::print_segment_predict_first_last(bool printIndex)
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
uint64_t EqualSplitKey<Type_Key>::size_of()
{
    //Constants
    uint64_t temp = sizeof(int) + sizeof(vector<Type_Key>) 
                    + sizeof(vector<pair<int,int>>)
                    + sizeof(Type_Key)*this->m_data.size() 
                    + sizeof(pair<int,int>)*this->m_segmentIndexes.size()
                    + sizeof(rmi::ModelType) + sizeof(rmi::SearchType);

    //Leaf Error (for search)
    if (leafMaxError != nullptr)
    {
        if (m_searchMethod == rmi::SearchType::BINARY_MAX)
        {
            temp += sizeof(int*) + sizeof(int);
        }
        else if (m_searchMethod == rmi::SearchType::BINARY_MAX_SEG)
        {
            temp += sizeof(int*) + sizeof(int)*this->m_noSegment;
        }
    }

    //Starting Keys
    if (startingKeys != nullptr)
    {
        temp += sizeof(Type_Key*) + sizeof(Type_Key)*this->m_noSegment;
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


}// namespace rmi
#endif