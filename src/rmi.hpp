#ifndef __RMI_HPP__
#define __RMI_HPP__

#pragma once
#include "base_index.hpp"
#include "helper_functions/helper_rmi.hpp"

using namespace std;

namespace rmi{

template<class Type_Key>
class RMI: public LearnedIndex<Type_Key>
{
private:
    //Variables
    int m_actualNoSegments;
    rmi::ModelType m_rootModel;
    rmi::ModelType m_leafModel;
    rmi::SearchType m_searchMethod;

    int* leafMaxError;
    modelRMI<Type_Key>* rootLevel;
    modelRMI<Type_Key>** leafLevel;
    
public:
    //Constructor & Destructor
    RMI();
    RMI(vector<Type_Key> & data, 
        rmi::ModelType rootSegMethod,
        rmi::ModelType leafSegMethod, 
        int noSegment=1000, 
        rmi::SearchType searchMethod = rmi::SearchType::BINARY_ENTIRE);
    ~RMI();

    //Segment data
    void segment_data();

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
    size_t exponential_search(Type_Key key, int predictedPos);
    size_t linear_search(Type_Key key, int predictedPos);

public:
    //Getters & Setters
    void set_root_segmentation_method(rmi::ModelType segMethod);
    void set_leaf_segmentation_method(rmi::ModelType segMethod);
    void set_search_method(rmi::SearchType searchMethod);

    int get_actual_no_segment();
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
RMI<Type_Key>::RMI()
{
    LOG_DEBUG("%s", "Start");
    this->m_rootModel = rmi::ModelType::NONE;
    this->m_leafModel = rmi::ModelType::NONE;
    this->m_searchMethod = rmi::SearchType::NONE;
    this->m_actualNoSegments = 0;
    rootLevel = nullptr;
    leafLevel = nullptr;
    leafMaxError = nullptr;
    LOG_DEBUG("%s", "End");
}

template<class Type_Key>
RMI<Type_Key>::
RMI(vector<Type_Key> & data, 
    rmi::ModelType rootSegMethod, rmi::ModelType leafSegMethod,  
    int noSegment, rmi::SearchType searchMethod)
:LearnedIndex<Type_Key>(data, noSegment)
{
    LOG_DEBUG("%s", "Start");
    this->m_rootModel = rootSegMethod;
    this->m_leafModel = leafSegMethod;
    this->m_searchMethod = searchMethod;
    this->m_actualNoSegments = noSegment;
    rootLevel = nullptr;
    leafLevel = new modelRMI<Type_Key>*[noSegment];
    fill_n(leafLevel, noSegment, nullptr);
    leafMaxError = nullptr;

    segment_data();
    LOG_DEBUG("%s", "End");
}

template<class Type_Key>
RMI<Type_Key>::~RMI()
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
}

/*
Segment Data
*/
template<class Type_Key>
void RMI<Type_Key>::segment_data()
{
    if (this->m_segmentIndexes.size() > 0) {return;} //Already segmented
    
    switch (m_rootModel)
    {
        case rmi::ModelType::CONSTANT: //Not reccomended to use constant model for root level (results in one segment)
            rootLevel = new constantRMI<Type_Key>(this->m_data.begin(),this->m_data.end());
            break;
        case rmi::ModelType::LINEAR_SPLINE:
            rootLevel = new LinearSplineRMI<Type_Key>(
                this->m_data.begin(),this->m_data.end(),
                static_cast<double>(this->m_noSegment)/this->m_data.size());
            break;
        case rmi::ModelType::LINEAR_REGRESSION:
            rootLevel = new LinearRegressionRMI<Type_Key>(
                this->m_data.begin(),this->m_data.end(),
                static_cast<double>(this->m_noSegment)/this->m_data.size());
            break;
        case rmi::ModelType::CUBIC_SPLINE:
            rootLevel = new CubicSplineRMI<Type_Key>(
                this->m_data.begin(),this->m_data.end(),
                static_cast<double>(this->m_noSegment)/this->m_data.size());
            break;
        default:
            LOG_ERROR("Invalid ModelType");
            abort();
            return;
    }

    size_t segmentStartIndex = 0;
    size_t segmentID = 0;
    this->m_segmentIndexes = vector<pair<int,int>>(this->m_noSegment);
    for (size_t i = 0; i != this->m_data.size(); ++i)
    {
        auto currentIt = this->m_data.begin()+i;
        size_t predictedSegmentID = get_segment_id(*currentIt);
        if (predictedSegmentID > segmentID)
        {
            generate_new_segment(segmentID,this->m_data.begin()+segmentStartIndex,currentIt);
            this->m_segmentIndexes[segmentID] = make_pair(segmentStartIndex,i-1);
            
            //In RMI robust: authors say to train all models between predictedID and segmentID with last key
            //in previous segment when predictedSegment != segmentID + 1 
            for (size_t j = segmentID+1; j < predictedSegmentID; ++j)
            {
                generate_new_segment(j,currentIt-1,currentIt);
                this->m_segmentIndexes[j] = make_pair(i-1,i-1);
            }

            segmentID = predictedSegmentID;
            segmentStartIndex = i;
        }
    }

    generate_new_segment(segmentID,this->m_data.begin()+segmentStartIndex,this->m_data.end());
    this->m_segmentIndexes[segmentID] = make_pair(segmentStartIndex,this->m_data.size()-1);
    for (size_t j = segmentID+1; j < this->m_noSegment; ++j)
    {
        generate_new_segment(j,this->m_data.end()-1,this->m_data.end());
        this->m_segmentIndexes[j] = make_pair(this->m_data.size()-1,this->m_data.size()-1);
    }
    
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
        default: //BINARY_ENTIRE, EXPONENTIAL or LINEAR (No need error)
            break;
    }
}

/*
Generate Segment Helper
*/
template<class Type_Key>
template <typename AnyIt>
inline void RMI<Type_Key>::generate_new_segment(int segIndex, AnyIt first, AnyIt last)
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
void RMI<Type_Key>::find_max_error_single()
{
    if (leafMaxError != nullptr)
    {
        delete[] leafMaxError;
    }
    leafMaxError = new int[1];

    int maxError = 0;
    for (int i = 0; i < m_actualNoSegments; ++i)
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
void RMI<Type_Key>::find_max_error_segment()
{
    if (leafMaxError != nullptr)
    {
        delete[] leafMaxError;
    }
    leafMaxError = new int[m_actualNoSegments];
    
    int maxError;
    for (int i = 0; i < m_actualNoSegments; ++i)
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
bool RMI<Type_Key>::search(Type_Key key)
{
    if (m_searchMethod == rmi::SearchType::NONE)
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
size_t RMI<Type_Key>::search_range(Type_Key key)
{
    if (m_searchMethod == rmi::SearchType::NONE)
    {
        return 0;
    }

    size_t predictedSegmentID = get_segment_id(key);
    int predictedPos = this->m_segmentIndexes[predictedSegmentID].first + 
                        clamp<int>(leafLevel[predictedSegmentID]->predict(key),
                        0,this->m_segmentIndexes[predictedSegmentID].second-this->m_segmentIndexes[predictedSegmentID].first+1);
    
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
inline size_t RMI<Type_Key>::exponential_search(Type_Key key, int predictedPos)
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
inline size_t RMI<Type_Key>::linear_search(Type_Key key, int predictedPos)
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
inline void RMI<Type_Key>::set_root_segmentation_method(rmi::ModelType segMethod)
{
    this->m_rootModel = segMethod;
}

template<class Type_Key>
inline void RMI<Type_Key>::set_leaf_segmentation_method(rmi::ModelType segMethod)
{
    this->m_leafModel = segMethod;
}

template<class Type_Key>
inline void RMI<Type_Key>::set_search_method(rmi::SearchType searchMethod)
{
    this->m_searchMethod = searchMethod;
}

template<class Type_Key>
inline size_t RMI<Type_Key>::get_segment_id(const Type_Key key)
{
    return clamp<double>(rootLevel->predict(key),0,this->m_noSegment-1);
}

template<class Type_Key>
inline int RMI<Type_Key>::get_actual_no_segment()
{
    return m_actualNoSegments;
}

template<class Type_Key>
double RMI<Type_Key>::get_entropy_of_leaf_level()
{
    double entropy = 0;
    for (int i = 0; i < m_actualNoSegments; ++i)
    {
        double pi = static_cast<double>(this->m_segmentIndexes[i].second - 
                    this->m_segmentIndexes[i].first + 1)/this->m_data.size();
        entropy += pi*log2(pi);
    }
    return -entropy;
}

template<class Type_Key>
vector<int> RMI<Type_Key>::get_segment_size()
{
    vector<int> segmentSize(m_actualNoSegments);
    for (int i = 0; i < m_actualNoSegments; ++i)
    {
        segmentSize[i] = this->m_segmentIndexes[i].second - 
                            this->m_segmentIndexes[i].first + 1;
    }
    return segmentSize;
}


template<class Type_Key>
vector<int> RMI<Type_Key>::get_data_error()
{
    vector<int> dataError;
    dataError.reserve(this->m_data.size());
    for (int i = 0; i < this->m_data.size(); ++ i)
    {
        size_t predictedSegmentID = get_segment_id(this->m_data[i]);
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
void RMI<Type_Key>::print_segment_predict_first_last(bool printIndex)
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
    
    for (int i = 1; i < m_actualNoSegments; ++i)
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
uint64_t RMI<Type_Key>::size_of()
{
    //Constants
    uint64_t temp = sizeof(int)*2 + sizeof(vector<Type_Key>) 
                    + sizeof(vector<pair<int,int>>)
                    + sizeof(Type_Key)*this->m_data.size() 
                    + sizeof(pair<int,int>)*this->m_segmentIndexes.size()
                    + sizeof(rmi::ModelType)*2 + sizeof(rmi::SearchType);

    //Leaf Error (for search)
    if (leafMaxError != nullptr)
    {
        if (m_searchMethod == rmi::SearchType::BINARY_MAX)
        {
            temp += sizeof(int*) + sizeof(int);
        }
        else if (m_searchMethod == rmi::SearchType::BINARY_MAX_SEG)
        {
            temp += sizeof(int*) + sizeof(int)*m_actualNoSegments;
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
        for (int i = 0; i < m_actualNoSegments; ++i)
        {
            temp += sizeof(modelRMI<Type_Key>*) + leafLevel[i]->size_of();
        }
    }

    return temp;
}

} // namespace rmi
#endif