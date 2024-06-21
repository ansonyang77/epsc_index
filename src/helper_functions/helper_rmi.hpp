#ifndef __HELPER_RMI_HPP__
#define __HELPER_RMI_HPP__

//Adapted from https://github.com/alhuan/analysis-rmi

#pragma once
#include <cmath>
#include <x86intrin.h>

using namespace std;

#ifdef DETAILED_TIME
#include "../../utils/timer_variables.hpp"
uint64_t rootPredictCycle = 0;
uint64_t rootCorrectCycle = 0;
uint64_t segmentPredictCycle = 0;
uint64_t segmentCorrectCycle = 0;
#endif

namespace rmi{

enum class ModelType {NONE = -1, CONSTANT = 0, LINEAR_SPLINE = 1, LINEAR_REGRESSION = 2, CUBIC_SPLINE = 3};
enum class SearchType {NONE = -1, BINARY_MAX = 0, BINARY_MAX_SEG = 1,  BINARY_ENTIRE = 2, EXPONENTIAL = 3, LINEAR = 4};

template<typename DataType>
class modelRMI
{
public:
    modelRMI() = default;

    virtual double predict(const DataType value) = 0;
    virtual uint64_t size_of() = 0;
};

template<typename DataType>
class constantRMI : public modelRMI<DataType>
{
private:
    double m_constant;

public:
    constantRMI()
    {
        m_constant = -1;
    }

    template<typename AnyIt>
    constantRMI(AnyIt first, AnyIt last, double compressFactor = 1.f)
    {
        size_t n = distance(first,last);
        switch (n)
        {
            case 0:
            case 1:
                m_constant = 0;
                return;
            default:
                break;
        }
        m_constant = 0.5*(n-1);
    }

    double predict(const DataType value)
    {
        return m_constant;
    }

    uint64_t size_of()
    {
        return sizeof(double);
    }
};


template<typename DataType>
class LinearSplineRMI : public modelRMI<DataType>
{
private:
    double m_slope;
    double m_intercept;

public:
    LinearSplineRMI()
    {
        m_slope = -1;
        m_intercept = -1;
    }

    template<typename AnyIt>
    LinearSplineRMI(AnyIt first, AnyIt last, double compressFactor = 1.f)
    {
        size_t n = distance(first,last);
        switch (n)
        {
            case 0:
            case 1:
                m_slope = 0;
                m_intercept = 0;
                return;
            default:
                break;
        }

        double denominator = static_cast<double>(*(last-1) - *first);
        m_slope = denominator != 0 ? static_cast<double>(n)/denominator * compressFactor: 0;
        m_intercept = compressFactor - m_slope * (*first);
    }

    double predict(const DataType value)
    {
        return fma(m_slope,static_cast<double>(value),m_intercept);
    }

    uint64_t size_of()
    {
        return sizeof(double)*2;
    }
};

template<typename DataType>
class LinearRegressionRMI : public modelRMI<DataType>
{
private:
    double m_slope;
    double m_intercept;

public:
    LinearRegressionRMI()
    {
        m_slope = -1;
        m_intercept = -1;
    }

    template<typename AnyIt>
    LinearRegressionRMI(AnyIt first, AnyIt last, double compressFactor = 1.f)
    {
        size_t n = distance(first,last);
        switch (n)
        {
            case 0:
            case 1:
                m_slope = 0;
                m_intercept = 0;
                return;
            default:
                break;
        }

        double mean_x = 0.0;
        double mean_y = 0.0;
        double c = 0.0;
        double m2 = 0.0;

        for (size_t i = 0; i != n; ++i) {
            auto x = *(first + i);
            size_t y = i;

            double dx = x - mean_x;
            mean_x += dx /  (i + 1);
            mean_y += (y - mean_y) / (i + 1);
            c += dx * (y - mean_y);

            double dx2 = x - mean_x;
            m2 += dx * dx2;
        }

        double cov = c / (n - 1);
        double var = m2 / (n - 1);

        if (var == 0.f) {
            m_slope  = 0.f;
            m_intercept = mean_y;
            return;
        }

        m_slope = cov / var * compressFactor;
        m_intercept = mean_y * compressFactor - m_slope * mean_x;
    }

    double predict(const DataType value)
    {
        return fma(m_slope,static_cast<double>(value),m_intercept);
    }

    uint64_t size_of()
    {
        return sizeof(double)*2;
    }
};

template<typename DataType>
class CubicSplineRMI : public modelRMI<DataType>
{
private:
    double m_a; //cubic coefficient (a*x^3)
    double m_b; //quadratic coefficient (b*x^2)
    double m_c; //linear coefficient (c*x)
    double m_d; //y-intercept

public:
    CubicSplineRMI()
    {
        m_a = -1; m_b = -1; m_c = -1; m_d = -1;
    }

    template<typename AnyIt>
    CubicSplineRMI(AnyIt first, AnyIt last, double compressFactor = 1.f)
    {
        size_t n = distance(first,last);
        if (n == 0 || n == 1 || *first == *(last-1))
        {
            m_a = 0; m_b = 0; m_c = 0; m_d = 0;
            return;
        }

        double xmin = static_cast<double>(*first);
        double ymin = 0;
        double xmax = static_cast<double>(*(last - 1));
        double ymax = static_cast<double>(n - 1) * compressFactor;

        double x1 = 0.0;
        double y1 = 0.0;
        double x2 = 1.0;
        double y2 = 1.0;

        double sxn, syn = 0.0;
        for (size_t i = 0; i != n; ++i) {
            double x = static_cast<double>(*(first + i));
            double y = static_cast<double>(i) * compressFactor;
            sxn = (x - xmin) / (xmax - xmin);
            if (sxn > 0.0) {
                syn = (y - ymin) / (ymax - ymin);
                break;
            }
        }
        double m1 = (syn - y1) / (sxn - x1);

        double sxp, syp = 0.0;
        for (std::size_t i = 0; i != n; ++i) {
            double x = static_cast<double>(*(first + i));
            double y = static_cast<double>(i) * compressFactor;
            sxp = (x - xmin) / (xmax - xmin);
            if (sxp < 1.0) {
                syp = (y - ymin) / (ymax - ymin);
                break;
            }
        }
        double m2 = (y2 - syp) / (x2 - sxp);

        if (pow(m1, 2.0) + pow(m2, 2.0) > 9.0) {
            double tau = 3.0 / sqrt(pow(m1, 2.0) + pow(m2, 2.0));
            m1 *= tau;
            m2 *= tau;
        }

        m_a = (m1 + m2 - 2.0)
            / pow(xmax - xmin, 3.0);

        m_b = -(xmax * (2.0 * m1 + m2 - 3.0) + xmin * (m1 + 2.0 * m2 - 3.0))
            / pow(xmax - xmin, 3.0);

        m_c = (m1 * pow(xmax, 2.0) + m2 * pow(xmin, 2.0) + xmax * xmin * (2.0 * m1 + 2.0 * m2 - 6.0))
            / pow(xmax - xmin, 3.0);

        m_d = -xmin * (m1 * pow(xmax, 2.0) + xmax * xmin * (m2 - 3.0) + pow(xmin, 2.0))
            / pow(xmax - xmin, 3.0);

        m_a *= ymax - ymin;
        m_b *= ymax - ymin;
        m_c *= ymax - ymin;
        m_d *= ymax - ymin;
        m_d += ymin;
    }

    double predict(const DataType value)
    {
        double x = static_cast<double>(value);
        return fma(fma(fma(m_a,x,m_b),x,m_c),x,m_d);
    }

    uint64_t size_of()
    {
        return sizeof(double)*4;
    }
};

// //Implement when needed
// class RadixRMI
// {
// private:

// };

} // namespace rmi

#endif