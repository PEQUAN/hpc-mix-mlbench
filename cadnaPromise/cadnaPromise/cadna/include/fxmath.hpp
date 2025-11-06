#ifndef FXMATH_H
#define FXMATH_H

#include <floatx.hpp>
#include <cmath>
#include <iostream>
#include <cassert>

#ifndef FXMATH_BACKEND_FLOAT
    #define BACKEND_FLOAT double
#else
    #define BACKEND_FLOAT float
#endif

#define ERROR_MESSAGE "Please enter a valid floatx type variable, \
         the built-in floating point type follows the usage of math.h."

#ifndef __CADNA__
template <class FLOAT, class T>
FLOAT rounding(const FLOAT *obj, const T& value) noexcept {
    return flx::floatxr(flx::get_exp_bits(*obj), flx::get_sig_bits(*obj), T(value));
}

// Trigonometric functions

template <class FLOAT> FLOAT sin(const FLOAT &x){
    BACKEND_FLOAT val = std::sin( (BACKEND_FLOAT) x );
    return rounding(&x, val);
}

template <class FLOAT> FLOAT cos(const FLOAT &x){
    
    BACKEND_FLOAT val = std::cos( (BACKEND_FLOAT) x );
    return rounding(&x, val);
}

template <class FLOAT> FLOAT tan(const FLOAT &x){
    BACKEND_FLOAT val = std::tan( (BACKEND_FLOAT) x );
    return rounding(&x, val);
}


template <class FLOAT> FLOAT acos(const FLOAT &x){
    BACKEND_FLOAT val = std::acos( (BACKEND_FLOAT) x );
    return rounding(&x, val);
}


template <class FLOAT> FLOAT atan(const FLOAT &x){
    BACKEND_FLOAT val = std::atan( (BACKEND_FLOAT) x );
    return rounding(&x, val);
}

template <class FLOAT1, class FLOAT2> FLOAT1 atan2(const FLOAT1 &x, const FLOAT2 &y){
    BACKEND_FLOAT val = std::atan2( (BACKEND_FLOAT) x, (BACKEND_FLOAT) y );
    return rounding(&x, val);
}


// Hyperbolic functions

template <class FLOAT> FLOAT cosh(const FLOAT &x){
    BACKEND_FLOAT val = std::cosh( (BACKEND_FLOAT) x );
    return rounding(&x, val);
}



template <class FLOAT> FLOAT sinh(const FLOAT &x){
    BACKEND_FLOAT val = std::sinh( (BACKEND_FLOAT) x );
    return rounding(&x, val);
}

template <class FLOAT> FLOAT tanh(const FLOAT &x){
    
    BACKEND_FLOAT val = std::tanh( (BACKEND_FLOAT) x );
    return rounding(&x, val);
}



template <class FLOAT> FLOAT acosh(const FLOAT &x){
    BACKEND_FLOAT val = std::acosh( (BACKEND_FLOAT) x );
    return rounding(&x, val);
}



template <class FLOAT> FLOAT asinh(const FLOAT &x){
    BACKEND_FLOAT val = std::asinh( (BACKEND_FLOAT) x );
    return rounding(&x, val);
}


template <class FLOAT> FLOAT atanh(const FLOAT &x){
    BACKEND_FLOAT val = std::atanh( (BACKEND_FLOAT) x );
    return rounding(&x, val);
}


// Exponential and logarithmic functions

template <class FLOAT> FLOAT exp(const FLOAT &x){
    BACKEND_FLOAT val = std::exp( (BACKEND_FLOAT) x );
    return rounding(&x, val);
}



template <class FLOAT> FLOAT frexp(const FLOAT &x, int* exp){
    BACKEND_FLOAT val = std::frexp( (BACKEND_FLOAT) x, exp);
    return rounding(&x, val);
}


template <class FLOAT> FLOAT ldexp(const FLOAT &x, const int exp){
    BACKEND_FLOAT val = std::ldexp( (BACKEND_FLOAT) x, exp);
    return rounding(&x, val);
}


template <class FLOAT1, class FLOAT2> FLOAT2 log(FLOAT1 &x){
    BACKEND_FLOAT val = std::log( (BACKEND_FLOAT) x );
    return rounding(&x, val);
}



template <class FLOAT1, class FLOAT2> FLOAT2 log10(const FLOAT1 &x){
    BACKEND_FLOAT val = std::log10( (BACKEND_FLOAT) x );
    return rounding(&x, val);
}



template <class FLOAT1, class FLOAT2> FLOAT1 modf(const FLOAT1 &x, const FLOAT2* intpart){
    BACKEND_FLOAT val = std::modf( (BACKEND_FLOAT) x, (BACKEND_FLOAT*) intpart);
    return rounding(&x, val);
}


template <class FLOAT> FLOAT exp2(const FLOAT &x){
    BACKEND_FLOAT val = std::exp2( (BACKEND_FLOAT) x );
    return rounding(&x, val);
}



template <class FLOAT> FLOAT expm1(const FLOAT &x){
    BACKEND_FLOAT val = std::expm1( (BACKEND_FLOAT) x );
    return rounding(&x, val);
}


template <class FLOAT> int ilogb(const FLOAT &x){
    return ilogb( (BACKEND_FLOAT) x );
}


template <class FLOAT> FLOAT log1p(const FLOAT &x){
    BACKEND_FLOAT val = std::log1p( (BACKEND_FLOAT) x );
    return rounding(&x, val);
}


template <class FLOAT> FLOAT log2(const FLOAT &x){
    BACKEND_FLOAT val = std::log2( (BACKEND_FLOAT) x );
    return rounding(&x, val);
}


template <class FLOAT> FLOAT logb(const FLOAT &x){
    BACKEND_FLOAT val = std::logb( (BACKEND_FLOAT) x );
    return rounding(&x, val);
}


template <class FLOAT> FLOAT scalbn(const FLOAT &x, const int n){
    BACKEND_FLOAT val = std::scalbn( (BACKEND_FLOAT) x, n);
    return rounding(&x, val);
}

template <class FLOAT> FLOAT scalbln(const FLOAT &x, const long int n){
    BACKEND_FLOAT val = std::scalbln( (BACKEND_FLOAT) x, n);
    return rounding(&x, val);
}


// Power functions

template <class FLOAT, class EXPTYPE> FLOAT pow(const FLOAT &x, const EXPTYPE &exponent){
    BACKEND_FLOAT val = std::pow( (BACKEND_FLOAT) x, (BACKEND_FLOAT) exponent);
    return rounding(&x, val);
}


template <class FLOAT> FLOAT cbrt(const FLOAT &x){
    BACKEND_FLOAT val = std::cbrt( (BACKEND_FLOAT) x );
    return rounding(&x, val);
}

template <class FLOAT1, class FLOAT2> FLOAT1 hypot(const FLOAT1 &x, const FLOAT2 &y){
    BACKEND_FLOAT val = std::hypot( (BACKEND_FLOAT) x, (BACKEND_FLOAT) y);
    return rounding(&x, val);
}


template <class FLOAT1, class FLOAT2> FLOAT2 sqrt(const FLOAT1 &x){
    BACKEND_FLOAT val = std::sqrt( (BACKEND_FLOAT) x );
    return rounding(&x, val);
}


// Error and gamma functions

template <class FLOAT> FLOAT erf(const FLOAT &x){
    BACKEND_FLOAT val = std::erf( (BACKEND_FLOAT) x );
    return rounding(&x, val);
}

template <class FLOAT> FLOAT erfc(const FLOAT &x){
    
    BACKEND_FLOAT val = erfc( (BACKEND_FLOAT) x );
    return rounding(&x, val);
}

template <class FLOAT> FLOAT tgamma(const FLOAT &x){
    BACKEND_FLOAT val = std::tgamma( (BACKEND_FLOAT) x );
    return rounding(&x, val);
}


template <class FLOAT> FLOAT lgamma(const FLOAT &x){
    BACKEND_FLOAT val = std::lgamma( (BACKEND_FLOAT) x );
    return rounding(&x, val);
}

// Minimum, maximum, difference functions

template <class FLOAT1, class FLOAT2> FLOAT1 fdim(const FLOAT1 &x, const FLOAT2 &y){
    BACKEND_FLOAT val = std::fdim( (BACKEND_FLOAT) x, (BACKEND_FLOAT) y );
    return rounding(&x, val);
}


template <class FLOAT1, class FLOAT2> FLOAT1 fmax(const FLOAT1 &x, const FLOAT2 &y){
    BACKEND_FLOAT val = std::fmax( (BACKEND_FLOAT) x, (BACKEND_FLOAT) y);
    return rounding(&x, val);
}


template <class FLOAT1, class FLOAT2> FLOAT1 fmin(const FLOAT1 &x, const FLOAT2 &y){
    BACKEND_FLOAT val = std::fmin( (BACKEND_FLOAT) x, (BACKEND_FLOAT) y);
    return rounding(&x, val);
}


// Other functions

template <class FLOAT> FLOAT fabs(const FLOAT &x){
    BACKEND_FLOAT val = std::fabs( (BACKEND_FLOAT) x );
    return rounding(&x, val);
}


template <class FLOAT> FLOAT abs(const FLOAT &x){
    BACKEND_FLOAT val = std::abs( (BACKEND_FLOAT) x );
    return rounding(&x, val);
}


template <class FLOAT1, class FLOAT2, class FLOAT3> FLOAT1 fma(const FLOAT1 &x, const FLOAT2 &y, const FLOAT3 &z){
    BACKEND_FLOAT val = std::fma( (BACKEND_FLOAT) x, (BACKEND_FLOAT) y, (BACKEND_FLOAT) z);
    return rounding(&x, val);
}

// Floating-point manipulation functions

template <class FLOAT1, class FLOAT2> FLOAT1 nexttoward(const FLOAT1 &x, const FLOAT2 &y){
    BACKEND_FLOAT val = std::nexttoward( (BACKEND_FLOAT) x, (BACKEND_FLOAT) y);
    return rounding(&x, val);
}


template <class FLOAT1, class FLOAT2> FLOAT1 nextafter(const FLOAT1 &x, const FLOAT2 &y){
    BACKEND_FLOAT val = std::nextafter( (BACKEND_FLOAT) x, (BACKEND_FLOAT) y);
    return rounding(&x, val);
}


template <class FLOAT1, class FLOAT2> FLOAT1 copysign(const FLOAT1 &x, const FLOAT2 &y){
    BACKEND_FLOAT val = std::copysign( (BACKEND_FLOAT) x, (BACKEND_FLOAT) y);
    return rounding(&x, val);
}

// Remainder 

template <class FLOAT1, class FLOAT2> FLOAT1 remainder(const FLOAT1 &x, const FLOAT2 &y){
    BACKEND_FLOAT val = std::remainder( (BACKEND_FLOAT) x, (BACKEND_FLOAT) y);
    return rounding(&x, val);
}


template <class FLOAT1, class FLOAT2> FLOAT1 remquo(const FLOAT1 &x, const FLOAT2 &y, int* quot){
    BACKEND_FLOAT val = std::remquo( (BACKEND_FLOAT) x, (BACKEND_FLOAT) y, quot);
    return rounding(&x, val);
}


template <class FLOAT> FLOAT nearbyint(const FLOAT &x){
    BACKEND_FLOAT val = std::nearbyint( (BACKEND_FLOAT) x );
    return rounding(&x, val);
}

// misc

template <class FLOAT> bool isnormal(const FLOAT &x){
    
    bool val = std::isnormal( (BACKEND_FLOAT) x );
    return val;
}

#endif

#endif // FXMATH_H