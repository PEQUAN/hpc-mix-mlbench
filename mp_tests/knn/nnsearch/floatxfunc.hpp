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

namespace fxmath{   

template <class FLOAT, class T>
FLOAT fx_rounding(const FLOAT *obj, const T& value) noexcept {
    return flx::floatxr(flx::get_exp_bits(*obj), flx::get_sig_bits(*obj), T(value));
}

// Trigonometric functions

template <class FLOAT> FLOAT fx_sin(const FLOAT &x){
    assert((ERROR_MESSAGE, !std::is_floating_point_v<FLOAT>));
    BACKEND_FLOAT val = sin( (BACKEND_FLOAT) x );
    return fx_rounding(&x, val);
}

template <class FLOAT> FLOAT fx_cos(const FLOAT &x){
    assert((ERROR_MESSAGE, !std::is_floating_point_v<FLOAT>));
    BACKEND_FLOAT val = cos( (BACKEND_FLOAT) x );
    return fx_rounding(&x, val);
}


template <class FLOAT> FLOAT fx_tan(const FLOAT &x){
    assert((ERROR_MESSAGE, !std::is_floating_point_v<FLOAT>));
    BACKEND_FLOAT val = tan( (BACKEND_FLOAT) x );
    return fx_rounding(&x, val);
}


template <class FLOAT> FLOAT fx_acos(const FLOAT &x){
    assert((ERROR_MESSAGE, !std::is_floating_point_v<FLOAT>));
    BACKEND_FLOAT val = acos( (BACKEND_FLOAT) x );
    return fx_rounding(&x, val);
}


template <class FLOAT> FLOAT fx_atan(const FLOAT &x){
    assert((ERROR_MESSAGE, !std::is_floating_point_v<FLOAT>));
    BACKEND_FLOAT val = atan( (BACKEND_FLOAT) x );
    return fx_rounding(&x, val);
}


template <class FLOAT1, class FLOAT2> FLOAT1 fx_atan2(const FLOAT1 &x, const FLOAT2 &y){
    assert((ERROR_MESSAGE, !std::is_floating_point_v<FLOAT1>));
    BACKEND_FLOAT val = atan2( (BACKEND_FLOAT) x, (BACKEND_FLOAT) y );
    return fx_rounding(&x, val);
}


// Hyperbolic functions

template <class FLOAT> FLOAT fx_cosh(const FLOAT &x){
    assert((ERROR_MESSAGE, !std::is_floating_point_v<FLOAT>));
    BACKEND_FLOAT val = cosh( (BACKEND_FLOAT) x );
    return fx_rounding(&x, val);
}


template <class FLOAT> FLOAT fx_sinh(const FLOAT &x){
    assert((ERROR_MESSAGE, !std::is_floating_point_v<FLOAT>));
    BACKEND_FLOAT val = sinh( (BACKEND_FLOAT) x );
    return fx_rounding(&x, val);
}


template <class FLOAT> FLOAT fx_tanh(const FLOAT &x){
    assert((ERROR_MESSAGE, !std::is_floating_point_v<FLOAT>));
    BACKEND_FLOAT val = tanh( (BACKEND_FLOAT) x );
    return fx_rounding(&x, val);
}


template <class FLOAT> FLOAT fx_acosh(const FLOAT &x){
    assert((ERROR_MESSAGE, !std::is_floating_point_v<FLOAT>));
    BACKEND_FLOAT val = acosh( (BACKEND_FLOAT) x );
    return fx_rounding(&x, val);
}


template <class FLOAT> FLOAT fx_asinh(const FLOAT &x){
    assert((ERROR_MESSAGE, !std::is_floating_point_v<FLOAT>));
    BACKEND_FLOAT val = asinh( (BACKEND_FLOAT) x );
    return fx_rounding(&x, val);
}


template <class FLOAT> FLOAT fx_atanh(const FLOAT &x){
    BACKEND_FLOAT val = atanh( (BACKEND_FLOAT) x );
    return fx_rounding(&x, val);
}


// Exponential and logarithmic functions
template <class FLOAT> FLOAT fx_exp(const FLOAT &x){
    assert((ERROR_MESSAGE, !std::is_floating_point_v<FLOAT>));
    BACKEND_FLOAT val = exp( (BACKEND_FLOAT) x );
    return fx_rounding(&x, val);
}


template <class FLOAT> FLOAT fx_frexp(const FLOAT &x, const int* exp){
    assert((ERROR_MESSAGE, !std::is_floating_point_v<FLOAT>));
    BACKEND_FLOAT val = frexp( (BACKEND_FLOAT) x, exp);
    return fx_rounding(&x, val);
}


template <class FLOAT> FLOAT fx_ldexp(const FLOAT &x, const int exp){
    assert((ERROR_MESSAGE, !std::is_floating_point_v<FLOAT>));
    BACKEND_FLOAT val = ldexp( (BACKEND_FLOAT) x, exp);
    return fx_rounding(&x, val);
}


template <class FLOAT> FLOAT fx_log(const FLOAT &x){
    assert((ERROR_MESSAGE, !std::is_floating_point_v<FLOAT>));
    BACKEND_FLOAT val = log( (BACKEND_FLOAT) x );
    return fx_rounding(&x, val);
}


template <class FLOAT> FLOAT fx_log10(const FLOAT &x){
    assert((ERROR_MESSAGE, !std::is_floating_point_v<FLOAT>));
    BACKEND_FLOAT val = log10( (BACKEND_FLOAT) x );
    return fx_rounding(&x, val);
}


template <class FLOAT1, class FLOAT2> FLOAT1 fx_modf(const FLOAT1 &x, const FLOAT2* intpart){
    assert((ERROR_MESSAGE, !std::is_floating_point_v<FLOAT1>));
    BACKEND_FLOAT val = modf( (BACKEND_FLOAT) x, (BACKEND_FLOAT*) intpart);
    return fx_rounding(&x, val);
}


template <class FLOAT> FLOAT fx_exp2(const FLOAT &x){
    assert((ERROR_MESSAGE, !std::is_floating_point_v<FLOAT>));
    BACKEND_FLOAT val = exp2( (BACKEND_FLOAT) x );
    return fx_rounding(&x, val);
}


template <class FLOAT> FLOAT fx_expm1(const FLOAT &x){
    assert((ERROR_MESSAGE, !std::is_floating_point_v<FLOAT>));
    BACKEND_FLOAT val = expm1( (BACKEND_FLOAT) x );
    return fx_rounding(&x, val);
}


template <class FLOAT> int fx_ilogb(const FLOAT &x){
    assert((ERROR_MESSAGE, !std::is_floating_point_v<FLOAT>));
    return ilogb( (BACKEND_FLOAT) x );
}


template <class FLOAT> FLOAT fx_log1p(const FLOAT &x){
    assert((ERROR_MESSAGE, !std::is_floating_point_v<FLOAT>));
    BACKEND_FLOAT val = log1p( (BACKEND_FLOAT) x );
    return fx_rounding(&x, val);
}


template <class FLOAT> FLOAT fx_log2(const FLOAT &x){
    assert((ERROR_MESSAGE, !std::is_floating_point_v<FLOAT>));
    BACKEND_FLOAT val = log2( (BACKEND_FLOAT) x );
    return fx_rounding(&x, val);
}


template <class FLOAT> FLOAT fx_logb(const FLOAT &x){
    assert((ERROR_MESSAGE, !std::is_floating_point_v<FLOAT>));
    BACKEND_FLOAT val = logb( (BACKEND_FLOAT) x );
    return fx_rounding(&x, val);
}


template <class FLOAT> FLOAT fx_scalbn(const FLOAT &x, const int n){
    assert((ERROR_MESSAGE, !std::is_floating_point_v<FLOAT>));
    BACKEND_FLOAT val = scalbn( (BACKEND_FLOAT) x, n);
    return fx_rounding(&x, val);
}


template <class FLOAT> FLOAT fx_scalbln(const FLOAT &x, const long int n){
    assert((ERROR_MESSAGE, !std::is_floating_point_v<FLOAT>));
    BACKEND_FLOAT val = scalbln( (BACKEND_FLOAT) x, n);
    return fx_rounding(&x, val);
}


// Power functions

template <class FLOAT, class EXPTYPE> FLOAT fx_pow(const FLOAT &x, const EXPTYPE &exponent){
    assert((ERROR_MESSAGE, !std::is_floating_point_v<FLOAT>));
    BACKEND_FLOAT val = pow( (BACKEND_FLOAT) x, (BACKEND_FLOAT) exponent);
    return fx_rounding(&x, val);
}


template <class FLOAT> FLOAT fx_cbrt(const FLOAT &x){
    assert((ERROR_MESSAGE, !std::is_floating_point_v<FLOAT>));
    BACKEND_FLOAT val = cbrt( (BACKEND_FLOAT) x );
    return fx_rounding(&x, val);
}


template <class FLOAT1, class FLOAT2> FLOAT1 fx_hypot(const FLOAT1 &x, const FLOAT2 &y){
    assert((ERROR_MESSAGE, !std::is_floating_point_v<FLOAT1>));
    BACKEND_FLOAT val = hypot( (BACKEND_FLOAT) x, (BACKEND_FLOAT) y);
    return fx_rounding(&x, val);
}


template <class FLOAT> FLOAT fx_sqrt(const FLOAT &x){
    assert((ERROR_MESSAGE, !std::is_floating_point_v<FLOAT>));
    BACKEND_FLOAT val = sqrt( (BACKEND_FLOAT) x );
    return fx_rounding(&x, val);
}


// Error and gamma functions

template <class FLOAT> FLOAT fx_erf(const FLOAT &x){
    assert((ERROR_MESSAGE, !std::is_floating_point_v<FLOAT>));
    BACKEND_FLOAT val = erf( (BACKEND_FLOAT) x );
    return fx_rounding(&x, val);
}


template <class FLOAT> FLOAT fx_erfc(const FLOAT &x){
    assert((ERROR_MESSAGE, !std::is_floating_point_v<FLOAT>));
    BACKEND_FLOAT val = erfc( (BACKEND_FLOAT) x );
    return fx_rounding(&x, val);
}


template <class FLOAT> FLOAT fx_tgamma(const FLOAT &x){
    assert((ERROR_MESSAGE, !std::is_floating_point_v<FLOAT>));
    BACKEND_FLOAT val = tgamma( (BACKEND_FLOAT) x );
    return fx_rounding(&x, val);
}


template <class FLOAT> FLOAT fx_lgamma(const FLOAT &x){
    assert((ERROR_MESSAGE, !std::is_floating_point_v<FLOAT>));
    BACKEND_FLOAT val = lgamma( (BACKEND_FLOAT) x );
    return fx_rounding(&x, val);
}


// Minimum, maximum, difference functions

template <class FLOAT1, class FLOAT2> FLOAT1 fx_fdim(const FLOAT1 &x, const FLOAT2 &y){
    assert((ERROR_MESSAGE, !std::is_floating_point_v<FLOAT1>));
    BACKEND_FLOAT val = fdim( (BACKEND_FLOAT) x, (BACKEND_FLOAT) y );
    return fx_rounding(&x, val);
}


template <class FLOAT1, class FLOAT2> FLOAT1 fx_fmax(const FLOAT1 &x, const FLOAT2 &y){
    assert((ERROR_MESSAGE, !std::is_floating_point_v<FLOAT1>));
    BACKEND_FLOAT val = fmax( (BACKEND_FLOAT) x, (BACKEND_FLOAT) y);
    return fx_rounding(&x, val);
}


template <class FLOAT1, class FLOAT2> FLOAT1 fx_fmin(const FLOAT1 &x, const FLOAT2 &y){
    assert((ERROR_MESSAGE, !std::is_floating_point_v<FLOAT1>));
    BACKEND_FLOAT val = fmin( (BACKEND_FLOAT) x, (BACKEND_FLOAT) y);
    return fx_rounding(&x, val);
}


// Other functions

template <class FLOAT> FLOAT fx_fabs(const FLOAT &x){
    assert((ERROR_MESSAGE, !std::is_floating_point_v<FLOAT>));
    BACKEND_FLOAT val = fabs( (BACKEND_FLOAT) x );
    return fx_rounding(&x, val);
}


template <class FLOAT> FLOAT fx_abs(const FLOAT &x){
    assert((ERROR_MESSAGE, !std::is_floating_point_v<FLOAT>));
    BACKEND_FLOAT val = abs( (BACKEND_FLOAT) x );
    return fx_rounding(&x, val);
}


template <class FLOAT1, class FLOAT2, class FLOAT3> FLOAT1 fx_fma(const FLOAT1 &x, const FLOAT2 &y, const FLOAT3 &z){
    assert((ERROR_MESSAGE, !std::is_floating_point_v<FLOAT1>));
    BACKEND_FLOAT val = fma( (BACKEND_FLOAT) x, (BACKEND_FLOAT) y, (BACKEND_FLOAT) z);
    return fx_rounding(&x, val);
}

// Floating-point manipulation functions

template <class FLOAT1, class FLOAT2> FLOAT1 fx_nexttoward(const FLOAT1 &x, const FLOAT2 &y){
    assert((ERROR_MESSAGE, !std::is_floating_point_v<FLOAT1>));
    BACKEND_FLOAT val = nexttoward( (BACKEND_FLOAT) x, (BACKEND_FLOAT) y);
    return fx_rounding(&x, val);
}


template <class FLOAT1, class FLOAT2> FLOAT1 fx_nextafter(const FLOAT1 &x, const FLOAT2 &y){
    assert((ERROR_MESSAGE, !std::is_floating_point_v<FLOAT1>));
    BACKEND_FLOAT val = nextafter( (BACKEND_FLOAT) x, (BACKEND_FLOAT) y);
    return fx_rounding(&x, val);
}


template <class FLOAT1, class FLOAT2> FLOAT1 fx_copysign(const FLOAT1 &x, const FLOAT2 &y){
    assert((ERROR_MESSAGE, !std::is_floating_point_v<FLOAT1>));
    BACKEND_FLOAT val = copysign( (BACKEND_FLOAT) x, (BACKEND_FLOAT) y);
    return fx_rounding(&x, val);
}


//  Remainder 

template <class FLOAT1, class FLOAT2> FLOAT1 fx_remainder(const FLOAT1 &x, const FLOAT2 &y){
    assert((ERROR_MESSAGE, !std::is_floating_point_v<FLOAT1>));
    BACKEND_FLOAT val = remainder( (BACKEND_FLOAT) x, (BACKEND_FLOAT) y);
    return fx_rounding(&x, val);
}


template <class FLOAT1, class FLOAT2> FLOAT1 fx_remquo(const FLOAT1 &x, const FLOAT2 &y, const int* quot){
    assert((ERROR_MESSAGE, !std::is_floating_point_v<FLOAT1>));
    BACKEND_FLOAT val = remquo( (BACKEND_FLOAT) x, (BACKEND_FLOAT) y, quot);
    return fx_rounding(&x, val);
}


template <class FLOAT> FLOAT fx_nearbyint(const FLOAT &x){
    assert((ERROR_MESSAGE, !std::is_floating_point_v<FLOAT>));
    BACKEND_FLOAT val = nearbyint( (BACKEND_FLOAT) x );
    return fx_rounding(&x, val);
}

// misc


template <class FLOAT> bool fx_isnormal(const FLOAT &x){
    assert((ERROR_MESSAGE, !std::is_floating_point_v<FLOAT>));
    bool val = std::isnormal( (BACKEND_FLOAT) x );
    return val;
}


} // end namespace


#endif // FXMATH_H


