// e8m7 -> bf16
// ripped from https://github.com/peeterjoot/floatexplorer/blob/master/floatexplorer.cc
#include <bitset>
#include <bit>
#include <cctype>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include "bf16.h"

// Type        Exponent-Size   Exponent-Bias Format    (Sign.Exponent.Mantissa)
// FP8 E4M3    4 bits          7                       1.4.3
// FP8 E5M2    5 bits          15                      1.5.2
// BF16        8 bits          127                     1.8.7
// FP16        5 bits          15                      1.5.10
union float_e4m3
{
    using UNSIGNED_TYPE = std::uint8_t;
    using SIGNED_TYPE = std::int8_t;

    static constexpr UNSIGNED_TYPE EXPONENT_BITS = 4;
    static constexpr UNSIGNED_TYPE MANTISSA_BITS = 3;

    static constexpr UNSIGNED_TYPE EXPONENT_MASK = ( ( UNSIGNED_TYPE( 1 ) << EXPONENT_BITS ) - 1 );
    static constexpr UNSIGNED_TYPE EXPONENT_BIAS = ( ( UNSIGNED_TYPE( 1 ) << ( EXPONENT_BITS - 1 ) ) - 1 );

    UNSIGNED_TYPE u;
    void fromFloat( float tf );
};

union float_e5m2
{
    using UNSIGNED_TYPE = std::uint8_t;
    using SIGNED_TYPE = std::int8_t;

    static constexpr UNSIGNED_TYPE EXPONENT_BITS = 5;
    static constexpr UNSIGNED_TYPE MANTISSA_BITS = 2;

    static constexpr UNSIGNED_TYPE EXPONENT_MASK = ( ( UNSIGNED_TYPE( 1 ) << EXPONENT_BITS ) - 1 );
    static constexpr UNSIGNED_TYPE EXPONENT_BIAS = ( ( UNSIGNED_TYPE( 1 ) << ( EXPONENT_BITS - 1 ) ) - 1 );

    UNSIGNED_TYPE u;
    void fromFloat( float tf );
};

union float_bf16
{
    using UNSIGNED_TYPE = std::uint16_t;
    using SIGNED_TYPE = std::int16_t;

    static constexpr UNSIGNED_TYPE EXPONENT_BITS = 8;
    static constexpr UNSIGNED_TYPE MANTISSA_BITS = 7;

    static constexpr UNSIGNED_TYPE EXPONENT_MASK = ( ( UNSIGNED_TYPE( 1 ) << EXPONENT_BITS ) - 1 );
    static constexpr UNSIGNED_TYPE EXPONENT_BIAS = ( ( UNSIGNED_TYPE( 1 ) << ( EXPONENT_BITS - 1 ) ) - 1 );

    UNSIGNED_TYPE u;
    void fromFloat( float tf );
};

union float_ieee32
{
    using UNSIGNED_TYPE = std::uint32_t;
    using SIGNED_TYPE = std::int32_t;

    static constexpr UNSIGNED_TYPE EXPONENT_BITS = 8;
    static constexpr UNSIGNED_TYPE MANTISSA_BITS = 23;

    static constexpr UNSIGNED_TYPE EXPONENT_MASK = ( ( UNSIGNED_TYPE( 1 ) << EXPONENT_BITS ) - 1 );
    static constexpr UNSIGNED_TYPE EXPONENT_BIAS = ( ( UNSIGNED_TYPE( 1 ) << ( EXPONENT_BITS - 1 ) ) - 1 );

    UNSIGNED_TYPE u;

    float s;
};

union float_ieee64
{
    using UNSIGNED_TYPE = std::uint64_t;
    using SIGNED_TYPE = std::int64_t;

    static constexpr UNSIGNED_TYPE EXPONENT_BITS = 11;
    static constexpr UNSIGNED_TYPE MANTISSA_BITS = 52;

    static constexpr UNSIGNED_TYPE EXPONENT_MASK = ( ( UNSIGNED_TYPE( 1 ) << EXPONENT_BITS ) - 1 );
    static constexpr UNSIGNED_TYPE EXPONENT_BIAS = ( ( UNSIGNED_TYPE( 1 ) << ( EXPONENT_BITS - 1 ) ) - 1 );

    UNSIGNED_TYPE u;
    double s;
};

template <class T>
void extract_float_representation( T f, typename T::UNSIGNED_TYPE& sign, typename T::SIGNED_TYPE& exponent,
                                   typename T::UNSIGNED_TYPE& mantissa )
{
    mantissa = f.u & ( ( typename T::UNSIGNED_TYPE( 1 ) << T::MANTISSA_BITS ) - 1 );
    typename T::UNSIGNED_TYPE exponent_with_bias = ( f.u >> T::MANTISSA_BITS ) & T::EXPONENT_MASK;

    if ( exponent_with_bias && exponent_with_bias != T::EXPONENT_MASK )
    {
        exponent = (typename T::SIGNED_TYPE)exponent_with_bias - T::EXPONENT_BIAS;    // Normal
    }
    else if ( exponent_with_bias == 0 && mantissa != 0 )
    {
        exponent = -( T::EXPONENT_BIAS - 1 );    // Denormal
    }
    else
    {
        exponent = 0;    // Zero
    }

    sign = f.u >> ( T::EXPONENT_BITS + T::MANTISSA_BITS );
}

template <class T>
double toDouble( T fu )
{
    float_ieee64 r;
    r.u = 0;

    using U = typename T::UNSIGNED_TYPE;
    using S = typename T::SIGNED_TYPE;
    using U64 = float_ieee64::UNSIGNED_TYPE;

    U s;
    S e;
    U m;

    extract_float_representation<T>( fu, s, e, m );

    U64 fsign = s;
    U64 fsignShift = float_ieee64::EXPONENT_BITS + float_ieee64::MANTISSA_BITS;
    fsign <<= fsignShift;

    // handle \pm 0: don't set exponent bits:
    U signTshift = T::EXPONENT_BITS + T::MANTISSA_BITS;
    U signT = U(s) << signTshift;
//    U notSignTmask = (U(1) << signTshift) - 1;
    U exponentMaskT = T::EXPONENT_MASK << T::MANTISSA_BITS;

    bool noExponent{};

    // inf: all exponent bits set, mantissa clear.
    // nan: all exponent bits set, and at least one mantissa bits set.
    // \pm 0: just sign bit set.
    if ( signT == fu.u ) {
        // don't set the exponent bits.
        noExponent = true;
    } else if ( (fu.u & exponentMaskT) == exponentMaskT ) {
        // \pm \infty, NaN
//        U64 notSignFmask = (U64(1) << fsignShift) - 1;

        r.u |= (float_ieee64::EXPONENT_MASK << float_ieee64::MANTISSA_BITS);
        noExponent = true;
    } else if ( (fu.u & exponentMaskT) == 0 ) {
        // denormalized mantissa.  No implied leading one
        U leading = std::countl_zero(m);
        U shift = 1 + leading - 8 * sizeof(U) + T::MANTISSA_BITS;
        m <<= shift;
        e -= shift;

        U mmask = ((U(1) << T::MANTISSA_BITS) - 1);
        m &= mmask;
    }

    if ( !noExponent ) {
        U64 fexponent = e;
        fexponent += float_ieee64::EXPONENT_BIAS;
        fexponent <<= float_ieee64::MANTISSA_BITS;

        r.u |= fexponent;
    }

    U64 fmantissa = m;
    fmantissa <<= ( float_ieee64::MANTISSA_BITS - T::MANTISSA_BITS);

    r.u |= fsign | fmantissa;

    return r.s;
}

template <class T>
typename T::UNSIGNED_TYPE fromFloatHelper( float tf )
{
    float_ieee32 f;
    f.s = tf;

    float_ieee32::UNSIGNED_TYPE s;
    float_ieee32::SIGNED_TYPE e;
    float_ieee32::UNSIGNED_TYPE m;

    extract_float_representation<float_ieee32>( f, s, e, m );

    float_ieee32::UNSIGNED_TYPE signbit32 = s << (float_ieee32::EXPONENT_BITS + float_ieee32::MANTISSA_BITS);

    T r;
    r.u = s << (T::EXPONENT_BITS + T::MANTISSA_BITS);

    // +- zero: don't set any exponent bits:
    if ( f.u == signbit32 ) {
       return r.u;
    }

    e += T::EXPONENT_BIAS;
    r.u |= ( e << T::MANTISSA_BITS );
    m >>= ( float_ieee32::MANTISSA_BITS - T::MANTISSA_BITS );
    r.u |= m;

    return r.u;
}

float e4m3_f(uint8_t v) {
  float_e4m3 tmp{ v };
  return (float)toDouble( tmp );
}

double e4m3_d(uint8_t v) {
  float_e4m3 tmp{ v };
  return toDouble( tmp );
}

float e5m2_f(uint8_t v) {
  float_e5m2 tmp{ v };
  return (float)toDouble( tmp );
}

double e5m2_d(uint8_t v) {
  float_e5m2 tmp{ v };
  return toDouble( tmp );
}

float e8m7_f(uint16_t v) {
  float_bf16 tmp{ v };
  return (float)toDouble( tmp );
}

double e8m7_d(uint16_t v) {
  float_bf16 tmp{ v };
  return toDouble( tmp );
}

uint8_t conv_e4m3(float f) {
  return fromFloatHelper<float_e4m3>(f);
}

uint8_t conv_e5m2(float f) {
  return fromFloatHelper<float_e5m2>(f);
}

uint16_t conv_e8m7(float f) {
    return fromFloatHelper<float_bf16>(f);
}