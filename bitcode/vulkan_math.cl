/*
 * Copyright (c) 2023 chipStar developers
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

// Math glue for the native Vulkan device library: the C math names map to
// OpenCL builtins, which libclc provides at link time.

#define OVLD __attribute__((overloadable))

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Targets of the c_to_opencl.def entries whose OpenCL counterpart is not a
// plain builtin.
//
// scalbln takes a long exponent and ldexp an int. ldexp already returns 0 or
// infinity for any finite nonzero x once |k| passes a few thousand, so
// clamping the exponent to int changes nothing.
static OVLD float __chip_scalbln(float x, long n) {
  return ldexp(x, (int)clamp(n, (long)INT_MIN, (long)INT_MAX));
}
static OVLD double __chip_scalbln(double x, long n) {
  return ldexp(x, (int)clamp(n, (long)INT_MIN, (long)INT_MAX));
}
// nexttowardf(x, y): y is a long double, a 64-bit double on spirv64. C
// returns y converted to float when x == y (which is how -0.0f steps to
// +0.0f), and otherwise the next float after x in the direction of y.
static float __chip_nexttoward(float x, double y) {
  if (isnan(y) || x == y)
    return (float)y;
  return nextafter(x, y > x ? INFINITY : -INFINITY);
}

// See c_to_opencl.def for details.
#define DEF_UNARY_FN_MAP(FROM_FN_, TO_FN_, TYPE_)                              \
  TYPE_ __chip_c2ocl_##FROM_FN_(TYPE_ x) { return TO_FN_(x); }
#define DEF_BINARY_FN_MAP(FROM_FN_, TO_FN_, TYPE_)                             \
  TYPE_ __chip_c2ocl_##FROM_FN_(TYPE_ x, TYPE_ y) { return TO_FN_(x, y); }
#define DEF_UNARY_FN_MAP_RET(FROM_FN_, TO_FN_, RET_TYPE_, TYPE_)               \
  RET_TYPE_ __chip_c2ocl_##FROM_FN_(TYPE_ x) { return TO_FN_(x); }
#define DEF_BINARY_FN_MAP_MIXED(FROM_FN_, TO_FN_, TYPE_, TYPE2_)               \
  TYPE_ __chip_c2ocl_##FROM_FN_(TYPE_ x, TYPE2_ y) { return TO_FN_(x, y); }
#include "c_to_opencl.def"
#define DEF_TERNARY_FN_MAP(FROM_FN_, TO_FN_, TYPE_)                            \
  TYPE_ __chip_c2ocl_##FROM_FN_(TYPE_ x, TYPE_ y, TYPE_ z) {                   \
    return TO_FN_(x, y, z);                                                    \
  }
OVLD static long __chip_vk_lround(float x) { return (long)round(x); }
OVLD static long __chip_vk_lround(double x) { return (long)round(x); }
OVLD static long __chip_vk_lrint(float x) { return (long)rint(x); }
OVLD static long __chip_vk_lrint(double x) { return (long)rint(x); }
#include "vulkan_c_to_opencl.def"
#undef UNARY_FN
#undef BINARY_FN

// OCML entry points with no OpenCL builtin counterpart, built on libclc's
// erf/erfc/exp/log/cbrt. erfinv/erfcinv start from Giles' approximation
// ("Approximating the erfinv function", GPU Computing Gems) evaluated on
// w = -log(y(2-y)), which stays exact near y = 0 for erfcinv, then take
// Newton steps on erfc.
#define GEN_SPECIALS(T, S)                                                     \
  static T giles_##S(T w) {                                                    \
    T p;                                                                       \
    if (w < (T)6.25) {                                                         \
      w -= (T)3.125;                                                           \
      p = (T)-3.6444120640178196996e-21;                                       \
      p = (T)-1.685059138182016589e-19 + p * w;                                \
      p = (T)1.2858480715256400167e-18 + p * w;                                \
      p = (T)1.115787767802518096e-17 + p * w;                                 \
      p = (T)-1.333171662854620906e-16 + p * w;                                \
      p = (T)2.0972767875968561637e-17 + p * w;                                \
      p = (T)6.6376381343583238325e-15 + p * w;                                \
      p = (T)-4.0545662729752068639e-14 + p * w;                               \
      p = (T)-8.1519341976054721522e-14 + p * w;                               \
      p = (T)2.6335093153082322977e-12 + p * w;                                \
      p = (T)-1.2975133253453532498e-11 + p * w;                               \
      p = (T)-5.4154120542946279317e-11 + p * w;                               \
      p = (T)1.051212273321532285e-09 + p * w;                                 \
      p = (T)-4.1126339803469836976e-09 + p * w;                               \
      p = (T)-2.9070369957882005086e-08 + p * w;                               \
      p = (T)4.2347877827932403518e-07 + p * w;                                \
      p = (T)-1.3654692000834678645e-06 + p * w;                               \
      p = (T)-1.3882523362786468719e-05 + p * w;                               \
      p = (T)0.0001867342080340571352 + p * w;                                 \
      p = (T)-0.00074070253416626697512 + p * w;                               \
      p = (T)-0.0060336708714301490533 + p * w;                                \
      p = (T)0.24015818242558961693 + p * w;                                   \
      p = (T)1.6536545626831027356 + p * w;                                    \
    } else if (w < (T)16.0) {                                                  \
      w = sqrt(w) - (T)3.25;                                                   \
      p = (T)2.2137376921775787049e-09;                                        \
      p = (T)9.0756561938885390979e-08 + p * w;                                \
      p = (T)-2.7517406297064545428e-07 + p * w;                               \
      p = (T)1.8239629214389227755e-08 + p * w;                                \
      p = (T)1.5027403968909827627e-06 + p * w;                                \
      p = (T)-4.013867526981545969e-06 + p * w;                                \
      p = (T)2.9234449089955446044e-06 + p * w;                                \
      p = (T)1.2475304481671778723e-05 + p * w;                                \
      p = (T)-4.7318229009055733981e-05 + p * w;                               \
      p = (T)6.8284851459573175448e-05 + p * w;                                \
      p = (T)2.4031110387097893999e-05 + p * w;                                \
      p = (T)-0.0003550375203628474796 + p * w;                                \
      p = (T)0.00095328937973738049703 + p * w;                                \
      p = (T)-0.0016882755560235047313 + p * w;                                \
      p = (T)0.0024914420961078508066 + p * w;                                 \
      p = (T)-0.0037512085075692412107 + p * w;                                \
      p = (T)0.005370914553590063617 + p * w;                                  \
      p = (T)1.0052589676941592334 + p * w;                                    \
      p = (T)3.0838856104922207635 + p * w;                                    \
    } else {                                                                   \
      w = sqrt(w) - (T)5.0;                                                    \
      p = (T)-2.7109920616438573243e-11;                                       \
      p = (T)-2.5556418169965252055e-10 + p * w;                               \
      p = (T)1.5076572693500548083e-09 + p * w;                                \
      p = (T)-3.7894654401267369937e-09 + p * w;                               \
      p = (T)7.6157012080783393804e-09 + p * w;                                \
      p = (T)-1.4960026627149240478e-08 + p * w;                               \
      p = (T)2.9147953450901080826e-08 + p * w;                                \
      p = (T)-6.7711997758452339498e-08 + p * w;                               \
      p = (T)2.2900482228026654717e-07 + p * w;                                \
      p = (T)-9.9298272942317002539e-07 + p * w;                               \
      p = (T)4.5260625972231537039e-06 + p * w;                                \
      p = (T)-1.9681778105531670567e-05 + p * w;                               \
      p = (T)7.5995277030017761139e-05 + p * w;                                \
      p = (T)-0.00021503011930044477347 + p * w;                               \
      p = (T)-0.00013871931833623122026 + p * w;                               \
      p = (T)1.0103004648645343977 + p * w;                                    \
      p = (T)4.8499064014085844221 + p * w;                                    \
    }                                                                          \
    return p;                                                                  \
  }                                                                            \
  /* erfcinv(y) for y in (0, 2); Newton on erfc from Giles' start. */          \
  static T vk_erfcinv_##S(T y) {                                               \
    if (!(y > (T)0 && y < (T)2))                                               \
      return y == (T)0 ? (T)INFINITY : y == (T)2 ? -(T)INFINITY : (T)NAN;      \
    T x = giles_##S(-log(y * ((T)2 - y))) * ((T)1 - y);                        \
    for (int i = 0; i < 2; ++i) {                                              \
      T e = erfc(x) - y;                                                       \
      x += e / ((T)1.1283791670955126 * exp(-x * x));                          \
    }                                                                          \
    return x;                                                                  \
  }                                                                            \
  T __ocml_erfcinv_##S(T y) { return vk_erfcinv_##S(y); }                      \
  T __ocml_erfinv_##S(T x) {                                                   \
    if (fabs(x) < (T)0.5) {                                                    \
      T r = giles_##S(-log(((T)1 - x) * ((T)1 + x))) * x;                      \
      for (int i = 0; i < 2; ++i)                                              \
        r -= (erf(r) - x) / ((T)1.1283791670955126 * exp(-r * r));             \
      return r;                                                                \
    }                                                                          \
    return vk_erfcinv_##S((T)1 - x);                                           \
  }                                                                            \
  T __ocml_ncdf_##S(T x) { return (T)0.5 * erfc(-x * (T)M_SQRT1_2); }          \
  T __ocml_ncdfinv_##S(T p) { return -(T)M_SQRT2 * vk_erfcinv_##S((T)2 * p); } \
  T __ocml_rcbrt_##S(T x) { return (T)1 / cbrt(x); }                           \
  T __ocml_erfcx_##S(T x) {                                                    \
    if (x < (T)0) {                                                            \
      T x2 = x * x;                                                            \
      return (T)2 * exp(x2) - __ocml_erfcx_##S(-x);                            \
    }                                                                          \
    if (x > (T)10) {                                                           \
      /* Asymptotic series; exp(x*x) would overflow first. */                  \
      T r = (T)1 / (x * x);                                                    \
      return (T)M_2_SQRTPI * (T)0.5 / x *                                      \
             ((T)1 + r * ((T)-0.5 + r * ((T)0.75 + r * (T)-1.875)));           \
    }                                                                          \
    T hi = x * x, lo = fma(x, x, -hi);                                         \
    return exp(hi) * ((T)1 + lo) * erfc(x);                                    \
  }

GEN_SPECIALS(float, f32)
GEN_SPECIALS(double, f64)

void __chip_sincospi_f32(float x, float *s, float *c) {
  *s = sinpi(x);
  *c = cospi(x);
}
void __chip_sincospi_f64(double x, double *s, double *c) {
  *s = sinpi(x);
  *c = cospi(x);
}
