#include "stddef.h"

#pragma function(memcpy, memset, logf)
#pragma optimize("", off)

int _fltused = 0x9875;

void *memcpy(void *dest, const void *src, size_t n) {
    volatile char *d = (volatile char *)dest;
    const volatile char *s = (const volatile char *)src;
    for (size_t i = 0; i < n; ++i) {
        d[i] = s[i];
    }
    return dest;
}

void *memset(void *s, int c, size_t n) {
    volatile unsigned char *p = (volatile unsigned char *)s;
    for (size_t i = 0; i < n; ++i) {
        p[i] = (unsigned char)c;
    }
    return s;
}

// IEEE 754 float natural log implementation with 1-ULP precision
float logf(float x) {
    if (x <= 0.0f) {
        return -3.402823466e+38F;
    }
    
    union {
        float f;
        unsigned int u;
    } num;
    num.f = x;
    
    int exp = (int)((num.u >> 23) & 0xFF) - 127;
    num.u = (num.u & 0x007FFFFF) | 0x3F800000; // x in [1.0, 2.0)
    
    double m = (double)num.f;
    if (m > 1.4142135623730950488) {
        m *= 0.5;
        exp += 1;
    }
    
    double z = (m - 1.0) / (m + 1.0);
    double z2 = z * z;
    
    double poly = z * (2.0 + z2 * (2.0/3.0 + z2 * (2.0/5.0 + z2 * (2.0/7.0 + z2 * (2.0/9.0 + z2 * (2.0/11.0 + z2 * (2.0/13.0 + z2 * (2.0/15.0 + z2 * (2.0/17.0)))))))));
    double res = poly + (double)exp * 0.693147180559945309417232;
    return (float)res;
}

float floorf(float x) {
    int i = (int)x;
    if (x < 0.0f && (float)i != x) {
        return (float)(i - 1);
    }
    return (float)i;
}

