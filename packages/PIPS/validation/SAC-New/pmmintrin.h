#include <stdint.h>

typedef union {
    int8_t c[8];
    int16_t s[4];
    int32_t i[2];
    int64_t l;
} __m64;

inline __m64 _mm_setzero_si64() {
    __m64 zeros;
    zeros.l=0;
    return zeros;
}

inline __m64 _m_pmaddwd(__m64 v0, __m64 v1) {
    __m64 tmp;
    int i;
    for(i = 0; i< 4; i++)
        tmp.s[i]= v0.s[i] * v1.s[i];
    tmp.i[0]= (int32_t)tmp.s[0] + (int32_t)tmp.s[1];
    tmp.i[1]= (int32_t)tmp.s[2] + (int32_t)tmp.s[3];
    return tmp;
}

inline __m64 _m_paddw(__m64 v0, __m64 v1) {
    __m64 tmp;
    int i;
    for(i = 0; i< 4; i++)
        tmp.s[i]= v0.s[i] + v1.s[i];
    return tmp;
}

inline int _m_to_int(__m64 v) {
    return v.i[0];
}

inline __m64 _m_psrlqi(__m64 v, int32_t count) {
    v.l >>= count;
    return v;
}

inline void _m_empty() {
}
