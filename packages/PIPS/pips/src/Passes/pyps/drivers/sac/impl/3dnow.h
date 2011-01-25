#include <mm3dnow.h>

typedef float a2sf[2] __attribute__ ((aligned (16)));
typedef __m64 v2sf;

typedef int	a2si[2] __attribute__ ((aligned (16)));
typedef __m64 v2si;

#define SIMD_LOAD_V2SF(vec, f)			\
		vec = *(const __m64 *) &(f)[0]
#define SIMD_LOAD_GENERIC_V2SF(vec, f0,f1)			\
		do {\
			a2sf __tmp;\
			__tmp[0]=f0;\
			__tmp[1]=f1;\
			vec = *(const __m64 *) &(__tmp)[0];\
			} while(0)

#define SIMD_STORE_V2SF(vec, f)			\
		*(__m64 *)(&(f)[0]) = vec

#define SIMD_MULPS(vec1, vec2, vec3)		\
		vec1 = _m_pfmul(vec2, vec3)

#define SIMD_ADDPS(vec1, vec2, vec3)		\
		vec1 = _m_pfadd(vec2, vec3)

#define SIMD_SUBPS(vec1, vec2, vec3)		\
		vec1 = _m_pfsub(vec2, vec3)

/* should not be there :$ */
#define SIMD_ZERO_V4SF(vec) \
		SIMD_SUBPS(vec,vec,vec)

#define SIMD_UMINPS(vec1, vec2)					\
		do {							\
		__m64 __pips_tmp;				\
		__pips_tmp = _m_pxor(__pips_tmp, __pips_tmp);	\
		SIMD_SUBPS(vec1, __pips_tmp, vec2);		\
		} while(0)

#define SIMD_LOAD_V2SI_TO_V2SF(vec, f)		\
		do {					\
		float __pips_f[2];		\
		__pips_f[0] = (f)[0];		\
		__pips_f[1] = (f)[1];		\
		SIMD_LOAD_V2SF(vec, __pips_f);	\
		} while (0)

