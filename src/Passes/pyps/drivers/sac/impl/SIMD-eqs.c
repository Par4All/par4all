#if !defined(RWBITS) || (RWBITS != 64 && RWBITS != 128 && RWBITS != 256 && RWBITS != 512)
	#error The register width variable RWBITS must be declared as 64,128,256 or 512 bits.
#endif

#define RW (RWBITS/8)
#define SIZEOF_VEC(T) (RW/sizeof(CTYPE_##T))
#define VW(T) SIZEOF_VEC(T)

#include <stdint.h>
#include <stdarg.h>

// Types definition
#define CTYPE_PD double
#define CTYPE_PS float
#define CTYPE_DI int64_t
#define CTYPE_D int32_t
#define CTYPE_W int16_t
#define CTYPE_B int8_t

// Types definition with "argument promotion" (used by va_arg)
#define CTYPEP_PD double
#define CTYPEP_PS double
#define CTYPEP_DI int64_t
#define CTYPEP_D int32_t
#define CTYPEP_W int32_t
#define CTYPEP_B int32_t

// Types for load/store/conv operations are not the same
// that the ones for mathematical operations. (Why ?)
// This is a conversion table !
#define LSTYPE_PD DF
#define LSTYPE_PS SF
#define LSTYPE_DI DI
#define LSTYPE_D  SI
#define LSTYPE_W  HI 
#define LSTYPE_B  QI 

#define _ALIGNED A
#define _UNALIGNED V

// This is a precomputed version of VW(T), needed for LOAD/STORE function names
// VM_##RWBITS##_##T
#define VW_64_PS 2
#define VW_64_PD 1
#define VW_64_DI 2
#define VW_64_D  2
#define VW_64_W 4
#define VW_64_B 8
#define VW_128_PS 4
#define VW_128_PD 2
#define VW_128_DI 2
#define VW_128_D  4
#define VW_128_W 8
#define VW_128_B 16
#define VW_256_PS 8
#define VW_256_PD 4
#define VW_256_DI 4
#define VW_256_D  8
#define VW_256_W 16
#define VW_256_B 32
#define VW_512_PS 16
#define VW_512_PD 8
#define VW_512_DI 8
#define VW_512_D  16
#define VW_512_W 32
#define VW_512_B 64

#define _DEF_FOR_TYPES(F,P)\
	F(P, PS)\
	F(P, PD)\
	F(P, DI)\
	F(P, D)\
	F(P, W)\
	F(P, B)

// Temporary SIMD equality macros definitions
#define SIMD_EQ_TYPE(A,T) _SIMD_EQ_TYPE(T,RWBITS,A)
#define _SIMD_EQ_TYPE(T,RWB,A) __SIMD_EQ_TYPE(T,RWB,A) // Process the "A" macro
#define __SIMD_EQ_TYPE(T,RWB,A) ___SIMD_EQ_TYPE(T,LSTYPE_##T,VW_##RWB##_##T,A) // Define the VM_XX_XX macro (defined above)
#define ___SIMD_EQ_TYPE(T,LST,VW,A) ____SIMD_EQ_TYPE(T,LST,VW,A) // Process the "VW" and "LST" macros
#define ____SIMD_EQ_TYPE(T,LST,VW,A)\
	void SIMD_EQ##T##_##A##VW##LST(CTYPE_##T dst[VW], CTYPE_##T src[VW])\
	{\
		int i;\
		for (i = 0; i < (VW); i++)\
			dst[i] = src[i];\
	}\
	void SIMD_EQ##T(CTYPE_##T dst[VW], CTYPE_##T src[VW])\
	{\
		int i;\
		for (i = 0; i < (VW); i++)\
			dst[i] = src[i];\
	}

#define SIMD_EQS(A) 	 _DEF_FOR_TYPES(SIMD_EQ_TYPE,A)

#define CTYPE_PD double
#define CTYPE_PS float
#define CTYPE_DI int64_t
#define CTYPE_D int32_t
#define CTYPE_W int16_t
#define CTYPE_B int8_t

// Temporary equality operations
//SIMD_EQS(_ALIGNED)
SIMD_EQS(_UNALIGNED)
