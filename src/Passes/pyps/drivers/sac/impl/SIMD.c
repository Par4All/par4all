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

// Operations definition
#define OP_ADD +
#define OP_SUB -
#define OP_DIV /
#define OP_MUL *

// Double-argument simple operaters definition
#define OP_F_TYPE(P, T)\
	CTYPE_##T F_##P##T(int i, va_list ap)\
	{\
		CTYPE_##T *v1,*v2,r;\
		v1 = va_arg(ap, CTYPE_##T *);\
		v2 = va_arg(ap, CTYPE_##T *);\
		r = v1[i] OP_##P v2[i];\
		return r;\
	}

// Muladd uses three parameters
// P is unused but present so that _DEF_FOR_TYPES can be used
#define OP_MULADD_TYPE(P,T)\
	CTYPE_##T F_MULADD##T(int i, va_list ap)\
	{\
		CTYPE_##T *v1,*v2,*v3;\
		v1 = va_arg(ap, CTYPE_##T *);\
		v2 = va_arg(ap, CTYPE_##T *);\
		v3 = va_arg(ap, CTYPE_##T *);\
		return v1[i] + v2[i]*v3[i];\
	}

// Unary-minus operation
// P is unused but present so that _DEF_FOR_TYPES can be used
#define OP_UMIN_TYPE(P,T)\
	CTYPE_##T F_UMIN##T(int i, va_list ap)\
	{\
		CTYPE_##T *v1;\
		v1 = va_arg(ap, CTYPE_##T *);\
		return - (v1[i]);\
	}

// SIMD operation macro definition
#define SIMD_OP_TYPE(P,T)\
	void SIMD_##P##T(CTYPE_##T *dst, ...)\
	{\
		int i;\
		va_list ap,ap_f;\
		va_start(ap,dst);\
		for (i = 0; i < (VW(T)); i++)\
		{\
			va_copy(ap_f,ap);\
			dst[i] = F_##P##T(i,ap_f);\
		}\
		va_end(ap);\
	}

#define _DEF_FOR_TYPES(F,P)\
	F(P, PS)\
	F(P, PD)\
	F(P, DI)\
	F(P, D)\
	F(P, W)\
	F(P, B)

#define SIMD_OP(P) _DEF_FOR_TYPES(SIMD_OP_TYPE,P)
#define OP_F(P) _DEF_FOR_TYPES(OP_F_TYPE,P)


// SIMD load/store macro definition
#define _ALIGNED A
#define _UNALIGNED V

#define SIMD_LOAD_TYPE(A,T) _SIMD_LOAD_TYPE(T,RWBITS,A)
#define _SIMD_LOAD_TYPE(T,RWB,A) __SIMD_LOAD_TYPE(T,RWB,A) // Process the "A" macro
#define __SIMD_LOAD_TYPE(T,RWB,A) ___SIMD_LOAD_TYPE(T,LSTYPE_##T,VW_##RWB##_##T,A) // Define the VM_XX_XX macro (defined above)
#define ___SIMD_LOAD_TYPE(T,LST,VW,A) ____SIMD_LOAD_TYPE(T,LST,VW,A) // Process the "VW" and "LST" macros
#define ____SIMD_LOAD_TYPE(T,LST,VW,A)\
	void SIMD_LOAD_##A##VW##LST(CTYPE_##T vec[VW], CTYPE_##T base[VW])\
	{\
		int i;\
		for (i = 0; i < (VW); i++)\
			vec[i] = base[i];\
	}\
	\
	void SIMD_LOAD_GENERIC_##A##VW##LST(CTYPE_##T vec[VW], ...)\
	{\
		int i;\
		va_list ap;\
		CTYPE_##T n;\
		va_start(ap, vec);\
		for (i = 0; i < (VW); i++)\
		{\
			n = (CTYPE_##T) va_arg(ap, CTYPEP_##T);\
			vec[i] = n;\
		}\
		va_end(ap);\
	}\
    \
    void SIMD_LOAD_BROADCAST_##A##VW##LST(CTYPE_##T vec[VW], CTYPE_##T base)\
    {\
		int i;\
		for (i = 0; i < (VW); i++)\
			vec[i] = base;\
	}\
    

#define SIMD_STORE_TYPE(A,T) _SIMD_STORE_TYPE(T,RWBITS,A)
#define _SIMD_STORE_TYPE(T,RWB,A) __SIMD_STORE_TYPE(T,RWB,A) // Process the "A" macro
#define __SIMD_STORE_TYPE(T,RWB,A) ___SIMD_STORE_TYPE(T,LSTYPE_##T,VW_##RWB##_##T,A) // Define the VM_XX_XX macro
#define ___SIMD_STORE_TYPE(T,LST,VW,A) ____SIMD_STORE_TYPE(T,LST,VW,A) // Process the "VW" and "LST" macro
#define ____SIMD_STORE_TYPE(T,LST,VW,A)\
	void SIMD_STORE_##A##VW##LST(CTYPE_##T vec[VW], CTYPE_##T base[VW])\
	{\
		int i;\
		for (i = 0; i < (VW); i++)\
			base[i] = vec[i];\
	}\
	\
	void SIMD_STORE_GENERIC_##A##VW##LST(CTYPE_##T vec[VW], ...)\
	{\
		int i;\
		va_list ap;\
		CTYPE_##T *pn;\
		va_start(ap, vec);\
		for (i = 0; i < (VW); i++)\
		{\
			pn = va_arg(ap, CTYPE_##T *);\
			*pn = vec[i];\
		}\
		va_end(ap);\
	}

// SIMD zero macros definitions
#define SIMD_ZERO_TYPE(A,T) _SIMD_ZERO_TYPE(T,RWBITS,A)
#define _SIMD_ZERO_TYPE(T,RWB,A) __SIMD_ZERO_TYPE(T,RWB,A) // Process the "A" macro
#define __SIMD_ZERO_TYPE(T,RWB,A) ___SIMD_ZERO_TYPE(T,LSTYPE_##T,VW_##RWB##_##T,A) // Define the VM_XX_XX macro (defined above)
#define ___SIMD_ZERO_TYPE(T,LST,VW,A) ____SIMD_ZERO_TYPE(T,LST,VW,A) // Process the "VW" and "LST" macros
#define ____SIMD_ZERO_TYPE(T,LST,VW,A)\
	void SIMD_ZERO_##A##VW##LST(CTYPE_##T vec[VW])\
	{\
		int i;\
		for (i = 0; i < (VW); i++)\
			vec[i] = 0;\
	}

// Shuffle function
#define SIMD_SHUFFLE_TYPE(A,T) _SIMD_SHUFFLE_TYPE(T,RWBITS,A)
#define _SIMD_SHUFFLE_TYPE(T,RWB,A) __SIMD_SHUFFLE_TYPE(T,RWB,A)
#define __SIMD_SHUFFLE_TYPE(T,RWB,A) ___SIMD_SHUFFLE_TYPE(T,LSTYPE_##T,VW_##RWB##_##T,A)
#define ___SIMD_SHUFFLE_TYPE(T,LST,VW,A) ____SIMD_SHUFFLE_TYPE(T,LST,VW,A)
#define ____SIMD_SHUFFLE_TYPE(T,LST,VW,A) \
	void SIMD_SHUFFLE_V##VW##LST(CTYPE_##T res[VW], CTYPE_##T vec[VW], ...)\
	{\
		int i;\
		int p;\
		va_list ap;\
		va_start(ap, vec);\
		for (i = 0; i < (VW); i++)\
		{\
			p = va_arg(ap, int);\
			res[p] = vec[i];\
		}\
	}

// Invert function
#define SIMD_INVERT_TYPE(A,T) _SIMD_INVERT_TYPE(T,RWBITS,A)
#define _SIMD_INVERT_TYPE(T,RWB,A) __SIMD_INVERT_TYPE(T,RWB,A)
#define __SIMD_INVERT_TYPE(T,RWB,A) ___SIMD_INVERT_TYPE(T,LSTYPE_##T,VW_##RWB##_##T,A)
#define ___SIMD_INVERT_TYPE(T,LST,VW,A) ____SIMD_INVERT_TYPE(T,LST,VW,A)
#define ____SIMD_INVERT_TYPE(T,LST,VW,A) \
	void SIMD_INVERT_V##VW##LST(CTYPE_##T res[VW], CTYPE_##T vec[VW])\
	{\
		int i;\
		for (i = 0; i < (VW); i++)\
			res[VW-i-1] = vec[i];\
	}

// Conversion functions

/* TO: original type
   TD: destination type
   RWD: register width in bits
   VWD: destination type vector length */
#define SIMD_LOAD_CONV(A,TO,TD) _SIMD_LOAD_CONV(A,TO,TD,RWBITS)
#define _SIMD_LOAD_CONV(A,TO,TD,RWB) __SIMD_LOAD_CONV(A, TO, TD, RWB)
#define __SIMD_LOAD_CONV(A,TO,TD,RWB) ___SIMD_LOAD_CONV(A,TO,TD,VW_##RWB##_##TD,LSTYPE_##TO,LSTYPE_##TD)
#define ___SIMD_LOAD_CONV(A,TO,TD,VWD,TOLST,TDLST) ____SIMD_LOAD_CONV(A,TO,TD,VWD,TOLST,TDLST)
#define ____SIMD_LOAD_CONV(A,TO,TD,VWD,TOLST,TDLST)\
	void SIMD_LOAD_##A##VWD##TOLST##_TO_##A##VWD##TDLST(CTYPE_##TD dst[VWD], CTYPE_##TO src[VWD])\
	{\
		int i;\
		for (i = 0; i < VWD; i++)\
			dst[i] = src[i];\
	}\
	\
	void SIMD_LOAD_GENERIC_##A##VWD##TOLST##_TO_##A##VWD##TDLST(CTYPE_##TD vec[VWD], ...)\
	{\
		int i;\
		va_list ap;\
		CTYPE_##TO n;\
		va_start(ap, vec);\
		for (i = 0; i < (VWD); i++)\
		{\
			n = (CTYPE_##TO) va_arg(ap, CTYPEP_##TO);\
			vec[i] = n;\
		}\
		va_end(ap);\
	}

#define SIMD_STORE_CONV(A,TO,TD) _SIMD_STORE_CONV(A,TO,TD,RWBITS)
#define _SIMD_STORE_CONV(A,TO,TD,RWB) __SIMD_STORE_CONV(A, TO, TD, RWB)
#define __SIMD_STORE_CONV(A,TO,TD,RWB) ___SIMD_STORE_CONV(A,TO,TD,VW_##RWB##_##TD,LSTYPE_##TO,LSTYPE_##TD)
#define ___SIMD_STORE_CONV(A,TO,TD,VWD,TOLST,TDLST) ____SIMD_STORE_CONV(A,TO,TD,VWD,TOLST,TDLST)
#define ____SIMD_STORE_CONV(A,TD,TO,VWD,TDLST,TOLST)\
	void SIMD_STORE_##A##VWD##TOLST##_TO_##A##VWD##TDLST(CTYPE_##TO src[VWD], CTYPE_##TD dst[VWD])\
	{\
		int i;\
		for (i = 0; i < VWD; i++)\
			dst[i] = src[i];\
	}\
	void SIMD_STORE_##A##VWD##TDLST##_TO_##A##VWD##TOLST(CTYPE_##TD src[VWD], CTYPE_##TO dst[VWD])\
	{\
		int i;\
		for (i = 0; i < VWD; i++)\
			dst[i] = src[i];\
	}

#define SIMD_LOADS(A)	 _DEF_FOR_TYPES(SIMD_LOAD_TYPE,A)
#define SIMD_STORES(A) 	 _DEF_FOR_TYPES(SIMD_STORE_TYPE,A)
#define SIMD_ZEROS(A) 	 _DEF_FOR_TYPES(SIMD_ZERO_TYPE,A)
#define SIMD_SHUFFLES(A) _DEF_FOR_TYPES(SIMD_SHUFFLE_TYPE,A)
#define SIMD_INVERTS(A)  _DEF_FOR_TYPES(SIMD_INVERT_TYPE,A)

#define CTYPE_PD double
#define CTYPE_PS float
#define CTYPE_DI int64_t
#define CTYPE_D int32_t
#define CTYPE_W int16_t
#define CTYPE_B int8_t

#define _DEF_ALL_CONV(F,A) \
	F(A, PS, PD)\
	F(A, DI, PD)\
	F(A, D, PD)\
	F(A, W, PD)\
	F(A, B, PD)\
	F(A, D, PS)\
	F(A, W, PS)\
	F(A, B, PS)\
	F(A, D, DI)\
	F(A, W, DI)\
	F(A, B, DI)\
	F(A, W, D)\
	F(A, D, W)\
	F(A, B, D)\
	F(A, B, W)

#define SIMD_LOAD_CONVS(A) _DEF_ALL_CONV(SIMD_LOAD_CONV,A)
#define SIMD_STORE_CONVS(A) _DEF_ALL_CONV(SIMD_STORE_CONV,A)

// Declare operation functions
OP_F(ADD)
OP_F(MUL)
OP_F(DIV)
OP_F(SUB)
_DEF_FOR_TYPES(OP_MULADD_TYPE,__UNUSED__)
_DEF_FOR_TYPES(OP_UMIN_TYPE,__UNUSED__)

// SIMD functions
SIMD_OP(ADD)
SIMD_OP(MUL)
SIMD_OP(DIV)
SIMD_OP(SUB)
SIMD_OP(MULADD)
SIMD_OP(UMIN)

// LOAD operations
SIMD_LOADS(_ALIGNED)
SIMD_LOADS(_UNALIGNED)

// STORE operations
SIMD_STORES(_ALIGNED)
SIMD_STORES(_UNALIGNED)

// ZERO operations
SIMD_ZEROS(_ALIGNED)
SIMD_ZEROS(_UNALIGNED)

// Shuffle operations (_aligned unused)
SIMD_SHUFFLES(_ALIGNED) 

// Invert operations (_aligned unused)
SIMD_INVERTS(_ALIGNED) 

// Define all possible conversions
SIMD_LOAD_CONVS(_UNALIGNED)
SIMD_LOAD_CONVS(_ALIGNED)

SIMD_STORE_CONVS(_UNALIGNED)
SIMD_STORE_CONVS(_ALIGNED)

