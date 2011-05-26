#include <arm_neon.h>


/* Uses 128-bits NEON instructions.
Notes :
 * NEON can also operate on 64-bits vectors.
 * NEON does not operate on double-precision float. However, VFP can work on double-precision 64-bits vectors,
   but VFP is not a simd unit : it processes vectors scalar by scalar.
 * __reminder__ : NEON data types : signed/unsigned 8-bit, 16-bit, 32-bit, 64-bit, single precision floating point
 * TODO: alignement: [1] says that each instruction has an alignement offset argument., but I can't find it in
   the intrinsics... An other option is to used an isntruction that set the alignement offset before each call,
   but it sounds like wasted cycles...
 * TODO: a feature of NEON is to load/store up to 4 vectors with just one instruction
   (see vst{1,2,3,4}/vld{1,2,3,4} variants and [2] for examples). This needs (I think) a modification in SAC.
 * [2] is a nice summary of the intrinsics used here

 [1] : http://infocenter.arm.com/help/topic/com.arm.doc.dui0489b/CIHGIAEH.html
 [2] : http://gcc.gnu.org/onlinedocs/gcc/ARM-NEON-Intrinsics.html
 */


typedef float32_t	a4sf[8] __attribute__ ((aligned (32)));
typedef int64_t		a2di[4] __attribute__ ((aligned (32)));
typedef int32_t		a4si[8] __attribute__ ((aligned (32)));
typedef int16_t		a8hi[16] __attribute__ ((aligned (32)));
typedef int8_t		a16qi[32] __attribute__ ((aligned (32)));

typedef float32x4_t	v4sf;
typedef int64x2_t	v2di;
typedef int32x4_t	v4si;
typedef int16x8_t	v8hi;
typedef int8x16_t	v16qi;

/* float */
#define SIMD_LOAD_V4SF(vec,arr) vec=vld1q_f32(arr)
#define SIMD_LOADA_V4SF(vec,arr) vec=vld1q_f32(arr)
#define SIMD_MULPS(vec1,vec2,vec3) vec1=vmulq_f32(vec2,vec3)
#define SIMD_DIVPS(vec1,vec2,vec3)\
	do {\
	vec3=vrecpeq_f32(vec3);\
	vec1=vmulq_f32(vec2,vec3);\
	}\
	while (0)

#define SIMD_ADDPS(vec1,vec2,vec3) vec1=vaddq_f32(vec2,vec3)
#define SIMD_SUBPS(vec1, vec2, vec3) vec1=vsubq_f32(vec2, vec3)
#define SIMD_MULADDPS(vec1, vec2, vec3, vec4) vec1=vmlaq_f32(vec2,vec3,vec4)
#define SIMD_UMINPS(vec1, vec2)	vec1=vnegq_f32(vec2)
#define SIMD_STORE_V4SF(vec,arr) vst1q_f32(arr,vec)
#define SIMD_STOREA_V4SF(vec,arr) vst1q_f32(arr,vec)
#define SIMD_STORE_GENERIC_V4SF(vec,v0,v1,v2,v3)			\
		do {								\
		float __pips_tmp[4] __attribute__ ((aligned (16)));	\
		SIMD_STOREA_V4SF(vec,&__pips_tmp[0]);			\
		*(v0)=__pips_tmp[0];					\
		*(v1)=__pips_tmp[1];					\
		*(v2)=__pips_tmp[2];					\
		*(v3)=__pips_tmp[3];					\
		} while (0)

#define SIMD_ZERO_V4SF(vec) vec = vsubq_f32(vec,vec)

#define SIMD_LOAD_GENERIC_V4SF(vec,v0,v1,v2,v3)				\
		do {								\
		float __pips_v[4] __attribute ((aligned (16)));\
		__pips_v[0]=v0;\
		__pips_v[1]=v1;\
		__pips_v[2]=v2;\
		__pips_v[3]=v3;\
		SIMD_LOADA_V4SF(vec,&__pips_v[0]);			\
		} while(0)

/* handle padded value, this is a very bad implementation ... */
#define SIMD_STORE_MASKED_V4SF(vec,arr)					\
		do {								\
		float __pips_tmp[4] __attribute__ ((aligned (16)));					\
		SIMD_STOREA_V4SF(vec,&__pips_tmp[0]);			\
		(arr)[0] = __pips_tmp[0];				\
		(arr)[1] = __pips_tmp[1];				\
		(arr)[2] = __pips_tmp[2];				\
		} while(0)

#define SIMD_LOAD_V4SI_TO_V4SF(v, f)		\
		do {					\
		float __pips_tmp[4];		\
		__pips_tmp[0] = (f)[0];		\
		__pips_tmp[1] = (f)[1];		\
		__pips_tmp[2] = (f)[2];		\
		__pips_tmp[3] = (f)[3];		\
		SIMD_LOAD_V4SF(v, __pips_tmp);	\
		} while(0)

/* int64_t */
#define SIMD_LOAD_V2DI(vec,arr) vec=vld1q_s64(arr)
#define SIMD_STORE_V2DI(vec,arr) vst1q_s64(arr,vec)

#define SIMD_ZERO_V2DI(vec) vec = veorq_s64(vec,vec)

#define SIMD_ADDDI(v1,v2,v3) v1=vaddq_s64(v2,v3)
#define SIMD_SUBDI(v1,v2,v3) v1=vsubq_s64(v2,v3)
#define SIMD_DIVDI(vec1,vec2,vec3)\
	do {\
	vec3=vrecpeq_s64(vec3);\
	vec1=vmulq_s64(vec2,vec3);\
	}\
	while (0)
#define SIMD_MULDI(v1,v2,v3) v1=vmulq_s64(v2,v3)
#define SIMD_MULADDDI(vec1, vec2, vec3, vec4) vec1=vmlaq_s64(vec2,vec3,vec4)

/* int32_t */
#define SIMD_LOAD_V4SI(vec,arr) vec=vld1q_s32(arr)
#define SIMD_STORE_V4SI(vec,arr) vst1q_s32(arr,vec)

#define SIMD_ZERO_V4SI(vec) vec = veorq_s32(vec,vec)

#define SIMD_ADDD(v1,v2,v3) v1=vaddq_s32(v2,v3)
#define SIMD_SUBD(v1,v2,v3) v1=vsubq_s32(v2,v3)
#define SIMD_DIVD(vec1,vec2,vec3)\
	do {\
	vec3=vrecpeq_s32(vec3);\
	vec1=vmulq_s32(vec2,vec3);\
	}\
	while (0)
#define SIMD_MULD(v1,v2,v3) v1=vmulq_s32(v2,v3)
#define SIMD_MULADDD(vec1, vec2, vec3, vec4) vec1=vmlaq_s32(vec2,vec3,vec4)

/* int16_t */
#define SIMD_LOAD_V8HI(vec,arr) vec=vld1q_s16(arr)
#define SIMD_STORE_V8HI(vec,arr) vst1q_s16(arr,vec)

#define SIMD_ZERO_V8HI(vec) vec = veorq_s16(vec,vec)

#define SIMD_ADDHI(v1,v2,v3) v1=vaddq_s16(v2,v3)
#define SIMD_SUBHI(v1,v2,v3) v1=vsubq_s16(v2,v3)
#define SIMD_DIVHI(vec1,vec2,vec3)\
	do {\
	vec3=vrecpeq_s16(vec3);\
	vec1=vmulq_s16(vec2,vec3);\
	}\
	while (0)
#define SIMD_MULHI(v1,v2,v3) v1=vmulq_s16(v2,v3)

#define SIMD_STORE_V8HI_TO_V8SI(vec,arr)\
	SIMD_STORE_V8HI(vec,arr)
#define SIMD_LOAD_V8SI_TO_V8HI(vec,arr)\
	SIMD_LOAD_V8HI(vec,arr)

#define SIMD_MULADDHI(vec1, vec2, vec3, vec4) vec1=vmlaq_s16(vec2,vec3,vec4)

/* int8_t */
#define SIMD_LOAD_V16QI(vec,arr) vec=vld1q_s8(arr)
#define SIMD_STORE_V16QI(vec,arr) vst1q_s8(arr,vec)

#define SIMD_ZERO_V16QI(vec) vec = veorq_s8(vec,vec)

#define SIMD_ADDQI(v1,v2,v3) v1=vaddq_s8(v2,v3)
#define SIMD_SUBQI(v1,v2,v3) v1=vsubq_s8(v2,v3)
#define SIMD_DIVQI(vec1,vec2,vec3)\
	do {\
	vec3=vrecpeq_s8(vec3);\
	vec1=vmulq_s8(vec2,vec3);\
	}\
	while (0)
#define SIMD_MULQI(v1,v2,v3) v1=vmulq_s8(v2,v3)

#define SIMD_MULADDQI(vec1, vec2, vec3, vec4) vec1=vmlaq_s8(vec2,vec3,vec4)
