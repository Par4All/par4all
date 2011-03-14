#ifndef _P4A_COMPLEX_H
#define _P4A_COMPLEX_H
/* for native complex handling */

#ifdef P4A_ACCEL_CUDA
#include <vector_types.h>
#include <math_functions.h>
#endif

#ifdef P4A_ACCEL_OPENCL
    #define __builtin_align__(a)
    #define __inline__ inline
    #define __host__ 
    #define __device__
struct __builtin_align__(8) float2{
    float x, y;
};
typedef struct float2 float2;
static __inline__ __host__ __device__ float2 make_float2(float x, float y){
    float2 t; t.x = x; t.y = y; return t;
}
#endif


struct p4a_complexf {
	float2 __v;
	p4a_complexf(float re=0.f, float im=0.f) {
		__v.x=re;
		__v.y=im;
	}
	inline __device__ __host__ float re() const { return __v.x; }
	inline __device__ __host__ float im() const { return __v.y; }
	inline __device__ __host__ float& re() { return __v.x; }
	inline __device__ __host__ float& im() { return __v.y; }

	inline __device__ __host__ p4a_complexf operator*(const p4a_complexf& other) const {
		return p4a_complexf(
			re()*other.re() - im()*other.im(),
			re()*other.im() + im()*other.re()
		);
	}
	inline __device__ __host__ p4a_complexf operator+(const p4a_complexf& other) const {
		return p4a_complexf(
			re()+other.re(),
 			im()+other.im()
		);
	}
	inline __device__ __host__ p4a_complexf operator-() const {
		return p4a_complexf(-re(),-im());
	}
	inline __device__ __host__ p4a_complexf operator=(const p4a_complexf& other) {
		re()=other.re();
		im()=other.im();
		return *this;
	}
	inline __device__ __host__ p4a_complexf operator+=(const p4a_complexf& other) {
		re()+=other.re();
		im()+=other.im();
		return *this;
	}
};
inline __device__ __host__ float crealf(const p4a_complexf& c) { return c.re(); }
inline __device__ __host__ float cimagf(const p4a_complexf& c) { return c.im(); }

inline __device__ __host__ p4a_complexf operator*(float self, const p4a_complexf& other) {
	return p4a_complexf( self*other.re() , self*other.im() );
}
inline __device__ __host__ p4a_complexf operator+(float self, const p4a_complexf& other) {
	return p4a_complexf( self+other.re(), other.im() );
}

inline __device__ __host__ p4a_complexf cexpf(p4a_complexf a) {
	return expf(a.re()) * p4a_complexf( cosf(a.im()), sinf(a.im()) );
}

// it should be a global const variable ... lack of time
#define I p4a_complexf(0.f,1.f)

// to avoid reinclusion of complex.h
#define _COMPLEX_H

#endif
