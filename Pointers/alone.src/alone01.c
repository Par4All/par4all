/* $Id: Hopping_Matrix.c,v 1.48 2009/01/18 12:55:31 urbach Exp $ */

/******************************************
 * Hopping_Matrix is the conventional Wilson 
 * hopping matrix
 *
 * \kappa\sum_{\pm\mu}(r+\gamma_\mu)U_{x,\mu}
 *
 * for ieo = 0 this is M_{eo}, for ieo = 1
 * it is M_{oe}
 *
 * l is the number of the output field
 * k is the number of the input field
 *
 ******************************************/

typedef struct
{
   double re,im;
} complex;
typedef struct 
{
   complex c00,c01,c02,c10,c11,c12,c20,c21,c22;
} su3;

typedef struct
{
   complex c0,c1,c2;
} su3_vector;


typedef struct
{
   su3_vector s0,s1,s2,s3;
} spinor;

typedef struct
{
  su3_vector s0, s1;
} halfspinor;


typedef struct
{
   spinor sp_up,sp_dn;
} bispinor;

#define _vector_sub_assign(r,s) \
   (r).c0.re-=(s).c0.re; \
   (r).c0.im-=(s).c0.im; \
   (r).c1.re-=(s).c1.re; \
   (r).c1.im-=(s).c1.im; \
   (r).c2.re-=(s).c2.re; \
   (r).c2.im-=(s).c2.im;
#define _vector_sub(r,s1,s2) \
   (r).c0.re=(s1).c0.re-(s2).c0.re; \
   (r).c0.im=(s1).c0.im-(s2).c0.im; \
   (r).c1.re=(s1).c1.re-(s2).c1.re; \
   (r).c1.im=(s1).c1.im-(s2).c1.im; \
   (r).c2.re=(s1).c2.re-(s2).c2.re; \
   (r).c2.im=(s1).c2.im-(s2).c2.im;
#define _vector_assign(r,s) \
   (r).c0.re=(s).c0.re; \
   (r).c0.im=(s).c0.im; \
   (r).c1.re=(s).c1.re; \
   (r).c1.im=(s).c1.im; \
   (r).c2.re=(s).c2.re; \
   (r).c2.im=(s).c2.im;
#define _vector_i_sub_assign(r,s) \
   (r).c0.re+=(s).c0.im; \
   (r).c0.im-=(s).c0.re; \
   (r).c1.re+=(s).c1.im; \
   (r).c1.im-=(s).c1.re; \
   (r).c2.re+=(s).c2.im; \
   (r).c2.im-=(s).c2.re;	
#define _vector_i_sub(r,s1,s2)	    \
   (r).c0.re=(s2).c0.re;	\
   (r).c0.re=(s1).c0.re+(s2).c0.im; \
   (r).c0.im=(s1).c0.im-(s2).c0.re; \
   (r).c1.re=(s1).c1.re+(s2).c1.im; \
   (r).c1.im=(s1).c1.im-(s2).c1.re; \
   (r).c2.re=(s1).c2.re+(s2).c2.im; \
   (r).c2.im=(s1).c2.im-(s2).c2.re;
#define _su3_multiply(r,u,s) \
   (r).c0.re= (u).c00.re*(s).c0.re-(u).c00.im*(s).c0.im  \
             +(u).c01.re*(s).c1.re-(u).c01.im*(s).c1.im  \
             +(u).c02.re*(s).c2.re-(u).c02.im*(s).c2.im; \
   (r).c0.im= (u).c00.re*(s).c0.im+(u).c00.im*(s).c0.re  \
             +(u).c01.re*(s).c1.im+(u).c01.im*(s).c1.re  \
             +(u).c02.re*(s).c2.im+(u).c02.im*(s).c2.re; \
   (r).c1.re= (u).c10.re*(s).c0.re-(u).c10.im*(s).c0.im  \
             +(u).c11.re*(s).c1.re-(u).c11.im*(s).c1.im  \
             +(u).c12.re*(s).c2.re-(u).c12.im*(s).c2.im; \
   (r).c1.im= (u).c10.re*(s).c0.im+(u).c10.im*(s).c0.re  \
             +(u).c11.re*(s).c1.im+(u).c11.im*(s).c1.re  \
             +(u).c12.re*(s).c2.im+(u).c12.im*(s).c2.re; \
   (r).c2.re= (u).c20.re*(s).c0.re-(u).c20.im*(s).c0.im  \
             +(u).c21.re*(s).c1.re-(u).c21.im*(s).c1.im  \
             +(u).c22.re*(s).c2.re-(u).c22.im*(s).c2.im; \
   (r).c2.im= (u).c20.re*(s).c0.im+(u).c20.im*(s).c0.re  \
             +(u).c21.re*(s).c1.im+(u).c21.im*(s).c1.re  \
             +(u).c22.re*(s).c2.im+(u).c22.im*(s).c2.re;   
#define _complex_times_vector(r,c,s) \
   (r).c0.re=(c).re*(s).c0.re-(c).im*(s).c0.im; \
   (r).c0.im=(c).re*(s).c0.im+(c).im*(s).c0.re; \
   (r).c1.re=(c).re*(s).c1.re-(c).im*(s).c1.im; \
   (r).c1.im=(c).re*(s).c1.im+(c).im*(s).c1.re; \
   (r).c2.re=(c).re*(s).c2.re-(c).im*(s).c2.im; \
   (r).c2.im=(c).re*(s).c2.im+(c).im*(s).c2.re;
#define _complex_times_vector2(r,c,s) \
   (r).c0.re=(c).re*(s).c0.re-(c).im*(s).c0.im; \
   (r).c1.re=(c).re*(s).c1.re-(c).im*(s).c1.im; \
   (r).c2.re=(c).re*(s).c2.re-(c).im*(s).c2.im; \
   (r).c0.im=(c).re*(s).c0.im+(c).im*(s).c0.re; \
   (r).c1.im=(c).re*(s).c1.im+(c).im*(s).c1.re; \
   (r).c2.im=(c).re*(s).c2.im+(c).im*(s).c2.re;

#define _vector_add_assign(r,s) \
   (r).c0.re+=(s).c0.re; \
   (r).c0.im+=(s).c0.im; \
   (r).c1.re+=(s).c1.re; \
   (r).c1.im+=(s).c1.im; \
   (r).c2.re+=(s).c2.re; \
   (r).c2.im+=(s).c2.im;
#define _vector_i_add_assign(r,s) \
   (r).c0.re-=(s).c0.im; \
   (r).c0.im+=(s).c0.re; \
   (r).c1.re-=(s).c1.im; \
   (r).c1.im+=(s).c1.re; \
   (r).c2.re-=(s).c2.im; \
   (r).c2.im+=(s).c2.re;
#define _vector_i_add(r,s1,s2) \
   (r).c0.re=(s1).c0.re-(s2).c0.im; \
   (r).c0.im=(s1).c0.im+(s2).c0.re; \
   (r).c1.re=(s1).c1.re-(s2).c1.im; \
   (r).c1.im=(s1).c1.im+(s2).c1.re; \
   (r).c2.re=(s1).c2.re-(s2).c2.im; \
   (r).c2.im=(s1).c2.im+(s2).c2.re;

#define _vector_add(r,s1,s2) \
   (r).c0.re=(s1).c0.re+(s2).c0.re; \
   (r).c0.im=(s1).c0.im+(s2).c0.im; \
   (r).c1.re=(s1).c1.re+(s2).c1.re; \
   (r).c1.im=(s1).c1.im+(s2).c1.im; \
   (r).c2.re=(s1).c2.re+(s2).c2.re; \
   (r).c2.im=(s1).c2.im+(s2).c2.im;
#define _complexcjg_times_vector(r,c,s) \
   (r).c0.re=(c).re*(s).c0.re+(c).im*(s).c0.im; \
   (r).c0.im=(c).re*(s).c0.im-(c).im*(s).c0.re; \
   (r).c1.re=(c).re*(s).c1.re+(c).im*(s).c1.im; \
   (r).c1.im=(c).re*(s).c1.im-(c).im*(s).c1.re; \
   (r).c2.re=(c).re*(s).c2.re+(c).im*(s).c2.im; \
   (r).c2.im=(c).re*(s).c2.im-(c).im*(s).c2.re;
#define _su3_inverse_multiply(r,u,s) \
   (r).c0.re= (u).c00.re*(s).c0.re+(u).c00.im*(s).c0.im  \
             +(u).c10.re*(s).c1.re+(u).c10.im*(s).c1.im  \
             +(u).c20.re*(s).c2.re+(u).c20.im*(s).c2.im; \
   (r).c0.im= (u).c00.re*(s).c0.im-(u).c00.im*(s).c0.re  \
             +(u).c10.re*(s).c1.im-(u).c10.im*(s).c1.re  \
             +(u).c20.re*(s).c2.im-(u).c20.im*(s).c2.re; \
   (r).c1.re= (u).c01.re*(s).c0.re+(u).c01.im*(s).c0.im  \
             +(u).c11.re*(s).c1.re+(u).c11.im*(s).c1.im  \
             +(u).c21.re*(s).c2.re+(u).c21.im*(s).c2.im; \
   (r).c1.im= (u).c01.re*(s).c0.im-(u).c01.im*(s).c0.re  \
             +(u).c11.re*(s).c1.im-(u).c11.im*(s).c1.re  \
             +(u).c21.re*(s).c2.im-(u).c21.im*(s).c2.re; \
   (r).c2.re= (u).c02.re*(s).c0.re+(u).c02.im*(s).c0.im  \
             +(u).c12.re*(s).c1.re+(u).c12.im*(s).c1.im  \
             +(u).c22.re*(s).c2.re+(u).c22.im*(s).c2.im; \
   (r).c2.im= (u).c02.re*(s).c0.im-(u).c02.im*(s).c0.re  \
             +(u).c12.re*(s).c1.im-(u).c12.im*(s).c1.re  \
             +(u).c22.re*(s).c2.im-(u).c22.im*(s).c2.re;

#define _memcpy_spinor(s,p)\
	(s).s0.c0.re = (p).s0.c0.re; (s).s0.c0.im = (p).s0.c0.im;\
	(s).s0.c1.re = (p).s0.c1.re; (s).s0.c0.im = (p).s0.c1.im;\
	(s).s0.c2.re = (p).s0.c2.re; (s).s0.c2.im = (p).s0.c2.im;\
	(s).s1.c0.re = (p).s1.c0.re; (s).s1.c0.im = (p).s1.c0.im;\
	(s).s1.c1.re = (p).s1.c1.re; (s).s1.c0.im = (p).s1.c1.im;\
	(s).s1.c2.re = (p).s1.c2.re; (s).s1.c2.im = (p).s1.c2.im;\
	(s).s2.c0.re = (p).s2.c0.re; (s).s2.c0.im = (p).s2.c0.im;\
	(s).s2.c1.re = (p).s2.c1.re; (s).s2.c0.im = (p).s2.c1.im;\
	(s).s2.c2.re = (p).s2.c2.re; (s).s2.c2.im = (p).s2.c2.im;\
	(s).s3.c0.re = (p).s3.c0.re; (s).s3.c0.im = (p).s3.c0.im;\
	(s).s3.c1.re = (p).s3.c1.re; (s).s3.c0.im = (p).s3.c1.im;\
	(s).s3.c2.re = (p).s3.c2.re; (s).s3.c2.im = (p).s3.c2.im;

   ////////////////////////////////////////////////////////////////////////////////////////
/// SEE stuffs ////



/* #define _vector_add_sse__(r,s1,s2) \ */
/* __asm__ __volatile__(\ */
/* 	"movupd %0, %%xmm0 \n\t" \ */
/*     "movupd %1, %%xmm1 \n\t" \ */
/*     "movupd %2, %%xmm2 \n\t" \ */
/* 	"movupd %3, %%xmm3 \n\t" \ */
/*     "movupd %4, %%xmm4 \n\t" \ */
/*     "movupd %5, %%xmm5 \n\t" \ */
/* 	"addpd %%xmm0, %%xmm3 \n\t" \ */
/* 	"addpd %%xmm1, %%xmm4 \n\t" \ */
/* 	"addpd %%xmm2, %%xmm5 \n\t" \ */
/* 	"movupd %%xmm0, %6 \n\t" \ */
/*     "movupd %%xmm1, %7 \n\t" \ */
/*     "movupd %%xmm2, %8 \n\t" \ */
/* 	: \ */
/*     "=m" ((r).c0), "=m" ((r).c1), "=m" ((r).c2) \ */
/*     : \ */
/*     "m" ((s1).c0), "m" ((s1).c1), \ */
/*     "m" ((s1).c2), "m" ((s2).c0), \ */
/*     "m" ((s2).c1), "m" ((s2).c2));  */

/* #define _vector_add_sse(r,s1,s2)     \ */
/*    v_1 = (__m128d *)&(s1);		 \ */
/*    v_2 = (__m128d *)&(s2);		 \ */
/*    v_3 = (__m128d *)&(r);		 \ */
/*    v_3[0] = _mm_add_pd(v_1[0], v_2[0]); \ */
/*    v_3[1] = _mm_add_pd(v_1[1], v_2[1]); \ */
/*    v_3[2] = _mm_add_pd(v_1[2], v_2[2]); */

///////////////////////////////////////////////////////////////////////////////////////////
#define VOLUME 32*16*16*16
//#include <smmintrin.h>
#include <stdio.h>  // for io
#include <sys/time.h> // for create thread 
//#include <libmisc.h>  // for malloc
#include <string.h> // for string copy
////#include <libspe.h> // for create thread 
#include <pthread.h>
#include <stdlib.h> 
#include <errno.h> 
#include <time.h> 
#include <math.h>
#define MAX_NB_THREADS 4  // Fix this to the max in your machine
//#define BIND 1 
#ifdef BIND
 #include <unistd.h>
 //#include <sys/syscall.h>
 #include <sched.h>
 #define __GNU_SOURCE
#endif
#define ALIGNMENT 16
//---------------------------------------------------------------
/* #include <xmmintrin.h>	// Need this for SSE compiler intrinsics */
/* #include <emmintrin.h>	// Need this for SSE2 compiler intrinsics */
/* #include <pmmintrin.h>	// Need this for SSE3 compiler intrinsics */
/* #include <tmmintrin.h>	// Need this for SSE3 compiler intrinsics */
/* #include <smmintrin.h>	// Need this for SSE4 compiler intrinsics */

#include <math.h>		// Needed for sqrt in CPU-only version
#include <malloc.h>
//#include "alone.h"
//---------------------------------------------------------------
////////////////////////////////////////////////////////////////////////////////////////////
/* align_size has to be a power of two !! */
void *aligned_malloc(size_t size, size_t align_size) {

  char *ptr,*ptr2,*aligned_ptr;
  int align_mask = align_size - 1;

  ptr=(char *)malloc(size + align_size + sizeof(int));
  if(ptr==NULL) return(NULL);

  ptr2 = ptr + sizeof(int);
  aligned_ptr = ptr2 + (align_size - ((size_t)ptr2 & align_mask));


  ptr2 = aligned_ptr - sizeof(int);
  *((int *)ptr2)=(int)(aligned_ptr - ptr);

  return((void *)aligned_ptr);
}


