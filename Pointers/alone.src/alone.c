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
#define _vector_add_sse__(r,s1,s2) \
__asm__ __volatile__(\
	"movupd %0, %%xmm0 \n\t" \
    "movupd %1, %%xmm1 \n\t" \
    "movupd %2, %%xmm2 \n\t" \
	"movupd %3, %%xmm3 \n\t" \
    "movupd %4, %%xmm4 \n\t" \
    "movupd %5, %%xmm5 \n\t" \
	"addpd %%xmm0, %%xmm3 \n\t" \
	"addpd %%xmm1, %%xmm4 \n\t" \
	"addpd %%xmm2, %%xmm5 \n\t" \
	"movupd %%xmm0, %6 \n\t" \
    "movupd %%xmm1, %7 \n\t" \
    "movupd %%xmm2, %8 \n\t" \
	: \
    "=m" ((r).c0), "=m" ((r).c1), "=m" ((r).c2) \
    : \
    "m" ((s1).c0), "m" ((s1).c1), \
    "m" ((s1).c2), "m" ((s2).c0), \
    "m" ((s2).c1), "m" ((s2).c2)); 

#define _vector_add_sse(r,s1,s2)     \
   v_1 = (__m128d *)&(s1);		 \
   v_2 = (__m128d *)&(s2);		 \
   v_3 = (__m128d *)&(r);		 \
   v_3[0] = _mm_add_pd(v_1[0], v_2[0]); \
   v_3[1] = _mm_add_pd(v_1[1], v_2[1]); \
   v_3[2] = _mm_add_pd(v_1[2], v_2[2]);

///////////////////////////////////////////////////////////////////////////////////////////
#define VOLUME 32*16*16*16
#include <smmintrin.h>
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
#include <xmmintrin.h>	// Need this for SSE compiler intrinsics
#include <emmintrin.h>	// Need this for SSE2 compiler intrinsics
#include <pmmintrin.h>	// Need this for SSE3 compiler intrinsics
#include <tmmintrin.h>	// Need this for SSE3 compiler intrinsics
#include <smmintrin.h>	// Need this for SSE4 compiler intrinsics

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

void aligned_free(void *ptr) {

  int *ptr2=(int *)ptr - 1;
  ptr -= *ptr2;
  free(ptr);
}
///////////////////////////////////////////////////

int ** g_iup;
int ** g_idn;
int * iup;
int * idn;
int * g_lexic2eosub;
int * g_eo2lexic;
su3 *U0;
int *loop;
int nb_cores=1;
int ieo;
spinor *l;
spinor *k;
spinor *copy;
unsigned int *site_dep;
int cyclic;
unsigned int nb_site_block, nb_site_dep_block, nb_block_total;
unsigned int *block_dep_id;
//// Claude ////
spinor * spinor_copy; /* 8-time expanded spinor for contiguous dependencies */
unsigned int * spinor_copy_indices_e; /* 8xV matrix for dependence indices */
unsigned int * spinor_copy_indices_o; /* 8xV matrix for dependence indices */
////////////////////////////////////////////////////////////
unsigned int endian()
{
	/* Check the endianess of the machine 0: little endian  1: big endian */
long x = 0x34333231;
char *y = (char *) &x;
unsigned int ind = 0;

if(strncmp(y,"1234",4))
 ind = 1; // printf("Big Endian");
else
 ind = 0; // printf("little Endian");
return ind;
}



void byte_swap(void * ptr, int nmemb){
  int j;
  char char_in[4];
  char * in_ptr;
  int * int_ptr;

  for(j = 0, int_ptr = (int *) ptr; j < nmemb; j++, int_ptr++){
    in_ptr = (char *) int_ptr;
    
    char_in[0] = in_ptr[0];
    char_in[1] = in_ptr[1];
    char_in[2] = in_ptr[2];
    char_in[3] = in_ptr[3];

    in_ptr[0] = char_in[3];
    in_ptr[1] = char_in[2];
    in_ptr[2] = char_in[1];
    in_ptr[3] = char_in[0];
  }
}
double now(){
   struct timeval t; double f_t;
   gettimeofday(&t, NULL); 
   f_t = t.tv_usec; f_t = f_t/1000000.0; f_t +=t.tv_sec;
   return f_t; 
}
int SwapEndianInt(int val)
{   int a;
    int b;
    unsigned int i,n = sizeof(int);
    a = val; b = val;
    unsigned char *src = (unsigned char *)&a;
    unsigned char *dst = (unsigned char *)&b;
    for(i=0;i<n;i++) dst[i]=src[n-i-1];
    return b;
}
double SwapEndianDouble(double val)
{   double a;
    double b;
    unsigned int i,n = sizeof(double);
    a = val; b = val;
    unsigned char *src = (unsigned char *)&a;
    unsigned char *dst = (unsigned char *)&b;
    for(i=0;i<n;i++) dst[i]=src[n-i-1];
    return b;
}

float SwapEndianFloat(float val)
{   float a;
    float b;
    unsigned int i,n = sizeof(float);
    a = val; b = val;
    unsigned char *src = (unsigned char *)&a;
    unsigned char *dst = (unsigned char *)&b;
    for(i=0;i<n;i++) dst[i]=src[n-i-1];
    return b;
}

void readInput(){
	//g_iup = [V][4] from iup  
    //g_idn = [V][4] from idn

 FILE *fp;
 fp=fopen("config_data", "r");
 fread(U0, sizeof(su3), 8*VOLUME, fp);
 fread(iup, sizeof(int), 4*VOLUME, fp);
 fread(idn, sizeof(int), 4*VOLUME, fp);
 fread(g_lexic2eosub, sizeof(int), VOLUME, fp);
 fread(g_eo2lexic, sizeof(int), VOLUME, fp);
 fclose(fp);
 if(endian()!=0){
  int i,n;
  double *d;
  int *e;
  d = (double *)U0; n=(8*VOLUME)*sizeof(su3)/8; for(i=0;i<n;i++) d[i]=SwapEndianDouble(d[i]);
  e = iup; n=(4*VOLUME); for(i=0;i<n;i++) e[i]=SwapEndianInt(e[i]);
  e = idn; n=(4*VOLUME); for(i=0;i<n;i++) e[i]=SwapEndianInt(e[i]);
  e = g_lexic2eosub; n=(VOLUME); for(i=0;i<n;i++) e[i]=SwapEndianInt(e[i]);
  e = g_eo2lexic; n=(VOLUME); for(i=0;i<n;i++) e[i]=SwapEndianInt(e[i]);
 }
 ////
 /*fp=fopen("config_data", "w");
 fwrite(U0, sizeof(double), (sizeof(su3)/sizeof(double))*8*VOLUME, fp);
 fwrite(iup, sizeof(int), 4*VOLUME, fp);
 fwrite(idn, sizeof(int), 4*VOLUME, fp);
 fwrite(g_lexic2eosub, sizeof(int), VOLUME, fp);
 fwrite(g_eo2lexic, sizeof(int), VOLUME, fp);
 fclose(fp);*/
 ////
}



void ShuffeSpinor(spinor *SP, spinor *P, int i_start, int V){
/* 
 This routine copy a spinor field P of size V into an exented spinor field SP of size 8*V
 such that for each i in [0...V], associated spinors of P[i] are SP[8*i]...SP[8*I+7].
 This ensure contiguous read access, mainly useful for reducing the number of data transfers 
 (not the volume of data transfered!) 
 Author: Claude Tadonki (claude.tadonki@u-psud.fr)
 Date: December 14, 2009
*/
int i,j,ix,s;
const size_t spinor_size = sizeof(spinor); 
unsigned int *indice_ptr;

int step=0;

printf("Start shuffling\n");
if(step==0){ /* Naïve implementation of the copy */
	for(i=i_start;i<i_start+V;i++){
		j = 8*(i-i_start); ix = 4*g_eo2lexic[i]; 
		memcpy(&SP[j],  &P[g_lexic2eosub[iup[ix+0]]],spinor_size);
		memcpy(&SP[j+1],&P[g_lexic2eosub[idn[ix+0]]],spinor_size);
		memcpy(&SP[j+2],&P[g_lexic2eosub[iup[ix+1]]],spinor_size);
		memcpy(&SP[j+3],&P[g_lexic2eosub[idn[ix+1]]],spinor_size);
		memcpy(&SP[j+4],&P[g_lexic2eosub[iup[ix+2]]],spinor_size);
		memcpy(&SP[j+5],&P[g_lexic2eosub[idn[ix+2]]],spinor_size);
		memcpy(&SP[j+6],&P[g_lexic2eosub[iup[ix+3]]],spinor_size);
		memcpy(&SP[j+7],&P[g_lexic2eosub[idn[ix+3]]],spinor_size);
		/*printf("%4d %4d %4d %4d %4d %4d %4d %4d\n",g_lexic2eosub[iup[ix+0]],g_lexic2eosub[idn[ix+0]],g_lexic2eosub[iup[ix+1]],g_lexic2eosub[idn[ix+1]],
			g_lexic2eosub[iup[ix+2]],g_lexic2eosub[idn[ix+2]],g_lexic2eosub[iup[ix+3]],g_lexic2eosub[idn[ix+3]]);*/
	}
}else{ /* Cache-optimal implementation of the copy */
	if(i_start==0) indice_ptr = spinor_copy_indices_e; else indice_ptr = spinor_copy_indices_o; 
	for(i=0;i<V;i++){
		for(j=0;j<8;j++){
          s = indice_ptr[8*i+j]; /* A copy of this spinor s should go to location t */
          memcpy(&SP[s],&P[i],spinor_size);
		}
	}
}
printf("End shuffling\n");
}

int copy_ieo, copy_ioff, copy_ioff2;
void print_su3(su3 u){
 printf("%10.9f + %10.9fi ",u.c00.re,u.c00.im);printf("%10.9f + %10.9fi ",u.c01.re,u.c01.im);printf("%10.9f + %10.9fi\n",u.c02.re,u.c02.im);
 printf("%10.9f + %10.9fi ",u.c10.re,u.c10.im);printf("%10.9f + %10.9fi ",u.c11.re,u.c11.im);printf("%10.9f + %10.9fi\n",u.c12.re,u.c12.im);
 printf("%10.9f + %10.9fi ",u.c20.re,u.c20.im);printf("%10.9f + %10.9fi ",u.c21.re,u.c21.im);printf("%10.9f + %10.9fi\n",u.c22.re,u.c22.im);
}

int find(unsigned int *V, unsigned int N, unsigned int index){
 unsigned int i;
 int t;
 t = -1;
 for(i=0;i<N;i++)if(V[i]==index){t=i; i=N;}
 
}
//////////////////////// CLAUDE MULTITHREAD ///////////////////////////////////////////////
int index_a(int t, int x, int y, int z, int LX, int LY, int LZ){
/* Provides the absolute lexicographic index of (t, x, y, z)
   Useful to walk over the blocks, maybe could be just g_ipt[t][x][y][z]
   Claude Tadonki (claude.tadonki@u-psud.fr)
*/
 return ((t*LX + x)*LY + y)*(LZ) + z;
}
void build_block_loop(int is_even_odd, int ieo, int LT, int LX, int LY, int LZ, int tile_dim, int *loop){
/* This routines generates a tiled iteration space tile_volume = tile_size^4 */  
   int x, y, z, t, i, j, k;
   int bx, by, bz, bt;
   int dT, dX, dY, dZ;
   dT = LT / tile_dim;  dX = LX / tile_dim;
   dY = LY / tile_dim;  dZ = LZ / tile_dim;
   i = 0; j = 0;
   for (t = 0; t < dT; t++) 
   for (x = 0; x < dX; x++) 
   for (y = 0; y < dY; y++) 
   for (z = 0; z < dZ; z++)
	  for(bt = 0; bt < tile_dim; bt++)
	  for(bx = 0; bx < tile_dim; bx++)
	  for(by = 0; by < tile_dim; by++)
	  for(bz = 0; bz < tile_dim; bz++){
		  k = (tile_dim*t + bt) + (tile_dim*x + bx) + (tile_dim*y + by) + (tile_dim*z + bz);
		  j = k%2; 
		  if((!is_even_odd)||(j==ieo)){
			  k = index_a(tile_dim*t + bt, tile_dim*x + bx, tile_dim*y + by, tile_dim*z + bz, LX, LY, LZ);
			  if(is_even_odd){if(k%2==0) k = k/2; else k = (k-1)/2; }
			  loop[i] = k;
		      i++; 
		  }
	  }
}
void build_block_dep_id(int is_even_odd, int *loop, int tile_dim,int N, unsigned int *site_dep){
 int i,j,t,ix,f,td,d,nn,m;
 spinor *c;
 unsigned int local_dep[8];
 td = tile_dim*tile_dim*tile_dim*tile_dim;
 if(is_even_odd) td = td/2;
 ////
 nb_site_block = td;
 nb_site_dep_block = td + 8*(td/tile_dim);
 nb_block_total = N / td;
 if(block_dep_id==NULL) block_dep_id = malloc(nb_site_block*nb_site_dep_block*sizeof(unsigned int));
 ////
 for(i=0;i<nb_block_total;i++){
	 nn = 0; m = i*nb_site_dep_block;
	 for(j=i*nb_site_block;j<(i+1)*nb_site_block;j++){
		 t = loop[j];   ix = 4*g_eo2lexic[t];
		 local_dep[0] = g_lexic2eosub[iup[ix+0]];
		 local_dep[1] = g_lexic2eosub[idn[ix+0]];
		 local_dep[2] = g_lexic2eosub[iup[ix+1]];
		 local_dep[3] = g_lexic2eosub[idn[ix+1]];
		 local_dep[4] = g_lexic2eosub[iup[ix+2]];
		 local_dep[5] = g_lexic2eosub[idn[ix+2]];
		 local_dep[6] = g_lexic2eosub[iup[ix+3]];
		 local_dep[7] = g_lexic2eosub[idn[ix+3]];
		 for(ix=0;ix<8;ix++){
           f = find(block_dep_id+m, nn, local_dep[ix]);
		   if(f==-1){f = nn; block_dep_id[m+f] = local_dep[ix]; nn+=1; }
           site_dep[8*j+ix] = f;
	     }
	 }
 }
}
void build_block_dep_data(spinor *copy, spinor *k, int tile_dim, int N, unsigned int *site_dep){
 int i,j,t,ix,f,td,d,nn,m;
 for(i=0;i<nb_block_total*nb_site_dep_block;i++){
	 //memcpy(copy + i, k + site_dep[i], sizeof(spinor));
   _memcpy_spinor(*(copy + i), *(k + site_dep[i]));
 }
} 
void Hopping_Matrix(int ieo, spinor * const l, spinor * const k, int start){
  double *ds, *du, *dr;
  int loop_i, loop_j;
  su3_vector psi1, psi2, psi, chi, phi1, phi3;
  spinor * k0;
  int i,j,n,ix,iy;
  int ioff,ioff2,loop_icx,icx,icy,icx_start,icx_end;
  //// SSE /////////////////
  if(0){
 __m128d *v_1,*v_2,*v_3,*v_4;
 __m128d v_i =  _mm_set_pd(1, -1);
 __m128d v_c_r;
 __m128d v_c_i;
 __m128d v_a, v_b, v_c, v_d, v_e, v_f;
 __m128d vv_a, vv_b, vv_c, vv_d;
 __m128d *v_u, *v_r, *v_s;
  }
  //////////////////////////
  su3 *up, *um;
  spinor *r, *sp, *sm;
  spinor temp;
  complex ka0,ka1,ka2,ka3;
  /////////////////////// Just for this 32x16 conf
  ka0.re = 0.160081; ka0.im = 0.015767;
  ka1.re = 0.160856; ka1.im = 0.000000;
  ka2.re = 0.160856; ka2.im = 0.000000;
  ka3.re = 0.160856; ka3.im = 0.000000;
  ///////////////////////
  if(ieo == 0) ioff = 0; else ioff = VOLUME/2;
 //  l[start].s1.c0.re=0;for(j=0;j<3;j++) for (loop_icx = ioff + start; loop_icx < (VOLUME/2 + ioff); loop_icx+=nb_cores) l[start].s1.c0.re+=cos(cos(loop_icx));
///////

  /**************** loop over all lattice sites ****************/
  /* #pragma ivdep*/
  for (loop_icx = ioff + start; loop_icx < (VOLUME/2 + ioff); loop_icx+=nb_cores){
    icx = loop[loop_icx]; //icx = loop_icx;
    ix = 4*g_eo2lexic[icx]; ;
    r=l+(icx-ioff);
		/*printf("%d: ",icx);
		printf("%5d ",g_lexic2eosub[iup[ix+0]]);
		printf("%5d ",g_lexic2eosub[idn[ix+0]]);
		printf("%5d ",g_lexic2eosub[iup[ix+1]]);
		printf("%5d ",g_lexic2eosub[idn[ix+1]]);
		printf("%5d ",g_lexic2eosub[iup[ix+2]]);
		printf("%5d ",g_lexic2eosub[idn[ix+2]]);
		printf("%5d ",g_lexic2eosub[iup[ix+3]]);
		printf("%5d ",g_lexic2eosub[idn[ix+3]]);
        printf("\n");*/
	/*
	if((loop_icx-start)/cyclic<10) printf("(%d, %d) ",start,icx);
	if((loop_icx-start)/cyclic==10) printf("(%d, %d)\n",start,icx);*/
    /*********************** direction +0 ************************/
    icy=g_lexic2eosub[iup[ix+0]];
    
   
    sp=k+icy;
    up=&(U0[8*(icx-ioff)]);

    _vector_add(psi,(*sp).s0,(*sp).s2);

    _su3_multiply(chi,(*up),psi);
    _complex_times_vector(psi,ka0,chi);
      
    _vector_assign(temp.s0,psi);
    _vector_assign(temp.s2,psi);

    _vector_add(psi,(*sp).s1,(*sp).s3);

    _su3_multiply(chi,(*up),psi);
    _complex_times_vector(psi,ka0,chi);
            
    _vector_assign(temp.s1,psi);
    _vector_assign(temp.s3,psi);

    /*********************** direction -0 ************************/

    icy=g_lexic2eosub[idn[ix+0]];

    sm=k+icy;
    um = up+1;

    _vector_sub(psi,(*sm).s0,(*sm).s2);

    _su3_inverse_multiply(chi,(*um),psi);
    _complexcjg_times_vector(psi,ka0,chi);

    _vector_add_assign(temp.s0,psi);
    _vector_sub_assign(temp.s2,psi);

    _vector_sub(psi,(*sm).s1,(*sm).s3);

    _su3_inverse_multiply(chi,(*um),psi);
    _complexcjg_times_vector(psi,ka0,chi);
      
    _vector_add_assign(temp.s1,psi);
    _vector_sub_assign(temp.s3,psi);

    /*********************** direction +1 ************************/

    icy=g_lexic2eosub[iup[ix+1]];

    sp=k+icy;
    up=um+1;
      
    _vector_i_add(psi,(*sp).s0,(*sp).s3);

    _su3_multiply(chi,(*up),psi);
    _complex_times_vector(psi,ka1,chi);

    _vector_add_assign(temp.s0,psi);
    _vector_i_sub_assign(temp.s3,psi);

    _vector_i_add(psi,(*sp).s1,(*sp).s2);

    _su3_multiply(chi,(*up),psi);
    _complex_times_vector(psi,ka1,chi);

    _vector_add_assign(temp.s1,psi);
    _vector_i_sub_assign(temp.s2,psi);

    /*********************** direction -1 ************************/

    icy=g_lexic2eosub[idn[ix+1]];

    sm=k+icy;
    um=up+1;

    _vector_i_sub(psi,(*sm).s0,(*sm).s3);

    _su3_inverse_multiply(chi,(*um),psi);
    _complexcjg_times_vector(psi,ka1,chi);

    _vector_add_assign(temp.s0,psi);
    _vector_i_add_assign(temp.s3,psi);

    _vector_i_sub(psi,(*sm).s1,(*sm).s2);

    _su3_inverse_multiply(chi,(*um),psi);
    _complexcjg_times_vector(psi,ka1,chi);

    _vector_add_assign(temp.s1,psi);
    _vector_i_add_assign(temp.s2,psi);

    /*********************** direction +2 ************************/

    icy=g_lexic2eosub[iup[ix+2]];

    sp=k+icy;
    up=um+1;
    _vector_add(psi,(*sp).s0,(*sp).s3);

    _su3_multiply(chi,(*up),psi);
    _complex_times_vector(psi,ka2,chi);

    _vector_add_assign(temp.s0,psi);
    _vector_add_assign(temp.s3,psi);

    _vector_sub(psi,(*sp).s1,(*sp).s2);

    _su3_multiply(chi,(*up),psi);
    _complex_times_vector(psi,ka2,chi);
      
    _vector_add_assign(temp.s1,psi);
    _vector_sub_assign(temp.s2,psi);


    /*********************** direction -2 ************************/

    icy=g_lexic2eosub[idn[ix+2]];

    sm=k+icy;
    um = up +1;

    _vector_sub(psi,(*sm).s0,(*sm).s3);

    _su3_inverse_multiply(chi,(*um),psi);
    _complexcjg_times_vector(psi,ka2,chi);

    _vector_add_assign(temp.s0,psi);
    _vector_sub_assign(temp.s3,psi);

    _vector_add(psi,(*sm).s1,(*sm).s2);

    _su3_inverse_multiply(chi,(*um),psi);
    _complexcjg_times_vector(psi,ka2,chi);
      
    _vector_add_assign(temp.s1,psi);
    _vector_add_assign(temp.s2,psi);

    /*********************** direction +3 ************************/

    icy=g_lexic2eosub[iup[ix+3]];

    sp=k+icy;
    up=um+1;
    _vector_i_add(psi,(*sp).s0,(*sp).s2);
      
    _su3_multiply(chi,(*up),psi);
    _complex_times_vector(psi,ka3,chi);

    _vector_add_assign(temp.s0,psi);
    _vector_i_sub_assign(temp.s2,psi);

    _vector_i_sub(psi,(*sp).s1,(*sp).s3);

    _su3_multiply(chi,(*up),psi);
    _complex_times_vector(psi,ka3,chi);

    _vector_add_assign(temp.s1,psi);
    _vector_i_add_assign(temp.s3,psi);

    /*********************** direction -3 ************************/

    icy=g_lexic2eosub[idn[ix+3]];

    sm=k+icy;
    um = up+1;

    _vector_i_sub(psi,(*sm).s0,(*sm).s2);

    _su3_inverse_multiply(chi,(*um),psi);
    _complexcjg_times_vector(psi,ka3,chi);
      
    _vector_add((*r).s0, temp.s0, psi);
    _vector_i_add((*r).s2, temp.s2, psi);

    _vector_i_add(psi,(*sm).s1,(*sm).s3);

    _su3_inverse_multiply(chi,(*um),psi);
    _complexcjg_times_vector(psi,ka3,chi);

    _vector_add((*r).s1, temp.s1, psi);
    _vector_i_sub((*r).s3, temp.s3, psi);
    /************************ end of loop ************************/
  }
  //printf("Executing thread %d\n",start);
}
///////////////////////////////////////////////////////////////////////////////////////////
double sum(spinor *P, int N){
 double s = 0;
 double *d = (double *)P;
 int i;
 for(i=0;i<24*N;i++) s = s + d[i];
 return s;
}
void  fill(spinor *P, int N){
 double s = 0;
 double *d = (double *)P;
 long i;
 for(i=0;i<24*N;i++) {s=i; d[i] = i;}
}
///////////////////////////////////////////////////////////////////////////////////////////
void normal_complex_time_vector(complex c, spinor *p, spinor *q, int N, int type){
	/* p = c x q */
	int i;
	su3_vector s;
	if(type==0) 
      for(i=0;i<N;i++){
	   _complex_times_vector(p[i].s0,c,q[i].s0);
	   _complex_times_vector(p[i].s1,c,q[i].s1);
	   _complex_times_vector(p[i].s2,c,q[i].s2);
	   _complex_times_vector(p[i].s3,c,q[i].s3);
	  }//
     if(type==1)
	  for(i=0;i<N;i++){
	   _complex_times_vector2(p[i].s0,c,q[i].s0);
	   _complex_times_vector2(p[i].s1,c,q[i].s1);
	   _complex_times_vector2(p[i].s2,c,q[i].s2);
	   _complex_times_vector2(p[i].s3,c,q[i].s3);
	  }
	 
}

///////////////////////////////////////////////////////////////////////////////////////////
void *thread_hopping(void *threadarg){
  int thread_id;
  thread_id = *(int *)threadarg;
#ifdef BIND
  cpu_set_t set;
  CPU_ZERO( &set );
  CPU_SET( thread_id, &set );
  if(sched_setaffinity( syscall( __NR_gettid ), sizeof( cpu_set_t ), &set ))
  {
	perror( "sched_setaffinity" );
	return NULL;
  }
#endif
  
  Hopping_Matrix(ieo, l, k, thread_id);
  return NULL;
}
void thread_Hopping_Matrix(int nb_threads){
   int i;
   pthread_t thread_ptr[MAX_NB_THREADS];
   int thread_ids[MAX_NB_THREADS];
   for(i=1;i<nb_threads;i++){
	    thread_ids[i] = i;
	    pthread_create(&thread_ptr[i], NULL, thread_hopping, (void *)&thread_ids[i]);
   }
   Hopping_Matrix(ieo, l, k, 0); // This part is axecuted by the main thread
   for(i=1;i<nb_threads;i++) pthread_join(thread_ptr[i], NULL);

}

///////////////////////////////////////////////////////////////////////////////////////////

int main(){
	//sse(); exit(0);
printf("ENDIANESS %d\n",endian());
double t0,t1;
complex c;
int tile_dim;
int i,j;
int LT, LX, LY, LZ;
int tt,tx,ty,tz;
loop = (int *)aligned_malloc(VOLUME*sizeof(int), ALIGNMENT);
spinor_copy = (spinor *)aligned_malloc(8*(VOLUME/2)*sizeof(spinor), ALIGNMENT);//malloc(8*(VOLUME/2)*sizeof(spinor));
spinor_copy_indices_e = (int *)aligned_malloc(8*(VOLUME/2)*sizeof(int), ALIGNMENT);
spinor_copy_indices_o = (int *)aligned_malloc(8*(VOLUME/2)*sizeof(int), ALIGNMENT);
U0 = (su3 *)aligned_malloc(8*VOLUME*sizeof(su3), ALIGNMENT);
iup  = (int *)aligned_malloc(4*(VOLUME)*sizeof(int), ALIGNMENT);
idn   = (int *)aligned_malloc(4*(VOLUME)*sizeof(int), ALIGNMENT);
g_lexic2eosub  = (int *)aligned_malloc(VOLUME*sizeof(int), ALIGNMENT);
g_eo2lexic   = (int *)aligned_malloc(VOLUME*sizeof(int), ALIGNMENT);
l   = (spinor *)aligned_malloc(VOLUME*sizeof(spinor), ALIGNMENT);
k   = (spinor *)aligned_malloc(VOLUME*sizeof(spinor), ALIGNMENT);
site_dep = (unsigned int *)aligned_malloc(8*(VOLUME/2)*sizeof(unsigned int), ALIGNMENT);
ieo = 0;
cyclic = 1;
memset(l,0,VOLUME*sizeof(spinor));memset(k,0,VOLUME*sizeof(spinor)); fill(k, VOLUME);
readInput();printf("Reading done (%10.7f) \n",sum(k,VOLUME/2));
LT = 32; LX = 16; LY = 16; LZ = 16;
tile_dim = 4;
t0 = now();
build_block_loop(1, ieo, LT, LX, LY, LZ, tile_dim, loop);
t1 = now();
printf("Blocking in %5.4f seconds\n",t1-t0);
t0 = now();
build_block_dep_id(1, loop, tile_dim, VOLUME/2, site_dep);
t1 = now();
printf("Build block dep in %9.7f seconds\n",t1-t0);
/*
for(j=0;j<20;j++){
memset(spinor_copy,0,nb_block_total*nb_site_dep_block*sizeof(spinor));
t0 = now();
build_block_dep_data(spinor_copy, k, tile_dim, VOLUME/2, site_dep);
t1 = now();
printf("Copy (%9.7f) from block in %9.7f seconds\n",sum(spinor_copy,nb_block_total*nb_site_dep_block),t1-t0);
}
*/


c.re=0.125; c.im = 2.365;
for(i=0;i<2;i++){
 t0 = now(); 
 normal_complex_time_vector(c, l,k, VOLUME, i);
 t1 = now();
 printf("Spinor time Complex (%10.7f) VERSION %d in %5.4f seconds\n",sum(l,VOLUME),i,t1-t0);
}
cyclic = 1;
for(tile_dim = 1; tile_dim <=16; tile_dim=2*tile_dim){
	printf("----------------------------------------------------------------------------------------\n");
	for(nb_cores=1;nb_cores<=MAX_NB_THREADS;nb_cores=2*nb_cores){
      memset(l,0,VOLUME*sizeof(spinor));
      build_block_loop(1, ieo, LT, LX, LY, LZ, tile_dim, loop);
      t0 = now(); thread_Hopping_Matrix(nb_cores); t1 = now();
      printf("Half-Hopping (%10.7f) %d thread(s)  and tile_size = %2d done in %5.4f seconds\n",sum(l,VOLUME/2),nb_cores,tile_dim,t1-t0);
    }
}
printf("----------------------------------------------------------------------------------------\n");

/*
cyclic = 2;
t0 = now();
thread_Hopping_Matrix();
t1 = now();
printf("Half-Hopping (%10.7f) 2 threads done in %5.4f seconds\n",sum(l,VOLUME/2),t1-t0);
*/
aligned_free(U0);
aligned_free(iup);
aligned_free(idn);
aligned_free(g_lexic2eosub);
aligned_free(g_eo2lexic);
aligned_free(loop);
aligned_free(block_dep_id);
aligned_free(site_dep);
return 0;
}

/*
	SSE_Tutorial
	This tutorial was written for supercomputingblog.com
	This tutorial may be freely redistributed provided this header remains intact
	_mm_malloc  _mm_free
*/

//#include "stdafx.h"

#define SSE_complex_prod(a, b, z)\
  num1 = _mm_loaddup_pd(&(a).re); \
  num2 = _mm_set_pd((b).im, (b).re);\
  num3 = _mm_mul_pd(num2, num1); \
  num1 = _mm_loaddup_pd(&(a).im);\
  num2 = _mm_shuffle_pd(num2, num2, 1); \
  num2 = _mm_mul_pd(num2, num1);\
  num3 = _mm_addsub_pd(num3, num2);\
  _mm_storeu_pd((double *)(&(z)), num3);



#define _complex_prod(a, b, z)\
 (z).re = ((a).re*(b).re - (a).im*b.im);\
 (z).im = ((a).im*(b).re + (a).re*b.im);

int mmain(int argc, char* argv[])
{
 double t0,t1,f;
 complex *a, *b, *z;
 __m128d num1, num2, num3;
 int i,j,n,m;
 n = atoi(argv[1]); m = atoi(argv[2]);
 a = aligned_malloc(n*sizeof(complex), ALIGNMENT);
 b = aligned_malloc(n*sizeof(complex), ALIGNMENT);
 z = aligned_malloc(n*sizeof(complex), ALIGNMENT);
 for(i=0;i<n;i++) {f=i; a[i].re=f/(f+1); a[i].im = f/(3+f);b[i].re=f/(f+6); b[i].im = f/(9+f);}
 
 if(m==0){
  printf("Starting NO SEE calculation %d...\n",n);
  t0 = now();
  for(i=0;i<n;i++){ _complex_prod(a[i],b[i],z[i]);}
 }
 else{
  printf("Starting SEE calculation %d...\n",n);
  t0 = now();
  for(i=0;i<n;i++){SSE_complex_prod(a[i],b[i],z[i]); } 
 }
 t1 = now();
 f=0; for(i=0;i<n;i++) f+=(z[i].re+z[i].im);
 printf("Calculation (%10.7f) done in %5.4f seconds\n",f,t1-t0);
 aligned_free(a); aligned_free(b); aligned_free(z);
 return 0;
}
