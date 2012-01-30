/************
QSORT.H
************/
#ifndef __QSORT_H__
#define __QSORT_H__

void sciqsort(char *a, char *tab,int flag, int n, int es, int es1, int (*cmp) (),int (*swapcode) (), int (*swapcodeind) ());
int swapcodeint(char * parmi,char * parmj,int n,int incr);

#define swapcodeind swapcodeint
#define swap(a, b) swapcode(a, b, 1,es)
#define swapind(a, b)  if ( flag==1) swapcodeind(a,b,1,es1)
#define vecswap(a, b, n) if ((n) > 0) swapcode(a, b, n/es,es)
#define vecswapind(a, b, n) if ((n) > 0 && flag == 1) swapcodeind(a,b,n/es1,es1) 
#define med3(res,tabres,a, b, c, xa,xb,xc,cmp) cmp(a, b,xa,xb) < 0 ?	\
   (cmp(b, c, xb, xc) < 0 ? (res=b,tabres=xb) : \
    (cmp(a, c, xa, xc) < 0 ? (res=c,tabres=xc) : (res=a,tabres=xa) ))	\
  :(cmp(b, c, xb, xc) > 0 ? (res=b,tabres=xb) : (cmp(a, c, xa, xc) < 0 ? (res=a,tabres=xa) : (res=c,tabres=xc) ))

#endif 


/************
 ISANAN.H
************/

int isanan(double *x);




/************
CORE_MATH.H
************/

#ifndef __BASE_MATH_H__
#define __BASE_MATH_H__
#include <limits.h>
#include <math.h>

#ifdef __STDC__
#include <stdlib.h>
#endif

#ifndef _MSC_VER 
#endif

#ifdef _MSC_VER 
		#include <float.h>
		#define finite(x) _finite(x) 
#endif /* _MSC_VER */

#ifdef _MSC_VER 
	#include <float.h>
	#define ISNAN(x) _isnan(x)
#else 
	#define ISNAN(x) isnan(x)
#endif 

#define Abs(x) ( ( (x) >= 0) ? (x) : -( x) )
#ifndef Min
#define Min(x,y)	(((x)<(y))?(x):(y))
#endif 

#ifndef Max 
#define Max(x,y)	(((x)>(y))?(x):(y))
#endif

#define PI0 (int *) 0
#define PD0 (double *) 0

/* angle conversion */
#define PI_OVER_180  0.01745329251994329576913914624236578987393
#define _180_OVER_PI 57.29577951308232087665461840231273527024
#define DEG2RAD(x) ((x) * PI_OVER_180  )
#define RAD2DEG(x) ((x) * _180_OVER_PI )

#define linint(x) ((int)  floor(x + 0.5 )) 
#define inint(x) ((int) floor(x + 0.5 ))  

#if (defined(sun) && defined(SYSV)) 
#include <ieeefp.h>
#endif

#if defined(_MSC_VER)
  #define M_PI 3.14159265358979323846
#else
  #if defined(HAVE_VALUES_H)
    #include <values.h>
  #else
    #if defined(HAVE_LIMITS_H)
     #include <limits.h>
    #endif
  #endif
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846 
#endif

#ifndef HAVE_EXP10
#define log_10_ 2.3025850929940456840179914546844
/* Provide a macro to do exp10 */
#define exp10(x) exp( (log_10_) * (x) )
#endif

#endif 





