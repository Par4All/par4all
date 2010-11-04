#include "bench.h"
/*
 * benchmark program:   n_real_updates.c
 * 
 * benchmark suite:     DSP-kernel
 * 
 * description:         n_real_updates - filter benchmarking
 * 
 * This program performs n real updates of the form 
 *           D(i) = C(i) + A(i)*B(i),
 * where A(i), B(i), C(i) and D(i) are real numbers,
 * and i = 1,...,N
 * 
 * reference code:      target assembly
 * 
 * f. verification:     simulator based
 * 
 *  organization:        Aachen University of Technology - IS2 
 *                       DSP Tools Group
 *                       phone:  +49(241)807887 
 *                       fax:    +49(241)8888195
 *                       e-mail: zivojnov@ert.rwth-aachen.de 
 *
 * author:              Juan Martinez Velarde
 * 
 * history:             25-05-94 creation fixed-point (Martinez Velarde)
 *                      16-03-95 adaption floating-point (Harald L. Schraut)
 *
 *                      $Author: schraut $
 *                      $Date: 1995/04/11 07:42:37 $
 *                      $Revision: 1.2 $
 */

#define STORAGE_CLASS register
#define TYPE          float
#define N             16

void
pin_down(TYPE pa[N], TYPE pb[N], TYPE pc[N], TYPE pd[N])
{
  STORAGE_CLASS int i ; 

  for (i=0 ; i < N ; i++)
    {
      pa[i] = 10 ; 
      pb[i] = 2 ; 
      pc[i] = 10 ; 
      pd[i] = 0 ; 
    }
}

TYPE main()
{
  static TYPE A[N], B[N], C[N], D[N] ; 
  
  int i ; 

  pin_down(A, B, C, D) ; 
  
  START_PROFILING; 
	  
  for (i = 0 ; i < N ; i++) 
    D[i]  = C[i] + A[i] * B[i] ;
  
  END_PROFILING;   
  
  pin_down(A, B, C, D) ; 
  return(0)  ; 
}
