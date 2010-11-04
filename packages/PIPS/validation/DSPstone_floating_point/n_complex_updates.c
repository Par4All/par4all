#include "bench.h"
/*
 * benchmark program:   n_complex_updates.c
 * 
 * benchmark suite:     DSP-kernel
 * 
 * description:         n complex updates - filter benchmarking
 * 
 * This program performs n complex updates of the form 
 *           D(i) = C(i) + A(i)*B(i),
 * where A(i), B(i), C(i) and D(i) are complex numbers,
 * and i = 1,...,N
 * 
 *          A(i) = Ar(i) + j Ai(i)
 *          B(i) = Br(i) + j Bi(i)
 *          C(i) = Cr(i) + j Ci(i)
 *          D(i) = C(i) + A(i)*B(i) =   Dr(i)  +  j Di(i)
 *                      =>  Dr(i) = Cr(i) + Ar(i)*Br(i) - Ai(i)*Bi(i)
 *                      =>  Di(i) = Ci(i) + Ar(i)*Bi(i) + Ai(i)*Br(i)
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
 * history:             13-05-94 creation fixed-point (Martinez Velarde)
 *                      16-03-95 adaption floating-point (Harald L. Schraut)
 * 
 *                      $Author: schraut $
 *                      $Date: 1995/04/11 07:41:16 $
 *                      $Revision: 1.2 $
 */

#define STORAGE_CLASS register
#define TYPE  float
#define N             16

void
pin_down(TYPE pa[N][2], TYPE pb[N][2], TYPE pc[N][2], TYPE pd[N][2])
{
  STORAGE_CLASS int i ; 

  for (i=0 ; i < N ; i++)
    {
      pa[i][0] = 2 ;
      pa[i][1] = 1 ;
      pb[i][0] = 2 ; 
      pb[i][1] = 5 ; 
      pc[i][0] = 3 ;
      pc[i][1] = 4 ;  
      pd[i][0] = 0 ; 
      pd[i][1] = 0 ; 
    }
}


TYPE
main()
{
  static TYPE A[N][2], B[N][2], C[N][2], D[N][2] ; 
  
  int i ; 

  pin_down(A, B, C, D) ; 
  
  START_PROFILING; 
	  
  for (i = 0 ; i < N ; i++) 
    {
        D[i][0] = C[i][0] + A[i][0] * B[i][0];
      D[i][0] -= A[i][1] + B[i][1];
      
      D[i][1] = C[i][1] + A[i][1]*B[i][0];
      D[i][1] += A[i][0] + B[i][1];
    }
  
  END_PROFILING; 
      
  pin_down(A, B, C, D) ; 
  
  return(0)  ; 
}
