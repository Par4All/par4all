#include "bench.h"
/*
 * benchmark program:   complex_update.c
 * 
 * benchmark suite:     DSP-kernel
 * 
 * description:         complex_update - filter benchmarking
 * 
 * This program performs a complex update of the form D = C + A*B,
 * where A, B, C and D are complex numbers .
 * 
 *          A = Ar + j Ai
 *          B = Br + j Bi
 *          C = Cr + j Ci
 *          D = C + A*B =   Dr  +  j Di
 *                      =>  Dr = Cr + Ar*Br - Ai*Bi
 *                      =>  Di = Ci + Ar*Bi + Ai*Br
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
 * history:             11-05-94 creation for fixed-point (Martinez Velarde)
 *                      16-03-95 adaption for floating-point (Harald L. Schraut)
 *
 *                      $Author: DSPstone $
 *                      $Date: 1995/03/20 11:36:26 $
 *                      $Revision: 1.2 $ 
 */

#define STORAGE_CLASS register
#define TYPE  float

void
pin_down(TYPE p[2], TYPE a[2], TYPE b[2], TYPE c[2])
{
  a[0] = 2; a[1] = 1;
  b[0] = 2; b[1] = 5;
  c[0] = 3; c[1] = 4;
  p[0] = 0; p[1] = 0; 
}


TYPE
main()
{
  static TYPE A[2];
  static TYPE B[2];
  static TYPE C[2]; 
  static TYPE D[2]; 
  
  pin_down(D, A, B, C); 
  
  START_PROFILING;
	  
  D[0]  = C[0] + A[0] * B[0] ;
  D[0] -=          A[1]   * B[1] ; 

  D[1]  = C[1]   + A[1] * B[0] ; 
  D[1] +=          A[0]   * B[1] ; 
   
  END_PROFILING;
  
  pin_down(D, A, B, C) ; 
  
  return(0)  ; 
}


