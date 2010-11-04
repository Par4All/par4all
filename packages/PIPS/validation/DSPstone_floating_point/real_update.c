#include "bench.h"
/*
 * benchmark program:   real_update.c
 * 
 * benchmark suite:     DSP-kernel
 * 
 * description:         real_update - filter benchmarking
 * 
 * This program performs a real update of the form D = C + A*B,
 * where A, B, C and D are real numbers .
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
 * history:             11-05-94 creation fixed-point (Martinez Velarde)
 *                      16-03-95 adaption floating-point (Harald L. Schraut)
 *                      12-10-95 changing pin-up (Armin Braun)
 * 
 *                      $Author: dehofman $
 *                      $Date: 1996/09/16 19:23:43 $
 *                      $Revision: 1.2 $
 */

#define STORAGE_CLASS register
#define TYPE  float

void
pin_down(TYPE *p,TYPE *q,TYPE *r,TYPE *s)
{

  *p = 10 ; 
  *q =  2 ;
  *r =  1 ;
  *s =  0 ;

}


TYPE
main()
{
  static TYPE A = 10 ;
  static TYPE B = 2 ;
  static TYPE C = 1 ; 
  static TYPE D = 0 ; 
  
  
  pin_down(&A,&B,&C,&D) ; 
  
  START_PROFILING; 
	  
  D  = C + A * B ;
   
  END_PROFILING; 
  
  pin_down(&A,&B,&C,&D) ; 
  
  return(0)  ; 
}
