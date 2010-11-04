#include "bench.h"
/*
 * benchmark program:   fir.c
 * 
 * benchmark suite:     DSP-kernel (floating-point)
 * 
 * description:         fir - filter benchmarking
 * 
 * reference code:      target assembly
 * 
 * f. verification:     simulator
 * 
 *  organization:        Aachen University of Technology - IS2 
 *                       DSP Tools Group
 *                       phone:  +49(241)807887 
 *                       fax:    +49(241)8888195
 *                       e-mail: zivojnov@ert.rwth-aachen.de 
 *
 * author:              Juan Martinez Velarde
 * 
 * history:             12-04-94 creation fixed-point (Martinez Velarde) 
 *                      15-02-95 adapting for floating-point (Harald L. Schraut)
 *
 *
 *                      $Date: 1995/04/11 07:15:34 $
 *                      $Author: schraut $
 *                      $Revision: 2.2 $
 */


#define STORAGE_CLASS register
#define TYPE  float
#define LENGTH 16

void
pin_down(TYPE px[LENGTH], TYPE ph[LENGTH], TYPE *y)
{
  STORAGE_CLASS int    i;
  
  for (i = 1; i <= LENGTH; i++) 
    {
      px[i-1] = i;
      ph[i-1] = i;
    }
  
}

TYPE
main()
{
  static TYPE  x[LENGTH];
  static TYPE  h[LENGTH];
  
  static TYPE  x0 = 100;
  TYPE y;

  int i ;
  
  pin_down(x, h, &y);
  
  START_PROFILING ; 

  y = 0;

  for (i = 0; i < LENGTH - 1; i++)
    {       
      y += h[LENGTH-1-i] * x[LENGTH-1-i] ; 
      x[LENGTH-1-i]=x[LENGTH-2-i];
    }
  
  y += h[0] * x[0] ;
  x[0] = x0 ; 
  
  END_PROFILING ;  
  
  pin_down(x, h,& y);

  return ((TYPE) y); 
}
