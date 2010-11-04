#include "bench.h"
/*
 * benchmark program:   mat1x3.c
 * 
 * benchmark suite:     DSP-kernel
 * 
 * description:         1x3 matrix - multiply benchmarking
 *
 *               |h11 h12 h13|   |x1|   |y1|   | h11*x1+h12*x2+h31*x3 |
 *               |h21 h22 h23| * |x2| = |y2| = | h21*x1+h22*x2+h23*x3 |
 *               |h31 h32 h33|   |x3|   |y3|   | h31*x1+h32*x2+h33*x3 |
 * 
 * Element are to store in following order:
 * 
 * matrix h[9]={h11,h12,h13, h21,h22,h23, h31,h32,h33}
 * vector x[3]={x1,x2,x3}
 * vector y[3]={y1,y1,y3}
 * 
 * reference code:       none
 * 
 * f. verification:      
 * 
 * organization:         Aachen University of Technology - IS2 
 *                       DSP Tools Group
 *                       phone:  +49(241)807887 
 *                       fax:    +49(241)8888195
 *                       e-mail: zivojnov@ert.rwth-aachen.de 
 *
 * author:              Juan Martinez Velarde
 * 
 * history:             31-03-94 creation fixed-point (Martinez Velarde)
 *                      16-03-95 adaption floating-point (Harald L. Schraut)
 *
 *                      $Author: schraut $
 *                      $Date: 1995/07/03 10:01:03 $
 *                      $Revision: 1.7 $
 */

#define STORAGE_CLASS  register
#define TYPE           float

TYPE
pin_down(TYPE h[3][3], TYPE y[3], TYPE x[3])
{
  int i,j;

  for(j=0; j<3; j++){
  for(i=0; i<3; i++){
    h[j][i] = j*3+i;
  }
  }

  for(i=0; i<3; i++){
    x[i] = i;
    y[i] = 0;
  }

  return(0);
}

TYPE
main()
{

  static TYPE h[3][3]; 
  static TYPE x[3];
  static TYPE y[3]; 

  int f,i ; 

  pin_down(h, y, x);


  START_PROFILING; 
  
  for (i = 0 ; i < 3; i++)
    {
      /* p_x points to the beginning of the input vector */
      y[i] = 0;     
      
       /* do matrix multiply */
      
      for (f = 0 ; f < 3; f++)
	  y[i] += h[i][f] * x[f] ;
      
    }

  END_PROFILING; 
    
  return(0)  ; 
  
}
