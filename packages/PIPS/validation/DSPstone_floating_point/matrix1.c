#include "bench.h"
/*
 * benchmark program:   matrix1.c
 * 
 * benchmark suite:     DSP-kernel
 * 
 * description:         generic matrix - multiply benchmarking
 *
 * This program performs a matrix multiplication of the form C=AB,
 * where A and B are two dimensional matrices of arbitrary dimension.
 * The only restriction os that the inner dimension of the arrays must
 * be greater than 1. 
 * 
 *          A[X x Y] * B[Y x Z] = C[X x Z]
 *                    
 *                  |a11     a12     ..      a1y|
 *                  |a21     a22     ..      a2y|
 * matrix A[X x Y]= |..      ..      ..     ..  |
 *                  |a(x-1)1 a(x-1)2 ..  a(x-1)y|
 *                  |ax1     ax2     ..      axy|
 *
 *
 *                  |b11     b12     ..     b1z|
 *                  |b21     b22     ..     b2z|
 * matrix B[Y x Z]= |..      ..      ..     .. |
 *                  |b(y-1)1 b(y-1)2 .. b(y-1)z|
 *                  |by1     by2     ..     byz|
 *
 *                  |c11     c12     ..     c1z|
 *                  |c21     c22     ..     c2z|
 * matrix C[X x Z]= |..      ..      ..     .. |
 *                  |c(x-1)1 c(x-1)2 .. c(x-1)z|
 *                  |cx1     cx2     ..     cxz|
 * 
 * matrix elements are stored as
 *
 * A[X x Y] = { a11, a12, .. , a1y, 
 *              a21, a22, .. , a2y, ..., 
 *              ax1, ax2, .. , axy}
 * 
 * B[Y x Z] = { b11, b21, .., b(y-1)1, by1, b12, b22, .. , b(y-1)z, byz} 
 * 
 * C[X x Z] = { c11, c21, .. , c(x-1)1, cx1, c12, c22, .. ,c(x-1)z, cxz }
 * 
 * 
 * reference code: 
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
 * history:             03-04-94 creation fixed-point (Martinez Velarde)
 *                      16-03-95 adaption floating-point (Harald L. Schraut)
 *
 *                      $Author: schraut $
 *                      $Date: 1995/04/11 07:37:58 $
 *                      $Revision: 1.4 $
 */

#define STORAGE_CLASS  register
#define TYPE           float

#define X 10 /* first dimension of array A */
#define Y 10 /* second dimension of array A, first dimension of array B */
#define Z 10 /* second dimension of array B */

TYPE
pin_down(TYPE A[X][Y], TYPE B[Y][Z], TYPE C[X][Z])
{
  int i,j ; 
  
  for (i = 0 ; i < X; i++)
  for (j = 0 ; j < Y; j++)
      A[i][j] = 1 ; 
  
  for (i = 0 ; i < Y ; i++)
  for (j = 0 ; j < Z ; j++)
      B[i][j] = 1 ; 
  
  for (i = 0 ; i < X ; i++)
  for (j = 0 ; j < Z ; j++)
      C[i][j] = 0 ; 
  
  return((TYPE)0) ; 

}

TYPE
main()
{ 
  static  TYPE A[X][Y] ; 
  static  TYPE B[Y][Z] ;
  static  TYPE C[X][Z] ;

  int i,f ;
  int k; 

  pin_down(A, B, C) ; 

  START_PROFILING; 
  
      for (i = 0 ; i < X; i++)
    {
      
  for (k = 0 ; k < Z ; k++)
	{
	  
	  C[i][k] =  0 ; 
	  
	  for (f = 0 ; f < Y; f++) /* do multiply */
	    C[i][k] += A[i][f] * B[f][k] ;
	  
	  	  
	}
    }
  
  END_PROFILING; 
  
  pin_down(A, B, C) ; 
    
  return(0)  ; 
}
