#include "bench.h"
/*
 *  benchmark program  : biquad_N_sections.c
 *
 *  benchmark suite    : DSP-kernel
 *
 *  description        : benchmarking of an iir biquad (N sections)
 *                       
 *	The equations of each biquad section filter are:
 *       w(n) =    x(n) - ai1*w(n-1) - ai2*w(n-2)
 *       y(n) = b0*w(n) + bi1*w(n-1) + bi2*w(n-2)
 *
 * Biquads are sequentally positioned. Input sample for biquad i is
 * xi-1(n). Output sample for biquad i is xi(n). 
 * System input sample is x0(n). System output sample is xN(n) = y(n) 
 * for N biquads. 
 * 
 * Each section performs following filtering (biquad i) : 
 * 
 *		              wi(n)
 *   xi-1(n) ---(-)---------->-|->---bi0---(+)-------> xi(n)
 *               A             |            A
 *               |           |1/z|          |
 *               |             | wi(n-1)    |
 *               |             v            |
 *               |-<--ai1----<-|->---bi1-->-|
 *               |             |            |
 *               |           |1/z|          |
 *               |             | wi(n-2)    |
 *               |             v            |
 *               |-<--ai2----<--->---bi2-->-|
 * 
 *     The values wi(n-1) and wi(n-2) are stored in wi1 and wi2
 * 
 *                                              
 *  reference code 
 *
 *  func. verification from separate computation
 *
 *  organization       Aachen University of Technology - IS2
 *                     DSP Tools Group
 *                     phone   : +49(241)807887
 *                     fax     : +49(241)8888195
 *                     e-mail  : zivojnov@ert.rwth-aachen.de
 *
 *  author             Juan Martinez Velarde
 *
 *  history            24-03-94 creation fixed-point (Martinez Velarde)
 *                     16-03-95 adaption floating-point (Harald L. Schraut)
 *
 *                     $Author: schraut $
 *                     $Date: 1995/04/11 07:18:00 $
 *                     $Revision: 1.3 $
 */


#define STORAGE_CLASS register
#define TYPE float

#define NumberOfSections 4

TYPE pin_down(TYPE x, TYPE coefficients[NumberOfSections][5], TYPE wi[NumberOfSections][2])
{
  int f,l; 

  for (f = 0 ; f < NumberOfSections; f++)
      for(l=0;l<5;l++)
          coefficients[f][l] = 7 ; 
  
  for (f = 0 ; f < NumberOfSections; f++)
      for(l=0;l<2;l++)
    wi[f][l] = 0 ; 
  
  return ((TYPE) 1) ;
}


TYPE main()
{
  
  STORAGE_CLASS TYPE w;
  int f; 

  /*STORAGE_CLASS*/ TYPE wi[NumberOfSections][2] ; 

  static TYPE coefficients[NumberOfSections][5]; 
  STORAGE_CLASS TYPE x,y ; 

  x = pin_down(x, coefficients, wi) ;
 
  START_PROFILING;
  
  y = x ; 
  
  for (f = 0 ; f < NumberOfSections ; f++)
    {
      w  = y -  coefficients[f][0] * wi[f][0] ; 
      w -= coefficients[f][1] * wi[f][1] ; 
      
      y  = coefficients[f][2] *  w ; 
      y += coefficients[f][3] * wi[f][0] ; 
      y += coefficients[f][4] * wi[f][1] ; 
      
      wi[f][1]=wi[f][0];
      wi[f][0]=w;
    }
  
  END_PROFILING;
    
  pin_down(y,coefficients,wi) ; 

  return((TYPE) y) ; 
}

