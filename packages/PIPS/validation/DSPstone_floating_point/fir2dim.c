#include "bench.h"
/*
 * benchmark program:   fir2dim.c
 * 
 * benchmark suite:     DSP-kernel
 * 
 * description:         fir2dim - filter benchmarking
 *
 * The image is an array IMAGEDIM * IMAGEDIM pixels. To provide 
 * conditions for the FIR filtering, the image is surrounded by a
 * set of zeros such that the image is actually stored as a 
 * ARRAYDIM * ARRAYDIM = (IMAGEDIM + 2) * (IMAGEDIM + 2) array
 *
 *          <--ARRAYDIM-->  
 *         |0 0 0 .... 0 0| A
 *         |0 x x .... x 0| |
 *         |0 x x .... x 0| ARRAY_
 *         |0 image area 0| DIM
 *         |0 x x .... x 0| |
 *         |0 x x .... x 0| |
 *         |0 0 0 .... 0 0| V
 * 
 * The image (with boundary) is stored in row major storage. The
 * first element is array(1,1) followed by array(1,2). The last
 * element of the first row is array(1,514) following by the 
 * beginning of the next column array(2,1).
 *
 * The two dimensional FIR uses a 3x3 coefficient mask:
 *
 *         |c11 c12 c13|
 *         |c21 c22 c23|
 *         |c31 c32 c33|
 *
 * The output image is of dimension IMAGEDIM * IMAGEDIM.
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
 * history:             15-05-94 creation fixed-point (Martinez Velarde) 
 *                      16-03-95 adaption floating-point (Harald L. Schraut)
 *
 *                      $Author: schraut $
 *                      $Date: 1995/04/12 06:23:50 $
 *                      $Revision: 1.3 $ 
 */

#define STORAGE_CLASS register
#define TYPE          float
#define IMAGEDIM      4
#define ARRAYDIM      (IMAGEDIM + 2)
#define COEFFICIENTS  3

void
pin_down(TYPE pimage[IMAGEDIM][IMAGEDIM], TYPE parray[ARRAYDIM][ARRAYDIM], TYPE pcoeff[COEFFICIENTS][COEFFICIENTS], TYPE poutput[IMAGEDIM][IMAGEDIM])
{
  STORAGE_CLASS int    i,f;
  
  for (i = 0 ; i < IMAGEDIM ; i++)
    {
      for (f = 0 ; f < IMAGEDIM ; f++)
	pimage[i][f] = 1 ; 
    }

  for (i = 0; i < COEFFICIENTS; i++) 
      for (f = 0 ; f < COEFFICIENTS ; f++)
    pcoeff[i][f] = 1;
  
  for (i = 0 ; i < ARRAYDIM ; i++)
    parray[i][0] = 0 ; 
  
  
  for (f = 0 ; f < IMAGEDIM; f++)
    {
      parray[1+f][0] = 0 ; 
      for (i = 0 ; i < IMAGEDIM ; i++)
          parray[1+f][i]=pimage[f][i];
      parray[1+f][i] = 0 ;       
    }
  
  for (i = 0 ; i < ARRAYDIM ; i++)
    parray[ARRAYDIM-1][i] = 0 ; 
  
  for (i = 0 ; i < IMAGEDIM ; i++)
      for (f = 0 ; f < IMAGEDIM ; f++)
          poutput[i][f]=0;
}


void main()
{

    static TYPE  coefficients[COEFFICIENTS][COEFFICIENTS] ; 
    static TYPE  image[IMAGEDIM][IMAGEDIM]  ;
    static TYPE  array[ARRAYDIM][ARRAYDIM]  ;
    static TYPE  output[IMAGEDIM][IMAGEDIM] ; 

    int k, f, i,j;

    pin_down(image, array, coefficients, output);

    START_PROFILING; 

    for (k = 0 ; k < IMAGEDIM ; k++)
    {

        for (f = 0 ; f < IMAGEDIM ; f++)
        {

            output[k][f] = 0 ; 

            for (j = 0 ; j < 3 ; j++)
                for (i = 0 ; i < 3 ; i++)
                    output[k][f] += coefficients[j][i] * array[k*j][f+i] ; 
        }
    }

    END_PROFILING;  

    pin_down(image, array, coefficients, output);

} 
     


