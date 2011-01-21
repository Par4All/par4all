/** @addtogroup P4AHyantes P4A version.

    @{
*/

/** @defgroup kernelHyantes The kernel of Hyantes

    @{
    The kernel of Hyantes invoked by P4A.
*/

#include "p4a_accel_wrapper.h"
#define rangex 290
#define rangey 299
#define nb 2878

#ifndef M_PI
#define M_PI 3.14
#endif

#ifdef USE_FLOAT
typedef float data_t;
#else
typedef double data_t;
#endif

#ifdef P4A_ACCEL_OPENMP 
#include <math.h>
#endif

#ifdef P4A_ACCEL_CUDA
#include <math.h>
#endif

typedef struct {
  data_t latitude;
  data_t longitude;
  data_t stock;
} town;



P4A_accel_kernel void kernel_hyantes(data_t xmin, data_t ymin, data_t step,data_t range, P4A_accel_global_address town pt[rangex][rangey], P4A_accel_global_address town t[nb], int i, int j)
{
  int k;
  
  pt[i][j].latitude =(xmin+step*i)*180/M_PI;
  pt[i][j].longitude =(ymin+step*j)*180/M_PI;
  pt[i][j].stock =0.;
  for(k=0;k<nb;k++) {
    data_t tmp =
      6368.* acos(cos(xmin+step*i)*cos( t[k].latitude ) * cos((ymin+step*j)-t[k].longitude) + sin(xmin+step*i)*sin(t[k].latitude));
    if( tmp < range )
      pt[i][j].stock += t[k].stock  / (1 + tmp) ;
  }
}

P4A_accel_kernel_wrapper kernel_wrapper(data_t xmin, data_t ymin, data_t step,data_t range, P4A_accel_global_address town pt[rangex][rangey], P4A_accel_global_address town t[nb])
{
  int i = P4A_vp_0;
  int j = P4A_vp_1;

  if (i < rangex && j < rangey)
    kernel_hyantes(xmin,ymin,step,range,pt,t,i,j);
}

/** @} */
/** @} */
