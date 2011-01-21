/** @file 
*/

/** @addtogroup P4AJacobi

    @{
*/

/** @addtogroup kernel1Jacobi The first kernel.

    @{
    Transforms the input to the output after averaging the pixels.

    It is composed of a kernel_wrapper and a core kernel.

*/

#include "p4a_accel_wrapper.h"
typedef float float_t;
#define SIZE 501


P4A_accel_kernel void kernel1(P4A_accel_global_address float_t space[SIZE][SIZE],P4A_accel_global_address float_t save[SIZE][SIZE],int i, int j) 
{ 
  save[i][j] = 0.25*(space[i-1][j]+space[i+1][j]+space[i][j-1]+space[i][j+1]);
}

P4A_accel_kernel_wrapper kernel1_wrapper(P4A_accel_global_address float_t space[SIZE][SIZE], P4A_accel_global_address float_t save[SIZE][SIZE])
{
    
  int i = P4A_vp_0;
  int j = P4A_vp_1;
  if(i >= 1 && i <= SIZE - 2 && j >= 1 && j <= SIZE - 2)
    kernel1(space, save, i, j);
}

/** @} */
/** @} */

