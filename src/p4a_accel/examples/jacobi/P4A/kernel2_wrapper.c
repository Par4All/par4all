/** @file 
*/

/** @defgroup Examples

    @{
*/

/** @addtogroup Jacobi

    @{
*/

/** @addtogroup kernel2Jacobi The second kernel.

    @{
    Overwrites the input from the output after averaging the pixels.
*/

#include "p4a_accel_wrapper.h"
typedef float float_t;
#define SIZE 501

P4A_accel_kernel void kernel2(P4A_accel_global_address float_t space[SIZE][SIZE], P4A_accel_global_address float_t save[SIZE][SIZE], int i, int j)
{
   space[i][j] = 0.25*(save[i-1][j]+save[i+1][j]+save[i][j-1]+save[i][j+1]);
}

P4A_accel_kernel_wrapper kernel2_wrapper(P4A_accel_global_address float_t space[SIZE][SIZE], P4A_accel_global_address float_t save[SIZE][SIZE])
{
  int j = P4A_vp_1;
  int i = P4A_vp_0;

  if (i >= 1 && i <= SIZE - 2 && j >= 1 && j <= SIZE - 2)
    kernel2(space, save, i, j);
}


/** @} */
/** @} */
/** @} */
