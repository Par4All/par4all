#ifndef P4A_ACCEL_WRAPPER_CL_H
#define P4A_ACCEL_WRAPPER_CL_H

/** @} */

/** A declaration attribute of a hardware-accelerated kernel in CL
    called from the GPU it-self
*/
#define P4A_accel_kernel inline void

/** A declaration attribute of a hardware-accelerated kernel called from
    the host in CL */
#define P4A_accel_kernel_wrapper __kernel void

/** The address space visible for all functions. 
    Allocation in the global memory pool.
 */
#define P4A_accel_global_address __global

/** The address space in the global memory pool but in read-only mode.
 */
#define P4A_accel_constant_address __constant

/** The address space visible by all work-items in a work group.
    This is the <<shared>> memory in the CUDA architecture.
    Can't be initialized :
    * __local float a = 1; is not allowed
    * __local float a;
              a = 1;       is allowed.
 */
#define P4A_accel_local_address __local


/** Get the coordinate of the virtual processor in X (first) dimension in
    CL
*/
#define P4A_vp_0 get_global_id(0)

/** Get the coordinate of the virtual processor in Y (second) dimension in
    CL
*/
#define P4A_vp_1 get_global_id(1)

/** Get the coordinate of the virtual processor in Z (second) dimension in
    CL
*/
#define P4A_vp_2 get_global_id(2)

/**
   @}
*/

#endif //P4A_ACCEL_WRAPPER_CL_H
typedef float float_t;
#define SIZE 501


P4A_accel_kernel kernel1(P4A_accel_global_address float_t space[SIZE][SIZE],P4A_accel_global_address float_t save[SIZE][SIZE],int i, int j) 
{ 
  save[i][j] = 0.25*(space[i-1][j]+space[i+1][j]+space[i][j-1]+space[i][j+1]);
}

P4A_accel_kernel_wrapper kernel1_wrapper(P4A_accel_global_address float_t space[SIZE][SIZE], P4A_accel_global_address float_t save[SIZE][SIZE])
{
    
  //int i = get_global_id(0);
  int i = P4A_vp_0;
  //int j = get_global_id(1);
  int j = P4A_vp_1;
  if(i >= 1 && i <= SIZE - 2 && j >= 1 && j <= SIZE - 2)
    kernel1(space, save, i, j);
}
