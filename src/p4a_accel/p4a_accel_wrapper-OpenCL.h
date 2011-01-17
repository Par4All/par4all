/** @file

    API of Par4All C to OpenCL for the kernel wrapper and kernel.

    Funded by the FREIA (French ANR), TransMedi\@ (French Pôle de
    Compétitivité Images and Network) and SCALOPES (Artemis European
    Project project), with the support of the MODENA project (French
    Pôle de Compétitivité Mer Bretagne)

    "mailto:Stephanie.Even@enstb.org"
    "mailto:Ronan.Keryell@hpc-project.com"

    This work is done under MIT license.
*/

#ifndef P4A_ACCEL_WRAPPER_CL_H
#define P4A_ACCEL_WRAPPER_CL_H

/** @defgroup P4A_qualifiers Kernels and arguments qualifiers

    @{
*/

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
