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

    This is the return type of the kernel.
    The type is here undefined and must be locally defined.
*/
/* change this define to void:
#define P4A_accel_kernel inline 
*/
#define P4A_accel_kernel void 

/** A declaration attribute of a hardware-accelerated kernel called from
    the host in CL 

    This is the return type of the kernel wrapper.
    It must be a void function.
    Type used in the protoizer.
*/
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

/*
The OpenCL extension cl_khr_byte_addressable_store removes certain
restrictions on built-in types char, uchar, char2, uchar2, short, and
half. An application that wants to be able to write to elements of a
pointer (or struct) that are of type char, uchar, char2, uchar2,
short, ushort, and half will need to include the #pragma OPENCL
EXTENSION cl_khr_byte_addressable_store : enable directive before any
code that performs writes that may not be supported. 
*/
#ifdef cl_khr_byte_addressable_store
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable 
#endif

/*
Pragma to support double floating point precision
* */

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
#warning "Your OpenCL device doesn't support double precision"
#endif

// Required for loop unrolling
#define MOD(x,n) ((x)%(n))
#define MAX0(x,n) max(x,n)



/**
   @}
*/

#endif //P4A_ACCEL_WRAPPER_CL_H
