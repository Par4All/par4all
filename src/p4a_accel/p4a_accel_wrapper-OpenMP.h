/** @file

    API of Par4All C to OpenMP for the kernel wrapper and kernel.

    Funded by the FREIA (French ANR), TransMedi\@ (French Pôle de
    Compétitivité Images and Network) and SCALOPES (Artemis European
    Project project), with the support of the MODENA project (French
    Pôle de Compétitivité Mer Bretagne)

    "mailto:Stephanie.Even@enstb.org"
    "mailto:Ronan.Keryell@hpc-project.com"

    This work is done under MIT license.
*/

#ifndef P4A_ACCEL_WRAPPER_OPENMP_H
#define P4A_ACCEL_WRAPPER_OPENMP_H

/** @defgroup P4A_qualifiers Kernels and arguments qualifiers

    @{
*/

/** A declaration attribute of a hardware-accelerated kernel in OpenMP

    This is the return type of the kernel.
    The type is here undefined and must be locally defined.
*/
#define P4A_accel_kernel 


/** A declaration attribute of a hardware-accelerated kernel called from
    the host in OpenMP

    This is the return type of the kernel wrapper.
    It must be a void function.
    Type used in the protoizer.
*/
#define P4A_accel_kernel_wrapper void


/** The global address qualifier.
    Address space visible by all kernels in the global memory pool.

    Nothing for OpenMP.
 */
#define P4A_accel_global_address 

/** The constant address qualifier.
    Memory space in the global memory pool but in read-only mode.
 */
#define P4A_accel_constant_address const

/** The local address qualifier only visible for a kernel group.
    
    Nothing for OpenMP.
 */
#define P4A_accel_local_address 
/** @} */

/** @defgroup P4A_index Indexations of the loop

    @{
*/


/** Note that in CUDA and OpenCL there is 3 dimensions max: */
enum { P4A_vp_dim_max = 3 };

/** This is a global variable used to simulate P4A virtual processor
    coordinates in OpenMP because we need to pass a local variable to a
    function without passing it in the arguments.

    Use thead local storage (TLS) to have it local to each OpenMP thread.
 */
extern __thread int P4A_vp_coordinate[P4A_vp_dim_max];


/** Get the coordinate of the virtual processor in X (first) dimension
    in OpenMP emulation. 
*/
#define P4A_vp_0 P4A_vp_coordinate[0]


/** Get the coordinate of the virtual processor in Y (second) dimension in
    OpenMP emulation 
*/
#define P4A_vp_1 P4A_vp_coordinate[1]


/** Get the coordinate of the virtual processor in Z (second) dimension in
    OpenMP emulation 
*/
#define P4A_vp_2 P4A_vp_coordinate[2]

/** @} */

#endif //P4A_ACCEL_WRAPPER_OPENMP_H
