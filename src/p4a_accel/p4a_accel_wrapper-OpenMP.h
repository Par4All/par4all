//#ifndef P4A_ACCEL_WRAPPER_OPENMP_H
//#define P4A_ACCEL_WRAPPER_OPENMP_H

enum { P4A_vp_dim_max = 3 };

extern __thread int P4A_vp_coordinate[P4A_vp_dim_max];

/** A declaration attribute of a hardware-accelerated kernel in OpenMP

    Nothing by default for OpenMP since it is normal C
*/
#define P4A_accel_kernel void


/** A declaration attribute of a hardware-accelerated kernel called from
    the host in OpenMP

    Nothing by default since it is homogeneous programming model
*/
#define P4A_accel_kernel_wrapper void


/** The address space visible for all functions. 
    Allocation in the global memory pool.
 */
#define P4A_accel_global_address 

/** The address space in the global memory pool but in read-only mode.
 */
#define P4A_accel_constant_address const

/** 
 */
#define P4A_accel_local_address 


/** Get the coordinate of the virtual processor in X (first) dimension in
    OpenMP emulation */
#define P4A_vp_0 P4A_vp_coordinate[0]


/** Get the coordinate of the virtual processor in Y (second) dimension in
    OpenMP emulation */
#define P4A_vp_1 P4A_vp_coordinate[1]


/** Get the coordinate of the virtual processor in Z (second) dimension in
    OpenMP emulation */
#define P4A_vp_2 P4A_vp_coordinate[2]


//#endif //P4A_ACCEL_WRAPPER_OPENMP_H
