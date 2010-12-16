/** @file

    API of Par4All C for accelerator

    @mainpage

    The Par4All API leverages the code to program an hardware accelerator
    with some helper functions.

    It allows simpler and more portable code to be generated with PIPS,
    but it can also be used to ease heterogeneous parallel manual
    programming.

    Should target Ter\@pix and SCALOPES accelerators, CUDA and OpenCL.

    Funded by the FREIA (French ANR), TransMedi\@ (French Pôle de
    Compétitivité Images and Network) and SCALOPES (Artemis European
    Project project)

    "mailto:Stephanie.Even@enstb.org"
    "mailto:Ronan.Keryell@hpc-project.com"
*/

/* License BSD */

#ifndef P4A_ACCEL_H
#define P4A_ACCEL_H

/* For size_t: */
#include <stddef.h>

/** Note that in CUDA and OpenCL there is 3 dimensions max: */
enum { P4A_vp_dim_max = 3 };

extern double P4A_accel_timer_stop_and_float_measure();

#if defined(P4A_ACCEL_CUDA) && defined(P4A_ACCEL_OPENMP)
#error "You cannot have both P4A_ACCEL_CUDA and P4A_ACCEL_OPENMP defined, yet"
#endif

/* Some common function prototypes. */

/** Prototype for allocating memory on the hardware accelerator.

    @param[out] address is the address of a variable that is updated by
    this macro to contains the address of the allocated memory block

    @param[in] size is the size to allocate in bytes
*/
void P4A_accel_malloc(void **address, size_t size);


/** Prototype for freeing memory on the hardware accelerator/

    @param[in] address points to a previously allocated memory zone for
    the hardware accelerator
*/
void P4A_accel_free(void *address);


/** Prototype for copying a scalar from the host to a memory zone in the
    hardware accelerator.
*/
void P4A_copy_to_accel(size_t element_size,
		       const void *host_address,
		       void *accel_address);


/** Prototype for copying a scalar from the hardware accelerator memory to
    the host.
*/
void P4A_copy_from_accel(size_t element_size,
			 void *host_address,
			 const void *accel_address);


/** Prototype for copying a 1D memory zone from the host to a compact
    memory zone in the hardware accelerator.
*/
void P4A_copy_to_accel_1d(size_t element_size,
			  size_t d1_size,
			  size_t d1_block_size,
			  size_t d1_offset,
			  const void *host_address,
			  void *accel_address);


/** Prototype for copying memory from the hardware accelerator to a 1D
    array in the host.
*/
void P4A_copy_from_accel_1d(size_t element_size,
			    size_t d1_size,
			    size_t d1_block_size,
			    size_t d1_offset,
			    void *host_address,
			    const void *accel_address);


/** Prototype for copying a 2D memory zone from the host to a compact
    memory zone in the hardware accelerator.
*/
void P4A_copy_to_accel_2d(size_t element_size,
			  size_t d1_size, size_t d2_size,
			  size_t d1_block_size, size_t d2_block_size,
			  size_t d1_offset,   size_t d2_offset,
			  const void *host_address,
			  void *accel_address);


/** Prototype for copying memory from the hardware accelerator to a 2D
    array in the host.
*/
void P4A_copy_from_accel_2d(size_t element_size,
			    size_t d1_size, size_t d2_size,
			    size_t d1_block_size, size_t d2_block_size,
			    size_t d1_offset, size_t d2_offset,
			    void *host_address,
			    const void *accel_address);


/** Prototype for copying a 3D memory zone from the host to a compact
    memory zone in the hardware accelerator.
*/
void P4A_copy_to_accel_3d(size_t element_size,
			  size_t d1_size, size_t d2_size, size_t d3_size,
			  size_t d1_block_size, size_t d2_block_size, size_t d3_block_size,
			  size_t d1_offset,   size_t d2_offset, size_t d3_offset,
			  const void *host_address,
			  void *accel_address);


/** Prototype for copying memory from the hardware accelerator to a 3D
    array in the host.
*/
void P4A_copy_from_accel_3d(size_t element_size,
			    size_t d1_size, size_t d2_size, size_t d3_size,
			    size_t d1_block_size, size_t d2_block_size, size_t d3_block_size,
			    size_t d1_offset, size_t d2_offset, size_t d3_offset,
			    void *host_address,
			    const void *accel_address);


/** Prototype for copying memory from a 4D array in the host to the hardware
    accelerator.
*/
void P4A_copy_to_accel_4d(size_t element_size,
          size_t d1_size, size_t d2_size, size_t d3_size, size_t d4_size,
          size_t d1_block_size, size_t d2_block_size, size_t d3_block_size, size_t d4_block_size,
          size_t d1_offset, size_t d2_offset, size_t d3_offset, size_t d4_offset,
          const void *host_address,
          void *accel_address);


/** Prototype for copying memory from the hardware accelerator to a 4D
    array in the host.
*/
void P4A_copy_from_accel_4d(size_t element_size,
          size_t d1_size, size_t d2_size, size_t d3_size, size_t d4_size,
          size_t d1_block_size, size_t d2_block_size, size_t d3_block_size, size_t d4_block_size,
          size_t d1_offset, size_t d2_offset, size_t d3_offset, size_t d4_offset,
          void *host_address,
          const void *accel_address);


/** A macro to enable or skip debug instructions

    Just define P4A_DEBUG to have debug information at runtime

    @param debug_stuff is some text that is included texto if P4A_DEBUG is
    defined
*/
#ifdef P4A_DEBUG
#define P4A_skip_debug(debug_stuff) debug_stuff
#else
#define P4A_skip_debug(debug_stuff)
#endif


/** Output a debug message à la printf */
#define P4A_dump_message(...)						\
  fprintf(stderr, " P4A: " __VA_ARGS__)

/** Output where we are */
#define P4A_dump_location()						\
  fprintf(stderr, " P4A: line %d of function %s in file \"%s\":\n",	\
          __LINE__, __func__, __FILE__)


/** A macro to compute an minimum value of 2 values

    Since it is a macro, beware of side effects...
*/
#define P4A_min(a, b) ((a) > (b) ? (b) : (a))

#ifdef P4A_ACCEL_CUDA
#include <p4a_accel-CUDA.h>
#else
#ifdef P4A_ACCEL_OPENMP
#include <p4a_accel-OpenMP.h>
#else
#error "You have to define either P4A_ACCEL_CUDA or P4A_ACCEL_OPENMP"
#endif
#endif

#endif //P4A_ACCEL_H
