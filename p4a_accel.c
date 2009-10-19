/** @file

    API of Par4All C to OpenMP

    This file is an implementation of the C to CUDA Par4All API.

    Funded by the FREIA (French ANR), TransMedi\@ (French Pôle de
    Compétitivité Images and Network) and SCALOPES (Artemis European
    Project project)

    "mailto:Stephanie.Even@enstb.org"
    "mailto:Ronan.Keryell@hpc-project.com"
*/

#include <p4a_accel.h>


#ifdef P4A_ACCEL_OPENMP

/** Stop a timer on the accelerator and get double ms time

    @addtogroup P4A_OpenMP_time_measure
*/
double P4A_accel_timer_stop_and_float_measure() {
  double run_time;
  gettimeofday(&p4a_time_end, NULL);
  /* Take care of the non-associativity in floating point :-) */
  run_time = (p4a_time_end.tv_sec - p4a_time_begin.tv_sec)
    + (p4a_time_end.tv_usec - p4a_time_begin.tv_usec)*1e-6;
  return run_time;
}


/** This is a global variable used to simulate P4A virtual processor
    coordinates in OpenMP because we need to pass a local variable to a
    function without passing it in the arguments.

    Use thead local storage to have it local to each OpenMP thread.

    With __thread, it looks like this declaration cannot be repeated in
    the .h without any extern.
 */
__thread int P4A_vp_coordinate[P4A_vp_dim_max];

#endif


#ifdef P4A_ACCEL_CUDA

#include <cutil_inline.h>

/** To do basic time measure. Do not nest... */

cudaEvent_t p4a_start_event, p4a_stop_event;

/** Stop a timer on the accelerator and get float time in second

    @addtogroup P4A_cuda_time_measure
*/

double P4A_accel_timer_stop_and_float_measure() {
  float execution_time;
  cutilSafeCall(cudaEventRecord(p4a_stop_event, 0));
  cutilSafeCall(cudaEventSynchronize(p4a_stop_event));
  /* Get the time in ms: */
  cutilSafeCall(cudaEventElapsedTime(&execution_time,
				     p4a_start_event,
				     p4a_stop_event));
  /* Return the time in second: */
  return execution_time*1e-3;
}

#endif
