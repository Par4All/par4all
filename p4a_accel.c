
/** Stop a timer on the accelerator and get double ms time

    @addtogroup P4A_OpenMP_time_measure
 */
/* #ifdef P4A_ACCEL_OPENMP

double P4A_ACCEL_TIMER_STOP_AND_FLOAT_MEASURE() {
  double run_time;
  gettimeofday(&p4a_time_end, NULL);
  // Take care of the non-associativity in floating point :-) 
  run_time = (p4a_time_end.tv_sec - p4a_time_begin.tv_sec)
    + (p4a_time_end.tv_usec - p4a_time_begin.tv_usec)*1e-6;
  return run_time;
}

#endif
*/


#ifdef P4A_ACCEL_CUDA

/** Stop a timer on the accelerator and get float time in second

    @addtogroup P4A_cuda_time_measure
 */
float P4A_ACCEL_TIMER_STOP_AND_FLOAT_MEASURE() {
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

#else
#include <p4a_accel.h>
/** Stop a timer on the accelerator and get double ms time

    @addtogroup P4A_OpenMP_time_measure
 */
double P4A_ACCEL_TIMER_STOP_AND_FLOAT_MEASURE() {
  double run_time;
  gettimeofday(&p4a_time_end, NULL);
  /* Take care of the non-associativity in floating point :-) */
  run_time = (p4a_time_end.tv_sec - p4a_time_begin.tv_sec)
    + (p4a_time_end.tv_usec - p4a_time_begin.tv_usec)*1e-6;
  return run_time;
}


#endif
