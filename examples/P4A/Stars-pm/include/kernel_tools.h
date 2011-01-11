#ifndef __KERNEL_TOOLS__H_
#define __KERNEL_TOOLS__H_


#include <stdio.h>

/*********************************
 * Derived from p4a_accel-Cuda.h
 */

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


/** Output a debug message Ã  la printf */
#define P4A_dump_message(...)           \
  fprintf(stderr, " P4A: " __VA_ARGS__)

/** Output where we are */
#define P4A_dump_location()           \
  fprintf(stderr, " P4A: line %d of function %s in file \"%s\":\n", \
          __LINE__, __func__, __FILE__)

#define toolTestExec(error)   checkErrorInline        (error, __FILE__, __LINE__)
#define toolTestExecMessage(message)  checkErrorMessageInline (message, __FILE__, __LINE__)

inline void checkErrorInline(cudaError_t error,
                             const char *currentFile,
                             const int currentLine) {
  if(cudaSuccess != error) {
    fprintf(stderr,
            "File %s - Line %i - The runtime error is %s\n",
            currentFile,
            currentLine,
            cudaGetErrorString(error));
    exit(-1);
  }
}

inline void checkErrorMessageInline(const char *errorMessage,
                                    const char *currentFile,
                                    const int currentLine) {
  cudaError_t error = cudaGetLastError();
  if(cudaSuccess != error) {
    fprintf(stderr,
            "File %s - Line %i - %s : %s\n",
            currentFile,
            currentLine,
            errorMessage,
            cudaGetErrorString(error));
    exit(-1);
  }
#ifdef P4A_DEBUG
  error = cudaThreadSynchronize();
  if(cudaSuccess != error) {
    fprintf(stderr, "File %s - Line %i - Error after ThreadSynchronize %s : %s\n", currentFile, currentLine, errorMessage, cudaGetErrorString(error));
    exit(-1);
  }
#endif
}

/** Start a timer on the accelerator in CUDA */
#define P4A_accel_timer_start toolTestExec(cudaEventRecord(p4a_start_event, 0))

/** Stop a timer on the accelerator in CUDA */
#define P4A_accel_timer_stop toolTestExec(cudaEventRecord(p4a_stop_event, 0))

#ifdef P4A_TIMING
// Set of routine for timing kernel executions
extern float p4a_timing_elapsedTime;
extern cudaEvent_t p4a_start_event, p4a_stop_event;
#define P4A_TIMING_accel_timer_start P4A_accel_timer_start
#define P4A_TIMING_accel_timer_stop \
{ P4A_accel_timer_stop; \
  cudaEventSynchronize(p4a_stop_event); \
}
#define P4A_TIMING_elapsed_time(elapsed) \
    cudaEventElapsedTime(&elapsed, p4a_start_event, p4a_stop_event);
#define P4A_TIMING_get_elapsed_time \
{   P4A_TIMING_elapsed_time(&p4a_timing_elapsedTime); \
    p4a_timing_elapsedTime; }
#define P4A_TIMING_display_elasped_time(msg) \
{ P4A_TIMING_elapsed_time(p4a_timing_elapsedTime); \
  P4A_dump_message("Time for '%s' : %fms\n",  \
                   #msg,      \
                   p4a_timing_elapsedTime ); }

#define P4A_init_timing \
  do {              \
    toolTestExec(cudaEventCreate(&p4a_start_event));  \
    toolTestExec(cudaEventCreate(&p4a_stop_event)); \
  } while (0);


#else
#define P4A_TIMING_accel_timer_start
#define P4A_TIMING_accel_timer_stop
#define P4A_TIMING_elapsed_time(elapsed)
#define P4A_TIMING_get_elapsed_time
#define P4A_TIMING_display_elasped_time(msg)
#define P4A_init_timing
#endif //P4A_TIMING

#define P4A_launch_kernel(dimGrid,dimBlock,kernel, ...)      \
  do {                  \
    P4A_skip_debug(P4A_dump_location());        \
    P4A_skip_debug(P4A_dump_message("Invoking kernel %s with %s\n", \
            #kernel,       \
            #__VA_ARGS__));      \
    P4A_TIMING_accel_timer_start; \
    kernel<<<dimGrid,dimBlock>>>(__VA_ARGS__);      \
    toolTestExecMessage("P4A CUDA kernel execution failed");      \
    P4A_TIMING_accel_timer_stop; \
    P4A_TIMING_display_elasped_time(kernel); \
  } while (0)


#endif // __KERNEL_TOOLS__H_
