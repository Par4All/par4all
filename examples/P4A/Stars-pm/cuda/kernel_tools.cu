#include "kernel_tools.h"

/**
 *  This is used by P4A_TIMING routine to compute elapsed time for a kernel
 *  execution (in ms)
 */
float p4a_timing_elapsedTime = -1;
cudaEvent_t p4a_start_event, p4a_stop_event;
