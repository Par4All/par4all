#ifdef P4A_ACCEL_CUDA


int p4a_max_threads_per_block = P4A_CUDA_THREAD_MAX;

void p4a_init_cuda_accel() {
  // Main p4a initialization
  p4a_main_init();

  // Timing stuff
  if(p4a_timing) {
    toolTestExec(cudaEventCreate(&p4a_start_event));
    toolTestExec(cudaEventCreate(&p4a_stop_event));
  }

  // Threads per blocks
  char *env_p4a_max_tpb = getenv ("P4A_MAX_TPB");
  if(env_p4a_max_tpb) {
    errno = 0;
    int tpb = strtol(env_p4a_max_tpb, NULL,10);
    if(errno==0) {
      P4A_dump_message("Setting max TPB to %d\n",tpb);
      p4a_max_threads_per_block = tpb;
    } else {
      fprintf(stderr,"Invalid value for P4A_MAX_TPB : %s\n",env_p4a_max_tpb);
    }
  }

  // We prefer to have more L1 cache and less shared since we don't make use of it
  cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 );
  toolTestExecMessage(cudaGetLastError(),"P4A CUDA cache config failed");

}



/** To do basic time measure. Do not nest... */

cudaEvent_t p4a_start_event, p4a_stop_event;



/** Allocate memory on the hardware accelerator in Cuda mode.

    @param[out] address is the address of a variable that is updated by
    this function to contains the address of the allocated memory block

    @param[in] size is the size to allocate in bytes
*/
void P4A_accel_malloc(void **address, size_t size) {
  toolTestExec(cudaMalloc(address, size));
}


/** Free memory on the hardware accelerator in Cuda mode

    @param[in] address points to a previously allocated memory zone for
    the hardware accelerator
*/
void P4A_accel_free(void *address) {
  toolTestExec(cudaFree(address));
}


/** Copy a scalar from the hardware accelerator to the host

 It's a wrapper around CudaMemCopy*.

 Do not change the place of the pointers in the API. The host address
 is always first...

 @param[in] element_size is the size of one element of the array in
 byte

 @param[out] host_address point to the element on the host to write
 into

 @param[in] accel_address refer to the compact memory area to read
 data. In the general case, accel_address may be seen as a unique idea (FIFO)
 and not some address in some memory space.
 */
void P4A_copy_from_accel(size_t element_size,
    void *host_address,
    void const*accel_address) {
  P4A_TIMING_accel_timer_start;

  toolTestExecMessage(cudaMemcpy(host_address,accel_address,element_size,cudaMemcpyDeviceToHost),
                      "It seems that cudaMemcpy failed, but to be sure run again with environment "
                      "variable P4A_DEBUG set to a value >1. This failure is probably an internal "
                      "Par4All bug, please report. The runtime error is:"    );

  if(p4a_timing) {
    P4A_TIMING_accel_timer_stop;
    P4A_TIMING_elapsed_time(p4a_timing_elapsedTime);
    P4A_dump_message("Copied %zd bytes of memory from accel %p to host %p : "
                     "%.1fms - %.2fGB/s\n",element_size, accel_address,host_address,
                      p4a_timing_elapsedTime,(float)element_size/(p4a_timing_elapsedTime*1000000));

  }
}

/** Copy a scalar from the host to the hardware accelerator

 It's a wrapper around CudaMemCopy*.

 Do not change the place of the pointers in the API. The host address
 is always before the accel address...

 @param[in] element_size is the size of one element of the array in
 byte

 @param[out] host_address point to the element on the host to write
 into

 @param[in] accel_address refer to the compact memory area to read
 data. In the general case, accel_address may be seen as a unique idea (FIFO)
 and not some address in some memory space.
 */
void P4A_copy_to_accel(size_t element_size,
    void const*host_address,
    void *accel_address) {
  P4A_TIMING_accel_timer_start;

  toolTestExecMessage(cudaMemcpy(accel_address,host_address,element_size,cudaMemcpyHostToDevice),
                      "It seems that cudaMemcpy failed, but to be sure run again with environment "
                      "variable P4A_DEBUG set to a value >1. This failure is probably an internal "
                      "Par4All bug, please report. The runtime error is:"    );

  if(p4a_timing) {
    P4A_TIMING_accel_timer_stop;
    P4A_TIMING_elapsed_time(p4a_timing_elapsedTime);
    P4A_dump_message("Copied %zd bytes of memory from host %p to accel %p : "
                      "%.1fms - %.2fGB/s\n",element_size, host_address,accel_address,
                      p4a_timing_elapsedTime,(float)element_size/(p4a_timing_elapsedTime*1000000));
  }
}




#endif //P4A_ACCEL_CUDA
