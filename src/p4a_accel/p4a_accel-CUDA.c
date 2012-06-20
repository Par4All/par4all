#ifdef P4A_ACCEL_CUDA


int p4a_max_threads_per_block = P4A_CUDA_THREAD_MAX;

struct cudacc {
  int major;
  int minor;
};

static int computeCoresForDevice(struct cudacc cc, int multiproc) {
  if(cc.major==1) {
    return 8*multiproc;
  } else if (cc.major==2 and cc.minor==0) {
    return 32*multiproc;
  } else if (cc.major==2 and cc.minor==1) {
    return 48*multiproc;
  } else if (cc.major==3 and cc.minor==0) {
    return 192*multiproc;
  } else {
    P4A_dump_message("Unknown architecture for compute capability %d.%d, assume 32 cores per SM, please report to support@par4all.org\n",cc.major,cc.minor);
    return 32*multiproc;
  }
}

// Print device properties
static void displayCudaDevices(int nDevices)
{
  // Derived from http://gpucoder.livejournal.com/1064.html
  for(int i = 0; i < nDevices; ++i) {
    // Get device properties
    P4A_dump_message("*********** CUDA Device #%d ***********\n", i);
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, i);
    P4A_dump_message("Major revision number:         %d\n",  devProp.major);
    P4A_dump_message("Minor revision number:         %d\n",  devProp.minor);
    P4A_dump_message("Name:                          %s\n",  devProp.name);
    P4A_dump_message("Number of multiprocessors:     %d\n",  devProp.multiProcessorCount);
    P4A_dump_message("Number of cores:               %d\n",  computeCoresForDevice((struct cudacc){devProp.major,devProp.minor},devProp.multiProcessorCount));
    P4A_dump_message("Total global memory:           %zu\n", devProp.totalGlobalMem);
    P4A_dump_message("Total shared memory per block: %zu\n", devProp.sharedMemPerBlock);
    P4A_dump_message("Total registers per block:     %d\n",  devProp.regsPerBlock);
    P4A_dump_message("Warp size:                     %d\n",  devProp.warpSize);
    P4A_dump_message("Maximum memory pitch:          %lu\n", devProp.memPitch);
    P4A_dump_message("Maximum threads per block:     %d\n",  devProp.maxThreadsPerBlock);
    for (int i = 0; i < 3; ++i)
      P4A_dump_message("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
    for (int i = 0; i < 3; ++i)
      P4A_dump_message("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
    P4A_dump_message("Clock rate:                    %d\n",  devProp.clockRate);
    P4A_dump_message("Total constant memory:         %lu\n",  devProp.totalConstMem);
    P4A_dump_message("Texture alignment:             %lu\n",  devProp.textureAlignment);
    P4A_dump_message("Concurrent copy and execution: %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));
    P4A_dump_message("Kernel execution timeout:      %s\n",  (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
  }
}


void p4a_init_cuda_accel() {
  // Main p4a initialization
  p4a_main_init();

  // Number of CUDA devices
  int num_devices;
  // This also initialize the CUDA runtime
  toolTestExecMessage(cudaGetDeviceCount(&num_devices),"p4a_init_cuda_accel() cannot find any CUDA capable device, aborting.\n");
  if(num_devices==0) {
    P4A_dump_message("p4a_init_cuda_accel() cannot find any CUDA capable device, aborting.\n");
    exit(-1);
  }
  P4A_skip_debug(1,P4A_dump_message("There are %d CUDA devices, increase debug level for details or set P4A_CUDA_DEVICE to chose a particular one.\n", num_devices));


  // Display devices information
  P4A_skip_debug(2,displayCudaDevices(num_devices))

  /*
   * Chose one device accordingly to an environment variable, or the one with the biggest number of cores
   */
  int chosen_device = 0;
  char *env_cuda_device = getenv ("P4A_CUDA_DEVICE");
  if(env_cuda_device) {
    P4A_dump_message("P4A_CUDA_DEVICE environment variable is set : '%s'\n",env_cuda_device);
    errno = 0;
    int env_device = strtol(env_cuda_device, NULL,10);
    if(errno==0) {
      // Sanity check
      if(env_device>=0 && env_device<num_devices) {
        P4A_dump_message("Choosing CUDA device %d\n",env_device);
        chosen_device = env_device;
      } else {
        P4A_dump_message("Invalid value for P4A_CUDA_DEVICE  : %s\n",env_cuda_device);
        exit(-1);
      }
    } else {
      P4A_dump_message("Invalid value for P4A_CUDA_DEVICE: %s\n",env_cuda_device);
      exit(-1);
    }
  } else {
    // Automatic selection: take the one with the max number of cores
    if (num_devices > 1) {
      int max_cores = 0;
      for (int dev=0; dev<num_devices; dev++) {
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, dev);
        struct cudacc cc = {devProp.major,devProp.minor};
        int cores = computeCoresForDevice(cc,devProp.multiProcessorCount);
        if (max_cores < cores) {
          max_cores = cores;
        }
      }
    }
  }

  // Set the device
  P4A_skip_debug(1,P4A_dump_message("CUDA device chosen : %d\n",chosen_device));
  cudaSetDevice(chosen_device);



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
      P4A_dump_message("Invalid value for P4A_MAX_TPB : %s\n",env_p4a_max_tpb);
    }
  }

#if CUDA_VERSION >= 4000
  // We prefer to have more L1 cache and less shared since we don't make use of it
  cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 );
  toolTestExecMessage(cudaGetLastError(),"P4A CUDA cache config failed");
#endif

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
