extern "C" {
#include <stdio.h>
#include <p4a_accel.h>
}

#include <map>

/**
 * This global handle mappings between host memory area and corresponding area
 * in the GPU memory
 */
typedef std::map<void *, void *> p4a_memory_mapping;
static p4a_memory_mapping memory_mapping;



/**
 * This function return a pointer on the area in the GPU memory corresponding to
 * an area in the host memory. If the mapping doesn't exist, an area is
 * allocated in the GPU memory and the mapping is created
 *
 * @params host_ptr is the pointer in the host memory
 * @params size is the size of the area, in case we need to allocate it
 * @return a pointer in the GPU memory corresponding to host_ptr
 */
void * P4A_runtime_host_ptr_to_accel_ptr(void *host_ptr, size_t size) {
  void *accel_ptr;


/* Mandatory in openmp accel emulation, because of macro implementation */
#pragma omp critical
  {
    // Try to convert host_ptr into corresponding accel_ptr
    p4a_memory_mapping::iterator it = memory_mapping.find(host_ptr);

    if(it != memory_mapping.end()) {
      // We found a mapping
      accel_ptr = it->second;
    } else {
      // If there's no existing mapping, then allocating a new one and register it

#ifdef P4A_RUNTIME_DEBUG
      fprintf(stderr,"[%s:%d] allocating %zd bytes of memory for host %p !\n",__FUNCTION__,__LINE__, size, host_ptr);
#endif

      // Alloc && map
      P4A_accel_malloc(&accel_ptr, size);
      memory_mapping.insert(it, p4a_memory_mapping::value_type(host_ptr,
                                                               accel_ptr));

      it = memory_mapping.find(host_ptr);
      if(it == memory_mapping.end()) {
        fprintf(stderr, "Error insert failed !\n");
        checkP4ARuntimeInitialized();
        exit(-1);
      }

    }
  }


#ifdef P4A_RUNTIME_DEBUG
  fprintf(stderr,"[%s:%d] Resolved accel adress for host %p is %p\n",__FUNCTION__,__LINE__, host_ptr, accel_ptr);
#endif


  return accel_ptr;
}



/**
 * This function copy "size" bytes from "host_ptr" to the corresponding area in
 * the GPU memory. The memory area can be allocated if there's none existing
 *
 * @params host_ptr is the pointer in the host memory
 * @params size is the size of the area
 * @return nothing
 */
void P4A_runtime_copy_to_accel(void *host_ptr, size_t size /* in bytes */) {
  // Resolve (possibly allocate) the pointer in the GPU memory space
  void *accel_ptr = P4A_runtime_host_ptr_to_accel_ptr(host_ptr,size);

#ifdef P4A_RUNTIME_DEBUG
    fprintf(stderr,"[%s:%d] : Copying %zd bytes of memory from host %p to accel %p !\n",__FUNCTION__,__LINE__,size, host_ptr,accel_ptr);
#endif

  // Copy data
  P4A_copy_to_accel(size,host_ptr,accel_ptr);

}




/**
 * This function copy "size" bytes to "host_ptr" from the corresponding area in
 * the GPU memory. An abort is raised if no mapping if found.
 *
 * @params host_ptr is the pointer in the host memory
 * @params size is the size of the area
 * @return nothing
 */
void P4A_runtime_copy_from_accel(void *host_ptr, size_t size /* in bytes */) {
  void *accel_ptr;

/* Mandatory in openmp accel emulation, because of macro implementation */
  #pragma omp critical
  {
    // Try to convert host_ptr into corresponding accel_ptr
    p4a_memory_mapping::iterator it = memory_mapping.find(host_ptr);

    if(it == memory_mapping.end()) {
      // We didn't found a mapping
      fprintf(stderr,
              "[%s:%d] Error : no mapping in the accel memory for host ptr %p ! Abort...\n",
              __FUNCTION__,
              __LINE__,
              host_ptr);
      checkP4ARuntimeInitialized()
      exit(-1);
    } else {
      // A mapping was found !
      accel_ptr = it->second;
    }
  }

#ifdef P4A_RUNTIME_DEBUG
  fprintf(stderr,"[%s:%d] Copying %zd bytes of memory from accel %p to host %p !\n",__FUNCTION__,__LINE__,size, accel_ptr,host_ptr);
#endif

  // Copy data
  P4A_copy_from_accel(size,host_ptr,accel_ptr);
}
