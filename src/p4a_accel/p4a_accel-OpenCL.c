#ifdef P4A_ACCEL_OPENCL

#include <string.h>

/** @author Stéphanie Even

    Aside the procedure defined for CUDA or OpenMP, OpenCL needs
    loading the kernel from a source file.

    Procedure to manage a kernel list have thus been created for the C version.
    In C++, this a map.

    C and C++ versions are implemented.
 */


/** @defgroup P4A_log Debug messages and errors tracking.
    
    @{
*/

cl_int p4a_global_error=0;
static cl_device_id p4a_device_id=0;
static cl_program p4a_program=0;

#define P4A_DEBUG_BUFFER_SIZE 10000
static char p4a_debug_buffer[P4A_DEBUG_BUFFER_SIZE];

/**
 * Callback for OpenCL runtime, display some error asynchronously
 */
void p4a_opencl_notify(const char *errinfo,
                  const void *private_info,
                  size_t cb,
                  void *user_data ) {
  P4A_dump_message("OpenCL runtime error, %s : '%s'"
      "(we ignore it at that time and the program is likely to fail...\n",
      (char *)user_data,(char *)errinfo);
}




#define p4a_opencl_dump_platform_info(platform, param_name) \
    p4a_opencl_dump_platform_info_(platform, param_name,#param_name)
/**
 * Dump an information about an OpenCL platform
 */
void p4a_opencl_dump_platform_info_(cl_platform_id platform,
                                   cl_platform_info param_name,
                                   const char header[]) {
  size_t str_size;
  // Get the string size for the requested param
  p4a_global_error=clGetPlatformInfo(platform, param_name, 0, NULL, &str_size);
  P4A_test_execution_with_message("clGetPlatformInfo");
  if(str_size<=0) {
    P4A_dump_message("Error : clGetPlatformInfo returns size <= 0 (%zu)\n",str_size);
    p4a_clean(EXIT_FAILURE);
  }

  // Allocate a buffer for the string
  char buffer[str_size+1];

  // Get the string for this param
  p4a_global_error=clGetPlatformInfo(platform, param_name, str_size, &buffer, NULL);
  P4A_test_execution_with_message("clGetPlatformInfo");

  // Dump it !
  fprintf(stderr,"    - %s: %s\n",header,buffer);
}

/**
 * Display and choose an OpenCL platform
 */
cl_platform_id p4a_opencl_get_platform() {
  // Get the number of plaforms available
  cl_uint num_platforms;
  p4a_global_error=clGetPlatformIDs(0, NULL, &num_platforms);
  P4A_test_execution_with_message("clGetPlatformIDs");

  if(num_platforms <= 0) {
    P4A_dump_message("No OpenCL platforms found :-(\n");
    p4a_clean(EXIT_FAILURE);
  }

  cl_platform_id p4a_platform_ids[num_platforms];
  p4a_global_error=clGetPlatformIDs(num_platforms, p4a_platform_ids, NULL);
  P4A_test_execution_with_message("clGetPlatformIDs");

  // Display the platforms
  P4A_skip_debug(1,
  {
    P4A_dump_message("Available OpenCL platforms :\n");
    for(cl_uint num=0; num<num_platforms; ++num) {
      P4A_dump_message(" ** platform %d **\n",num);
      p4a_opencl_dump_platform_info(p4a_platform_ids[num],CL_PLATFORM_NAME);
      p4a_opencl_dump_platform_info(p4a_platform_ids[num],CL_PLATFORM_VENDOR);
      p4a_opencl_dump_platform_info(p4a_platform_ids[num],CL_PLATFORM_VERSION);
      p4a_opencl_dump_platform_info(p4a_platform_ids[num],CL_PLATFORM_PROFILE);
      p4a_opencl_dump_platform_info(p4a_platform_ids[num],CL_PLATFORM_EXTENSIONS);
    }
  });

  /* Choose the platform, check if an environment variable is set, else
   * take the first available platform...
   */
  int chosen_platform = 0;
  char *env_opencl_platform = getenv ("P4A_OPENCL_PLATFORM");
  if(env_opencl_platform) {
    P4A_dump_message("P4A_OPENCL_PLATFORM environment variable is set : '%s'\n",env_opencl_platform);
    errno = 0;
    int env_platform_id = strtol(env_opencl_platform, NULL,10);
    if(errno==0) {
      // Sanity check
      if(env_platform_id>=0 && env_platform_id<num_platforms) {
        P4A_dump_message("Choosing OpenCL platform %d\n",env_platform_id);
        chosen_platform=env_platform_id;
      } else {
        P4A_dump_message("Invalid value for P4A_OPENCL_PLATFORM: %s\n",env_opencl_platform);
        p4a_clean(EXIT_FAILURE);
      }
    } else {
      P4A_dump_message("Invalid value for P4A_OPENCL_PLATFORM: %s\n",env_opencl_platform);
    }
  }

  P4A_skip_debug(1,P4A_dump_message("OpenCL platform chosen : %d\n",chosen_platform));
  return p4a_platform_ids[chosen_platform];
}


#define p4a_opencl_dump_device_info(device, param_name) \
    p4a_opencl_dump_device_info_(device, param_name,#param_name)
/**
 * Dump an information about an OpenCL device
 */
void p4a_opencl_dump_device_info_(cl_device_id device,
                                  cl_device_info param_name,
                                  const char header[]) {
  size_t str_size;
  // Get the string size for the requested param
  p4a_global_error=clGetDeviceInfo(device, param_name, 0, NULL, &str_size);
  P4A_test_execution_with_message("clGetDeviceInfo");
  if(str_size<=0) {
    P4A_dump_message("Error : clGetDeviceInfo returns size <= 0 (%zu)\n",str_size);
    p4a_clean(EXIT_FAILURE);
  }

  // Allocate a buffer for the string
  char buffer[str_size+1];

  // Get the string for this param
  p4a_global_error=clGetDeviceInfo(device, param_name, str_size, &buffer, NULL);
  P4A_test_execution_with_message("clGetDeviceInfo");

  // Dump it !
  fprintf(stderr,"    - %s: %s\n",header,buffer);

}
/**
 * Display and choose a device for a given platform
 */
void p4a_opencl_init_devices(cl_platform_id platform_id) {
  // Get the number of devices
  cl_uint num_devices;
  p4a_global_error = clGetDeviceIDs(platform_id,
                                    CL_DEVICE_TYPE_ALL,
                                    0,
                                    NULL,
                                    &num_devices);
  P4A_test_execution_with_message("clGetDeviceIDs");

  if(num_devices <= 0) {
    P4A_dump_message("No devices found associated to this OpenCL platform :-(\n");
    p4a_clean(EXIT_FAILURE);
  }

  // Allocate spaces for devices
  cl_device_id devices[num_devices];

  // Get devices list
  p4a_global_error=clGetDeviceIDs(platform_id,
                                  CL_DEVICE_TYPE_ALL,
                                  num_devices,
                                  devices,
                                  NULL);
  P4A_test_execution_with_message("clGetDeviceIDs");

  /* Create a context for all devices */
  p4a_context = clCreateContext(0,
                                num_devices,
                                devices,
                                p4a_opencl_notify,
                                "from 'context'",
                                &p4a_global_error);
  P4A_test_execution_with_message("clCreateContext");

  P4A_skip_debug(1,
  {
    /* show info for each device in the context */
    P4A_dump_message("Available OpenCL devices for platform");
    p4a_opencl_dump_platform_info_(platform_id,CL_PLATFORM_NAME,"");
    for(int num = 0; num < num_devices; ++num ) {
      P4A_dump_message(" ** device %d **\n",num);
      p4a_opencl_dump_device_info(devices[num],CL_DEVICE_NAME);

      // Get the device type
      cl_device_type type;
      p4a_global_error=clGetDeviceInfo(devices[num],
                                       CL_DEVICE_TYPE,
                                       sizeof(cl_device_type),
                                       &type,
                                       NULL);
      P4A_test_execution_with_message("clGetDeviceInfo");
      fprintf(stderr,"    - CL_DEVICE_TYPE :");
      switch(type) {
        case CL_DEVICE_TYPE_CPU: fprintf(stderr,"CPU"); break;
        case CL_DEVICE_TYPE_GPU: fprintf(stderr,"GPU"); break;
        case CL_DEVICE_TYPE_ACCELERATOR: fprintf(stderr,"ACCELERATOR"); break;
        case CL_DEVICE_TYPE_DEFAULT: fprintf(stderr,"DEFAULT"); break;
        default: fprintf(stderr,"Unknown type"); p4a_clean(EXIT_FAILURE);
      }
      fprintf(stderr,"\n");

      p4a_opencl_dump_device_info(devices[num],CL_DEVICE_EXTENSIONS );
      p4a_opencl_dump_device_info(devices[num],CL_DEVICE_PROFILE);
      p4a_opencl_dump_device_info(devices[num],CL_DEVICE_VENDOR);
      p4a_opencl_dump_device_info(devices[num],CL_DEVICE_VERSION);
      p4a_opencl_dump_device_info(devices[num],CL_DRIVER_VERSION);
    }
  });

  /*
   * Chose one device accordingly to an environment variable, or take the first
   * one !
   */
  int chosen_device = 0;
  char *env_opencl_device = getenv ("P4A_OPENCL_DEVICE");
  if(env_opencl_device) {
    P4A_dump_message("P4A_OPENCL_DEVICE environment variable is set : '%s'\n",env_opencl_device);
    errno = 0;
    int env_device = strtol(env_opencl_device, NULL,10);
    if(errno==0) {
      // Sanity check
      if(env_device>=0 && env_device<num_devices) {
        P4A_dump_message("Choosing OpenCL device %d\n",env_device);
        chosen_device = env_device;
      } else {
        P4A_dump_message("Invalid value for P4A_OPENCL_DEVICE  : %s\n",env_opencl_device);
        p4a_clean(EXIT_FAILURE);
      }
    } else {
      P4A_dump_message("Invalid value for P4A_OPENCL_DEVICE: %s\n",env_opencl_device);
    }
  }

  P4A_skip_debug(1,P4A_dump_message("OpenCL device chosen : %d\n",chosen_device));
  /* Record the chosen device globaly */
  p4a_device_id=devices[chosen_device];

  cl_command_queue_properties p4a_queue_properties;
  if(p4a_timing) {
    p4a_queue_properties = CL_QUEUE_PROFILING_ENABLE;
  } else {
    p4a_queue_properties = 0;
  }

  /* Create an in-order queue for this device */
  p4a_queue=clCreateCommandQueue(p4a_context,
                                 p4a_device_id,
                                 p4a_queue_properties,
                                 &p4a_global_error);
  P4A_test_execution_with_message("clCreateCommandQueue");


}

void p4a_init_opencl() {
  // Common initialization for par4all Runtime
  p4a_main_init();

  /* Get an OpenCL platform */
  cl_platform_id p4a_platform_id = p4a_opencl_get_platform();

  /* Initialize the devices, could be a collection of device */
  p4a_opencl_init_devices(p4a_platform_id);

}


/** @author : Stéphanie Even

    In OpenCL, errorToString doesn't exists by default ...
    Macros have been picked from cl.h
 */
char * p4a_error_to_string(int error) 
{
  switch (error)
    {
    case CL_SUCCESS:
      return (char *)"Success";
    case CL_DEVICE_NOT_FOUND:
      return (char *)"Device Not Found";
    case CL_DEVICE_NOT_AVAILABLE:
      return (char *)"Device Not Available";
    case CL_COMPILER_NOT_AVAILABLE:
      return (char *)"Compiler Not Available";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:
      return (char *)"Mem Object Allocation Failure";
    case CL_OUT_OF_RESOURCES:
      return (char *)"Out Of Ressources";
    case CL_OUT_OF_HOST_MEMORY:
      return (char *)"Out Of Host Memory";
    case CL_PROFILING_INFO_NOT_AVAILABLE:
      return (char *)"Profiling Info Not Available";
    case CL_MEM_COPY_OVERLAP:
      return (char *)"Mem Copy Overlap";
    case CL_IMAGE_FORMAT_MISMATCH:
      return (char *)"Image Format Mismatch";
    case CL_IMAGE_FORMAT_NOT_SUPPORTED:
      return (char *)"Image Format Not Supported";
    case CL_BUILD_PROGRAM_FAILURE:
	#define CL_BUILD_PROGRAM_FAILURE_MSG "Build Program Failure : "
	strncat(p4a_debug_buffer,CL_BUILD_PROGRAM_FAILURE_MSG,P4A_DEBUG_BUFFER_SIZE);
	clGetProgramBuildInfo(p4a_program,
  				p4a_device_id,
  				CL_PROGRAM_BUILD_LOG ,
			  	P4A_DEBUG_BUFFER_SIZE,
			  	p4a_debug_buffer+strlen(CL_BUILD_PROGRAM_FAILURE_MSG),
			  	NULL);
      return (char *)p4a_debug_buffer;
    case CL_MAP_FAILURE:
      return (char *)"Map Failure";
    case CL_INVALID_VALUE:
      return (char *)"Invalid Value";
    case CL_INVALID_DEVICE_TYPE:
      return (char *)"Invalid Device Type";
    case CL_INVALID_PLATFORM:
      return (char *)"Invalid Platform";
    case CL_INVALID_DEVICE:
      return (char *)"Invalid Device";
    case CL_INVALID_CONTEXT:
      return (char *)"Invalid Context";
    case CL_INVALID_QUEUE_PROPERTIES:
      return (char *)"Invalid Queue Properties";
    case CL_INVALID_COMMAND_QUEUE:
      return (char *)"Invalid Command Queue";
    case CL_INVALID_HOST_PTR:
      return (char *)"Invalid Host Ptr";
    case CL_INVALID_MEM_OBJECT:
      return (char *)"Invalid Mem Object";
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
      return (char *)"Invalid Image Format Descriptor";
    case CL_INVALID_IMAGE_SIZE:
      return (char *)"Invalid Image Size";
    case CL_INVALID_SAMPLER:
      return (char *)"Invalid Sampler";
    case CL_INVALID_BINARY:
      return (char *)"Invalid Binary";
    case CL_INVALID_BUILD_OPTIONS:
      return (char *)"Invalid Build Options";
    case CL_INVALID_PROGRAM:
      return (char *)"Invalid Program";
    case CL_INVALID_PROGRAM_EXECUTABLE:
      return (char *)"Invalid Program Executable";
    case CL_INVALID_KERNEL_NAME:
      return (char *)"Invalid Kernel Name";
    case CL_INVALID_KERNEL_DEFINITION:
      return (char *)"Invalid Kernel Definition";
    case CL_INVALID_KERNEL:
      return (char *)"Invalid Kernel";
    case CL_INVALID_ARG_INDEX:
      return (char *)"Invalid Arg Index";
    case CL_INVALID_ARG_VALUE:
      return (char *)"Invalid Arg Value";
    case CL_INVALID_ARG_SIZE:
      return (char *)"Invalid Arg Size";
    case CL_INVALID_KERNEL_ARGS:
      return (char *)"Invalid Kernel Args";
    case CL_INVALID_WORK_DIMENSION:
      return (char *)"Invalid Work Dimension";
    case CL_INVALID_WORK_GROUP_SIZE:
      return (char *)"Invalid Work Group Size";
    case CL_INVALID_WORK_ITEM_SIZE:
      return (char *)"Invalid Work Item Size";
    case CL_INVALID_GLOBAL_OFFSET:
      return (char *)"Invalid Global Offset";
    case CL_INVALID_EVENT_WAIT_LIST:
      return (char *)"Invalid Event Wait List";
    case CL_INVALID_EVENT:
      return (char *)"Invalid Event";
    case CL_INVALID_OPERATION:
      return (char *)"Invalid Operation";
    case CL_INVALID_GL_OBJECT:
      return (char *)"Invalid GL Object";
    case CL_INVALID_BUFFER_SIZE:
      return (char *)"Invalid Buffer Size";
    case CL_INVALID_MIP_LEVEL:
      return (char *)"Invalid Mip Level";
    case CL_INVALID_GLOBAL_WORK_SIZE:
      return (char *)"Invalid Global Work Size";
    default:
      break;
    }
  return "Unknown";
} 

/** To quit properly OpenCL.
 */
void p4a_clean(int exitCode)
{
  //if(p4a_program)clReleaseProgram(p4a_program);
  if(p4a_kernel)clReleaseKernel(p4a_kernel);  
  if(p4a_queue)clReleaseCommandQueue(p4a_queue);
  if(p4a_context)clReleaseContext(p4a_context);
  checkP4ARuntimeInitialized();
  exit(exitCode);
}

/* @} */

/** @addtogroup P4A_time_measure

    @{
 */

double p4a_time = 0.;
double p4a_host_time = 0.;
cl_event p4a_event = NULL;

/** Counts the number of kernel to display nice timer / kernel load*/
int nKernel = 0;


/* @} */

/** @addtogroup P4A_kernel_call Accelerator kernel call

    @{
*/
cl_context p4a_context = NULL;
cl_command_queue p4a_queue = NULL;
cl_kernel p4a_kernel = NULL;  

#ifdef __cplusplus
std::map<std::string,struct p4a_cl_kernel *> p4a_kernels_map;
#else
//struct p4a_cl_kernel *p4a_kernels=NULL;
// Only used when creating the kernel list
//struct p4a_cl_kernel *last_kernel=NULL;

struct p4a_cl_kernel p4a_kernels_list[MAX_K];


/** @addtogroup P4A_call_parameters Parameters invocation.

    @{
    The C function to invoke clSetKernelArg where the reference to the
    parameter is the first var_arg.
*/
void p4a_setArguments(int i, char* varname, size_t size, void *ref_arg)
{
  //fprintf(stderr,"Argument %d : size = %u\n",i,size);
  p4a_global_error = clSetKernelArg(p4a_kernel,i,size, ref_arg); 
  P4A_test_execution_with_message("clSetKernelArg");
}
/** 
    @}
*/

/** Creation of a new kernel, initialisation and insertion in the kernel list.
 */
struct p4a_cl_kernel *new_p4a_kernel(const char *kernel)
{
  if (nKernel >= MAX_K) {
    printf("Over max kernel number: %d\n",MAX_K);
    exit(EXIT_FAILURE);
  }
  if (strlen(kernel) >= MAX_CAR-10) {
    printf("Kernel name toolong: %d\n",MAX_CAR);
    exit(EXIT_FAILURE);
  }
    
  strcpy(p4a_kernels_list[nKernel].name,kernel);
  p4a_kernels_list[nKernel].kernel = NULL;
  char* kernelFile;
  asprintf(&kernelFile,"./%s.cl",kernel);
  strcpy(p4a_kernels_list[nKernel].file_name,kernelFile);
  return &p4a_kernels_list[nKernel];
}

/** Search if the <kernel> is already in the list.
 */

struct p4a_cl_kernel *p4a_search_current_kernel(const char *kernel)
{
  int i = 0;
  while (i < nKernel && strcmp(p4a_kernels_list[i].name,kernel) != 0) {
    i++;
  }
  if (i < nKernel)
    return &p4a_kernels_list[i];
  else
    return NULL;
}

#endif //!__cplusplus

/** When launching the kernel
    - need to create the program from sources
    - select the kernel call within the program source

    Arguments are pushed from the ... list.
    Two solutions are proposed :
    - One, is actually commented : the parameters list (...)
      must contain the number of parameters, the sizeof(type)
      and the parameter as a reference &;
    - Second, the program source is parsed and the argument list is analyzed.
      The result of the parser is :
      - the number of arguments
      - the sizeof(argument type)

    @param kernel: Name of the kernel and the source MUST have the same
    name with .cl extension.
 */

void p4a_load_kernel(const char *kernel,...)
{
    
#ifdef __cplusplus
  std::string scpy(kernel);
  struct p4a_cl_kernel *k = p4a_kernels_map[scpy];
  
#else
  struct p4a_cl_kernel *k = p4a_search_current_kernel(kernel);
#endif

  // If not found ...
  if (!k) {
    P4A_skip_debug(1,P4A_dump_message("The kernel %s is loaded for the first time\n",kernel));

#ifdef __cplusplus
    k = new p4a_cl_kernel(kernel);
    p4a_kernels_map[scpy] = k;
#else
    new_p4a_kernel(kernel);
    k = &p4a_kernels_list[nKernel];
#endif
    
    nKernel++; 
    char* kernelFile =  k->file_name;
    P4A_skip_debug(2,P4A_dump_message("Program and Kernel creation from %s\n",kernelFile));
    size_t kernelLength=0;
    const char *comment = "// This kernel was generated for P4A\n";
  // Same design as the NVIDIA oclLoadProgSource
   char* cSourceCL = p4a_load_prog_source(kernelFile,
					   comment,
					   &kernelLength);
    if (cSourceCL == NULL) {
      P4A_dump_message("Error : loading source for kernel %s\n",kernelFile);
      p4a_clean(EXIT_FAILURE);
    }
      
    P4A_skip_debug(4,P4A_dump_message("Kernel length = %lu\n",kernelLength));
    
    /*Create and compile the program : 1 for 1 kernel */
    p4a_program=clCreateProgramWithSource(p4a_context,1,
					  (const char **)&cSourceCL,
					  &kernelLength,
					  &p4a_global_error);
    P4A_test_execution_with_message("clCreateProgramWithSource");
    p4a_global_error=clBuildProgram(p4a_program,0,NULL,NULL,NULL,NULL);
    P4A_test_execution_with_message("clBuildProgram");
    p4a_kernel=clCreateKernel(p4a_program,kernel,&p4a_global_error);
    k->kernel = p4a_kernel;
    P4A_test_execution_with_message("clCreateKernel");
    free(cSourceCL);
  }
  p4a_kernel = k->kernel;
 }

/** @author : Stéphanie Even

    Load and store the content of the kernel file in a string.
    Replace the oclLoadProgSource function of NVIDIA.
 */
char *p4a_load_prog_source(char *cl_kernel_file,const char *head,size_t *length)
{
  // Initialize the size and memory space
  struct stat buf;
  stat(cl_kernel_file,&buf);
  size_t size = buf.st_size;
  size_t len = strlen(head);
  char *source = (char *)malloc(len+size+1);
  strncpy(source,head,len);

  // A string pointer referencing to the position after the head
  // where the storage of the file content must begin
  char *p = source+len;

  // Open the file
  int in = open(cl_kernel_file,O_RDONLY);
  if (in==-1) {
    fprintf(stderr,"Bad kernel source reference : %s\n",cl_kernel_file);
    exit(EXIT_FAILURE);
  }
  
  // Read the file content
  int n=0;
  if ((n = read(in,(void *)p,size)) != (int)size) {
    fprintf(stderr,"Read was not completed : %d / %lu octets\n",n,size);
    exit(EXIT_FAILURE);
  }
  
  // Final string marker
  source[len+n]='\0';
  close(in);
  *length = size+len;
  return source;
}

/** @} */

/** @defgroup P4A_memory_allocation_copy Memory allocation and copy 

    @{
*/
/** Allocate memory on the hardware accelerator in OpenCL mode.

    @param[out] address is the address of a variable that is updated by
    this function to contains the address of the allocated memory block

    @param[in] size is the size to allocate in bytes
*/
void P4A_accel_malloc(void **address, size_t size) {
  // The mem flag can be : 
  // CL_MEM_READ_ONLY, 
  // CL_MEM_WRITE_ONLY, 
  // CL_MEM_READ_WRITE, 
  // CL_MEM_USE_HOST_PTR
  *address=(void *)clCreateBuffer(p4a_context,
				  CL_MEM_READ_WRITE,
				  size,
				  NULL,
				  &p4a_global_error);
  P4A_test_execution_with_message("Memory allocation via clCreateBuffer");
}

/** Free memory on the hardware accelerator in OpenCL mode

    @param[in] address points to a previously allocated memory zone for
    the hardware accelerator
*/
void P4A_accel_free(void *address) 
{
  p4a_global_error=clReleaseMemObject((cl_mem)address);
  P4A_test_execution_with_message("clReleaseMemObject");
}

/** Copy a scalar from the hardware accelerator to the host

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
			 const void *accel_address) 
{
#ifdef P4A_TIMING
  P4A_TIMING_accel_timer_start;
#endif

  p4a_global_error=clEnqueueReadBuffer(p4a_queue,
				       (cl_mem)accel_address,
				       CL_TRUE, // synchronous read
				       0,
				       element_size,
				       host_address,
				       0,
				       NULL,
				       &p4a_event);
  P4A_test_execution_with_message("clEnqueueReadBuffer");

  if(p4a_timing) {
    P4A_TIMING_elapsed_time(p4a_timing_elapsedTime);
    P4A_dump_message("Copied %zd bytes of memory from accel %p to host %p : "
                      "%.1fms - %.2fGB/s\n", element_size, accel_address, host_address,
                      p4a_timing_elapsedTime,(float)element_size/(p4a_timing_elapsedTime*1000000));
  }
}

/** Copy a scalar from the host to the hardware accelerator

  @param[in] element_size is the size of one element of the array in
 byte

 @param[out] host_address point to the element on the host to write
 into

 @param[in] accel_address refer to the compact memory area to read
 data. In the general case, accel_address may be seen as a unique idea (FIFO)
 and not some address in some memory space.
 */
void P4A_copy_to_accel(size_t element_size,
		       const void *host_address,
		       void *accel_address) 
{
  p4a_global_error=clEnqueueWriteBuffer(p4a_queue,
					(cl_mem)accel_address,
					CL_FALSE,// asynchronous write
					0,
					element_size,
					host_address,
					0,
					NULL,
					&p4a_event);
  P4A_test_execution_with_message("clEnqueueWriteBuffer");

  if(p4a_timing) {
    P4A_TIMING_elapsed_time(p4a_timing_elapsedTime);
    P4A_dump_message("Copied %zd bytes of memory from host %p to accel %p : "
                      "%.1fms - %.2fGB/s\n",element_size, host_address,accel_address,
                      p4a_timing_elapsedTime,(float)element_size/(p4a_timing_elapsedTime*1000000));
  }
}

/* @} */
#endif // P4A_ACCEL_OPENCL
