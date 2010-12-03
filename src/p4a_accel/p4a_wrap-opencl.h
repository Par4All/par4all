#define p4a_error_to_string(error)	(char *)checkError(error)

inline char * checkError(int error) 
{
  switch (error)
    {
    case CL_SUCCESS:
      return (char *)"P4A : Success";
    case CL_DEVICE_NOT_FOUND:
      return (char *)"P4A : Device Not Found";
    case CL_DEVICE_NOT_AVAILABLE:
      return (char *)"P4A : Device Not Available";
    case CL_COMPILER_NOT_AVAILABLE:
      return (char *)"P4A : Compiler Not Available";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:
      return (char *)"P4A : Mem Object Allocation Failure";
    case CL_OUT_OF_RESOURCES:
      return (char *)"P4A : Out Of Ressources";
    case CL_OUT_OF_HOST_MEMORY:
      return (char *)"P4A : Out Of Host Memory";
    case CL_PROFILING_INFO_NOT_AVAILABLE:
      return (char *)"P4A : Profiling Info Not Available";
    case CL_MEM_COPY_OVERLAP:
      return (char *)"P4A : Mem Copy Overlap";
    case CL_IMAGE_FORMAT_MISMATCH:
      return (char *)"P4A : Image Format Mismatch";
    case CL_IMAGE_FORMAT_NOT_SUPPORTED:
      return (char *)"P4A : Image Format Not Supported";
    case CL_BUILD_PROGRAM_FAILURE:
      return (char *)"P4A : Build Program Failure";
    case CL_MAP_FAILURE:
      return (char *)"P4A : Map Failure";
    case CL_INVALID_VALUE:
      return (char *)"P4A : Invalid Value";
    case CL_INVALID_DEVICE_TYPE:
      return (char *)"P4A : Invalid Device Type";
    case CL_INVALID_PLATFORM:
      return (char *)"P4A : Invalid Platform";
    case CL_INVALID_DEVICE:
      return (char *)"P4A : Invalid Device";
    case CL_INVALID_CONTEXT:
      return (char *)"P4A : Invalid Context";
    case CL_INVALID_QUEUE_PROPERTIES:
      return (char *)"P4A : Invalid Queue Properties";
    case CL_INVALID_COMMAND_QUEUE:
      return (char *)"P4A : Invalid Command Queue";
    case CL_INVALID_HOST_PTR:
      return (char *)"P4A : Invalid Host Ptr";
    case CL_INVALID_MEM_OBJECT:
      return (char *)"P4A : Invalid Mem Object";
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
      return (char *)"P4A : Invalid Image Format Descriptor";
    case CL_INVALID_IMAGE_SIZE:
      return (char *)"P4A : Invalid Image Size";
    case CL_INVALID_SAMPLER:
      return (char *)"P4A : Invalid Sampler";
    case CL_INVALID_BINARY:
      return (char *)"P4A : Invalid Binary";
    case CL_INVALID_BUILD_OPTIONS:
      return (char *)"P4A : Invalid Build Options";
    case CL_INVALID_PROGRAM:
      return (char *)"P4A : Invalid Program";
    case CL_INVALID_PROGRAM_EXECUTABLE:
      return (char *)"P4A : Invalid Program Executable";
    case CL_INVALID_KERNEL_NAME:
      return (char *)"P4A : Invalid Kernel Name";
    case CL_INVALID_KERNEL_DEFINITION:
      return (char *)"P4A : Invalid Kernel Definition";
    case CL_INVALID_KERNEL:
      return (char *)"P4A : Invalid Kernel";
    case CL_INVALID_ARG_INDEX:
      return (char *)"P4A : Invalid Arg Index";
    case CL_INVALID_ARG_VALUE:
      return (char *)"P4A : Invalid Arg Value";
    case CL_INVALID_ARG_SIZE:
      return (char *)"P4A : Invalid Arg Size";
    case CL_INVALID_KERNEL_ARGS:
      return (char *)"P4A : Invalid Kernel Args";
    case CL_INVALID_WORK_DIMENSION:
      return (char *)"P4A : Invalid Work Dimension";
    case CL_INVALID_WORK_GROUP_SIZE:
      return (char *)"P4A : Invalid Work Group Size";
    case CL_INVALID_WORK_ITEM_SIZE:
      return (char *)"P4A : Invalid Work Item Size";
    case CL_INVALID_GLOBAL_OFFSET:
      return (char *)"P4A : Invalid Global Offset";
    case CL_INVALID_EVENT_WAIT_LIST:
      return (char *)"P4A : Invalid Event Wait List";
    case CL_INVALID_EVENT:
      return (char *)"P4A : Invalid Event";
    case CL_INVALID_OPERATION:
      return (char *)"P4A : Invalid Operation";
    case CL_INVALID_GL_OBJECT:
      return (char *)"P4A : Invalid GL Object";
    case CL_INVALID_BUFFER_SIZE:
      return (char *)"P4A : Invalid Buffer Size";
    case CL_INVALID_MIP_LEVEL:
      return (char *)"P4A : Invalid Mip Level";
    case CL_INVALID_GLOBAL_WORK_SIZE:
      return (char *)"P4A : Invalid Global Work Size";
    default:
      break;
    }
} 


