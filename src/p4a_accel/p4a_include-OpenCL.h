#ifndef INCLUDE_CL_H
#define INCLUDE_CL_H

#include <opencl.h>
#include <stdbool.h>
#include <stdarg.h>

/** A data structure to store the arguments types for a given 
    kernel arguments list.
*/
struct arg_type {
  char *type_name;
  size_t size;
  struct arg_type *next;
};

/** A kernel list.
*/
struct p4a_kernel_list {
  cl_kernel kernel;
  char *name;
  char *file_name;
  int n_args;
  struct arg_type *args;
  struct arg_type *current;
  struct p4a_kernel_list *next;
};

struct type_substitution {
  // Definition of a type
  char *definition;
  // Its substitute via typedef
  char *substitute;
  struct type_substitution * next;
};


/** A timer Tag to know when to print p4a_time_copy
 */
extern bool p4a_time_tag;
/** Total execution time.
    In OpenCL, time is mesured for each lunch of the kernel.
    We also need to cumulate all the particulate times.
 */
extern double p4a_time_execution;
/** Total time for copy of data (transfers host <-> device).
 */
extern double p4a_time_copy;
/** Global error in absence of a getLastError equivalent in OpenCL */
extern cl_int p4a_global_error;
/** Events for timing in CL: */
extern cl_event p4a_event_execution, p4a_event_copy;
/** The OpenCL context ~ a CPU process 
- a module
- data
- specific memory address 
*/
extern cl_context p4a_context;
/** The OpenCL command queue, that allows delayed execution
 */
extern cl_command_queue p4a_queue;

extern cl_command_queue_properties p4a_queue_properties;


/** The selected device */
//extern cl_device_id p4a_device_id;  
/** The OpenCL platform : a set of devices available 
    The device_id is the selected device from his type
    CL_DEVICE_TYPE_GPU
    CL_DEVICE_TYPE_CPU
*/
//extern cl_platform_id p4a_platform_id;  
/** The OpenCL programm composed of a set of modules
 */
extern cl_program p4a_program;  
/** The current module selected in the program
 */
extern cl_kernel p4a_kernel; 

/** Begin of the kernels list
 */
extern struct p4a_kernel_list *p4a_kernels;
/** Pointer to the current kernel descriptor
 */
extern struct p4a_kernel_list *current_kernel;

extern struct type_substitution * substitution_list;
extern struct type_substitution * current_substitution;
  
char * p4a_error_to_string(int error);
void p4a_clean(int exitCode);
void p4a_error(cl_int error,const char *currentFile,const int currentLine);
void p4a_message(const char *message,const char *currentFile,const int currentLine);
void p4a_load_kernel_arguments(const char *kernel,...);
struct p4a_kernel_list* new_p4a_kernel(const char *kernel);
char *p4a_load_prog_source(char *cl_kernel_file,const char *head,size_t *length);
struct p4a_kernel_list *p4a_search_current_kernel(const char *kernel);
struct arg_type * new_type(char *str_type);
//void set_type(char *str_type);
struct type_substitution * new_typedef(char *str1,char *str2);
char * search_type_for_substitute(char *str);
void parse_file(char *name);
void * argument_reference(va_list ap, struct arg_type *type);

#endif
