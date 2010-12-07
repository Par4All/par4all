#define P4A_error_to_string(error) (char *)p4a_error_to_string_inline(error)


struct arg_type {
  char *type_name;
  size_t size;
  struct arg_type *next;
};

extern cl_program p4a_program; 
extern cl_kernel p4a_kernel;
extern int args;
extern struct arg_type *args_type,*current_type;

#define P4A_log_and_exit(code,...)               \
  do {						 \
  fprintf(stdout,__VA_ARGS__);			 \
  exit(code);					 \
  } while(0)

#ifdef P4A_DEBUG
#define P4A_log(...)               fprintf(stdout,__VA_ARGS__)
#else
#define P4A_log(...)   
#endif


inline char * p4a_error_to_string_inline(int error) 
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

/** Skips any line comment beginning with // and ending at '\n'
 */

inline void p4a_skip_line_comments(char *p)
{
  int n;
  int count = 0;
  bool comment = false;
  char buf[1000];
  char *ref = p;

  // Reads a succession of string ...
  while ((n=sscanf(p,"%s",buf)) != EOF) {
    char *tmp = buf;
    // If comment status is false, try to find a // to begin a comment
    while (*tmp != '\0' && (*tmp != '/' || *(tmp+1) != '/')) 
      tmp++;
    
    // At "//" position
    if (*tmp != '\0') {
      // Change the status of the comment
      comment = true;
      //ref is the position of a cursor in the source where the comment
      // must begin to be skipped
      ref = p + (strlen(buf) - strlen(tmp));
      // Count is the cumulated length of the comment
      count += strlen(tmp);

      //printf("Comment %s %s %d : longueur %d\n",buf,tmp,comment,count);
    }
    else {
      // To cumulate the length of the strings in the comment
      if (comment == true) {
	count += strlen(buf);
	//printf("Length of comment %s %s %d : longueur %d\n",buf,tmp,comment,count);
      }
    }

    int len = strlen(buf);
    p += len;
    // For a good position of the pointer for reading, skips any blank
    while (*p == ' ' || *p == '\t' || *p == '\n') {
      // To cumulate the blank in the comment
      if (comment == true) {
	count++;
	//printf("Length of comment %s %s %d : longueur %d\n",buf,tmp,comment,count);
      }
      // At the end of the comment
      if (*p == '\n' && comment == true) {
	comment = false;
	// We skip the remaining line
	int l = strlen(ref)-count;
	// +1 for the '\0' char
	memmove(ref,ref+count,l+1);
	p = ref;
	//printf("Après move : %s\n",p);
	count = 0;
      }
      p++;
    }
  }
}


/** Skips any bloc comment 
 */

inline void p4a_skip_bloc_comments(char *p)
{
  int n;
  int count = 0;
  bool comment = false;
  char buf[1000];
  char *ref = p;

  // Reads a succession of string ...
  while ((n=sscanf(p,"%s",buf)) != EOF) {
    char *tmp = buf;
    // If comment status is false, try to find a /* to begin comment
    if (comment == false) {
      while (*tmp != '\0' && (*tmp != '/' || *(tmp+1) != '*')) 
	tmp++;
    }
    // If comment status is true, try to find a */ to end the comment
    else {
      while (*tmp != '\0' && (*tmp != '*' || *(tmp+1) != '/'))
	tmp++;
    }
    // At "/*" or "*/" position
    if (*tmp != '\0') {
      // Change the status of the comment
      comment = comment == false ? true : false;
      if (comment == true) {
	//ref is the position of a cursor in the source where the comment
	// must begin to be skipped
	ref = p + (strlen(buf) - strlen(tmp));
	// Count is the cumulated length of the comment
	count += strlen(tmp);
      }
      else 
	// +2 for the final */ that must be skipped 
	count += strlen(buf) - strlen(tmp) +2;
      
      //printf("Comment %s %s %d : longueur %d\n",buf,tmp,comment,count);
      if (comment == false) {
	// We skip the bloc comment
	int l = strlen(ref)-count;
	// +1 for the '\0' char
	memmove(ref,ref+count,l+1);
	p = ref;
	//printf("Après move : %s\n",p);
	count = 0;
      }
    }
    else {
      // To cumulate the length of the strings in the comment
      if (comment == true) {
	count += strlen(buf);
	//printf("Length of comment %s %s %d : longueur %d\n",buf,tmp,comment,count);
      }
    }

    int len = strlen(buf);
    p += len;
    // For a good position of the pointer for reading, skips any blank
    while (*p == ' ' || *p == '\t' || *p == '\n') {
      p++;
      // To cumulate the blank in the comment
      if (comment == true) {
	count++;
	//printf("Length of comment %s %s %d : longueur %d\n",buf,tmp,comment,count);
      }
    }
  }
}

/** Analyse the typedef lines to memorise some re-definitions of types
 */
inline void p4a_analyse_typedef(char *p,
				const int max,
				char *definition[],
				char *substitute[],
				int *N)
{
  char buf[1000];
  int n;
  int count = 0;
  // Statut de typedef
  bool tdf=false;
  // Contents fields of the typedef : type and replacing name
  char *def[max];

  while ((n=sscanf(p,"%s",buf)) != EOF) {
    if (strcmp(buf,"typedef")==0) {
      printf("Typedef begins\n");
      tdf = true;
    }
    else if (tdf == true && count < max)
      def[count++]=strdup(buf);
      
    int len = strlen(buf);
    if (tdf == true) printf("char final %c\n",buf[len-1]);
    
    if (tdf == true && buf[len-1] == ';') {
      printf("Typedef ends : %d\n",count);
      tdf = false;
      if (count == 2) {
	if (def[1][len-1] == ';')
	  def[1][len-1]='\0';
	printf("def 0 : %s %d\n",def[0],*N);
	definition[*N] = (char *)strdup(def[0]);
	printf("def 1 : %s\n",def[1]);
	substitute[*N] = (char *)strdup(def[1]);
	printf("Typedef %d : %s %s\n",*N,definition[*N],substitute[*N]);
	(*N)++;
      }
      count = 0;
    }
   
    p += len;
    // For a good position of the pointer for reading, skips any blank
    while (*p == ' ' || *p == '\t' || *p == '\n')
      p++;
  } 
}


/** Read and skip strings in the current source up to the kernel call 
    referenced via the ref key word.

    Returns the pointer at the position of the next string.
 */
inline char * p4a_skip_up_to_kernel_call(char *p,
					 char buf[], 
					 int N, 
					 const char *ref)
{
  int n;
  int count = 0;

  while ((n=sscanf(p,"%s",buf)) != EOF) {
    int len = strlen(buf);
    p += len;
    // For a good position of the pointer for reading, skips any blank
    while (*p == ' ' || *p == '\t' || *p == '\n')
      p++;
    if (strcmp(buf,ref) == 0)
      count++;
    if (count == N)
      break;
  }
  if (n == EOF)
    return NULL;
  return p;  
}


/** Read and skip strings in the source up to the beginning of the argument 
    list (the char '(').
    
    Returns the pointer to the part of the string after the char c, 
    skipping the char c and any blank.
 */
inline char * p4a_skip_up_to_argument_list(char *p,char buf[])
{
  int n;
  char c='(';

  while ((n=sscanf(p,"%s",buf)) != EOF) {
    // Search from the instance of char c in the current string
    char *tmp=buf;
    while (*tmp != c && *tmp != '\0')
      tmp++;
    // Move the pointer p to the position of the char c
    p += tmp-buf;
    if (*tmp != '\0')
      // We skip the char c
      p++;
    // Skipping blank as usual
    // for a good position of the pointer p in the source
    // If the char c was at the end of the string, for instance
    // Or simply if not found
    while (*p == ' ' || *p == '\t' || *p == '\n')
      p++;
    if (*tmp != '\0')
      break;
  }
  if (n == EOF)
    return NULL;
  else
    return p;
}

/** Read strings in the source and quits when EOF or ')' are found.
    
    Betwwen each ',' or ')', analyses of the strings to memorise
    types.
    
    Returns the pointer to the part of the string after ','
    or NULL.
 */
inline void p4a_read_up_to_argument_list_end(char *p,char buf[])
{
  int n;
  //bool end = false;
  char *t[2];
  int count=0;

  while ((n=sscanf(p,"%s",buf)) != EOF) {
    //printf("Après lecture : %s\n",buf);
    // Search from the instance of , and ) in the current string
    char *tmp=buf;
    while (*tmp != ')' && *tmp != ',' && *tmp != '\0') 
      tmp++;
    
    // if , and ) found => new argument
    if (*tmp == ',' || *tmp == ')') {
      // The variable name can be linked to the , or )
      int l = strlen(buf);
      if (count < 2 && l > 1) {
	t[count++] = strdup(buf);
	t[count-1][l-1]='\0';
      }

      /*
      for (int i = 0;i < count;i++)
	printf("Count %d : %s\n",i,t[i]);
      */

      args++;
      struct arg_type *type = (struct arg_type*)malloc(sizeof(struct arg_type));
      // By default, gloabal and constant key word have been skipped...
      // The following is necessarily the type in t[0]
      // If the type ends with *
      // or if the variable name contained in t[1] begins with *
      // or ends with ]
      // Thus => pointer
      type->size=-1;
      type->type_name = strdup(t[0]);
      l = strlen(t[0]);
      if (t[0][l-1] == '*') 
	type->type_name = strdup("cl_mem");
      else if (count > 1) {
	if (t[1][0] == '*') 
	  type->type_name = strdup("cl_mem");
	else {
	  l = strlen(t[1]);
	  if (t[1][l-1] == ']')
	    type->type_name = strdup("cl_mem");
	} 
      }

      printf("Args %d : t[0] = %s\n",args,type->type_name);
      
      if (args_type==NULL)
	args_type = type;
      else {
	current_type->next = type;
      }
      current_type = type;
      type->next = NULL;
      count = 0;
    }
    //printf("Après tri : %s\n",tmp);
    // Skip some elements of the argument list
    if (strcmp(buf,"P4A_accel_global_address") != 0
	&&strcmp(buf,"P4A_accel_constant_address") != 0
	&& *tmp != ',' && *tmp != ')') {
      printf("Argument %d : type %s (%d)\n",args,buf,count);
      if (count < 2)
	t[count++] = strdup(buf);
    }

    // For the following ...
    // Move the pointer p to the position of the char c
    p += tmp-buf;
    if (*tmp != '\0')
      // We skip the char c
      p++;
    // Skipping blank as usual
    // for a good position of the pointer p in the source
    // If the char c was at the end of the string, for instance
    // Or simply if not found
    while (*p == ' ' || *p == '\t' || *p == '\n')
      p++;
    if (*tmp == ')')
      break;
  }
}

/** This procedure analyses the argument list of the kernel to
    - extract the number of arguments
    - identify their types.
 */

inline void p4a_parse_kernel(char *source,const char *comment)
{
  //A buffer to read in the sucessive string in source
  char buf[1000];
  char *p = source;
  // Skip the head comment
  int len = strlen(comment);
  p += len;

  p4a_skip_line_comments(p);
  p4a_skip_bloc_comments(p);

  const int max_def = 10;
  char *definitions[max_def];
  char *substitutes[max_def];
  int n=0;
  
  p4a_analyse_typedef(p,max_def,definitions,substitutes,&n);

  /*
  for (int i = 0;i < n;i++)
    printf("%d : Def %s = subst %s\n",i,definitions[i],substitutes[i]);
  */

  p = p4a_skip_up_to_kernel_call(p,buf,2,"P4A_accel_kernel_wrapper");
  p = p4a_skip_up_to_argument_list(p,buf);

  p4a_read_up_to_argument_list_end(p,buf);
  //printf("On s'est arrêté normalement : %d\n",args);

  struct arg_type *type = args_type;
  while (type) {
    //printf("%s\n",type->type_name);
    for (int i = 0;i < n;i++) {
      if (strcmp(type->type_name,substitutes[i]) == 0)
	type->type_name = definitions[i];
    }

    if (strcmp(type->type_name,"cl_mem")==0)
      type->size = sizeof(cl_mem);
    if (strcmp(type->type_name,"int")==0)
      type->size = sizeof(int);
    if (strcmp(type->type_name,"cl_int")==0)
      type->size = sizeof(cl_int);
    if (strcmp(type->type_name,"float")==0)
      type->size = sizeof(float);
    if (strcmp(type->type_name,"cl_float")==0)
      type->size = sizeof(cl_float);
    if (strcmp(type->type_name,"double")==0)
      type->size = sizeof(double);
    if (strcmp(type->type_name,"char")==0)
      type->size = sizeof(char);
    if (strcmp(type->type_name,"cl_char")==0)
      type->size = sizeof(cl_char);
    if (strcmp(type->type_name,"size_t")==0)
      type->size = sizeof(size_t);

    //printf("\tsize = %lu\n",type->size);
    type = type->next;
  }
}

/** Load and store the content of the kernel file in a string.
    Replace the oclLoadProgSource function of NVIDIA.
 */
inline char *p4a_load_prog_source(char *cl_kernel_file,const char *head,size_t *length)
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
  if (!in) 
    P4A_log_and_exit(EXIT_FAILURE,"Bad kernel source reference : %s\n",cl_kernel_file);
  
  // Read the file content
  int n=0;
  if ((n = read(in,(void *)p,size)) != (int)size) 
    P4A_log_and_exit(EXIT_FAILURE,"Read was not completed : %d / %lu octets\n",n,size);
  
  // Final string marker
  source[len+n]='\0';
  close(in);
  *length = size+len;

  p4a_parse_kernel(source,head);
  
  return source;
}

