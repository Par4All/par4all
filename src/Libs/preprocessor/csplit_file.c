/* $Id$
 *
 * Split one C file into one compilation module declaration file and one
 * file for each C function found. The input C file is assumed
 * preprocessed.
 *
 * $Log: csplit_file.c,v $
 * Revision 1.2  2003/08/01 05:59:04  irigoin
 * Intermediate version installed to let Production link
 *
 * Revision 1.1  2003/07/29 15:12:28  irigoin
 * Initial revision
 *
 *
 * 
 *
 */

#include <stdio.h>

extern char * strdup(char *);

#include "genC.h"

#include "misc.h"

/* To import FILE_SEP_STRING... */
#include "linear.h"
#include "ri.h"
#include "ri-util.h"

/* Kind of useless since a file is used to mimic fsplit */
static list module_name_list = list_undefined;

void init_module_name_list()
{
  pips_assert("module_name_list is undefined",
	      list_undefined_p(module_name_list));
  module_name_list = NIL;
}

void reset_module_name_list()
{
  pips_assert("module_name_list is not undefined",
	      !list_undefined_p(module_name_list));
  gen_free_list(module_name_list);
  module_name_list = list_undefined;
}

/* No checking, to be called from a (future) error handling module. */
void error_reset_module_name_list()
{
  if(!list_undefined_p(module_name_list))
    reset_module_name_list();
}

static FILE * compilation_unit_file = NULL; /* Compilation unit*/
static string splitc_input_file_name = string_undefined;
static FILE * splitc_in_append = NULL; /* Used to generate the compilation unit */
static int current_input_line = 0; /* In previous file */
/* static FILE * splitc_in_copy = NULL; */ /* Used to generate the modules */
static string current_compilation_unit_name = string_undefined;
static FILE * module_list_file = NULL;
static string current_workspace_name = string_undefined;

/* Disambiguate the compilation unit base name (special character to avoid
 * conflicts with function names and rank if same basename exists
 * elsewhere in user files), 
 *
 * Do not create the corresponding directory and within it the compilation unit file.
 *
 * Initialize compilation_unit_file by opening this last file. 
 *
 * Set the current_compilation_unit_name.
 * */
void csplit_open_compilation_unit(string input_file_name)
{
  string unambiguous_name = string_undefined;
  string simpler_file_name = pips_basename(input_file_name, ".cpp_processed.c");
  string compilation_unit_name = string_undefined;

  pips_debug(1, "Open compilation unit \"%s\"\n", input_file_name);

  /* Step 1: Define the compilation unit name from the input file name. */
  unambiguous_name = strdup(concatenate(current_workspace_name,
					"/", simpler_file_name,
					FILE_SEP_STRING,
					".c",
					NULL));
  compilation_unit_file = safe_fopen(unambiguous_name, "w");
  pips_assert("No current compilation unit",
	      string_undefined_p(current_compilation_unit_name));
  current_compilation_unit_name = unambiguous_name;

  /* Keep track of the new compilation unit as a "module" */
  compilation_unit_name = strdup(concatenate(current_workspace_name,
					     "/", simpler_file_name,
					     FILE_SEP_STRING,
					     ".c",
					     NULL));
  fprintf(module_list_file, "%s %s\n", simpler_file_name, compilation_unit_name);
}

void csplit_close_compilation_unit(string name)
{
  safe_fclose(compilation_unit_file, name);
}

void csplit_append_to_compilation_unit(int last_line)
{

  pips_assert("last_line is positive", last_line>=0);
  pips_assert("if last_line is strictly less than current_input_line, then last_line is 0",
	      last_line >= current_input_line || last_line==0);

  pips_assert("The compilation unit file is open", compilation_unit_file != NULL);

  if(last_line==0) 
    current_input_line = 0;
  else {
    /* In some cases, e.g. two module definitions are contiguous, nothing
       has to be copied. */
    while(current_input_line<last_line) {
      char c = fgetc(splitc_in_append);

      fputc(c, compilation_unit_file);
      if(c=='\n')
	current_input_line++;
    }
  }
}

/*
static void csplit_skip(FILE * f, int lines)
{
  int i = 0;

  pips_assert("the number of lines to be skipped is positive", lines>0);

  while(i<lines) {
    char c = fgetc(f);

    if(c=='\n')
      i++;
  }
}
*/

/* Create the module directory and file, copy the definition of the module and add the module name to the module name list*/
void csplit_copy(string module_name, string signature, int first_line, int last_line, bool is_static_p)
{
  FILE * mfd = NULL;
  string unambiguous_module_file_name
    = strdup(concatenate(current_workspace_name, "/", module_name, ".c", NULL));
  /* string unambiguous_module_name = module_name; */

  pips_debug(1, "Begin for %s module \"%s\" from line %d to line %d in compilation unit %s towards %s\n",
	     is_static_p? "static" : "global",
	     module_name,
	     first_line, last_line, splitc_input_file_name,
	     unambiguous_module_file_name);

  pips_assert("First line is strictly positive and lesser than last_line",
	      first_line>0 && first_line<last_line);

  /* Step 1: Define an unambiguous name */
  if(is_static_p) {
    /* Concatenate the unambigous compilation unit name and the module name */
    /* Note: the same static function may be declared twice in the
       compilation unit because the user is mistaken. */
    pips_internal_error("Not implemented yet.");
  }
  else {
    /* The name should be unique in the workspace, but the user may have
       provided several module with the same name */
    if((mfd=fopen(unambiguous_module_file_name, "r"))!=NULL) {
      /* Such a module already exists */
      pips_user_warning("Duplicate function %s.\nCopy in file %s from input file %s is ignored\n",
			module_name, 
			unambiguous_module_file_name,
			splitc_input_file_name);
      pips_internal_error("Not implemented yet.");
    }
    else if((mfd=fopen(unambiguous_module_file_name, "w"))==NULL) {
      /* Possible access right problem? */
      pips_internal_error("Not implemented yet.");
    }
  }

  pips_assert("The module file descriptor is defined", mfd!=NULL);

  /* Step 2: Copy the compilation unit*/
  csplit_append_to_compilation_unit(first_line-1);

  pips_assert("Current position is OK", current_input_line==first_line-1);

  /* Step 3: Copy the function declaration in the compilation unit */
  while(current_input_line<last_line) {
    char c = fgetc(splitc_in_append);

    fputc(c, mfd);
    if(c=='\n')
      current_input_line++;
  }


  /* Step 4: Copy the function definition */
  fprintf(compilation_unit_file, "extern %s;\n", signature);

  /* Step 5: Keep track of the new module */
  fprintf(module_list_file, "%s %s\n", module_name, unambiguous_module_file_name);

  safe_fclose(mfd, unambiguous_module_file_name);
  free(unambiguous_module_file_name);
}

/* Close open files and reset variables */
void csplit_error_handler()
{
}

void csplit_reset() 
{
}

/* Returns an error message or NULL if no error has occured. */
string  csplit(
	       char * dir_name,
	       char * file_name,
	       FILE * out /* File open to record module and compilation unit names */
)
{
  extern FILE * splitc_in;
  extern void init_keyword_typedef_table(void);
  extern void splitc_parse();

  /* */
  debug_on("CSPLIT_DEBUG_LEVEL");

  pips_debug(1, "Begin in directory %s for file %s\n", dir_name, file_name);

  init_keyword_typedef_table();

  /* The same file is opened twice for parsing, for copying the
     compilation unit and for copying the modules. */

  if ((splitc_in = fopen(file_name, "r")) == NULL) {
    fprintf(stderr, "csplit: cannot open %s\n", file_name);
    return "cannot open file";
  }
  splitc_input_file_name = file_name;

  current_workspace_name = dir_name;

  splitc_in_append = safe_fopen(file_name, "r");
  /* splitc_in_copy = safe_fopen(file_name, "r"); */

  module_list_file = out;
  csplit_open_compilation_unit(file_name);

  splitc_parse();

  csplit_close_compilation_unit(file_name);
  safe_fclose(splitc_in, file_name);
  splitc_in = NULL;
  splitc_input_file_name = string_undefined;
  safe_fclose(splitc_in_append, file_name);
  splitc_in_append = NULL;
  /*
    safe_fclose(splitc_in_copy, file_name);
    splitc_in_copy = NULL;
  */
  /* No close, because this file descriptor is managed by the caller. */
  module_list_file = NULL;

  csplit_reset();

  debug_off();

  return NULL;
}
