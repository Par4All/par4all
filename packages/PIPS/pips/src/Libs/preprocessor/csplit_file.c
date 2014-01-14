/*

  $Id$

  Copyright 1989-2014 MINES ParisTech

  This file is part of PIPS.

  PIPS is free software: you can redistribute it and/or modify it
  under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  any later version.

  PIPS is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.

  See the GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with PIPS.  If not, see <http://www.gnu.org/licenses/>.

*/
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif
/*
 * Split one C file into one compilation module declaration file and one
 * file for each C function found. The input C file is assumed
 * preprocessed.
 */

#include <stdio.h>
#include <string.h>
#include <ctype.h>

#include "genC.h"
#include "misc.h"

/* To import FILE_SEP_STRING... */
#include "constants.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "preprocessor.h"
#include "properties.h"
#include "c_syntax.h"
#include "splitc.h"

/* used to keep track of include level  */
char * current_include_file_path = NULL;
char * current_file_path = NULL;


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

static string splitc_input_file_name = string_undefined;
/* The FILE descripton used to generate the compilation unit: */
static FILE * splitc_in_append = NULL;
static int current_input_line = 0; /* In file just above */

void reset_current_input_line()
{
current_input_line = 0;
}

static string current_compilation_unit_file_name = string_undefined;
static string current_compilation_unit_name = string_undefined; /* includes FILE_SEP_STRING as a suffix. */
static FILE * compilation_unit_file = NULL; /* Compilation unit*/

static FILE * module_list_file = NULL;

static string current_workspace_name = string_undefined;

/* Disambiguate the compilation unit base name (special character to avoid
 * conflicts with function names and rank if same basename exists
 * elsewhere in user files).
 *
 * Do not create the corresponding directory and within it the compilation unit file.
 *
 * Initialize compilation_unit_file by opening this last file.
 *
 * Set the current_compilation_unit_file_name.
 * */
void csplit_open_compilation_unit(string input_file_name)
{
  string unambiguous_file_name = string_undefined;
  string simpler_file_name = pips_basename(input_file_name, PP_C_ED);
  /* string compilation_unit_name = string_undefined; */
  /* string compilation_unit_file_name = string_undefined; */

  pips_debug(1, "Open compilation unit \"%s\"\n", input_file_name);

  pips_assert("No current compilation unit",
	      string_undefined_p(current_compilation_unit_file_name));

  pips_assert("No current compilation unit",
	      string_undefined_p(current_compilation_unit_name));

  /* Step 1: Define the compilation unit name from the input file name. */
  unambiguous_file_name = strdup(concatenate(current_workspace_name,
					     "/",
					     simpler_file_name,
					     FILE_SEP_STRING,
					     C_FILE_SUFFIX,
					     NULL));

  /* Loop with a counter until the open is OK. Two or more files with the
     same local names may be imported from different directories. */
  /* This does not work because this file is later moved in the proper directory. */
  /*
  if(fopen(unambiguous_file_name, "r")!=NULL) {
    pips_internal_error("Two source files (at least) with same name: \"%s\"",
			simpler_file_name);
  }
  */
  compilation_unit_file = safe_fopen(unambiguous_file_name, "w");

  /* Loop over counter not implemented. */

  pips_assert("compilation_unit_file is defined",
	      compilation_unit_file != NULL);

  current_compilation_unit_file_name = unambiguous_file_name;
  current_compilation_unit_name
    = strdup(concatenate(simpler_file_name, FILE_SEP_STRING, NULL));

  /* Does this current compilation unit already exist? */
  if(fopen(concatenate(current_workspace_name,
		       "/",
		       simpler_file_name,
		       FILE_SEP_STRING,
		       "/",
		       simpler_file_name,
		       FILE_SEP_STRING,
		       C_FILE_SUFFIX, NULL)
	   , "r")!=NULL) {
    pips_user_error("Two source files (at least) with same name: \"%s"
		    C_FILE_SUFFIX "\".\n"
		    "Not supported yet.\n",
		    simpler_file_name);
  }

  /* Keep track of the new compilation unit as a "module" stored in a file */

  fprintf(module_list_file, "%s %s\n",
	  current_compilation_unit_name,
	  current_compilation_unit_file_name);
  free(simpler_file_name);
}

void csplit_close_compilation_unit()
{
  safe_fclose(compilation_unit_file, current_compilation_unit_file_name);
  free(current_compilation_unit_name);
  free(current_compilation_unit_file_name);
  current_compilation_unit_name = string_undefined;
  current_compilation_unit_file_name = string_undefined;
}


/* Copy data from on file to another up to an offset.

   The encountered newlines increase the value of global variable
   current_input_line.

   If the character just after the "up_to_offset" one is a newline, it is
   output in the destination, since it is nicer to have line oriented
   source files.

   @param greedy_spaces is no longer used (used to be: if true, the copy
   is going on if encounter some spaces or comments on the same current
   line. The idea is to get a comment attached to the current statement
   but try to keep the comment for a function.
*/
void copy_between_2_fd_up_to_offset(FILE * source,
				    FILE * destination,
				    unsigned long long up_to_offset,
				    bool greedy_spaces __attribute__ ((__unused__))) {
  int c = EOF;
  int next_c = EOF;
  while(((unsigned) ftell(source)) < up_to_offset) {
    /* There is something to copy: */
      c = fgetc(source);
      if (c == EOF)
	break;

      ifdebug(5)
	putc(c, stderr);
      fputc(c, destination);
      if (c == '\n')
	current_input_line++;
  }
  /* If the next character is a new line, we include it in the file: */
  if ((next_c = fgetc(source)) == '\n') {
    fputc(next_c, destination);
    current_input_line++;
    ifdebug(5)
      putc(next_c, stderr);
  }
  else {
    /* Oops. It was not, "unread" it: */
    ungetc(next_c, source);
    /* But if the last character was not a '\n', end the file with one,
       that is cleaner: */
    if (c != EOF && c != '\n') {
      ifdebug(5)
	putc('\n', stderr);
      fputc('\n', destination);
    }
  }
  /* Remove the greedy stuff since it should be dealt by the
     lexer... Well, not... */
#if 0
  while(isspace(c = fgetc(source))) {
    ifdebug(5)
      putc(c, stderr);
    fputc(c, destination);
    if (c == '\n')
      current_input_line++;
  }
  /* Oops. It was not, "unread" it: */
  ungetc(c, source);
#endif
}


/* Copy the input file to the compilation unit between the function
   declarations up to the current function definition. */
void csplit_append_to_compilation_unit(int last_line,
				       unsigned long long last_offset) {
  pips_debug(2, "append to compilation unit up-to line %d (from %d) or offset %llu\n",
	     last_line, current_input_line, last_offset);

  if (last_offset != 0) {
    /* We are in the offset mode instead of line mode */
    pips_debug(2, "copying to compilation unit file up to offset %llu, we are at currently at offset %lu\n",
	       last_offset, ftell(splitc_in_append));
    copy_between_2_fd_up_to_offset(splitc_in_append,
				   compilation_unit_file,
				   last_offset,
				   true /* Copy up to function begin */);
  }
  else {
    /* We are in the line-oreiented mode: */
    pips_assert("last_line is positive", last_line >= 0);
    pips_assert("if last_line is strictly less than current_input_line, then last_line is 0",
		last_line >= current_input_line || last_line == 0);

    pips_assert("The compilation unit file is open", compilation_unit_file != NULL);

    if(last_line == 0)
      current_input_line = 0;
    else {
      /* In some cases, e.g. two module definitions are contiguous, nothing
	 has to be copied. */
      while(current_input_line < last_line) {
	char c = fgetc(splitc_in_append);

	ifdebug(5)
	  putc(c, stderr);
	fputc(c, compilation_unit_file);
	if(c == '\n')
	  current_input_line++;
      }
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
static bool path_header_p(const char * filepath) {
    return filepath &&
        filepath[strlen(filepath)-1] != 'c';
}

/* Create the module directory and file, copy the definition of the module
   and add the module name to the module name list.

   The compilation unit name used for static functions is retrieved from a
   global variable set by csplit_open_compilation_unit(),
   current_compilation_unit_name.

   If first_offset and last_offset are not both 0, the module is found in the
   source file between these file offset instead of between lines
   first_line and int last_line.
 */
void csplit_copy(const char* module_name,
		 string signature,
		 int first_line,
		 int last_line,
		 size_t first_offset,
		 size_t last_offset,
		 int user_first_line, bool is_static_p)
{
  FILE * mfd = NULL;
  /* Unambiguous, unless the user has given the same name to two functions. */
  string unambiguous_module_file_name;
  const char* unambiguous_module_name = is_static_p?
    strdup(concatenate(current_compilation_unit_name, /* MODULE_SEP_STRING,*/ module_name, NULL)) :
    module_name;
  /* string unambiguous_module_name = module_name; */

  /* pips_assert("First line is strictly positive and lesser than last_line",
     first_line>0 && first_line<last_line); */
  if(!(first_line > 0 && first_line <= last_line)) {
    pips_user_error("Definition of function %s starts at line %d and ends a t line %d\n"
		    "PIPS assumes the function definition to start on a new line "
		    "after the function signature\n", module_name, first_line, last_line);
  }
  pips_assert("current_compilation_unit_name is defined",
	      !string_undefined_p(current_compilation_unit_name));

  /* Step 1: Define an unambiguous name and open the file if possible */
  if(is_static_p) {
    /* Concatenate the unambigous compilation unit name and the module name */
    /* Note: the same static function may be declared twice in the
       compilation unit because the user is mistaken. */
    /* pips_internal_error("Not implemented yet."); */
    unambiguous_module_file_name
      = strdup(concatenate(current_workspace_name, "/", current_compilation_unit_name,
			   /* FILE_SEP_STRING, */ module_name, C_FILE_SUFFIX, NULL));
  }
  else {
    /* The name should be unique in the workspace, but the user may have
       provided several module with the same name */
    unambiguous_module_file_name
      = strdup(concatenate(current_workspace_name, "/", module_name, C_FILE_SUFFIX, NULL));
  }

  /* Open the module code file for writing as the mfd FILE descriptor: */
  pips_debug(1, "Begin for %s module \"%s\" from line %d to line %d (offset [%zd-%zd]) in compilation unit %s towards %s\n",
	     is_static_p? "static" : "global",
	     module_name,
	     first_line, last_line, first_offset, last_offset,
	     splitc_input_file_name,
	     unambiguous_module_file_name);

  if((mfd=fopen(unambiguous_module_file_name, "r"))!=NULL) {
    /* Such a module already exists */
    pips_user_error("Duplicate function \"%s\".\n"
		    "Copy in file %s from input file %s is ignored\n"
		    "Check source code with a compiler or set property %s\n",
		    module_name,
		    unambiguous_module_file_name,
		    splitc_input_file_name,
		    "PIPS_CHECK_FORTRAN");
  }
  else if((mfd=fopen(unambiguous_module_file_name, "w"))==NULL) {
    /* Possible access right problem? */
    pips_user_error("Access or creation right denied for %s\n",
		    unambiguous_module_file_name);
  }

  pips_assert("The module file descriptor is defined", mfd!=NULL);

  /* Step 2: Copy the file source from the end of the last function
     definition up to the begin of the current one into the compilation
     unit to get variable and type declarations, etc. */
  csplit_append_to_compilation_unit(first_line - 1, first_offset);

  pips_assert("Current position is OK", /* Only bother in line-oriented mode */
	      (first_offset != 0 || last_offset != 0) || current_input_line == first_line-1);

  /* Step 3: Copy the function declaration in its module file, starting
     with its line number in the original file. */
  fprintf(mfd, "# %d\n", user_first_line);
  if (first_offset == 0 && last_offset == 0) {
    pips_debug(2, "copying to module file lines [%d-%d]\n",
	       current_input_line, last_line);
    /* Begin and end are specified as line numbers: */
    while(current_input_line<last_line) {
      char c = fgetc(splitc_in_append);
      ifdebug(5)
	putc(c, stderr);
      fputc(c, mfd);
      if(c=='\n')
	current_input_line++;
    }
  }
  else {
    pips_debug(2, "copying to module file offset [%zd-%zd]\n",
	       first_offset, last_offset);
    /* Begin and end are specified as file offsets. First seek at the begin
       of the function: */
    //safe_fseek(splitc_in_append, first_offset, SEEK_SET, "splitc_in_append");
    /* Copy up to the function end: */
    copy_between_2_fd_up_to_offset(splitc_in_append,
				   mfd,
				   last_offset,
				   false /* Do not include trailing spaces */);
  }

  /* Step 4: Copy the function definition */
  /* Fabien: you could add here anything you might want to unsplit the
     file later.
     SG: what about inline static ? why add an extern qualifier ?
     */
  /* check for the static qualifier */
  char * where;
  if( (where = strstr(signature,"static") ) &&
          isspace(where[sizeof("static")-1]) &&
              ( where == signature || isspace(where[-1]) )
    ) {
    fprintf(compilation_unit_file, "%s;\n", signature);
  }
  /* or the extern qualifier */
  else if ( (where = strstr(signature,"extern") ) &&
          isspace(where[sizeof("extern")-1]) &&
              ( where == signature || isspace(where[-1]) )
    ){
    fprintf(compilation_unit_file, "%s;\n", signature);
  }
  /* default to extern qualifier */
  else
    fprintf(compilation_unit_file, "extern %s;\n", signature);

  /* Step 5: Keep track of the new module */
  /* SG hook: do not keep track of module declared inside a header
   * not very reliable in the presence of used inline function in user header,
   * so left apart as of now
   */
  if(true || !get_bool_property("IGNORE_FUNCTION_IN_HEADER")
          || !path_header_p(current_include_file_path)) {
      fprintf(module_list_file, "%s %s\n", unambiguous_module_name, unambiguous_module_file_name);
  }


  safe_fclose(mfd, unambiguous_module_file_name);
  free(unambiguous_module_file_name);
  /* Do not free unambiguous_module_name since it is already done in
     reset_csplit_current_beginning() */
  //free(unambiguous_module_name);
}

void keep_track_of_typedef(string type_name)
{
  hash_put(keyword_typedef_table, type_name,(void *) TK_NAMED_TYPE);
  pips_debug(2,"Add typedef name %s to hash table\n", type_name);
  if(strcmp(type_name, "v1")==0) {
    pips_debug(1, "v1 added as typedef\n");
  }
}

/* Close open files and reset variables */
void csplit_error_handler()
{
  /* Reset keyword table */
  reset_current_input_line();
  reset_csplit_line_number();
  reset_keyword_typedef_table();
}

void csplit_reset()
{
  /* Reset keyword table */
  reset_current_input_line();
  reset_csplit_line_number();
  reset_keyword_typedef_table();
}

void csplit_close_files(string file_name)
{
    csplit_close_compilation_unit();
    current_include_file_path = NULL;
    free(current_file_path);
    current_file_path=NULL;
    ForceResetTypedefStack();
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
}

/** Split a C file into one file per module (function or procedure) plus

    @param dir_name the directory name where the input file is to pick
    @param file_name the C input file name to split
    @param out file opened to record module and compilation unit names

    @return an error message or NULL if no error has occurred.
*/
string current_file_name = string_undefined;
string  csplit(
	       char * dir_name,
	       char * file_name,
	       FILE * out
)
{
  string error_message = string_undefined;
  current_file_name = file_name; /* In case a error occurs*/

  /* */
  debug_on("CSPLIT_DEBUG_LEVEL");

  pips_debug(1, "Begin in directory %s for file %s\n", dir_name, file_name);

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

  init_keyword_typedef_table();

  module_list_file = out;
  csplit_open_compilation_unit(file_name);
  MakeTypedefStack();
  current_include_file_path = NULL;
  current_file_path = NULL;

  CATCH(any_exception_error) {
    error_message = "parser error";
  }
  TRY {
    splitc_parse();
    error_message = NULL;
    UNCATCH(any_exception_error);
  }

  if(error_message==NULL) {
    /* Do not forget to catch what could remain after the last function up
       to the end of file: */
    csplit_append_to_compilation_unit(INT_MAX, ULLONG_MAX);
    csplit_close_files(file_name);

    csplit_reset();
  }
  debug_off();

  return error_message;
}
