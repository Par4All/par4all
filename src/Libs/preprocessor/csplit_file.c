/* $Id$
 *
 * Split one C file into one compilation module declaration file and one
 * file for each C function found. The input C file is assumed
 * preprocessed.
 *
 * $Log: csplit_file.c,v $
 * Revision 1.1  2003/07/29 15:12:28  irigoin
 * Initial revision
 *
 *
 * 
 *
 */

#include <stdio.h>

#include "genC.h"

#include "misc.h"

/* Returns an error message or NULL if no error has occured. */
string  csplit(char * dir_name, char * file_name, FILE * out)
{
  extern FILE * splitc_in;
  extern void init_keyword_typedef_table(void);
  extern void splitc_parse();

  /* */
  debug_on("CSPLIT_DEBUG_LEVEL");
  init_keyword_typedef_table();
  splitc_in = safe_fopen(file_name, "r");
  splitc_parse();
  safe_fclose(splitc_in, file_name);
  pips_internal_error("Not implemented yet");
  debug_off();

  return NULL;
}
