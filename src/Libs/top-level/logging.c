/*

  $Id$

  Copyright 1989-2012 MINES ParisTech

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
#include <stdio.h>
#include <unistd.h>

#include "genC.h"

#include "linear.h"

#include "ri.h"
#include "misc.h"
#include "database.h"
#include "ri-util.h"
#include "pipsdbm.h"
#include "properties.h"

#define LOG_FILE "Logfile"

/* The log file is closed by default
 */
static FILE *log_file = NULL;

FILE * get_log_file()
{
    return log_file;
}

void
close_log_file(void)
{
   if (log_file != NULL && get_bool_property("USER_LOG_P") )
      if (fclose(log_file) != 0) {
	  pips_internal_error("Could not close");
         perror("close_log_file");
         abort();
      }
   log_file = NULL;
}


void
open_log_file(void)
{
    if (log_file != NULL)
	close_log_file();

    if (get_bool_property("USER_LOG_P") ) 
    {
	string 
	    dir = db_get_current_workspace_directory(),
	    log_file_name = strdup(concatenate(dir, "/", LOG_FILE, NULL));
	free(dir);
	
	if ((log_file = fopen(log_file_name, "a")) == NULL) {
	    perror("[open_log_file]");
	    pips_user_error("Cannot open log file in workspace %s. "
			    "Check access rights.");
	}

	free(log_file_name);
    }
}


void
log_on_file(char chaine[])
{
   if (log_file != NULL /* && get_bool_property("USER_LOG_P") */) {
      if (fprintf(log_file, "%s", chaine) <= 0) {
         perror("log_on_file");
         abort();
      }
      else
         fflush(log_file);
   }
}

/* log message on stderr and in the log file
 * can be used as pips_log_handler
 */
void smart_log_handler(const char *fmt, va_list args)
{
	FILE * log_file = get_log_file();

	/* It goes to stderr to have only displayed files on stdout.
	 */

	/* To be C99 compliant, a va_list can be used only once...
		 Also to avoid exploding on x86_64: */
	va_list args_copy;
	va_copy (args_copy, args);

	vfprintf(stderr, fmt, args);
	fflush(stderr);

	if (!log_file || !get_bool_property("USER_LOG_P"))
		return;

	if (vfprintf(log_file, fmt, args_copy) <= 0) {
		perror("user_log");
		abort();
	}
	else fflush(log_file);
}

/* The # "stringificator" only works in a macro expansion... */
#define PIPS_THANKS_STRING(arch)                                        \
  "%s (ARCH=" arch ")\n  running as %s\n"                               \
  "\n"                                                                  \
  "  (c) 1988-2013 Centre de Recherche en Informatique,\n"              \
  "                Unite de Recherche Mathematiques et Systemes,\n"     \
  "                MINES ParisTech, France.\n"                          \
  "\n"                                                                  \
  "  CRI URL: http://www.cri.mines-paristech.fr/\n"                     \
  "  PIPS URL: http://pips4u.org/\n"                                    \
  "  EMAIL: pips-support at cri dot mines-paristech dot fr\n"           \
  "\n"                                                                  \
  "  This software is provided as is, under the terms of the GPL.\n"    \
  "  It includes and uses software from GNU (gnulib, readline),\n"      \
  "  Berkeley (fsplit), INRIA, IRISA and others (polylib, janus)...\n"  \
  "\n"

/* display pips thanks on startup, if it on a tty.
 */
void
pips_thanks(string name, string path)
{
  if (isatty(fileno(stdout)))
  {
    fprintf(stdout, PIPS_THANKS_STRING(STRINGIFY(SOFT_ARCH)), name, path);
    fflush(stdout);
  }
}
