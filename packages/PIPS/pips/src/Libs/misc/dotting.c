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
 /* Sorry to keep you waiting... 
  *
  * Emit a message and then a waiting character every second till it is
  * notified to shut up.
  *
  * Francois Irigoin, 18 April 1990
  *
  * Modification:
  *  - installed in Lib/misc/dotting.c, 21 April 1990
  *  - no more dotting: in wpips, the window io associated to each
  * dotting requires a malloc(); as it may occur at any time, malloc()
  * (or free()) may be already active and wpips core dumps;
  * (Francois Irigoin, Bruno Baron, ?? ???? 1990)
  */

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <signal.h>

#include "genC.h"

#include "misc.h"

/*
static FILE * dotting_file;
static char dotting_character;
*/

/*
static void print_dot()
{
    if (dotting_file == stdout) 
	user_log("%c", dotting_character);
    else {
	(void) putc(dotting_character, dotting_file);
	if(dotting_file!=stderr) fflush(dotting_file);
    }
    alarm(1);
}
*/

/* start_dotting: the profile should be changed and varargs used so as
 * to let user emit the initial message as with a fprintf
 * the dotting character could be a constant (e.g. '.') or be passed
 * as first argument
 * start_dotting(dotting_file, dotting_character, format, args)
 */

/*VARARGS3*/
void
start_dotting(FILE * dotting_file,
              const char dotting_character,
              const char * fmt,
              ...)
{
   pips_debug(9, "dotting with '%c'", dotting_character);

   va_list args;

   va_start(args, fmt);

   if (dotting_file == stdout) 
      (* pips_log_handler)(fmt, args);
   else {
      vfprintf(dotting_file, fmt, args);
      if(dotting_file!=stderr) fflush(dotting_file);
   }
   va_end(args);
/*
  if((int) signal(SIGALRM, print_dot)==-1) {
  pips_internal_error("signal error");
  exit(1);
  }
  alarm(1);
  */
}

void stop_dotting()
{
/*    alarm(0);
    (void) signal(SIGALRM, SIG_DFL);

    Vire le "\n" qui interfère... RK.
    if (dotting_file == stdout) 
	user_log("\n");
    else {
	(void) putc('\n', dotting_file);
	if(dotting_file!=stderr) fflush(dotting_file);
    }
*/
}
