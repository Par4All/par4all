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
  these versions of functions 'dialogue', 'message', ... are used to replaced
  XView versions when a batch version of pips is compiled.
*/

#include <stdio.h>
#include <stdarg.h>

/*VARARGS0*/
void dialogue(char* fmt, ...)
{
    va_list args;

    va_start(args, fmt);
    (void) vfprintf(stderr, fmt, args);
    va_end(args);
}

/*VARARGS0*/
void show_message(char *fmt, ...)
{
    va_list args;

    va_start(args, fmt);
    (void) vfprintf(stderr, fmt, args);
    va_end(args);
}
