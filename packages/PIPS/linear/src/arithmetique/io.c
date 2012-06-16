/*

  $Id$

  Copyright 1989-2012 MINES ParisTech

  This file is part of Linear/C3 Library.

  Linear/C3 Library is free software: you can redistribute it and/or modify it
  under the terms of the GNU Lesser General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  any later version.

  Linear/C3 Library is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.

  See the GNU Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public License
  along with Linear/C3 Library.  If not, see <http://www.gnu.org/licenses/>.

*/

/* package arithmetic
 *
 * IO on a Value
 */
#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif
#include <stdio.h>
#include <string.h>

#include "arithmetique.h"

void print_Value(Value v)
{
    (void) printf(VALUE_FMT, v);
}

void fprint_Value(FILE *f, Value v)
{
    (void) fprintf(f, VALUE_FMT, v);
}

void fprint_string_Value(FILE *f, char * blah, Value v)
{
    fputs(blah,f);
    fprint_Value(f, v);
}

void sprint_Value(char *s, Value v)
{
    (void) sprintf(s, VALUE_FMT, v);
}

int fscan_Value(FILE *f, Value *pv)
{
    return fscanf(f, VALUE_FMT, pv);
}

int scan_Value(Value *pv)
{
    return scanf(VALUE_FMT, pv);
}

int sscan_Value(char *s, Value *pv)
{
    return sscanf(s, VALUE_FMT, pv);
}

/* this seems a reasonnable upperbound
 */
#define BUFFER_SIZE 50
char * Value_to_string(Value v)
{
    static char buf[BUFFER_SIZE];
    sprintf(buf, VALUE_FMT, v);
    return buf;
}
