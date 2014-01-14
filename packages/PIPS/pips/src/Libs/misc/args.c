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
 * there was some static buffer management here.
 * now it relies on dynamic arrays.
 */
#include <stdlib.h>
#include <stdio.h>

#include "genC.h"
#include "misc.h"

void
list_to_array(list l, gen_array_t a)
{
    int index = 0;
    MAP(STRING, s, gen_array_addto(a, index++, s), l);
    gen_free_list(l);
}

/* Just modify the strings in a list from an array of strings. The
   array of string must have at least as much as strings as in the
   list. No free is done. */
void
update_list_from_array(list l, gen_array_t a)
{
    int index = 0;
    MAPL(scons, STRING(CAR(scons)) = gen_array_item(a, index++), l);
}


/* Sort a list of strings.
 */
void
sort_list_of_strings(list l)
{
    size_t number_of_strings = gen_length(l);
    gen_array_t a = gen_array_make(number_of_strings);
    list_to_array(l, a);
    pips_assert("same length", number_of_strings == gen_array_nitems(a));
    gen_array_sort(a);
    update_list_from_array(l, a);
    gen_array_free(a);
}


/* Return the malloc()ed version of the concatenation of all the strings in
   the list */
string
list_to_string(list l)
{
  string result = NULL;
  if (l == NIL) return strdup("");
  MAP(STRING,s, {
      if (result == NULL)
	result = strdup((const char *)s);
      else {
	string new_result = strdup(concatenate(result,s,NULL));
	free(result);
	result = new_result;
      }
    }, l);
  return result;
}
