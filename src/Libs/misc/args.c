/* $Id$
 *
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
    int number_of_strings = gen_length(l);
    gen_array_t a = gen_array_make(number_of_strings); 
    list_to_array(l, a);
    pips_assert("same length", number_of_strings==gen_array_nitems(a));
    gen_array_sort(a);
    update_list_from_array(l, a);
    gen_array_free(a);
}
