/* $Id$
 *
 * there was some static buffer management here. 
 * now it relies on dynamic arrays.
 */
#include <stdlib.h>
#include <stdio.h>
#include <setjmp.h>

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

/************************************************************ OBSOLETE STUFF */

/* 
 * Quite a poor code in my opinion. FC.
 */

/* given li build argv, update *argc and free li */
void
list_to_arg(list li,
            int * pargc,
            char * argv[])
{
   list l2=li;
   int index=0;

   pips_user_warning("obsolete function, update to gen_array_t\n");

   pips_assert("not too long", gen_length(li)<ARGS_LENGTH);

   while(l2!=NIL) {
       argv[index++] = STRING(CAR(l2));
       l2= CDR(l2);
   }

   gen_free_list(li), li= NIL;
   (*pargc) = index;
}


/* Just modify the strings in a list from an array of strings. The
   array of string must have at least as much as strings as in the
   list. No free is done. */
void
update_list_from_arg(
    list l,
    char ** the_strings)
{
    int i = 0;

    pips_user_warning("obsolete function, update to gen_array_t\n");
    pips_assert("long enough", gen_length(l)<ARGS_LENGTH);
    MAPL(scons, STRING(CAR(scons)) = the_strings[i++], l);
}


void 
args_free(
    int *pargc,
    char *argv[])
{
    int i;

    pips_user_warning("obsolete function, update to gen_array_t\n");
    pips_assert("length ok", *pargc<ARGS_LENGTH);

    for (i = 0; i < *pargc; i++)
	free(argv[i]);
    (*pargc) = 0;
}



void 
args_add(
    int *pargc,
    char *argv[],
    char *arg)
{

    pips_user_warning("obsolete function, update to gen_array_t\n");
    pips_assert("not too many", *pargc<ARGS_LENGTH);
    argv[*pargc] = arg;
    (*pargc) ++;
}


int
compare_args_string_for_qsort(const void *x,
                              const void *y)
{
   return strcmp(* (char **) x, * (char **) y);

}


/* Sort an arg list: */
void
args_sort(int argc,
          char *argv[])
{

   pips_user_warning("obsolete function, update to gen_array_t\n");
   qsort((char *) argv,
         argc,
         sizeof(char *),
         compare_args_string_for_qsort);
}
