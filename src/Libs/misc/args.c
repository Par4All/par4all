/* 
 * Quite a poor code in my opinion. FC.
 */
#include <stdlib.h>
#include <stdio.h>
#include <setjmp.h>

#include "genC.h"
#include "misc.h"


/* given li build argv, update *argc and free li */
void
list_to_arg(list li,
            int * pargc,
            char * argv[])
{
   list l2=li;
   int index=0;

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
    pips_assert("long enough", gen_length(l)<ARGS_LENGTH);
    MAPL(scons, STRING(CAR(scons)) = the_strings[i++], l);
}


void 
args_free(
    int *pargc,
    char *argv[])
{
    int i;
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
   qsort((char *) argv,
         argc,
         sizeof(char *),
         compare_args_string_for_qsort);
}


/* Sort a list of string: */
void
sort_list_of_strings(list l)
{
    int arg_number = 0;
    int number_of_strings = gen_length(l);
    char * argsv[ARGS_LENGTH];

    pips_assert("not too long", number_of_strings<ARGS_LENGTH);
    list_to_arg(l, &arg_number, argsv);

    pips_assert("The size of the args and the list should be the same",
		arg_number == number_of_strings);

    args_sort(arg_number, argsv);
    update_list_from_arg(l, argsv);
}
