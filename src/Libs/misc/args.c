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

   *pargc = 0;
   while(l2!=NIL) {
      argv[(*pargc)++] = STRING(CAR(l2));
      l2= CDR(l2);
   }
   gen_free_list(li);
   li= NIL;
}


/* Just modify the strings in a list from an array of strings. The
   array of string must have at least as much as strings as in the
   list. No free is done. */
void
update_list_from_arg(list l,
		     char ** the_strings)
{
    int i = 0;
    MAPL(scons,
	{
	    STRING(CAR(scons)) = the_strings[i++];
	},
	l);
}


void args_free(pargc, argv)
int *pargc;
char *argv[];
{
    int i;

    for (i = 0; i < *pargc; i++)
	free(argv[i]);

    *pargc = 0;
}



void args_add(pargc, argv, arg)
int *pargc;
char *argv[];
char *arg;
{
    if (*pargc >= ARGS_LENGTH) {
	user_warning("args_add", "too many arguments; recompile after \
increasing ARGS_LENGTH in constants.h\n");
    }

    argv[*pargc] = arg;

    *pargc += 1;
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
    int arg_number;
    int number_of_strings = gen_length(l);    
    char ** the_strings = malloc(number_of_strings*sizeof(char *));
    
    list_to_arg(l, &arg_number, the_strings);
    pips_assert("The size of the args and the list should be the same",
		arg_number == number_of_strings);
    args_sort(arg_number, the_strings);
    update_list_from_arg(l, the_strings);
}
