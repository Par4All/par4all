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
