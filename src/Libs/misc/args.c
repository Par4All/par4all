#include <stdio.h>
#include "genC.h"
#include "misc.h"



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
