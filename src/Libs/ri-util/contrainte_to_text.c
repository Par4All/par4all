/*
 * help-to-debug functions; 
 * previously in /rice/debug.c
 *
 * BA, september 3, 1993
 */
#include <stdio.h>
#include <string.h>

#include "boolean.h"
#include "vecteur.h"
#include "contrainte.h"

#include "genC.h"
#include "ri.h"
#include "ri-util.h"

void inegalite_debug(c)
Pcontrainte c;
{
    inegalite_fprint(stderr, c, entity_local_name);
}

void egalite_debug(c)
Pcontrainte c;
{
    egalite_fprint(stderr, c, entity_local_name);
}

int
contrainte_gen_allocated_memory(
    Pcontrainte pc)
{
    int result = 0;
    for(; pc; pc=pc->succ)
	result += sizeof(Scontrainte) + 
	    vect_gen_allocated_memory(pc->vecteur);
    return result;
}

/*   That is all
 */
