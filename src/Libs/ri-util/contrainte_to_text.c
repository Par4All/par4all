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

extern int fprintf();


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

/*   That is all
 */
