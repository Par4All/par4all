/* package convex effects :  Be'atrice Creusillet 5/97
 *
 * File: utils.c
 * ~~~~~~~~~~~~~
 *
 * This File contains the interfaces with pipsmake which compute the various
 * types of convex regions by using the generic functions.
 *
 */
#include <stdio.h>
#include <string.h>

#include "genC.h"

#include "effects-generic.h"
#include "effects-convex.h"


bool 
empty_convex_context_p(transformer context)
{
    Psysteme sc_context = predicate_system(transformer_relation(context));
    return sc_empty_p(sc_context);
}
