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
    Psysteme sc_context;

    if (transformer_undefined_p(context))

	/* this happens to the CONTINUE statement of the exit node
	 * even if unreachable. Thus transformer are not computed,
	 * orderings are not set... however gen_multi_recurse goes there.
	 * I just store NIL, what seems reasonnable an answer.
	 * It seems to be sufficient for other passes. 
	 * I should check that it is indeed the exit node?
	 * FC.
	 */
	return TRUE;

    /* else 
     */

    sc_context = predicate_system(transformer_relation(context));
    return sc_empty_p(sc_context);
}
