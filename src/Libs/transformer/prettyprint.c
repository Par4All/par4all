/* 
 * This section has been revised and cleaned.
 * The main change is to sort the arguments for 
 * the preconditions print.
 * 
 * Modification : Arnauld LESERVOT
 * Date         : 92 08 27
 * Old version  : prettyprint.old.c
 *
 * $Id$
 *
 * $Log: prettyprint.c,v $
 * Revision 1.7  2001/07/19 18:06:02  irigoin
 * Better type casting somewhere. Minor change.
 *
 *
 */

#include <stdio.h>
#include <string.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"

#include "constants.h"

#include "misc.h"
#include "properties.h"

#include "transformer.h"

string transformer_to_string(transformer tf)
{
    pips_internal_error("not implemenented anymore\n");
    return string_undefined;
}

string precondition_to_string(pre)
transformer pre;
{
    pips_internal_error("not implemenented anymore\n");
    return string_undefined;
}

string arguments_to_string(string s, list args)
{
    pips_internal_error("not implemenented anymore\n");
    return string_undefined;
}

string 
relation_to_string(
    string s,
    Psysteme ps,
    char * (*variable_name)(entity))
{
    pips_internal_error("not implemenented anymore\n");
    return string_undefined;
}

char * pips_user_value_name(entity e)
{
    if(e == (entity) TCST) {
	return "";
    }
    else {
	(void) gen_check((gen_chunk *) e, entity_domain);
	return entity_has_values_p(e)? entity_minimal_name(e) :
	    external_value_name(e);
    }
}
