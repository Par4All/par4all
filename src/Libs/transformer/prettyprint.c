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
 * Revision 1.9  2001/12/05 17:09:38  irigoin
 * Modification to handle value names in a safer way
 *
 * Revision 1.8  2001/10/22 15:57:34  irigoin
 * Temporary values taken into account in generic_value_name()
 *
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

char * generic_value_name(entity e)
{
  string n = string_undefined;

  if(e == (entity) TCST) {
    n = "";
  }
  else {
    (void) gen_check((gen_chunk *) e, entity_domain);
    if(local_temporary_value_entity_p(e)) {
      n = entity_minimal_name(e);
    }
    /* else if (entity_has_values_p(e)){ */
    else if (value_entity_p(e)){
      /* n = external_value_name(e); */
      n = pips_user_value_name(e);
    }
    else {
      n = entity_minimal_name(e);
    }
  }
  return n;
}
