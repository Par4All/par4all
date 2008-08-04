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
    pips_internal_error("not implemenented anymore, tf=%p\n", tf);
    return string_undefined;
}

string precondition_to_string(pre)
transformer pre;
{
    pips_internal_error("not implemenented anymore, pre=%p\n", pre);
    return string_undefined;
}

string arguments_to_string(string s, list args)
{
    pips_internal_error("not implemenented anymore, s=\"%s\", args=%p\n", s, args);
    return string_undefined;
}

string 
relation_to_string(
    string s,
    Psysteme ps,
    char * (*variable_name)(entity))
{
    pips_internal_error("not implemenented anymore, s=\"%s\", ps=%p, variable_name=%p\n", s, ps, variable_name);
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
    else if (!hash_value_to_name_undefined_p()
	     && value_entity_p(e)){
      /* n = external_value_name(e); */
      n = pips_user_value_name(e);
    }
    else {
      n = entity_minimal_name(e);
    }
  }
  return n;
}
