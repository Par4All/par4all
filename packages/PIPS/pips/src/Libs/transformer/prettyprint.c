/*

  $Id$

  Copyright 1989-2014 MINES ParisTech

  This file is part of PIPS.

  PIPS is free software: you can redistribute it and/or modify it
  under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  any later version.

  PIPS is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.

  See the GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with PIPS.  If not, see <http://www.gnu.org/licenses/>.

*/
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif
/*
 * This section has been revised and cleaned.
 * The main change is to sort the arguments for
 * the preconditions print.
 *
 * Modification : Arnauld LESERVOT
 * Date         : 92 08 27
 * Old version  : prettyprint.old.c
 */

#include <stdio.h>
#include <string.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"

#include "constants.h"

#include "misc.h"
#include "preprocessor.h"

#include "transformer.h"

string transformer_to_string(transformer tf)
{
    pips_internal_error("not implemenented anymore, tf=%p", tf);
    return string_undefined;
}

string precondition_to_string(pre)
transformer pre;
{
    pips_internal_error("not implemenented anymore, pre=%p", pre);
    return string_undefined;
}

string arguments_to_string(string s, list args)
{
    pips_internal_error("not implemenented anymore, s=\"%s\", args=%p", s, args);
    return string_undefined;
}

string
relation_to_string(
    string s,
    Psysteme ps,
    char * (*variable_name)(entity))
{
    pips_internal_error("not implemenented anymore, s=\"%s\", ps=%p, variable_name=%p", s, ps, variable_name);
    return string_undefined;
}

const char * generic_value_name(entity e)
{
  const char* n = string_undefined;

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
