/*

  $Id$

  Copyright 1989-2010 MINES ParisTech

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
 /* package transformer - IOs
  *
  * Francois Irigoin, 21 April 1990
  */

#include <stdio.h>

#include "genC.h"
#include "misc.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"

#include "boolean.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"

#include "transformer.h"

/* print_transformer(tf): not a macro because of dbx and gdb */
transformer print_transformer(transformer tf)
{
  (void) fprint_transformer(stderr, tf,
			    (get_variable_name_t) external_value_name);
  return tf;
}

/* For debugging without problem from temporary values */
transformer print_any_transformer(transformer tf)
{
    return fprint_transformer(stderr, tf,
			      (get_variable_name_t) entity_local_name);
}

list print_transformers(list tl)
{
    return fprint_transformers(stderr, tl,
			       (get_variable_name_t) external_value_name);
}

transformer
fprint_transformer(FILE * fd,
		   transformer tf,
		   get_variable_name_t value_name)
{
    /* print_transformer returns an int to be compatible with the debug()
       function; however, debug being a function and not a macro, its
       arguments are ALWAYS evaluated regardless of the debug level;
       so a call to print_transformer passed as an argument to debug
       is ALWAYS effective */

  int dn = transformer_domain_number(tf);

  // For debugging with gdb, dynamic type checking
  if(dn!=transformer_domain) {
    (void) fprintf(stderr,"Arg. \"e\"is not an expression.\n");
  }
  else if(tf==transformer_undefined)
    (void) fprintf(stderr,"TRANSFORMER UNDEFINED\n");
  else {
	cons * args = transformer_arguments(tf);
	Psysteme sc = (Psysteme) predicate_system(transformer_relation(tf));

	/* print argument list */
	(void) fprintf(fd,"arguments:");
	print_homogeneous_arguments(args, (const char* (*) (entity))value_name);

	/* print relation */
	if(SC_UNDEFINED_P(sc))
	    pips_internal_error("undefined relation");
	(void) fprintf(fd,"relation:");
	sc_fprint(fd,
		  sc,
		  value_name);
    }

    return tf;
}

list fprint_transformers(FILE * fd,
			 list tl,
			 get_variable_name_t value_name)
{
  if(ENDP(tl)) {
    // FI: I changed my mind; this is a way to represent a non
    //feasible transformer
    //pips_internal_error("transformer lists should never be empty.");
    fprintf(fd, "Empty transformer list\n");
  }
  else {
    FOREACH(TRANSFORMER, tf, tl) {
      fprint_transformer(fd, tf, value_name);
    }
  }
  return tl;
}

/* char * dump_value_name(e): used as functional argument because
 * entity_name is a macro
 *
 * FI: should be moved in ri-util/entity.c
 */
char * dump_value_name(e)
entity e;
{
    return entity_name(e);
}

void dump_transformer(tf)
transformer tf;
{
    (void) fprint_transformer(stderr, tf, (get_variable_name_t) dump_value_name);
}
