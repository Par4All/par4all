/*

  $Id$

  Copyright 1989-2009 MINES ParisTech

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

/* print_transformer(tf): not a macro because of dbx */
transformer
print_transformer(transformer tf)
{
    return fprint_transformer(stderr, tf, (get_variable_name_t) external_value_name);
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

    if (tf!=transformer_undefined) {
	cons * args = transformer_arguments(tf);
	Psysteme sc = (Psysteme) predicate_system(transformer_relation(tf));

	/* print argument list */
	(void) fprintf(fd,"arguments:");
	print_homogeneous_arguments(args, value_name);

	/* print relation */
	if(SC_UNDEFINED_P(sc))
	    pips_error("fprint_transformer", "undefined relation\n");
	(void) fprintf(fd,"relation:");
	sc_fprint(fd,
		  sc,
		  value_name);
    }
    else
	(void) fprintf(fd, "TRANSFORMER_UNDEFINED\n");
    return tf;
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
