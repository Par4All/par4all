 /* package transformer - IOs 
  *
  * Francois Irigoin, 21 April 1990
  */

#include <stdio.h>

#include "genC.h"
#include "misc.h"
#include "ri.h"
#include "ri-util.h"

#include "boolean.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"

#include "transformer.h"

/* print_transformer(tf): not a macro because of dbx */
int print_transformer(tf)
transformer tf;
{
    return fprint_transformer(stderr, tf, external_value_name);
}

int fprint_transformer(fd, tf, value_name)
char * (*value_name)();
FILE * fd;
transformer tf;
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
	print_arguments(args);

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
    return (int) tf;
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
    (void) fprint_transformer(stderr, tf, dump_value_name);
}
