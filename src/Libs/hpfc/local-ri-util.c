/* $RCSfile: local-ri-util.c,v $ ($Date: 1995/04/10 18:49:37 $, )
 * version $Revision$
 *
 * HPFC (c) Fabien Coelho May 1993
 */

#include "defines-local.h"

type type_variable_dup(t)
type t;
{
    if(type_variable_p(t))
    {
	variable
	    v = type_variable(t);

	return(MakeTypeVariable(variable_basic(v),
				ldimensions_dup(variable_dimensions(v))));
    }
    else
	return(t); /* !!! means sharing */
}

/*  library functions...
 */
static string fortran_library[] =
{ 
  "TIME",
  (string) NULL
} ;

bool fortran_library_entity_p(e)
entity e;
{
    string *s, name=entity_local_name(e);

    if (!top_level_entity_p(e)) return(FALSE);
    for (s=fortran_library; *s!=(string) NULL; s++)
	if (same_string_p(*s, name)) return(TRUE);

    return(FALSE); /* else not found */
}

/* that is all
 */
