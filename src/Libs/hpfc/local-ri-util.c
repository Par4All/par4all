/* SCCS Stuff:
 * $RCSfile: local-ri-util.c,v $ ($Date: 1994/11/17 14:19:13 $, )
 * version $Revision$
 * got on %D%, %T%
 *
 * Fabien Coelho May 1993
 */


#include <stdio.h>
#include <string.h>

extern int fprintf();

#include "genC.h"

#include "ri.h"
#include "hpf.h"
#include "hpf_private.h"

#include "ri-util.h"
#include "misc.h"
#include "text-util.h"
#include "hpfc.h"
#include "defines-local.h"

/*
 * type_variable_dup
 */
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

/*
 * that is all
 */
