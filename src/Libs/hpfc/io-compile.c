/*
 * HPFC module by Fabien COELHO
 *
 * SCCS stuff:
 * $RCSfile: io-compile.c,v $ ($Date: 1994/03/23 17:45:03 $, ) version $Revision$,
 * got on %D%, %T%
 * $Id$
 */

/*
 * Standard includes
 */
 
#include <stdio.h>
#include <string.h> 
extern fprintf();

/*
 * Psystems stuff
 */

#include "types.h"
#include "boolean.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"

/*
 * Newgen stuff
 */

#include "genC.h"

#include "ri.h" 
#include "hpf.h" 
#include "hpf_private.h"

/*
 * PIPS stuff
 */

#include "ri-util.h" 
#include "misc.h" 
#include "control.h"
#include "regions.h"
#include "semantics.h"
#include "effects.h"

/* 
 * my own local includes
 */

#include "hpfc.h"
#include "defines-local.h"

/*----------------------------------------------------------------
 *
 * IO EFFICIENT COMPILATION
 */

/*
 * bool io_efficient_compilable_p(stat)
 * statement stat;
 *
 * checks for the compilability of the statement.
 *
 * ??? to be implemented
 */
bool io_efficient_compilable_p(stat)
statement stat;
{
    pips_assert("io_efficient_compilable_p",
		load_statement_only_io(stat)==TRUE);

    user_warning("io_efficient_compilable_p", 
		 "not implemented yet, returning FALSE!\n"); 

    return(FALSE);
}

void io_efficient_compile(stat, hp, np)
statement stat, *hp, *np;
{
    


    pips_error("io_efficient_compile",
	       "not implemented yet\n");
}


/*
 * that's all
 */
