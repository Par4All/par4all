/*
 * $Id$
 *
 * $Log: initial.c,v $
 * Revision 1.4  1997/09/08 09:39:11  coelho
 * *** empty log message ***
 *
 * Revision 1.3  1997/09/08 09:35:29  coelho
 * transformer -> precondition.
 *
 * Revision 1.2  1997/09/08 08:51:14  coelho
 * the code is not printed. name fixed.
 *
 * Revision 1.1  1997/09/08 08:45:50  coelho
 * Initial revision
 *
 *
 * Computation of initial transformers that allow to collect
 * global initializations of BLOCK DATA and so.
 */

#include <stdio.h>
#include <string.h>

#include "genC.h"

#include "ri.h"
#include "ri-util.h"

#include "database.h"
#include "pipsdbm.h"
#include "resources.h"

#include "misc.h"

#include "effects-generic.h"
#include "effects-simple.h"

#include "transformer.h"

#include "semantics.h"

/******************************************************** PIPSMAKE INTERFACE */

/* Compute an initial transformer.
 */
bool 
initial_preconditions(string name)
{
    pips_internal_error("not implemented yet");
    return FALSE;
}

bool
program_preconditions(string name)
{
    pips_internal_error("not implemented yet");
    return FALSE;
}

/*********************************************************** PRETTY PRINTERS */

bool 
print_initial_preconditions(string name)
{
    pips_internal_error("not implemented yet");
    return FALSE;
}

bool 
print_program_preconditions(string name)
{
    pips_internal_error("not implemented yet");
    return FALSE;
}

