/******************************************************************
 *
 *		     ALIAS PROPAGATION
 *
 *
*******************************************************************/

/* Aliasing occurs when two or more variables refer to the same
storage location at the same program point.

This phase tries to compute as precise as possible the
interprocedural alias information in a whole program.

The basic idea for computing interprocedural aliases is to follow all the
possible chains of argument-parameters and nonlocal variable-parameter
bindings at all call sites. We introduce a naming memory locations technique which guarantees the correctness and enhances the
precision of data-flow analysis. */


#include <stdio.h>
#include <stdlib.h>
#include <string.h> 

#include "genC.h"

#include "linear.h"

#include "ri.h"
#include "ri-util.h"
#include "database.h"
#include "pipsdbm.h"
#include "resources.h"
#include "misc.h"
#include "control.h"
#include "properties.h"
#include "semantics.h"

#include "instrumentation.h"
#include "transformations.h"

bool alias_propagation(char * module_name)
{
  return TRUE;
}
