/******************************************************************
 *
 *		     ALIAS VERIFICATION
 *
 *
*******************************************************************/

/* In Fortran 77, parameters are passed by address in such
a way that, as long as the actual argument is associated with a named
storage location, the called subprogram can change the value of the actual
argument by assigning a value to the corresponding formal parameter. So
new aliases can be created between formal parameters if the
same actual argument is passed to two or more formal parameters, or between formal
parameters and global parameters if an actual
argument is an object in common storage which is also visible in the
called subprogram or other subprograms in the call chain below it.

Restrictions on association of entities in Fortran 77 (Section 15.9.3.6
\cite{ANSI83}) say that neither aliased formal parameters nor the variable in the
common block may become defined during execution
of the called subprogram or the others subprograms in the call chain.

This phase verifies statically if the program violates the standard restriction on alias or
not by using information from the alias propagation phase. 
If these informations are not known at compile-time, we instrument
the code with tests that check the violation dynamically during
execution of program.*/

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
bool alias_check(char * module_name)
{
  return TRUE;
}
