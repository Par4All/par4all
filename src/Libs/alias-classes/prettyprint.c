
#include <stdio.h>
#include <string.h>

#include <setjmp.h>

#include "genC.h"
#include "ri.h"
#include "database.h"

#include "ri-util.h"
#include "constants.h"
#include "control.h"
#include "misc.h"
#include "text.h"

#include "effects.h"
#include "regions.h"
#include "semantics.h"

#include "pipsdbm.h"
#include "resources.h"

#define BACKWARD TRUE
#define FORWARD FALSE

bool
print_in_alias_pairs( string module_name )
{
return(TRUE);
}

bool
print_out_alias_pairs( string module_name )
{
return(TRUE);
}
