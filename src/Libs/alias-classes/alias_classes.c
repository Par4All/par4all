/* $Id$
 */

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

#include "effects-generic.h"
#include "effects-simple.h"
#include "effects-convex.h"

#include "pipsdbm.h"
#include "resources.h"

bool
alias_equivalence_classes ( string name )
{
    return TRUE;
}

