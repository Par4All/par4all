/* interface with pipsmake
 */

#include <stdio.h>
#include <string.h>

#include "genC.h"
#include "constants.h"
#include "misc.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "database.h"
#include "pipsdbm.h"
#include "resources.h"

#include "control.h"
#include "conversion.h"
/* #include "generation.h" */

#include "transformations.h"

bool
loop_interchange(string module_name)
{
    bool return_status = FALSE;

    return_status = interactive_loop_transformation(module_name, interchange);
    
    return return_status;
}
