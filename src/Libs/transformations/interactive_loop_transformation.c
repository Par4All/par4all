/* Interface with pipsmake for interactive loop transformations: loop
 *interchange, hyperplane method,...
 *
 * $Id$
 *
 * $Log: interactive_loop_transformation.c,v $
 * Revision 1.1  1998/10/11 20:02:58  irigoin
 * Initial revision
 * */

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

entity selected_label;

bool selected_loop_p(loop l)
{
    return loop_label(l) == selected_label;
}

bool
interactive_loop_transformation(string module_name, statement (*loop_transformation)())
{
    char lp_label[6];
    entity module = local_name_to_top_level_entity(module_name);
    statement s;
    string resp;
    bool return_status = FALSE;

    pips_assert("interactive_loop_transformation", entity_module_p(module));

    /* DBR_CODE will be changed: argument "pure" should take FALSE but
       this would be useless since there is only *one* version of code;
       a new version will be put back in the data base after transforming
       the loops */
    s = (statement) db_get_memory_resource(DBR_CODE, module_name, TRUE);

    /* Get the loop label form the user */
    resp = user_request("Which loop do you want to transform?\n"
			"(give its label): ");
    if (resp[0] == '\0') {
	user_log("Interactive loop transformation has been cancelled.\n");
	return_status = FALSE;
    }
    else {    
	sscanf(resp, "%s", lp_label);
	selected_label = find_label_entity(module_name, lp_label);
	if (selected_label==entity_undefined) {
	    pips_user_error("loop label `%s' does not exist\n", lp_label);
	}

	debug_on("INTERACTIVE_LOOP_TRANSFORMATION_DEBUG_LEVEL");

	look_for_nested_loop_statements(s, loop_transformation, selected_loop_p);

	debug_off();

	/* Reorder the module, because new statements have been generated. */
	module_reorder(s);

	DB_PUT_MEMORY_RESOURCE(DBR_CODE, 
			       strdup(module_name), 
			       (char*) s);
	return_status = TRUE;
    }
    
    return return_status;
}
