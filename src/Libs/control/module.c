#include <stdio.h>
#include <strings.h>
#include <string.h>

#include "linear.h"

#include "genC.h"
#include "ri.h"
#include "database.h"

#include "ri-util.h"
#include "control.h"
#include "pipsdbm.h"

#include "resources.h"
#include "properties.h"
#include "constants.h"

#include "misc.h"

/* interface with pipsdbm and pipsmake */

bool controlizer(string module_name)
{
  entity m = local_name_to_top_level_entity(module_name);
   
    statement module_stat;
    
    set_current_module_entity(m);

    module_stat = 
	copy_statement((statement)
	    db_get_memory_resource(DBR_PARSED_CODE, module_name, TRUE)) ;
    
       /* *module_stat can be re-used because control_graph reallocates
	  statements; do not show that to any student!
	  statement_instruction(module_stat) =
	  make_instruction(is_instruction_unstructured,
	        control_graph(module_stat));
    Maintenant que je nettoie le code aussi avant le controlizer,
       un programme sans instruction ne contient qu'un statement
       RETURN, c'est a` dire un statement de CALL vers RETURN avec le
       label 00000.  Comme la ligne ci-dessus recycle le label et le
       commentaire, on se retrouve avec un unstructured avec le label
       et le commentaire du RETURN qu'il contient... En plus il y avait
       une re'cursion de l'unstructured vers module_stat. :-(

       So now correct the label and the comment: */

    module_stat = make_statement(entity_empty_label(), 
				 STATEMENT_NUMBER_UNDEFINED,
				 MAKE_ORDERING(0,1),
				 empty_comments,
				 make_instruction(is_instruction_unstructured,
						  control_graph(module_stat)));
    ifdebug(5) {
	statement_consistent_p(module_stat);
    }

    /* By setting this property, we try to unspaghettify the control
       graph of the module: */

    if (get_bool_property("UNSPAGHETTIFY_IN_CONTROLIZER")) {

	/* To have the debug in unspaghettify_statement() working: */
	set_current_module_statement(module_stat);
       
	unspaghettify_statement(module_stat);

	/* Reorder the module, because statements may have been
           changed. */
	module_reorder(module_stat);
	reset_current_module_statement();
    }

    ifdebug(5) {
	statement_consistent_p(module_stat);
    }

    DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name, module_stat);
    
    reset_current_module_entity();

    return TRUE;
}
