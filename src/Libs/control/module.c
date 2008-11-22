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

/* FI: a short-term solution to fix declarations lost due to
   unstructured building by controlizer. */
static statement update_unstructured_declarations(statement module_stat)
{
  list dl = statement_to_declarations(module_stat);
  list vl = statement_to_referenced_entities(module_stat);
  list fl = statement_to_called_user_entities(module_stat);
  /* To preserve the order, it would be better to collect variables
     and functions at the same time with a
     statement_to_referenced_or_called-entities()*/
  list vfl = gen_nconc(vl, fl);
  list udl = NIL;
  entity m = get_current_module_entity();
  entity cu = module_entity_to_compilation_unit_entity(m);
  list cudl = code_declarations(value_code(entity_initial(cu)));

  MAP(ENTITY, e, {
    if(!gen_in_list_p(e, dl) && !gen_in_list_p(e, cudl)
       && !gen_in_list_p(e, udl) && !formal_parameter_p(e)
       && !member_entity_p(e))
      udl = gen_nconc(udl, CONS(ENTITY, e, NIL));
  }, vfl);

  if(!ENDP(udl)) {
    ifdebug(8) {
      pips_debug(8, "Lost declarations: ");
      print_entities(udl);
    }
    if(statement_block_p(module_stat))
      statement_declarations(module_stat)
	= gen_nconc(statement_declarations(module_stat), udl);
    else if(statement_unstructured_p(module_stat)) {
      /* might be OK... */
      statement_declarations(module_stat)
	= gen_nconc(statement_declarations(module_stat), udl);
    }
    else {
      module_stat = make_block_statement(CONS(STATEMENT, module_stat, NIL));
      statement_declarations(module_stat)
	= gen_nconc(statement_declarations(module_stat), udl);
    }
  }
  return module_stat;
}

/* interface with pipsdbm and pipsmake */

bool controlizer(string module_name)
{
  entity m = module_name_to_entity(module_name);
   
  statement module_stat, parsed_mod_stat;
    
  set_current_module_entity(m);

  parsed_mod_stat = (statement) db_get_memory_resource(DBR_PARSED_CODE, module_name, TRUE);
  module_stat =  copy_statement(parsed_mod_stat) ;
    
  debug_on("CONTROL_DEBUG_LEVEL");

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
						control_graph(module_stat)),
			       NIL /* gen_copy_seq(statement_declarations(parsed_mod_stat))*/,
			       NULL);
  ifdebug(5) {
    statement_consistent_p(module_stat);
    pips_debug(5, "New statement before unspaghettify:\n");
    print_statement(module_stat);
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

    /* With C code, some local declarations may have been lost by the
       (current) restructurer */
    if(c_module_p(m)) 
      module_stat = update_unstructured_declarations(module_stat);

    ifdebug(5) {
	statement_consistent_p(module_stat);
    }

    DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name, module_stat);
    
    reset_current_module_entity();

    debug_off();

    return TRUE;
}
