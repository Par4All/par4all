/*
 * HPFC module by Fabien COELHO
 *
 * SCCS stuff:
 * $RCSfile: host_node_entities.c,v $ ($Date: 1994/09/01 15:48:11 $, ) version $Revision$,
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


/*
 * Host and Node Entities Management
 *
 */

GENERIC_CURRENT_MAPPING(host_new, entity, entity);
GENERIC_CURRENT_MAPPING(host_old, entity, entity);
GENERIC_CURRENT_MAPPING(node_new, entity, entity);
GENERIC_CURRENT_MAPPING(node_old, entity, entity);

void store_new_node_variable(new, old)
entity new, old;
{
    pips_assert("store_new_node_variable",
		(!entity_undefined_p(new)) || (!entity_undefined_p(old)));

    store_entity_node_new(old, new);
    store_entity_node_old(new, old);
}

void store_new_host_variable(new, old)
entity new, old;
{
    pips_assert("store_new_host_variable",
		(!entity_undefined_p(new)) || (!entity_undefined_p(old)));

    store_entity_host_new(old, new);
    store_entity_host_old(new, old);
}

void make_host_node_maps()
{
    make_host_new_map();
    make_host_old_map();
    make_node_new_map();
    make_node_old_map();
}

void free_host_node_maps()
{
    free_host_new_map();
    free_host_old_map();
    free_node_new_map();
    free_node_old_map();
}

string hpfc_module_suffix(module)
entity module;
{
    if (module==node_module) return(NODE_NAME);
    if (module==host_module) return(HOST_NAME);

    /* else */

    pips_error("hpfc_module_suffix", "unexpected module\n");
    return(string_undefined);
}
  

/*
 * updates
 */

static entity_mapping
    current_entity_map = hash_table_undefined;

entity_mapping hpfc_map_of_module(module)
entity module;
{
    if (module==node_module) return(get_node_new_map());
    if (module==host_module) return(get_host_new_map());

    /* else */

    pips_error("hpfc_map_of_module", "unexpected module\n");
    return(hash_table_undefined);
}

/* shift the references to the right variable, in the module
 */
static void update_reference_for_module_rewrite(ref)
reference ref;
{
    entity
	var = reference_variable(ref),
	new = (entity) GET_ENTITY_MAPPING(current_entity_map, var);

    if (new != (entity) HASH_UNDEFINED_VALUE)
	reference_variable(ref) = new;
}

/* shift the calls to the right variable, in the module
 */
static void update_call_for_module_rewrite(c)
call c;
{
    entity
	fun = call_function(c),
	new = (entity) GET_ENTITY_MAPPING(current_entity_map, fun);

    if (new != (entity) HASH_UNDEFINED_VALUE)
	call_function(c) = new;
}

void update_object_for_module(object, module)
chunk *object;
entity module;
{
    entity_mapping
	saved = current_entity_map;

    current_entity_map = hpfc_map_of_module(module);

    gen_recurse(object, 
		reference_domain, 
		gen_true, 
		update_reference_for_module_rewrite);

    gen_recurse(object, 
		call_domain, 
		gen_true, 
		update_call_for_module_rewrite);

    current_entity_map = saved;
}

void update_list_for_module(l, module)
list l;
entity module;
{
    MAPL(cx,
     {
	 update_object_for_module(CHUNK(CAR(cx)), module);
     },
	 l);
}

/* removed unreferenced items in the common
 * the global map refenreced_variables should be set and ok
 * the variables updated are those local to the common...
 */
void clean_common_declaration(common)
entity common;
{
    entity
	var = entity_undefined;
    type
	t = entity_type(common);
    list
	l = NIL,
	lnew = NIL;

    pips_assert("clean_common_declaration", type_area_p(t));

    l = area_layout(type_area(t));

    MAPL(ce,
     {
	 var = ENTITY(CAR(ce));

	 if (load_entity_referenced_variables(var)==TRUE &&
	     local_entity_of_module_p(var, common))
	     lnew = CONS(ENTITY, var, lnew);
     },
	 l);

    gen_free_list(l);
    area_layout(type_area(t)) = lnew;
}

/*
 * that is all
 */

/*
 *
 *
 *    OLD FUNCTIONS TO BE CLEANED...
 *
 *
 *
 */

/*
 * UpdateExpressionForModule
 *
 * this function creates a new expression using the mapping of
 * old to new variables map.
 *
 * some of the structures generated may be shared...
 */

expression UpdateExpressionForModule(module, ex)
entity module;
expression ex;
{
    expression
	new = copy_expression(ex);

    update_object_for_module(new, module);

    return(new);
}


list lUpdateExpr(module, l)
entity module;
list l;
{
    list
	new = NIL,
	rev = NIL;

    MAPL(cx,
     {
	 rev = CONS(EXPRESSION,
		    copy_expression(EXPRESSION(CAR(cx))),
		    rev);
     },
	 l);

    new = gen_nreverse(rev); 
    update_list_for_module(new, module);
    
    return(new);
}


/* new
list lNewVariableForModule(module, le)
entity module;
list le;
{

}
*/

list lNewVariableForModule(module, le)
entity module;
list le;
{
    return((ENDP(le) ?
	    (NIL) :
	    CONS(ENTITY,
		 NewVariableForModule(module, ENTITY(CAR(le))),
		 lNewVariableForModule(module ,CDR(le)))));
}

entity NewVariableForModule(module,e)
entity module;
entity e;
{
    entity_mapping
	map = hpfc_map_of_module(module);

    return((entity) GET_ENTITY_MAPPING(map,e));
}


statement UpdateStatementForModule(module, stat)
entity module;
statement stat;
{
    statement
	updatedstat = statement_undefined;
    instruction 
	inst = statement_instruction(stat);

    debug(7, "UpdateStatementForModule", "updating...\n");

    switch(instruction_tag(inst))
    {
    case is_instruction_block:
    {
	list
	    lstat = NIL;
	
	debug(8, "UpdateStatementForModule", "block\n");

	MAPL(cs,
	 {
	     statement
		 stmp = UpdateStatementForModule(module, STATEMENT(CAR(cs)));

	     lstat = 
		 gen_nconc(lstat, CONS(STATEMENT, stmp, NULL));
	 },
	     instruction_block(inst));

	updatedstat = MakeStatementLike(stat, 
					is_instruction_block, 
					get_node_gotos_map());
	instruction_block(statement_instruction(updatedstat)) = lstat;
	break;
    }
    case is_instruction_test:
    {
	test
	    t = instruction_test(inst);

	debug(8, "UpdateStatementForModule", "test\n");

	updatedstat = MakeStatementLike(stat, 
					is_instruction_test, 
					get_node_gotos_map());
	instruction_test(statement_instruction(updatedstat)) = 
	    make_test(UpdateExpressionForModule(module, test_condition(t)),
		      UpdateStatementForModule(module, test_true(t)),
		      UpdateStatementForModule(module, test_false(t)));
	break;
    }
    case is_instruction_loop:
    {
	loop
	    l = instruction_loop(inst);
	range
	    r = loop_range(l);
	entity
	    nindex = NewVariableForModule(node_module, loop_index(l));

	debug(8, "UpdateStatementForModule", "loop\n");

	updatedstat = MakeStatementLike(stat, 
					is_instruction_loop, 
					get_node_gotos_map());
	instruction_loop(statement_instruction(updatedstat)) = 
	    make_loop(nindex,
		      make_range(UpdateExpressionForModule(module, 
							   range_lower(r)),
				 UpdateExpressionForModule(module, 
							   range_upper(r)),
				 UpdateExpressionForModule(module, 
							   range_increment(r))),
		      UpdateStatementForModule(module, loop_body(l)),
		      loop_label(l),
		      make_execution(is_execution_sequential,UU),
		      NULL);
	break;
    }
    case is_instruction_goto:
    {
	debug(8, "UpdateStatementForModule", "goto\n");

	updatedstat = MakeStatementLike(stat, 
					is_instruction_goto, 
					get_node_gotos_map());
	instruction_goto(statement_instruction(updatedstat)) = 
	    instruction_goto(inst);

	break;
    }
    case is_instruction_call:
    {
	call
	    c = instruction_call(inst);

	debug(8, "UpdateStatementForModule", 
	      "call to %s\n", 
	      entity_name(call_function(c)));

	updatedstat = MakeStatementLike(stat,
					is_instruction_call,
					get_node_gotos_map());
	instruction_call(statement_instruction(updatedstat)) = 
	    make_call(call_function(c), lUpdateExpr(module, call_arguments(c)));

	break;
    }
    case is_instruction_unstructured:
    {
	control_mapping 
	    ctrmap = MAKE_CONTROL_MAPPING();
	unstructured 
	    u=instruction_unstructured(inst);
	control 
	    ct = unstructured_control(u),
	    ce = unstructured_exit(u);
	list 
	    blocks = NIL;

	debug(8, "UpdateStatementForModule", "unstructured\n");

	CONTROL_MAP(c,
		{
		    statement
			statc = control_statement(c);
		    control
			ctr;

		    ctr = make_control(UpdateStatementForModule(module, statc),
				       NULL,
				       NULL);
		    SET_CONTROL_MAPPING(ctrmap, c, ctr);
		},
		    ct,
		    blocks);

	MAPL(cc,
	 {
	     control
		 c = CONTROL(CAR(cc));

	     update_control_lists(c, ctrmap);
	 },
	     blocks);

	updatedstat = MakeStatementLike(stat,
					is_instruction_unstructured,
					get_node_gotos_map());
	statement_instruction(instruction_unstructured(updatedstat)) =
	    make_unstructured((control) GET_CONTROL_MAPPING(ctrmap, ct),
			      (control) GET_CONTROL_MAPPING(ctrmap, ce));

	gen_free_list(blocks);
	FREE_CONTROL_MAPPING(ctrmap);
	break;
    }
    default:
	pips_error("UpdateStatementForModule","unexpected instruction tag\n");
	break;
    }
    
    debug(7, "UpdateStatementForModule", "end of update\n");
    return(updatedstat);
}

