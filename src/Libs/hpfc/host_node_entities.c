/*
 * HPFC module by Fabien COELHO
 *
 * SCCS stuff:
 * $RCSfile: host_node_entities.c,v $ ($Date: 1994/12/30 16:49:15 $, ) version $Revision$,
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
    assert(!entity_undefined_p(new) || !entity_undefined_p(old));

    store_entity_node_new(old, new);
    store_entity_node_old(new, old);
}

void store_new_host_variable(new, old)
entity new, old;
{
    assert(!entity_undefined_p(new) || !entity_undefined_p(old));

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

static void update_for_module_rewrite(pe)
entity *pe;
{
    entity
	new = (entity) GET_ENTITY_MAPPING(current_entity_map, *pe);

    if (new != (entity) HASH_UNDEFINED_VALUE) *pe = new;
}

/* shift the references to the right variable, in the module
 */
static void update_reference_for_module_rewrite(ref)
reference ref;
{
    update_for_module_rewrite(&reference_variable(ref));
}

/* shift the calls to the right variable, in the module
 */
static void update_call_for_module_rewrite(c)
call c;
{
    update_for_module_rewrite(&call_function(c));
}

static void update_code_for_module_rewrite(c)
code c;
{
    MAPL(ce,
     {
	 update_for_module_rewrite(&ENTITY(CAR(ce)));
     },
	 code_declarations(c));
}

static void update_loop_for_module_rewrite(l)
loop l;
{
    update_for_module_rewrite(&loop_index(l));
}

void update_object_for_module(obj, module)
gen_chunk *obj; /* loosely typed, indeed */
entity module;
{
    entity_mapping
	saved = current_entity_map;

    debug(8, "update_object_for_module", "updating (%s) 0x%x\n",
	  gen_domain_name(gen_type(obj)), (unsigned int) obj);

    current_entity_map = hpfc_map_of_module(module);

    gen_multi_recurse(obj, 
		      /* 
		       *   REFERENCES
		       */
		      reference_domain, 
		      gen_true, 
		      update_reference_for_module_rewrite,
		      /*
		       *   LOOPS (indexes)
		       */
		      loop_domain,
		      gen_true,
		      update_loop_for_module_rewrite,
		      /*
		       *   CALLS
		       */
		      call_domain, 
		      gen_true, 
		      update_call_for_module_rewrite,
		      /*
		       *   CODES
		       */
		      code_domain,
		      gen_true,
		      update_code_for_module_rewrite,
		      NULL);

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

    assert(type_area_p(t));

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

list lNewVariableForModule(module, le)
entity module;
list le;
{
    list result, last;

    if (ENDP(le)) return(NIL);

    for (result = CONS(ENTITY, 
		       NewVariableForModule(module, ENTITY(CAR(le))),
		       NIL),
	 last = result, le = CDR(le);
	 !ENDP(le);
	 le = CDR(le), last = CDR(last))
	CDR(last) = CONS(ENTITY, NewVariableForModule(module, ENTITY(CAR(le))),
			 NIL);

    return(result);
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
	new_stat = copy_statement(stat);
    
    update_object_for_module(new_stat, module);

    return(new_stat);
}

/*   That is all
 */
