/*

  $Id$

  Copyright 1989-2014 MINES ParisTech

  This file is part of PIPS.

  PIPS is free software: you can redistribute it and/or modify it
  under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  any later version.

  PIPS is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.

  See the GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with PIPS.  If not, see <http://www.gnu.org/licenses/>.

*/
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif
/* HPFC module by Fabien COELHO
 */

#include "defines-local.h"

#include "effects-generic.h"
#include "effects-simple.h"
#include "effects-convex.h"

/*      HOST AND NODE ENTITIES MANAGEMENT
 */
GENERIC_GLOBAL_FUNCTION(new_host, entitymap)
GENERIC_GLOBAL_FUNCTION(old_host, entitymap)
GENERIC_GLOBAL_FUNCTION(new_node, entitymap)
GENERIC_GLOBAL_FUNCTION(old_node, entitymap)

void
debug_host_node_variables(entity e)
{
    fprintf(stderr, "variable %s:\n\tnh=%s\n\toh=%s\n\tnn=%s\n\ton=%s\n",
	    entity_name(e),
	    bound_new_host_p(e)? entity_name(load_new_host(e)): "<undef>",
	    bound_old_host_p(e)? entity_name(load_old_host(e)): "<undef>",
	    bound_new_node_p(e)? entity_name(load_new_node(e)): "<undef>",
	    bound_old_node_p(e)? entity_name(load_old_node(e)): "<undef>");
}

void 
store_new_node_variable(entity new, entity old)
{
    pips_assert("defined", !entity_undefined_p(new)&&!entity_undefined_p(old));

    store_or_update_new_node(old, new);
    store_or_update_old_node(new, old);
}

void 
store_new_host_variable(entity new, entity old)
{
    pips_assert("defined", !entity_undefined_p(new)&&!entity_undefined_p(old));

    store_or_update_new_host(old, new);
    store_or_update_old_host(new, old);
}

void 
store_new_host_node_variable(
    entity neh /* host version */, 
    entity nen /* node version */, 
    entity old /* initial entity */)
{
    store_new_host_variable(neh, old);
    store_new_host_variable(neh, nen);
    store_new_node_variable(nen, old);
    store_new_node_variable(nen, neh);
}


void init_entity_status()
{
    init_new_host();
    init_old_host();
    init_new_node();
    init_old_node();
}

entity_status get_entity_status()
{
    return make_entity_status(get_new_host(),
			      get_new_node(),
			      get_old_host(),
			      get_old_node(),
			      entity_int_undefined);
}

void set_entity_status(entity_status s)
{
    set_new_host(entity_status_new_host(s));
    set_new_node(entity_status_new_node(s));
    set_old_host(entity_status_old_host(s));
    set_old_node(entity_status_old_node(s));
}

void reset_entity_status()
{
    reset_new_host();
    reset_old_host();
    reset_new_node();
    reset_old_node();
}

void close_entity_status()
{
    close_new_host();
    close_old_host();
    close_new_node();
    close_old_node();
}

string hpfc_module_suffix(module)
entity module;
{
    if (module==node_module) return(NODE_NAME);
    if (module==host_module) return(HOST_NAME);
    /* else
     */
    pips_internal_error("unexpected module");
    return string_undefined; /* to avoid a gcc warning */
}
  

/****************************************************************** UPDATES */

static bool (*bound_p)(entity) = (bool(*)(entity)) gen_false;
static entity (*load)(entity) = (entity(*)(entity)) gen_identity;

static void update_for_module_rewrite(
    entity *pe)
{
    if (bound_p(*pe)) 
    {
	entity n = load(*pe);
	pips_debug(10, "%s -> %s\n", entity_name(*pe), entity_name(n));
	*pe = n;
    }
}

/* shift the references to the right variable, in the module
 */
static void update_reference_for_module_rewrite(reference ref)
{
    update_for_module_rewrite(&reference_variable(ref));
}

/* shift the calls to the right variable, in the module
 */
static void update_call_for_module_rewrite(call c)
{
    update_for_module_rewrite(&call_function(c));
}

static void update_code_for_module_rewrite(code c)
{
    MAPL(ce, 
	 update_for_module_rewrite((entity*) &(CAR(ce).p)),
	 code_declarations(c));
}

static void update_loop_for_module_rewrite(loop l)
{
    update_for_module_rewrite(&loop_index(l));
}

void update_object_for_module(
    void * obj, /* loosely typed, indeed */
    entity module)
{
    bool (*saved_bound)(entity);
    entity (*saved_load)(entity);

    pips_debug(8, "updating (%s) %p\n", gen_domain_name(gen_type(obj)), obj);

    saved_bound = bound_p, saved_load = load; /* push the current functions */

    if (module==host_module)
    {
	pips_debug(8, "for host\n");
	bound_p = bound_new_host_p;
	load = load_new_host;
    }
    else
    {
	pips_debug(8, "for node\n");
	bound_p = bound_new_node_p;
	load = load_new_node;
    }

    gen_multi_recurse
	(obj, 
	 reference_domain, gen_true, update_reference_for_module_rewrite,
	 loop_domain, gen_true, update_loop_for_module_rewrite,
	 call_domain, gen_true, update_call_for_module_rewrite,
	 code_domain, gen_true, update_code_for_module_rewrite,
	 NULL);

    bound_p = saved_bound, load = saved_load; /* pop the initial functions */
}

void update_list_for_module(list l, entity module)
{
    MAPL(cx, update_object_for_module(CHUNK(CAR(cx)), module), l);
}

/* this function creates a new expression using the mapping of
 * old to new variables map.
 * some of the structures generated may be shared...
 */

expression UpdateExpressionForModule(module, ex)
entity module;
expression ex;
{
    expression new = copy_expression(ex);
    update_object_for_module(new, module);
    return(new);
}

/* used for compiling calls.
 */
list 
lUpdateExpr_but_distributed(
    entity module,
    list /* of expression */ l)
{
    list new = NIL;

    MAP(EXPRESSION, e,
    {
	if (!array_distributed_p(expression_to_entity(e)))
	    new = CONS(EXPRESSION, copy_expression(e), new);
    },
	l);

    new = gen_nreverse(new);
    update_list_for_module(new, module);    
    return new;
}

list 
lUpdateExpr(entity module, list /* of anything */ l)
{
    list new = gen_full_copy_list(l);
    update_list_for_module(new, module);    
    return new;
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

entity NewVariableForModule(
    entity module,
    entity e)
{
    if (module==host_module)
    {
	if (bound_new_host_p(e)) 
	    return load_new_host(e);
    }
    else
    {
	if (bound_new_node_p(e))
	    return load_new_node(e);
    }

    pips_internal_error("unexpected entity %s", entity_name(e));

    return entity_undefined;
}

statement UpdateStatementForModule(
    entity module,
    statement stat)
{
    statement new_stat = copy_statement(stat);
    update_object_for_module(new_stat, module);
    return(new_stat);
}

/*   That is all
 */
