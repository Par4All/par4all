/*
 * $Id$
 *
 * $Log: initial.c,v $
 * Revision 1.14  1998/05/28 19:05:47  irigoin
 * Filtering by effects added to avoid spurious variables as in Spec/fppp.f
 * and data02.f
 *
 * Revision 1.13  1998/04/14 21:25:21  coelho
 * linear.h
 *
 * Revision 1.12  1997/12/10 14:29:36  coelho
 * free text.
 *
 * Revision 1.11  1997/09/30 06:57:15  coelho
 * gen_array_* used...
 *
 * Revision 1.10  1997/09/11 13:42:10  coelho
 * check consistency...
 *
 * Revision 1.9  1997/09/11 12:34:16  coelho
 * duplicates instead of relying on pipsmake/pipsdbm...
 *
 * Revision 1.8  1997/09/11 12:00:10  coelho
 * none.
 *
 * Revision 1.7  1997/09/10 09:43:30  irigoin
 * Explicit free_value_mappings() added. Context set for
 * program_precondition() and the prettyprinter modules.
 *
 * Revision 1.6  1997/09/09 11:03:15  coelho
 * initial preconditions should be ok.
 *
 * Revision 1.5  1997/09/08 17:52:05  coelho
 * some version for testing.
 *
 * Revision 1.3  1997/09/08 09:35:29  coelho
 * transformer -> precondition.
 *
 * Revision 1.2  1997/09/08 08:51:14  coelho
 * the code is not printed. name fixed.
 *
 * Revision 1.1  1997/09/08 08:45:50  coelho
 * Initial revision
 *
 *
 * Computation of initial transformers that allow to collect
 * global initializations of BLOCK DATA and so.
 *
 * This computation is not safe because pipsmake does not provide a real
 * link-edit. The set of modules called $ALL may contain modules which
 * are never called by the main.
 *
 * The initial precondition of each module must be filtered with respect
 * to the effects of the MAIN in order to eliminate information about
 * unused variables.
 *
 * The filtering cannot be performed at the module level because some
 * modules such as BLOCKDATA do not have any proper effects and because
 * a module can initialize a variable in a common without using it.
 *
 */

#include <stdio.h>
#include <string.h>

#include "genC.h"

#include "linear.h"
#include "ri.h"
#include "ri-util.h"

#include "database.h"
#include "pipsdbm.h"
#include "resources.h"

#include "misc.h"

#include "effects-generic.h"
#include "effects-simple.h"

#include "prettyprint.h"
#include "transformer.h"

#include "semantics.h"

/******************************************************************** UTILS */

static entity
get_main_entity(void)
{
    entity m;
    gen_array_t modules = db_get_module_list();
    int nmodules = gen_array_nitems(modules), i;
    pips_assert("some modules in the program", nmodules>0);

    for (i=0; i<nmodules; i++)
    {
	m = local_name_to_top_level_entity(gen_array_item(modules, i));
	if (entity_main_module_p(m)) {
	    gen_array_full_free(modules);
	    return m;
	}
    }

    /* ??? some default if there is no main... */
    pips_user_warning("no main found, returning %s instead\n", 
		      gen_array_item(modules,0));
    m = local_name_to_top_level_entity(gen_array_item(modules, 0));
    gen_array_full_free(modules);
    return m;
}

/******************************************************** PIPSMAKE INTERFACE */

/* Compute an initial transformer.
 */
bool 
initial_precondition(string name)
{
    entity module = local_name_to_top_level_entity(name);
    transformer prec;

    debug_on("SEMANTICS_DEBUG_LEVEL");

    set_current_module_entity(module);
    set_current_module_statement( (statement) 
	db_get_memory_resource(DBR_CODE, name, TRUE));
    set_cumulated_rw_effects((statement_effects) 
		   db_get_memory_resource(DBR_CUMULATED_EFFECTS, name, TRUE));
    module_to_value_mappings(module);

    prec = all_data_to_precondition(module);

    ifdebug(1) 
	pips_assert("consistent initial precondition before filtering", 
		    transformer_consistency_p(prec));

    /* Filter out local and unused variables from the local precondition?
     * No, because it is not possible to guess what is used or unused at
     * the program level. Filtering is postponed to program_precondition().
     */

    DB_PUT_MEMORY_RESOURCE(DBR_INITIAL_PRECONDITION, strdup(name), (char*) prec);

    reset_current_module_entity();
    reset_current_module_statement();
    reset_cumulated_rw_effects();

    free_value_mappings();

    debug_off();
    return TRUE;
}

/* returns t1 inter= t2;
 */
static void
intersect(
    transformer t1,
    transformer t2)
{
    predicate_system_(transformer_relation(t1)) = (char*)
	sc_append((Psysteme) predicate_system(transformer_relation(t1)),
		  (Psysteme) predicate_system(transformer_relation(t2)));
}

#define pred_debug(level, msg, trans) \
  ifdebug(level) { pips_debug(level, msg); dump_transformer(trans);}

/* Compute the union of all initial preconditions.
 */
bool
program_precondition(string name)
{
    transformer t = transformer_identity();
    entity the_main = get_main_entity();
    int i, nmodules;
    gen_array_t modules;
    list e_inter = NIL;

    pips_assert("main was found", the_main!=entity_undefined);

    debug_on("SEMANTICS_DEBUG_LEVEL");
    pips_debug(1, "considering program \"%s\" with main \"%s\"\n", name,
	       module_local_name(the_main));

    set_current_module_entity(the_main);
    set_current_module_statement( (statement) 
				  db_get_memory_resource(DBR_CODE,
							 module_local_name(the_main),
							 TRUE));
    set_cumulated_rw_effects((statement_effects) 
			     db_get_memory_resource(DBR_CUMULATED_EFFECTS,
						    module_local_name(the_main),
						    TRUE));

    /* e_inter = effects_to_list(get_cumulated_rw_effects(get_current_module_statement())); */

    e_inter = effects_to_list( (effects)
			       db_get_memory_resource(DBR_SUMMARY_EFFECTS,
						      module_local_name(the_main),
						      TRUE));

    module_to_value_mappings(the_main);
    
    /* Unavoidable pitfall: initializations in uncalled modules may be
     * taken into account. It all depends on the "create" command.
     */
    modules = db_get_module_list();
    nmodules = gen_array_nitems(modules);
    pips_assert("some modules in the program", nmodules>0);

    for(i=0; i<nmodules; i++) 
    {
	transformer tm;
	string mname = gen_array_item(modules, i);
	pips_debug(1, "considering module %s\n", mname);
	
	tm = transformer_dup((transformer) /* no dup & FALSE => core */
			     db_get_memory_resource(DBR_INITIAL_PRECONDITION,  
						    mname, TRUE));

	pred_debug(3, "current: t =\n", t);
	pred_debug(2, "to be added: tm =\n", tm);
	translate_global_values(the_main, tm); /* modifies tm! */
	pred_debug(3, "to be added after translation:\n", tm);
	tm = transformer_intra_to_inter(tm, e_inter);
	/* tm = transformer_normalize(tm, 2); */
	if(!transformer_consistency_p(tm)) {
	    (void) print_transformer(tm);
	    pips_error("program_precondition",
		       "Non-consistent filtered initial precondition\n");
	}
	pred_debug(3, "to be added after filtering:\n", tm);

	intersect(t, tm); 
	free_transformer(tm);
    }

    pred_debug(1, "resulting program precondition:\n", t);

    ifdebug(1) 
	pips_assert("consistent program precondition", 
		    transformer_consistency_p(t));

    DB_PUT_MEMORY_RESOURCE(DBR_PROGRAM_PRECONDITION, strdup(name), t);

    reset_current_module_entity();
    reset_current_module_statement();
    reset_cumulated_rw_effects();

    free_value_mappings();
    gen_array_full_free(modules);

    debug_off();
    return TRUE;
}


/*********************************************************** PRETTY PRINTERS */

bool 
print_initial_precondition(string name)
{
    bool ok;
    entity module = local_name_to_top_level_entity(name);
    transformer t = (transformer) 
	db_get_memory_resource(DBR_INITIAL_PRECONDITION, name, TRUE);
    
    debug_on("SEMANTICS_DEBUG_LEVEL");

    set_current_module_entity(module);
    set_current_module_statement( (statement) 
	db_get_memory_resource(DBR_CODE, name, TRUE));
    set_cumulated_rw_effects((statement_effects) 
		   db_get_memory_resource(DBR_CUMULATED_EFFECTS,
					  name,
					  TRUE));
    module_to_value_mappings(module);
 
    ok = make_text_resource_and_free(name, DBR_PRINTED_FILE, ".ipred",
				     text_transformer(t));

    reset_current_module_entity();
    reset_current_module_statement();
    reset_cumulated_rw_effects();

    free_value_mappings();

    debug_off();

    return ok;
}

bool 
print_program_precondition(string name)
{
    bool ok;
    transformer t = (transformer) 
	db_get_memory_resource(DBR_PROGRAM_PRECONDITION, "", TRUE);
    entity m = get_main_entity();
    
    debug_on("SEMANTICS_DEBUG_LEVEL");
    pips_debug(1, "for \"%s\"\n", name);

    set_current_module_entity(m);
    set_current_module_statement( (statement) 
	db_get_memory_resource(DBR_CODE,
			       module_local_name(m),
			       TRUE));
    set_cumulated_rw_effects((statement_effects) 
		   db_get_memory_resource(DBR_CUMULATED_EFFECTS,
					  module_local_name(m),
					  TRUE));
    module_to_value_mappings(m);
 
    ok = make_text_resource_and_free(name, DBR_PRINTED_FILE, ".pipred",
				     text_transformer(t));

    reset_current_module_entity();
    reset_current_module_statement();
    reset_cumulated_rw_effects();

    free_value_mappings();

    debug_off();

    return ok;
}

