/*

  $Id$

  Copyright 1989-2010 MINES ParisTech

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
/*
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
#include "effects.h"
#include "ri-util.h"
#include "effects-util.h"

#include "database.h"
#include "pipsdbm.h"
#include "resources.h"

#include "misc.h"

#include "effects-generic.h"
#include "effects-simple.h"

#include "prettyprint.h"
#include "transformer.h"

#include "semantics.h"


/******************************************************** PIPSMAKE INTERFACE */

/* Compute an initial transformer.
 */
bool initial_precondition(string name)
{
    entity module = module_name_to_entity(name);
    transformer prec;

    debug_on("SEMANTICS_DEBUG_LEVEL");

    set_current_module_entity(module);
    set_current_module_statement( (statement)
	db_get_memory_resource(DBR_CODE, name, true));
    set_cumulated_rw_effects((statement_effects)
		   db_get_memory_resource(DBR_CUMULATED_EFFECTS, name, true));
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
    return true;
}

/* returns t1 inter= t2;
 */
static void
intersect(
    transformer t1,
    transformer t2)
{
    predicate_system(transformer_relation(t1)) =
	sc_append((Psysteme) predicate_system(transformer_relation(t1)),
		  (Psysteme) predicate_system(transformer_relation(t2)));
}

#define pred_debug(level, msg, trans) \
  ifdebug(level) { pips_debug(level, msg); dump_transformer(trans);}

/* Compute the union of all initial preconditions.
 */
bool program_precondition(string name)
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
							 true));
    set_cumulated_rw_effects((statement_effects)
			     db_get_memory_resource(DBR_CUMULATED_EFFECTS,
						    module_local_name(the_main),
						    true));

    /* e_inter = effects_to_list(get_cumulated_rw_effects(get_current_module_statement())); */

    e_inter = effects_to_list( (effects)
			       db_get_memory_resource(DBR_SUMMARY_EFFECTS,
						      module_local_name(the_main),
						      true));

    module_to_value_mappings(the_main);

    /* Unavoidable pitfall: initializations in uncalled modules may be
     * taken into account. It all depends on the "create" command.
     */
    modules = db_get_module_list();
    nmodules = gen_array_nitems(modules);
    pips_assert("some modules in the program", nmodules>0);

    for(i=0; i<nmodules; i++) {
	transformer tm;
	string mname = gen_array_item(modules, i);
	pips_debug(1, "considering module %s\n", mname);

	tm = transformer_dup((transformer) /* no dup & false => core */
			     db_get_memory_resource(DBR_INITIAL_PRECONDITION,
						    mname, true));

	pred_debug(3, "current: t =\n", t);
	pred_debug(2, "to be added: tm =\n", tm);
	translate_global_values(the_main, tm); /* modifies tm! */
	pred_debug(3, "to be added after translation:\n", tm);
	tm = transformer_intra_to_inter(tm, e_inter);
	/* tm = transformer_normalize(tm, 2); */
	if(!transformer_consistency_p(tm)) {
	    (void) print_transformer(tm);
	    pips_internal_error("Non-consistent filtered initial precondition");
	}
	pred_debug(3, "to be added after filtering:\n", tm);

	intersect(t, tm);
	free_transformer(tm);
    }

    pred_debug(1, "resulting program precondition:\n", t);

    ifdebug(1)
	pips_assert("consistent program precondition",
		    transformer_consistency_p(t));

    DB_PUT_MEMORY_RESOURCE(DBR_PROGRAM_PRECONDITION, "", t);

    reset_current_module_entity();
    reset_current_module_statement();
    reset_cumulated_rw_effects();

    free_value_mappings();
    gen_array_full_free(modules);

    debug_off();
    return true;
}

/* The program correctness postcondition cannot be infered. It should be
   provided by the user. */
bool
program_postcondition(string name)
{
  transformer post = transformer_identity();
  entity the_main = get_main_entity();

  debug_on("SEMANTICS_DEBUG_LEVEL");
  pips_debug(1, "considering program \"%s\" with main \"%s\"\n", name,
	     module_local_name(the_main));

  pred_debug(1, "assumed program postcondition:\n", post);

  ifdebug(1) 
    pips_assert("consistent program postcondition", 
		transformer_consistency_p(post));

  DB_PUT_MEMORY_RESOURCE(DBR_PROGRAM_POSTCONDITION, "", post);

  debug_off();
  return true;
}


/*********************************************************** PRETTY PRINTERS */

bool 
print_initial_precondition(string name)
{
    bool ok;
    entity module = module_name_to_entity(name);
    transformer t = (transformer) 
	db_get_memory_resource(DBR_INITIAL_PRECONDITION, name, true);
    
    debug_on("SEMANTICS_DEBUG_LEVEL");

    set_current_module_entity(module);
    set_current_module_statement( (statement) 
	db_get_memory_resource(DBR_CODE, name, true));
    set_cumulated_rw_effects((statement_effects) 
		   db_get_memory_resource(DBR_CUMULATED_EFFECTS,
					  name,
					  true));
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
	db_get_memory_resource(DBR_PROGRAM_PRECONDITION, "", true);
    entity m = get_main_entity();
    
    debug_on("SEMANTICS_DEBUG_LEVEL");
    pips_debug(1, "for \"%s\"\n", name);

    set_current_module_entity(m);
    set_current_module_statement( (statement) 
	db_get_memory_resource(DBR_CODE,
			       module_local_name(m),
			       true));
    set_cumulated_rw_effects((statement_effects) 
		   db_get_memory_resource(DBR_CUMULATED_EFFECTS,
					  module_local_name(m),
					  true));
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

