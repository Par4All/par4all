/*
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

/**
 * @file kernels.c
 * kernels manipulation
 * @author Serge Guelton <serge.guelton@enst-bretagne.fr>
 * @date 2010-01-03
 */
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif
#include <ctype.h>


#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "ri-util.h"
#include "effects-util.h"
#include "text.h"
#include "pipsdbm.h"
#include "resources.h"
#include "properties.h"
#include "misc.h"
#include "control.h"
#include "callgraph.h"
#include "effects-generic.h"
#include "effects-simple.h"
#include "effects-convex.h"
#include "text-util.h"
#include "transformations.h"
#include "transformer.h"
#include "semantics.h"
#include "parser_private.h"
#include "accel-util.h"


/* Generate a communication around a statement instead of plain memory
   access if it is a call to a function named module_name

   @param[in,out] s is the statment to isolate with some communication arounds

   @param[in] module_name is the name of the function to isolate
*/
static void
kernel_load_store_generator(statement s, const char* module_name)
{
    if(statement_call_p(s))
    {
        call c = statement_call(s);
	/* Generate communication operations around the call to a function
	   named "module_name". */
        if(same_string_p(module_local_name(call_function(c)),module_name))
        {
	  const char* prefix =  get_string_property ("KERNEL_LOAD_STORE_VAR_PREFIX");
	  const char* suffix =  get_string_property ("KERNEL_LOAD_STORE_VAR_SUFFIX");
	  pips_debug (5, "kernel_load_store used prefix : %s\n", prefix);
	  pips_debug (5, "kernel_load_store used suffix : %s\n", suffix);
	  do_isolate_statement(s, prefix, suffix);
        }
    }
}

/* run kernel load store using either region or effect engine

   @param[in] module_name is the name of the function we want to isolate
   with communications and memory allocations

   @param[in] enginerc is the name of the resources to use to analyse
   which data need to be allocated and transfers. It can be DBR_REGIONS
   (more precise) or DBR_EFFECTS
 */
static bool kernel_load_store_engine(const char* module_name,
				     const string enginerc) {
    /* generate a load stores on each caller */

    debug_on("KERNEL_LOAD_STORE_DEBUG_LEVEL");

    callees callers = (callees)db_get_memory_resource(DBR_CALLERS,module_name,true);
    FOREACH(STRING,caller_name,callees_callees(callers)) {
        /* prelude */
        set_current_module_entity(module_name_to_entity( caller_name ));
        set_current_module_statement((statement) db_get_memory_resource(DBR_CODE, caller_name, true) );
        set_cumulated_rw_effects((statement_effects)db_get_memory_resource(enginerc, caller_name, true));
        module_to_value_mappings(get_current_module_entity());
        set_precondition_map( (statement_mapping) db_get_memory_resource(DBR_PRECONDITIONS, caller_name, true) );
        /*do the job */
        gen_context_recurse(get_current_module_statement(),(char*)module_name,statement_domain,gen_true,kernel_load_store_generator);
        /* validate */
        module_reorder(get_current_module_statement());
        DB_PUT_MEMORY_RESOURCE(DBR_CODE, caller_name,get_current_module_statement());
        DB_PUT_MEMORY_RESOURCE(DBR_CALLEES, caller_name, compute_callees(get_current_module_statement()));

        /*postlude*/
        reset_precondition_map();
        free_value_mappings();
        reset_cumulated_rw_effects();
        reset_current_module_entity();
        reset_current_module_statement();
    }

    /*flag the module as kernel if not done */
    callees kernels = (callees)db_get_memory_resource(DBR_KERNELS,"",true);
    bool found = false;
    FOREACH(STRING,kernel_name,callees_callees(kernels))
        if( (found=(same_string_p(kernel_name,module_name))) ) break;
    if(!found)
        callees_callees(kernels)=CONS(STRING,strdup(module_name),callees_callees(kernels));
    db_put_or_update_memory_resource(DBR_KERNELS,"",kernels,true);

    debug_off();

    return true;
}


/** Generate malloc/copy-in/copy-out on the call sites of this module.
  * based on convex array regions
  */
bool kernel_load_store(const char* module_name) {
    return kernel_load_store_engine(module_name, DBR_REGIONS);
}



/**
 * create a statement eligible for outlining into a kernel
 * #1 find the loop flagged with loop_label
 * #2 make sure the loop is // with local index
 * #3 perform strip mining on this loop to make the kernel appear
 * #4 perform two outlining to separate kernel from host
 *
 * @param s statement where the kernel can be found
 * @param loop_label label of the loop to be turned into a kernel
 *
 * @return true as long as the kernel is not found
 */
static
bool do_kernelize(statement s, entity loop_label)
{
    if( same_entity_p(statement_label(s),loop_label) ||
            (statement_loop_p(s) && same_entity_p(loop_label(statement_loop(s)),loop_label)))
    {
        if( !instruction_loop_p(statement_instruction(s)) )
            pips_user_error("you choosed a label of a non-doloop statement\n");



        loop l = instruction_loop(statement_instruction(s));

        /* gather and check parameters */
        int nb_nodes = get_int_property("KERNELIZE_NBNODES");
        while(!nb_nodes)
        {
            string ur = user_request("number of nodes for your kernel?\n");
            nb_nodes=atoi(ur);
        }

        /* verify the loop is parallel */
        if( execution_sequential_p(loop_execution(l)) )
            pips_user_error("you tried to kernelize a sequential loop\n");
        if( !entity_is_argument_p(loop_index(statement_loop(s)),loop_locals(statement_loop(s))) )
            pips_user_error("you tried to kernelize a loop whose index is not private\n");

        if(nb_nodes >1 )
        {
            /* we can strip mine the loop */
            loop_strip_mine(s,nb_nodes,-1);
            /* unfortunately, the strip mining does not exactly does what we
               want, fix it here

               it is legal because we know the loop index is private,
               otherwise the end value of the loop index may be used
               incorrectly...
               */
            {
                statement s2 = loop_body(statement_loop(s));
                entity outer_index = loop_index(statement_loop(s));
                entity inner_index = loop_index(statement_loop(s2));
                replace_entity(s2,inner_index,outer_index);
                loop_index(statement_loop(s2))=outer_index;
                replace_entity(loop_range(statement_loop(s2)),outer_index,inner_index);
                if(!ENDP(loop_locals(statement_loop(s2)))) replace_entity(loop_locals(statement_loop(s2)),outer_index,inner_index);
                loop_index(statement_loop(s))=inner_index;
                replace_entity(loop_range(statement_loop(s)),outer_index,inner_index);
                RemoveLocalEntityFromDeclarations(outer_index,get_current_module_entity(),get_current_module_statement());
                loop_body(statement_loop(s))=make_block_statement(make_statement_list(s2));
                AddLocalEntityToDeclarations(outer_index,get_current_module_entity(),loop_body(statement_loop(s)));
                l = statement_loop(s);
            }
        }

        const char* kernel_name=get_string_property_or_ask("KERNELIZE_KERNEL_NAME","name of the kernel ?");
        const char* host_call_name=get_string_property_or_ask("KERNELIZE_HOST_CALL_NAME","name of the fucntion to call the kernel ?");

        /* validate changes */
        callees kernels=(callees)db_get_memory_resource(DBR_KERNELS,"",true);
        callees_callees(kernels)= CONS(STRING,strdup(host_call_name),callees_callees(kernels));
        DB_PUT_MEMORY_RESOURCE(DBR_KERNELS,"",kernels);

        entity cme = get_current_module_entity();
        statement cms = get_current_module_statement();
        module_reorder(get_current_module_statement());
        DB_PUT_MEMORY_RESOURCE(DBR_CODE, get_current_module_name(),get_current_module_statement());
        DB_PUT_MEMORY_RESOURCE(DBR_CALLEES, get_current_module_name(), compute_callees(get_current_module_statement()));
        reset_current_module_entity();
        reset_current_module_statement();

        /* recompute effects */
        proper_effects(module_local_name(cme));
        cumulated_effects(module_local_name(cme));
        set_current_module_entity(cme);
        set_current_module_statement(cms);
        set_cumulated_rw_effects((statement_effects)db_get_memory_resource(DBR_CUMULATED_EFFECTS, get_current_module_name(), true));

        /* outline the work and kernel parts*/
        outliner(kernel_name,make_statement_list(loop_body(l)));
        (void)outliner(host_call_name,make_statement_list(s));
        reset_cumulated_rw_effects();

        /* job done */
        gen_recurse_stop(NULL);

    }
    return true;
}


/**
 * turn a loop flagged with LOOP_LABEL into a kernel (GPU, terapix ...)
 *
 * @param module_name name of the module
 *
 * @return true
 */
bool kernelize(char * module_name)
{
  /* prelude */
  set_current_module_entity(module_name_to_entity( module_name ));
  set_current_module_statement((statement) db_get_memory_resource(DBR_CODE, module_name, true) );

  /* retreive loop label */
  const char* loop_label_name = get_string_property_or_ask("LOOP_LABEL","label of the loop to turn into a kernel ?\n");
  entity loop_label_entity = find_label_entity(module_name,loop_label_name);
  if( entity_undefined_p(loop_label_entity) )
    pips_user_error("label '%s' not found in module '%s' \n",loop_label_name,module_name);


  /* run kernelize */
  gen_context_recurse(get_current_module_statement(),loop_label_entity,statement_domain,do_kernelize,gen_null);

  /* validate */
  module_reorder(get_current_module_statement());
  DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name,get_current_module_statement());
  DB_PUT_MEMORY_RESOURCE(DBR_CALLEES, module_name, compute_callees(get_current_module_statement()));

  /*postlude*/
  reset_current_module_entity();
  reset_current_module_statement();
  return true;
}

bool flag_kernel(char * module_name)
{
  if (!db_resource_p(DBR_KERNELS, ""))
    pips_internal_error("kernels not initialized");
  callees kernels=(callees)db_get_memory_resource(DBR_KERNELS,"",true);
  callees_callees(kernels)= CONS(STRING,strdup(module_name),callees_callees(kernels));
  DB_PUT_MEMORY_RESOURCE(DBR_KERNELS,"",kernels);
  return true;
}

bool bootstrap_kernels(__attribute__((unused)) char * module_name)
{
  if (db_resource_p(DBR_KERNELS, ""))
    pips_internal_error("kernels already initialized");
  callees kernels=make_callees(NIL);
  DB_PUT_MEMORY_RESOURCE(DBR_KERNELS,"",kernels);
  return true;
}


