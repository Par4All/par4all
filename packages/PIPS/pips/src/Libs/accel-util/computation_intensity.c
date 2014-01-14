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

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdlib.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "complexity_ri.h"
#include "text.h"

#include "ri-util.h"
#include "effects-util.h"
#include "effects-generic.h"
#include "effects-convex.h"
#include "text-util.h"
#include "database.h"
#include "pipsdbm.h"
#include "resources.h"

#include "properties.h"

#include "complexity.h"
#include "accel-util.h"

typedef struct {
    int startup_overhead;
    int bandwidth;
    int frequency;
    const char *pragma;
} computation_intensity_param;

/* read properties to initialize cost model */
static void init_computation_intensity_param(computation_intensity_param* p) {
    p->startup_overhead=get_int_property("COMPUTATION_INTENSITY_STARTUP_OVERHEAD");
    p->bandwidth=get_int_property("COMPUTATION_INTENSITY_BANDWIDTH");
    p->frequency=get_int_property("COMPUTATION_INTENSITY_FREQUENCY");
    p->pragma=get_string_property("COMPUTATION_INTENSITY_PRAGMA");
}

/* a loop statement is considered as compute intensive if
 * transfer costs a re greater than execution cost
 */
static bool do_computation_intensity(statement s, computation_intensity_param* p) {
    if(statement_loop_p(s)) {
        list regions = load_cumulated_rw_effects_list(s);
        complexity comp = load_statement_complexity(s);
        /* compute instruction contribution to execution time */
        Ppolynome instruction_time = polynome_dup(complexity_polynome(comp));
        polynome_scalar_mult(&instruction_time,1.f/p->frequency);

        /* compute transfer contribution to execution time */
        Ppolynome transfer_time = POLYNOME_NUL;
        FOREACH(REGION,reg,regions) {
            Ppolynome reg_footprint= region_enumerate(reg); // may be we should use the rectangular hull ?
            polynome_add(&transfer_time,reg_footprint);
            polynome_rm(&reg_footprint);
        }
        polynome_scalar_mult(&transfer_time,1.f/p->bandwidth);
        polynome_scalar_add(&transfer_time,p->startup_overhead);

        /* now let's compare them, using their difference */
        polynome_negate(&transfer_time);
        polynome_add(&instruction_time,transfer_time);
        polynome_rm(&transfer_time);
        /* heuristic to check the behavior at the infinite:
         * assumes all variables are positives,
         * take the higher degree monomial
         * and decide upon its coefficient
         */
        int max_degree = polynome_max_degree(instruction_time);
        float coeff=-1.f;
        for(Ppolynome p = instruction_time; !POLYNOME_NUL_P(p); p = polynome_succ(p)) {
            int curr_degree =  (int)vect_sum(monome_term(polynome_monome(p)));
            if(curr_degree == max_degree) {
                coeff = monome_coeff(polynome_monome(p));
                break;
            }
        }
        polynome_rm(&instruction_time);
        /* seems a computation intensive loop ! */
        if(coeff> 0) {
            add_pragma_str_to_statement(s,p->pragma,true);
            return false;
        }
    }
    return true;
}

/* mark all loops (do / for /while) that have sufficient computation intensity
 * with a pragma, suitable for treatment by other phases.
 * The computation intensity is derived from the complexity and the memory footprint.
 * It assumes the cost model: execution_time = startup_overhead + memory_footprint / bandwidth + complexity / frequency
 */
bool computation_intensity(const char *module_name) {
    /* init stuff */
    set_current_module_entity(module_name_to_entity( module_name ));
    set_current_module_statement((statement) db_get_memory_resource(DBR_CODE, module_name, true) );
    set_complexity_map( (statement_mapping) db_get_memory_resource(DBR_COMPLEXITIES, module_name, true));
    set_cumulated_rw_effects((statement_effects)db_get_memory_resource(DBR_REGIONS, module_name, true));

    computation_intensity_param p;
    init_computation_intensity_param(&p);

    /* do it now ! */
    gen_context_recurse(get_current_module_statement(),&p,
            statement_domain,do_computation_intensity,gen_null);

    /* validate changes */
    DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name,get_current_module_statement());

    /* reset */
    reset_current_module_entity();
    reset_current_module_statement();
    reset_complexity_map();
    reset_cumulated_rw_effects();

    return true;
}
