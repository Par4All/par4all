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

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"

#include "resources.h"

#include "misc.h"
#include "ri-util.h"
#include "effects-util.h"
#include "pipsdbm.h"
#include "reductions.h"
#include "expressions.h"

#include "dg.h"

typedef dg_arc_label arc_label;
typedef dg_vertex_label vertex_label;

#include "graph.h"
#include "ricedg.h"


#include "control.h"


/* sg: this function does a huge pattern matching :) 
 * and is not very smart 
 */
static void do_reduction_propagation(graph dg) {
    set seen = set_make(set_pointer);
    FOREACH(VERTEX, v,graph_vertices(dg))
    {
        statement s = vertex_to_statement(v);
        if(!set_belong_p(seen,s)) {
            set_add_element(seen,seen,s);
            /* lazy ... s = f(sigma) */
            if(assignment_statement_p(s)) {
                expression assigned_exp = binary_call_lhs(statement_call(s));
                /* lazier */
                if(expression_reference_p(assigned_exp)) {
                    expression rhs = binary_call_rhs(statement_call(s));
                    if(expression_call_p(rhs)) {
                        reference assigned_ref = expression_reference(assigned_exp);
                        if(reference_scalar_p(assigned_ref)) {
                            call rhs_call = expression_call(rhs);
                            entity rhs_op = call_function(rhs_call);

                            FOREACH(SUCCESSOR,su,vertex_successors(v)) {
                                vertex sv = successor_vertex(su);
                                statement ssv = vertex_to_statement(sv);
                                /* even lazier */
                                if(assignment_statement_p(ssv)) {
                                    list reductions = reductions_list(load_proper_reductions(ssv));
                                    /* we got some simple reductions here, try to propagate them backward */
                                    if(gen_length(reductions) == 1 ) {
                                        reduction red = REDUCTION(CAR(reductions));
                                        if(same_entity_p(reduction_operator_entity(reduction_op(red)),rhs_op)) {
                                            FOREACH(CONFLICT,con,dg_arc_label_conflicts(successor_arc_label(su))) {
                                                if(effect_read_p(conflict_sink(con)) &&
                                                        effect_write_p(conflict_source(con)) &&
                                                        reference_equal_p(effect_any_reference(conflict_source(con)),assigned_ref)) {
                                                    expression rhs_fst_arg =EXPRESSION(CAR(call_arguments(rhs_call)));
                                                    if(expression_scalar_p(rhs_fst_arg)) {
                                                        reference rhs_fst_ref = copy_reference(expression_reference(rhs_fst_arg));
                                                        expression red_exp = reference_to_expression(reduction_reference(red));
                                                        replace_entity_by_expression(s,reference_variable(rhs_fst_ref),red_exp);
                                                        replace_reference(ssv,assigned_ref,reference_variable(rhs_fst_ref));
                                                        replace_entity_by_expression(s,reference_variable(assigned_ref),red_exp);
                                                        free_reference(rhs_fst_ref);
                                                        syntax_reference(expression_syntax(red_exp))=reference_undefined;
                                                        free_expression(red_exp);
                                                    }
                                                    else pips_user_warning("replacement of non reference expression not implemented yet \n");
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    set_free(seen);
}
/* now try to backward propagate the reduction, if it is meaningful.
 * the pattern checked is
 * a = b + c;
 * r = r + a;
 * which should become
 * r = r +b ;
 * r = r +c ;
 */

bool reduction_propagation(const char * mod_name) {
    /* get the resources */
    statement mod_stmt = (statement)
        db_get_memory_resource(DBR_CODE, mod_name, TRUE);

    set_current_module_statement(mod_stmt); 
    set_proper_reductions((pstatement_reductions) db_get_memory_resource(DBR_PROPER_REDUCTIONS, mod_name, true));
    set_current_module_entity(module_name_to_entity(mod_name));
	set_ordering_to_statement(mod_stmt);
    graph dependence_graph = 
        (graph) db_get_memory_resource(DBR_DG, mod_name, TRUE);

    simplify_c_operator(get_current_module_statement());

    /* do the job */
    do_reduction_propagation(dependence_graph);

    /* validate computation */
    module_reorder(mod_stmt);
    DB_PUT_MEMORY_RESOURCE(DBR_CODE, mod_name, mod_stmt);

    /* update/release resources */
    reset_proper_reductions();
	reset_ordering_to_statement();
    reset_current_module_statement();
    reset_current_module_entity();
    return true;
}
