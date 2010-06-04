/*
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

/**
 * @file insert_statement.c
 * check if a statement can be inserted without too much side effect
 * @author Serge Guelton <serge.guelton@enst-bretagne.fr>
 * @date 2010-06-01
 */
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif
#include <ctype.h>


#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "text.h"
#include "pipsdbm.h"
#include "resources.h"
#include "properties.h"
#include "misc.h"
#include "conversion.h"
#include "control.h"
#include "callgraph.h"
#include "effects-generic.h"
#include "effects-simple.h"
#include "effects-convex.h"
#include "preprocessor.h"
#include "text-util.h"
#include "transformations.h"
#include "parser_private.h"
#include "syntax.h"
#include "c_syntax.h"
#include "locality.h"
#include "expressions.h"
#include "semantics.h"
#include "transformer.h"
#include "accel-util.h"
/* generate a pcontrainte corresponding to dimensions, with a preset list of phis
 * ex: int a[10][n]; and (phi0,phi1)
 * will result in 0<=phi0<=9 and 0<=phi1<=n-1
 */
static Pcontrainte dimensions_to_psysteme(list dims,list phis)
{
    pips_assert("as many dimensions as phis",gen_length(dims) == gen_length(phis));
    Pvecteur vect_new = VECTEUR_NUL;
    FOREACH(DIMENSION,dim,dims)
    {
        entity phi = ENTITY(CAR(phis));
        expression lower = dimension_lower(dim);
        expression upper = dimension_upper(dim);
        NORMALIZE_EXPRESSION(lower);
        NORMALIZE_EXPRESSION(upper);

        if(normalized_linear_p(expression_normalized(lower)) && normalized_linear_p(expression_normalized(upper)))
        {

            Pvecteur vlower = vect_dup(normalized_linear(expression_normalized(lower)));
            Pvecteur vupper = vect_dup(normalized_linear(expression_normalized(upper)));
            vect_add_elem(&vlower,phi,1);
            vect_add_elem(&vupper,phi,1);
            vect_chg_sgn(vlower);
            for(Pvecteur iter=vlower;!VECTEUR_NUL_P(iter);iter=vecteur_succ(iter))
                vect_add_elem(&vect_new,vecteur_var(iter),vecteur_val(iter));
            for(Pvecteur iter=vupper;!VECTEUR_NUL_P(iter);iter=vecteur_succ(iter))
                vect_add_elem(&vect_new,vecteur_var(iter),vecteur_val(iter));
            vect_rm(vlower);
            vect_rm(vupper);
        }
        else {
            vect_rm(vect_new);
            return NULL;
        }
        POP(phis);
    }
    return contrainte_make(vect_new);
}

/* returns true if the statement writes or read an already defined area */
static bool statement_insertion_no_conflicting_access(list regions,enum action_utype tag)
{
    FOREACH(REGION,reg,regions)
    {
        reference r = region_any_reference(reg);
        entity e = reference_variable(r);
        Pcontrainte dims_sc = dimensions_to_psysteme(variable_dimensions(type_variable(entity_type(e))),reference_indices(r));
        Psysteme access_syst = region_system(reg);
        Psysteme inter_syst = sc_dup(access_syst);
        sc_add_inegalite(inter_syst,dims_sc);// there should not be any basis issue, they share the same ... */
        volatile bool feasible;
        CATCH(overflow_error)
        {	
            pips_debug(3, "overflow error \n");
            feasible = true;
        }
        TRY
        {    
            feasible = sc_integer_feasibility_ofl_ctrl(inter_syst,FWD_OFL_CTRL, true);
            UNCATCH(overflow_error);

            contrainte_rm(dims_sc);
            if(feasible && (tag == is_action_write) ) {
                sc_free(inter_syst);
                pips_user_warning("inserted statement modifies the store");
                return false;
            }
            else {
                /*no interference with the store, go on and update dimensions to avoid out of bound exceptions*/
                volatile Psysteme stmp ,sr;
                CATCH(overflow_error)
                {
                    pips_debug(1, "overflow error\n");
                    return false;
                }
                TRY
                {
                    stmp = sc_make(contrainte_new(),sc_inegalites(inter_syst));
                    sr = sc_cute_convex_hull(access_syst, stmp);
                    sc_rm(stmp);
                    sc_nredund(&sr);
                    UNCATCH(overflow_error);
                }
                /* if we reach this point, we are ready for backward translation from vecteur to dimensions :) */
                list new_dimensions = NIL;
                FOREACH(EXPRESSION,phi,reference_indices(r))
                {
                    Pcontrainte lower,upper;
                    constraints_for_bounds(phi, &sc_inegalites(sr), &lower, &upper);
                    if( !CONTRAINTE_UNDEFINED_P(lower) && !CONTRAINTE_UNDEFINED_P(upper))
                    {
                        expression elower = constraints_to_loop_bound(lower,phi,true,entity_intrinsic(DIVIDE_OPERATOR_NAME));
                        expression eupper = constraints_to_loop_bound(upper,phi,false,entity_intrinsic(DIVIDE_OPERATOR_NAME));
                        new_dimensions=CONS(DIMENSION, make_dimension(elower,eupper),new_dimensions);
                    }
                    else {
                        pips_user_warning("failed to translate region\n");
                        return false;
                    }
                }
                new_dimensions=gen_nreverse(new_dimensions);
                gen_full_free_list(variable_dimensions(type_variable(entity_type(e))));
                variable_dimensions(type_variable(entity_type(e)))=new_dimensions;
            }

        }
    }
        return true;
}

static bool do_statement_insertion(statement s)
{
    /* first find a statement with the relevant pragma */
    string inserted_pragma = get_string_property("STATEMENT_INSERTION_PRAGMA");
    if(empty_string_p(inserted_pragma)) {
        pips_user_warning("STATEMENT_INSERTION_PRAGMA property should not be empty\n");
    }
    else {
        list flagged_statements = find_statements_with_pragma(s,inserted_pragma);
        if(ENDP(flagged_statements)) {
            pips_user_warning("no statement with pragma '%s' found\n",inserted_pragma);
        }
        else {
            FOREACH(STATEMENT,flagged_statement,flagged_statements) {
                list regions = load_cumulated_rw_effects_list(flagged_statement);
                list read_regions = regions_read_regions(regions);
                list write_regions = regions_write_regions(regions);

                statement_insertion_no_conflicting_access(write_regions,is_action_write);
                statement_insertion_no_conflicting_access(read_regions,is_action_read);
            }
        }

    }
        return false;
}

bool statement_insertion(const char *module_name)
{
    /* init */
    set_current_module_entity(module_name_to_entity( module_name ));
    set_current_module_statement((statement) db_get_memory_resource(DBR_CODE, module_name, true) );
    set_cumulated_rw_effects((statement_effects)db_get_memory_resource(DBR_REGIONS, module_name, true));
    /* do */
    if(do_statement_insertion(get_current_module_statement()))
    {
        /* validate */
        module_reorder(get_current_module_statement());
        DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name,get_current_module_statement());
    }

    reset_current_module_entity();
    reset_current_module_statement();
    reset_cumulated_rw_effects();
    return true;
}
