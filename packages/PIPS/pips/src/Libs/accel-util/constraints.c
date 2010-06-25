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
 * @file constraintes.c
 * solve constraints equation for restricted hardware
 * @author Serge Guelton <serge.guelton@enst-bretagne.fr>
 * @date 2010-05-01
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
#include "conversion.h"
#include "control.h"
#include "effects-generic.h"
#include "effects-simple.h"
#include "effects-convex.h"
#include "text-util.h"
#include "parser_private.h"
#include "semantics.h"
#include "transformer.h"
#include "accel-util.h"


/* only works for hypercube */
static
expression region_enumerate(region r,transformer tr)
{
    reference ref = region_any_reference(r);
    Psysteme sc = sc_dup(region_system(r));
    sc_transform_eg_in_ineg(sc);
    expression volume = expression_undefined;
    FOREACH(EXPRESSION,index,reference_indices(ref))
    {
        Variable phi = expression_to_entity(index);
        Pcontrainte lower,upper;
        constraints_for_bounds(phi, &sc_inegalites(sc), &lower, &upper);
        if( !CONTRAINTE_UNDEFINED_P(lower) && !CONTRAINTE_UNDEFINED_P(upper))
        {
            /* this is a constant : the volume is 1 */
            if(bounds_equal_p(phi,lower,upper))
            {
                volume=int_to_expression(1);
            }
            /* this is a range : the dimension is eupper-elower +1 and the offset is elower */
            else
            {
                expression elower = constraints_to_loop_bound(lower,phi,true,entity_intrinsic(DIVIDE_OPERATOR_NAME));
                expression eupper = constraints_to_loop_bound(upper,phi,false,entity_intrinsic(DIVIDE_OPERATOR_NAME));
                if(expression_minmax_p(elower))
                    simplify_minmax_expression(elower,tr);
                if(expression_minmax_p(eupper))
                    simplify_minmax_expression(eupper,tr);
                expression dim = make_op_exp(MINUS_OPERATOR_NAME,eupper,elower);
                dim = make_op_exp(PLUS_OPERATOR_NAME,dim,int_to_expression(1));
                if(expression_undefined_p(volume))
                    volume=dim;
                else
                    volume=binary_intrinsic_expression(MULTIPLY_OPERATOR_NAME,volume,dim);
            }
        }
        else {
            pips_user_warning("failed to analyse region\n");
            free_expression(volume);
            return expression_undefined;
        }
    }
    return volume;
}

/* the equation is given by sum(e) { | REGION_READ(e) U REGION_WRITE(e) | } < VOLUME */
static bool do_solve_hardware_constraints(statement s, Pvecteur * solution)
{
    list regions = load_cumulated_rw_effects_list(s);

    list read_regions = regions_read_regions(regions);
    list write_regions = regions_write_regions(regions);
    transformer tr = transformer_range(load_statement_precondition(s));

    set visited_entities = set_make(set_pointer);
    expression volume_used = int_to_expression(0);
    FOREACH(REGION,reg,regions)
    {
        reference r = region_any_reference(reg);
        entity e = reference_variable(r);
        /* check we have not already dealt with this variable */
        if(!set_belong_p(visited_entities,e))
        {
            set_add_element(visited_entities,visited_entities,e);
            if(entity_array_p(e)) {
                /* get the associated read and write regions */
                region read_region = find_region_on_entity(e,read_regions);
                region write_region = find_region_on_entity(e,write_regions);
                /* compute their convex hull */
                region rw_region = 
                    region_undefined_p(read_region)?write_region:
                    region_undefined_p(write_region)?read_region:
                    regions_must_convex_hull(read_region,write_region);
                /* then compute their surface */
                expression local_volume=region_enumerate(rw_region,tr);
                if(expression_undefined_p(local_volume)){
                    pips_user_warning("unable to compute volume of the region of %s\n",entity_user_name(e));
                    free_expression(volume_used);
                    return false;
                }
                volume_used=binary_intrinsic_expression(PLUS_OPERATOR_NAME,volume_used,local_volume);
            }
        }
    }
    int max_volume= get_int_property("SOLVE_HARDWARE_CONSTRAINTS_LIMIT");
    /* ok now we have our global volume, lets find the solution to volume_used<=max_volume 
     * we do not implement a generic solution */
    NORMALIZE_EXPRESSION(volume_used);
    if(normalized_complex_p(expression_normalized(volume_used)))
    {
        pips_user_warning("do not know how to optimize the non linear expression of the volume\n");
        print_expression(volume_used);
        free_expression(volume_used);
        return false;
    }
    Pvecteur vvolume = normalized_linear(expression_normalized(volume_used));
    int vvolume_sz = vect_size(vvolume);
    if(vvolume_sz == 0) {
        pips_user_warning("empty volume ??");
        return false;
    }
    else if(vvolume_sz <=2 ) {
        entity sym=entity_undefined;
        int sym_coeff=0;
        int constant=0;
        for (Pvecteur iter=vvolume;iter != VECTEUR_NUL ;iter = vecteur_succ(iter)) {
            if (term_cst(iter) ) {
                constant=vecteur_val(iter);
            } 
            else {
                if(!entity_undefined_p(sym)) {
                    pips_user_warning("do not know how to optimize the linear expression of the volume\n");
                    return false;
                }
                sym_coeff=vecteur_val(iter);
                sym=(entity)vecteur_var(iter);
            }
        }
        *solution=vect_new(sym,(max_volume-constant)/sym_coeff);
    }
    else {
        pips_user_warning("do not know how to optimize the linear expression of the volume\n");
        print_expression(volume_used);
        free_expression(volume_used);
        return false;
    }





    set_free(visited_entities);
    gen_free_list(read_regions);
    gen_free_list(write_regions);
    free_transformer(tr);
    return true;
}

bool solve_hardware_constraints(const char * module_name)
{
    set_current_module_entity(module_name_to_entity( module_name ));
    set_current_module_statement((statement) db_get_memory_resource(DBR_CODE, module_name, true) );
    set_cumulated_rw_effects((statement_effects)db_get_memory_resource(DBR_REGIONS, module_name, true));
    module_to_value_mappings(get_current_module_entity());
    set_precondition_map( (statement_mapping) db_get_memory_resource(DBR_PRECONDITIONS, module_name, true) );

    string stmt_label=get_string_property("SOLVE_HARDWARE_CONSTRAINTS_LABEL");
    statement equation_to_solve = find_statement_from_label_name(get_current_module_statement(),get_current_module_name(),stmt_label);

    Pvecteur solution = VECTEUR_NUL;
    bool result =false;
    if(!statement_undefined_p(equation_to_solve))
    {
        if((result=do_solve_hardware_constraints(equation_to_solve,&solution))) {
            pips_user_warning("found a solution\n");
            vect_fprint(stderr,solution,(get_variable_name_t)entity_user_name);
        }

        /* validate */
        module_reorder(get_current_module_statement());
        DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name,get_current_module_statement());
    }

    reset_current_module_entity();
    reset_current_module_statement();
    reset_cumulated_rw_effects();
    reset_precondition_map();
    free_value_mappings();
    return result;
}
