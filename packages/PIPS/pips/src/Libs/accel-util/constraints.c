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
#include "control.h"
#include "conversion.h"
#include "expressions.h"
#include "effects-generic.h"
#include "effects-simple.h"
#include "effects-convex.h"
#include "text-util.h"
#include "parser_private.h"
#include "semantics.h"
#include "transformer.h"
#include "accel-util.h"

static bool do_solve_hardware_constraints_on_nb_proc(entity e, statement s) {
    list regions = load_cumulated_rw_effects_list(s);

    list read_regions = regions_read_regions(regions);
    list write_regions = regions_write_regions(regions);
    transformer tr = transformer_range(load_statement_precondition(s));
    /* add a new constraint to the system */
    Pvecteur pv = vect_new(e,-1);
    vect_add_elem(&pv,TCST,3);
    sc_add_inegalite(predicate_system(transformer_relation(tr)),
            contrainte_make(pv));

    set visited_entities = set_make(set_pointer);

    expression max_dim = expression_undefined;
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

                region hregion = rw_region;//region_hypercube(rw_region);
                entity phi = expression_to_entity(EXPRESSION(CAR(reference_indices(r))));
                list ephis = expressions_to_entities(reference_indices(r));
                Pbase phis = list_to_base(ephis);
                gen_free_list(ephis);
                Psysteme hsc = sc_rectangular_hull(region_system(hregion),phis);
                base_rm(phis);

                Pcontrainte lower,upper;
                constraints_for_bounds(phi,&sc_inegalites(hsc),&lower,&upper);
                if(!CONTRAINTE_UNDEFINED_P(lower) && !CONTRAINTE_UNDEFINED_P(upper))
                {
                    expression elower = constraints_to_loop_bound(lower,phi,true,entity_intrinsic(DIVIDE_OPERATOR_NAME));
                    simplify_minmax_expression(elower,tr);
                    expression eupper = constraints_to_loop_bound(upper,phi,false,entity_intrinsic(DIVIDE_OPERATOR_NAME));
                    simplify_minmax_expression(eupper,tr);
                    expression dist = make_op_exp(MINUS_OPERATOR_NAME,eupper,elower);
                    dist=make_op_exp(PLUS_OPERATOR_NAME,dist,int_to_expression(1));

                    if(expression_undefined_p(max_dim))
                        max_dim = dist;
                    else {
                        max_dim = MakeBinaryCall(
                                entity_intrinsic(MAX_OPERATOR_NAME),
                                max_dim,
                                dist
                                );
                        simplify_minmax_expression(max_dim,tr);
                    }

                }
                else
                    pips_user_error("failed to gather enough information for entity %s\n",entity_user_name(e));
            }
        }
    }
    int max_volume= get_int_property("SOLVE_HARDWARE_CONSTRAINTS_LIMIT");
    if(max_volume<=0) pips_user_error("constraint limit must be greater than 0\n");
    /* solve the equation if it is linear */
    max_dim=make_op_exp(MINUS_OPERATOR_NAME,max_dim,int_to_expression(max_volume));
    NORMALIZE_EXPRESSION(max_dim);
    normalized n = expression_normalized(max_dim);
    if(normalized_linear_p(n)) {
        /* create a system with preconditions information and the constraint limit*/
        Psysteme sc = sc_dup(predicate_system(transformer_relation(tr)));
        sc_add_egalite(sc,contrainte_make(normalized_linear(n)));
        /* find numerical constraints over unknown e */
        Value min,max;
        if(sc_minmax_of_variable(sc,e,&min,&max)) {
            expression soluce = Value_to_expression(min);
            /* SG: this is over optimistic, 
             * we should verify all elements of soluce are store-independant
             */
            free_value(entity_initial(e));
            entity_initial(e)=make_value_expression(
                    soluce);
        }
        else /* welcome in the real life (RK(C) we cannot solve this equation at commile time ... never mind let's do it at runtime ! */
            pips_user_error("unable to solve the equation at compile time\n");
    }
    else pips_user_error("unable to get a linear expression for nbproc\n");

    set_free(visited_entities);
    gen_free_list(read_regions);
    gen_free_list(write_regions);
    free_transformer(tr);
    return true;
}

static bool do_solve_hardware_constraints_on_volume(entity unknown, statement s) {
    list regions = load_cumulated_rw_effects_list(s);

    list read_regions = regions_read_regions(regions);
    list write_regions = regions_write_regions(regions);
    transformer tr = transformer_range(load_statement_precondition(s));

    set visited_entities = set_make(set_pointer);

    Ppolynome volume_used = POLYNOME_NUL;
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

                region hregion = rw_region;//region_hypercube(rw_region);
                Ppolynome p = region_enumerate(hregion);
                if(!POLYNOME_UNDEFINED_P(p)) {
                    polynome_add(&volume_used,p);
                    polynome_rm(&p);
                }
                else
                    pips_user_error("unable to compute volume of the region of %s\n",entity_user_name(e));
            }
        }
    }
    int max_volume= get_int_property("SOLVE_HARDWARE_CONSTRAINTS_LIMIT");
    if(max_volume<=0) pips_user_error("constraint limit must be greater than 0\n");
    /* create a string representation of all polynomials gathered */
    Pmonome m = monome_constant_new(-max_volume);
    polynome_monome_add(&volume_used,m);
    monome_rm(&m);
    /* try to solve the polynomial */
    Pvecteur roots = polynome_roots(volume_used,unknown);
    expression eroot;
    if(VECTEUR_UNDEFINED_P(roots)) { // that is volume_used is independent of unknown
        expression cst = polynome_to_expression(volume_used);
        partial_eval_expression_and_regenerate(&cst,predicate_system(transformer_relation(tr)),load_cumulated_rw_effects(s));
        intptr_t val;
        if(expression_integer_value(cst,&val) && val < 0 )
            eroot=int_to_expression(42);//any value is ok
        else
            pips_user_error("no solution possible for this limit\n");
    }
    else {
        Ppolynome root = (Ppolynome)vecteur_var(roots);//yes take the first without thinking more ...
        eroot = polynome_to_expression(root);
        if(expression_constant_p(eroot) &&!expression_integer_constant_p(eroot)) {
            /* this takes the floor of the floating point expression ...*/
            int ival = (int)expression_to_float(eroot);
            free_expression(eroot);
            eroot=int_to_expression(ival);
        }
    }

    /* insert solution ~ this is an approximation ! 
     * it only works if root is increasing
     */
    insert_statement(get_current_module_statement(),
            make_assign_statement(entity_to_expression(unknown),eroot),
            true);

    /* tidy */
    for(Pvecteur v = roots;!VECTEUR_NUL_P(v);v=vecteur_succ(v))
        polynome_rm((Ppolynome*)&vecteur_var(v));
    vect_rm(roots);
    set_free(visited_entities);
    gen_free_list(read_regions);
    gen_free_list(write_regions);
    free_transformer(tr);
    return true;
}

/* the equation is given by sum(e) { | REGION_READ(e) U REGION_WRITE(e) | } < VOLUME */
static bool do_solve_hardware_constraints(statement s)
{

    /* retreive the unknown variable { */
    entity unknown = string_to_entity(get_string_property("SOLVE_HARDWARE_CONSTRAINTS_UNKNOWN"),get_current_module_entity());
    if(entity_undefined_p(unknown))
        pips_user_error("must provide the unknown value\n");
    /* } */
    const char* constraint_type = get_string_property("SOLVE_HARDWARE_CONSTRAINTS_TYPE");
    if(same_string_p(constraint_type, "VOLUME"))
        return do_solve_hardware_constraints_on_volume(unknown,s);
    else if(same_string_p(constraint_type, "NB_PROC"))
        return do_solve_hardware_constraints_on_nb_proc(unknown,s);
    else {
        pips_user_error("constraint type '%s' unknown\n",constraint_type);
        return false;
    }
}

bool solve_hardware_constraints(const char * module_name)
{
    debug_on("SOLVE_HARDWARE_CONSTRAINTS");
    set_current_module_entity(module_name_to_entity( module_name ));
    set_current_module_statement((statement) db_get_memory_resource(DBR_CODE, module_name, true) );
    set_cumulated_rw_effects((statement_effects)db_get_memory_resource(DBR_REGIONS, module_name, true));
    module_to_value_mappings(get_current_module_entity());
    set_precondition_map( (statement_mapping) db_get_memory_resource(DBR_PRECONDITIONS, module_name, true) );

    const char* stmt_label=get_string_property("SOLVE_HARDWARE_CONSTRAINTS_LABEL");
    statement equation_to_solve = find_statement_from_label_name(get_current_module_statement(),get_current_module_name(),stmt_label);

    bool result =false;
    if(!statement_undefined_p(equation_to_solve))
    {
        if((result=do_solve_hardware_constraints(equation_to_solve))) {
            /* validate */
            module_reorder(get_current_module_statement());
            DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name,get_current_module_statement());
        }
    }

    reset_current_module_entity();
    reset_current_module_statement();
    reset_cumulated_rw_effects();
    reset_precondition_map();
    free_value_mappings();
    debug_off();
    return result;
}
