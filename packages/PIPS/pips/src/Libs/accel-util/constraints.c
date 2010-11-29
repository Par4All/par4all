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
#include "control.h"
#include "conversion.h"
#include "effects-generic.h"
#include "effects-simple.h"
#include "effects-convex.h"
#include "text-util.h"
#include "parser_private.h"
#include "semantics.h"
#include "transformer.h"
#include "accel-util.h"


/* computes the volume of a region
 * output it as a string representing the associated polynomial
 * maybe one day, conversion from string to expression will be possible ?
 *
 * the string is here to communicate with polylib.
 * It would be far better to convert polylib view into an expression
 *
 * the result is in number of bytes used
 */
static
char* region_enumerate(region reg)
{
    char * volume_used = NULL;
    Psysteme r_sc = region_system(reg);
    sc_fix(r_sc);
    Pbase sorted_base = region_sorted_base_dup(reg);
    sc_base(r_sc)=sorted_base;
    Pbase local_base = BASE_NULLE;
    for(Pbase b = sc_base(r_sc);!BASE_NULLE_P(b);b=b->succ)
        if(!variable_phi_p((entity)b->var))
            local_base=base_add_variable(local_base,b->var);

    const char * base_names [sc_dimension(r_sc)];
    int i=0;
    for(Pbase b = local_base;!BASE_NULLE_P(b);b=b->succ)
        base_names[i++]=entity_user_name((entity)b->var);

    ifdebug(1) print_region(reg);

    Pehrhart er = sc_enumerate(r_sc,
            local_base,
            base_names);
    if(er) {
        char ** ps = Pehrhart_string(er,base_names);
        /* use the smallest ... */
        volume_used = *ps;
        for(char **iter=ps +1;*iter ; ++iter)
            if(strlen(volume_used) > strlen(*iter)) volume_used=*iter;
        for(char ** iter = ps; *iter; iter++) if(*iter!=volume_used) free(*iter);
        free(ps);
    }
    return volume_used;
}

#define SCILAB_PSOLVE "psolve"

#if 0
static bool statement_parent_walker(statement s, statement *param)
{
    if(s == param[0]) {
        param[1]=(statement)gen_get_ancestor(statement_domain,s);
        gen_recurse_stop(NULL);
    }
    return true;
}
static statement statement_parent(statement root, statement s)
{
    statement args[] = { s, statement_undefined };
    gen_context_recurse(root,args,statement_domain,statement_parent_walker,gen_null);
    return args[1] == NULL || statement_undefined_p(args[1]) ? s : args[1];
}
#endif
static bool do_solve_hardware_constraints_on_nb_proc(entity e, statement s) {
    list regions = load_cumulated_rw_effects_list(s);

    list read_regions = regions_read_regions(regions);
    list write_regions = regions_write_regions(regions);
    transformer tr = transformer_range(load_statement_precondition(s));

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
    NORMALIZE_EXPRESSION(max_dim);
    normalized n = expression_normalized(max_dim);
    expression soluce = expression_undefined;
    if(normalized_linear_p(n)) {
        Pvecteur pv = VECTEUR_NUL;
        Value coeff = VALUE_ZERO;
        for(Pvecteur v = normalized_linear(n);!VECTEUR_NUL_P(v); v=vecteur_succ(v)) {
            if(vecteur_var(v) == e ) {
                coeff = vecteur_val(v);
            }
            else {
                pv = vect_chain(pv,vecteur_var(v),vecteur_val(v));
            }
        }
        soluce=Pvecteur_to_expression(pv);
        soluce=make_op_exp(
                MINUS_OPERATOR_NAME,
                int_to_expression(max_volume),
                soluce);
        soluce=make_op_exp(
                DIVIDE_OPERATOR_NAME,
                soluce,
                int_to_expression((_int)coeff)
                );
        /* SG: this is over optimistic, 
         * we should verify all elements of soluce are store-independant
         */
        free_value(entity_initial(e));
        entity_initial(e)=make_value_expression(
                soluce);
    }

    set_free(visited_entities);
    gen_free_list(read_regions);
    gen_free_list(write_regions);
    free_transformer(tr);
    return true;
}

static bool do_solve_hardware_constraints_on_volume(entity e, statement s) {
    list regions = load_cumulated_rw_effects_list(s);

    list read_regions = regions_read_regions(regions);
    list write_regions = regions_write_regions(regions);
    transformer tr = transformer_range(load_statement_precondition(s));

    set visited_entities = set_make(set_pointer);
    char * volume_used[gen_length(regions)];
    size_t volume_index=0;

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
                char * vused = region_enumerate(hregion);
                if(vused)
                    volume_used[volume_index++]=vused;
                else
                    pips_user_error("unable to compute volume of the region of %s\n",entity_user_name(e));
            }
        }
    }
    int max_volume= get_int_property("SOLVE_HARDWARE_CONSTRAINTS_LIMIT");
    if(max_volume<=0) pips_user_error("constraint limit must be greater than 0\n");
    /* create a string representation of all polynome gathered */
    char* full_poly =strdup(itoa(-max_volume));
    for(int i=0;i<volume_index;i++) {
        char *tmp;
        asprintf(&tmp,"%s+%s",full_poly,volume_used[i]);
        free(full_poly);
        full_poly=tmp;
    }
    /* call an external solver */
    char *scilab_cmd;
    asprintf(&scilab_cmd,SCILAB_PSOLVE " '%s' '%s'",full_poly,entity_user_name(e));
    /* must put the pragma on a new statement, because the pragma will be changed into a statement later */
    statement holder = make_continue_statement(entity_empty_label());
    add_pragma_str_to_statement(holder,scilab_cmd,false);
    //statement parent = statement_parent(get_current_module_statement(),s);
    insert_statement(get_current_module_statement(),holder,true);

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
    string constraint_type = get_string_property("SOLVE_HARDWARE_CONSTRAINTS_TYPE");
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

    string stmt_label=get_string_property("SOLVE_HARDWARE_CONSTRAINTS_LABEL");
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
