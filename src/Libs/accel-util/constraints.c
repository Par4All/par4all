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
#if 0
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
#endif
static Variable sort_key;
static int shc_sort(Pvecteur *v0, Pvecteur *v1)
{
    if((*v0)->var==sort_key) return 1;
    if((*v1)->var==sort_key) return -1;
    return compare_entities((entity*)&(*v0)->var,(entity*)&(*v1)->var);
}

#define SCILAB_PSOLVE "./psolve"

/* the equation is given by sum(e) { | REGION_READ(e) U REGION_WRITE(e) | } < VOLUME */
static bool do_solve_hardware_constraints(statement s)
{
    list regions = load_cumulated_rw_effects_list(s);

    list read_regions = regions_read_regions(regions);
    list write_regions = regions_write_regions(regions);
    transformer tr = transformer_range(load_statement_precondition(s));

    set visited_entities = set_make(set_pointer);
    char * volume_used[gen_length(regions)];
    size_t volume_index=0;

    /* retreive the unknown variable { */
    entity e = string_to_entity(get_string_property("SOLVE_HARDWARE_CONSTRAINTS_UNKNOWN"),get_current_module_entity());
    if(entity_undefined_p(e))
        pips_user_error("must provide the unknown value\n");
    Pbase pb = (Pbase) vect_new(e,-1);
    sort_key = e;
    /* } */

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

                Psysteme rw_sc = region_system(rw_region);
                vect_sort_in_place(&sc_base(rw_sc),shc_sort);
                const char * base_names [sc_dimension(rw_sc)];
                int i=0;
                for(Pbase b = sc_base(rw_sc);!BASE_NULLE_P(b);b=b->succ)
                    base_names[i]=entity_user_name((entity)b->var);

                Pehrhart er = sc_enumerate(rw_sc,pb,base_names);
                if(er) {
                    char ** ps = Pehrhart_string(er,base_names);
                    volume_used[volume_index++]=ps[0];/* use first ... */
                    for(char ** iter = ps +1; *iter; iter++) free(*iter);
                    free(ps);
                }
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
    asprintf(&scilab_cmd,SCILAB_PSOLVE " '%s'",full_poly);
    FILE* scilab_response = popen(scilab_cmd,"r");
    if(!scilab_response) pips_user_error("failed to solve polynomial %s\n",full_poly);
    float fresponse=0.;
    if(fscanf(scilab_response,"%f",&fresponse)!=1)
        pips_user_error("failed to scan "SCILAB_PSOLVE"response\n");
    if(pclose(scilab_response))
        pips_user_error("failed to call "SCILAB_PSOLVE" %s\n",full_poly);

    /* if the result is an integer, we have won, otherwise, try near integers */
    int iresponse = (int)fresponse;
    /* assume it will be ok with nearset integer , should do more check there ... */
    insert_statement(s,
            make_assign_statement(entity_to_expression(e),int_to_expression(iresponse)),
            true);

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
    return result;
}
