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
 * @file isolate_statement.c
 * transfer statement to isolate memory
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
#include "preprocessor.h"
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
#include "callgraph.h"
#include "expressions.h"
#include "accel-util.h"
#include "hpfc.h"


/**
 * isolate_statement
 */

struct dma_pair {
    entity new_ent;
    enum region_to_dma_switch s;
};


/* Some constant intended to have a more readable code */
static const int dmaScalar = 0;
static const int dma1D = 1;

static size_t get_dma_dimension(region reg_from) {
    /* It should be the number of variable phis and stop at first field*/
    size_t n=0;
    FOREACH(EXPRESSION,exp,reference_indices(region_any_reference(reg_from))) {
        if(expression_reference_p(exp)) {
            reference ref = expression_reference(exp);
            entity var = reference_variable(ref);
            if(variable_phi_p(var) ) {
                    ++n;
            }
            else {
                pips_assert("not a  phi means a field",entity_field_p(var));
                break;
            }
        }
    }
    return n;
}

/**
 * converts a region_to_dma_switch to corresponding dma name
 * according to properties
 */
static const char* get_dma_name(enum region_to_dma_switch m, size_t d) {
    const char *seeds[] = {
        "KERNEL_LOAD_STORE_LOAD_FUNCTION",
        "KERNEL_LOAD_STORE_STORE_FUNCTION",
        "KERNEL_LOAD_STORE_ALLOCATE_FUNCTION",
        "KERNEL_LOAD_STORE_DEALLOCATE_FUNCTION"
    };
    const char * propname = seeds[(int)m];
    /* If the DMA is not scalar, the DMA function name is in the property
       of the form KERNEL_LOAD_STORE_LOAD/STORE_FUNCTION_dD: */
    char * apropname= NULL;
    if(d > 0 /* not scalar*/ && (int)m < 2)
        asprintf(&apropname,"%s_%dD", seeds[(int)m], (int)d);
    const char* dmaname = get_string_property(apropname?apropname:propname);
    if(apropname) free(apropname);
    return dmaname;
}

#if 0
/*
 * Build a fake region and try to build dimensions from it
 */
static bool declaration_region_to_dimensions(effect reg,
                                      transformer tr,
                                      list *dimensions,
                                      list * offsets,
                                      expression *condition) {
  pips_assert("effects are regions\n", effect_region_p(reg));
  bool success_p=true;

  /* Apply partial eval on the declaration, better chance to get a
   * "usable" region
   */
  effects dummy_effect = make_effects(NIL);
  entity e = reference_variable(region_any_reference(reg));
  set_live_loop_indices();
  partial_eval_declaration(e,
                           predicate_system(transformer_relation(tr)),
                           dummy_effect);
  reset_live_loop_indices();
  free_effects(dummy_effect);

  /* Create a new region corresponding to the declaration dimensions */
  effect decl_region = region_dup(reg);
  descriptor_convex(effect_descriptor(decl_region)) = entity_declaration_sc(e);

  /* Try again to compute the dimensions, based on the array declaration
   * Mehdi: I don't understand why we want to use a "region" to do that ?
   */
  if(!region_to_minimal_dimensions(reg,
                                   tr,
                                   dimensions,
                                   offsets,
                                   true,
                                   condition)) {
    success_p = false;
    pips_user_warning("failed to compute DMA from the array declaration based region\n");
    ifdebug(4) {
      pips_user_warning("Original Region:");
      print_region(decl_region);
      pips_user_warning("Declaration based Region:");
      print_region(decl_region);
      pips_user_warning("Entity is :");
      print_entity_variable(e);
    }
  }
  region_free(decl_region);
  return success_p;
}
#endif


// Build DMA dimensions directly from the entity declaration
static bool declarations_to_dimensions(entity e, list *dimensions, list * offsets) {
  if(!entity_array_p(e)) {
    return false;
  }

  // Populate "dimensions" and "offsets" by looking to the declaration
  list ldim = variable_dimensions(type_variable(ultimate_type(entity_type(e))));
  FOREACH(dimension, d, ldim) {
    // FIXME Unsure who cares about the free....
    dimension d_dup = copy_dimension(d);
    expression dl = dimension_lower(d);

    // Check if we have an offset (impossible in C!)
    intptr_t offset;
    bool dl_is_int_p=expression_integer_value(dl,&offset);
    pips_assert("lower bound is int",dl_is_int_p);
    if(offset!=0) {
      pips_internal_error("We shouldn't have an offset in the array declaration"
          ", are you using Fortran?");
    }

    *dimensions = CONS(DIMENSION, d_dup, *dimensions );
    *offsets = CONS(EXPRESSION, dl, *offsets);
  }
  return true;

}


void region_to_dimensions(effect reg, transformer tr, list *dimensions, list * offsets, expression *condition) {
    pips_assert("effects are regions\n",effect_region_p(reg));
    if( ! region_to_minimal_dimensions(reg,tr,dimensions,offsets,true,condition) ) {
        pips_user_warning("failed to convert regions to minimal array dimensions, using whole array instead\n");

        bool success_p = false;
        // Try using a region based on the declaration (Old way)
        //success_p=declaration_region_to_dimensions(reg,tr,dimensions,offsets,condition);

        // Use directly the dimensions from the declaration without building a region!
        entity e = reference_variable(region_any_reference(reg));
        success_p=declarations_to_dimensions(e, dimensions, offsets);
 
        if(!success_p) {
          // Don't know how to recover from that...
          pips_internal_error("Abort");
        }
    }
}


static expression region_to_address(region reg) {
    bool is_fortran = fortran_module_p(get_current_module_entity());
    int indice_first = (is_fortran == true) ? 1 : 0;
    expression address = entity_to_expression(reference_variable(region_any_reference(reg)));
    FOREACH(EXPRESSION,exp,reference_indices(region_any_reference(reg))) {
        pips_assert("region indices only contain references",expression_reference_p(exp));
        reference expr = expression_reference(exp);
        entity var =reference_variable(expr);
        if(variable_phi_p(var)) {
            expression index = int_to_expression(indice_first);
            if(expression_reference_p(address)) {
                reference_indices(expression_reference(address))=
                    gen_nconc(reference_indices(expression_reference(address)),
                        make_expression_list(index));
            }
            else 
            address= make_expression(
                    make_syntax_subscript(
                        make_subscript(address,make_expression_list(index))
                        )
                    ,
                    normalized_undefined
                    );
        }
        else {
            address= MakeBinaryCall(
                    entity_intrinsic(FIELD_OPERATOR_NAME),
                    address,
                    copy_expression(exp)
                    );
        }
    }
    expression result;
    if (fortran_module_p(get_current_module_entity()))
        result = MakeUnaryCall(entity_intrinsic(C_LOC_FUNCTION_NAME),
                address);
    else
        result = MakeUnaryCall(entity_intrinsic(ADDRESS_OF_OPERATOR_NAME),
                address);
    return result;
}

/* generate an expression of the form
 * sizeof(typeof(variable[indices]))
 *
 * It also handles the fields:
 * fields reference are converted to proper expression
 * then an approximation is made to ensure there is no stride
 * e.g
 * struct { int x,y; } a [10];
 * a[1][x] -> sizeof(a[1])
 * struct { int x[10], y[10] } a;
 * a[x][1] -> sizeof(a.x)
 */

static expression get_sizeofexpression_for_region(region reg) {
    expression sizeof_exp;
    reference rr = region_any_reference(reg);

    reference r = make_reference(reference_variable(rr),NIL);

    /* extract the constant indices of the region */
    FOREACH(EXPRESSION,exp, reference_indices(rr)) {
        pips_assert("region indices are only references",expression_reference_p(exp)) ;
        reference ref = expression_reference(exp);
        entity var = reference_variable(ref);
        if(entity_field_p(var)) {
            break;
        }
        else if(variable_phi_p(var)) {
                reference_indices(r)=CONS(EXPRESSION,int_to_expression(0),reference_indices(r));
        }
        else pips_internal_error("not a field and not a phi variable ?");
    }
    reference_indices(r)= gen_nreverse(reference_indices(r));

    type element_type = make_type_variable(make_variable(basic_of_reference(r),
                NIL,
                NIL));

    /* Here we make a special case for struct because of nvcc/C++ doesn't like construct like :
     *  sizeof(struct {data_t latitude; data_t longitude; data_t stock;})
     * so we produce a sizeof(var); instead
     */
    if(type_struct_variable_p(element_type)) {
        expression exp = region_reference_to_expression(r);
        sizeof_exp = MakeSizeofExpression(exp);
        free_type(element_type);
    } else {
        sizeof_exp = MakeSizeofType(element_type);
    }
    return sizeof_exp;
}

list variable_to_dimensions(region reg_from) {
    /* We may skip the size of the first dimension since it is not
       used in address calculation. But since it depends of Fortran
       or C in the runtime, postpone this micro-optimization... */
    reference r =region_any_reference(reg_from);
    entity from = reference_variable(r);
    variable from_tv = type_variable(ultimate_type(entity_type(from)));
    list dims = variable_dimensions(from_tv);
    list out=NIL;
    FOREACH(EXPRESSION,exp, reference_indices(r)) {
        pips_assert("a region only contains references", expression_reference_p(exp));
        reference rexp = expression_reference(exp);
        entity var = reference_variable(rexp);
        if(variable_phi_p(var)) {
            /* we are in trouble, as we don't know anything about the size of the underlying array. It is the fist dimension, so not very important anyway ... except for FORTRAN :-( But there are no pointer in FORTRAN :p */
            if(ENDP(dims) && basic_pointer_p(variable_basic(from_tv))) {
                pips_assert("no pointer in fortran", c_module_p(get_current_module_entity()));
                expression eupper = expression_undefined;
                Psysteme sc_reg = sc_dup(region_system(reg_from));
                sc_transform_eg_in_ineg(sc_reg);
                Pcontrainte lower,upper;
                constraints_for_bounds(var,
                        &sc_inegalites(sc_reg), &lower, &upper);
                if( !CONTRAINTE_UNDEFINED_P(upper))
                {
                    eupper = constraints_to_loop_bound(upper,var,false,entity_intrinsic(DIVIDE_OPERATOR_NAME));
                }
                else {
                    pips_internal_error("Should not happen at this point, should it ?");
                }

                out=CONS(DIMENSION,
                        make_dimension(
                            int_to_expression(0), eupper
                            ), out);
                from_tv = type_variable(ultimate_type(basic_pointer(variable_basic(from_tv))));
                dims = variable_dimensions(from_tv);
            }
            else {
                pips_assert("some dims stacked", !ENDP(dims));
                out=CONS(DIMENSION, copy_dimension(DIMENSION(CAR(dims))), out);
                POP(dims);
            }
        }
        else if(entity_field_p(var)) {
            break;
        }
    }
    return gen_nreverse(out);
}

/**
 * converts dimensions to a dma call from a memory @a from to another memory @a to
 *
 * @param from expression giving the adress of the input memory
 * @param to expression giving the adress of the output memory
 * @param ld list of dimensions to analyze
 * @param m kind of call to generate
 *
 * @return
 */
call dimensions_to_dma(effect reg_from,
        entity to,
        list/*of dimensions*/ ld,
        list/*of offsets*/    lo,
        enum region_to_dma_switch m)
{
    entity from = reference_variable(region_any_reference(reg_from));
    expression dest;
    list args = NIL;
    const char* function_name = get_dma_name(m,get_dma_dimension(reg_from));

    entity mcpy = module_name_to_entity(function_name);
    if (entity_undefined_p(mcpy)) {
        mcpy=make_empty_subroutine(function_name,copy_language(module_language(get_current_module_entity())));
        pips_user_warning("Cannot find \"%s\" method. Are you sure you have set\n"
                "KERNEL_LOAD_STORE_..._FUNCTION "
                "to a defined entity and added the correct .c file?\n",function_name);
    }
    else if (!fortran_module_p(get_current_module_entity())) {
        AddEntityToModuleCompilationUnit(mcpy,get_current_module_entity());
    } else
        AddEntityToModule(mcpy,get_current_module_entity());

    /* Scalar detection: */
    bool scalar_entity = entity_scalar_p(from);

    if (dma_allocate_p(m)) {
        /* Need the address for the allocator to modify the pointer itself: */
        dest = MakeUnaryCall(entity_intrinsic(ADDRESS_OF_OPERATOR_NAME),entity_to_expression(to));
        /* Generate a "void **" type: */
        type voidpp = make_type_variable(
                make_variable(
                    make_basic_pointer(
                        make_type_variable(
                            make_variable(
                                make_basic_pointer(
                                    make_type_void(NIL)
                                    ),
                                NIL,NIL
                                )
                            )
                        ),
                    NIL,NIL
                    )
                );
        /* dest = "(void **) &to" */
        dest = make_expression(
                make_syntax_cast(
                    make_cast(voidpp,dest)
                    ),
                normalized_undefined);
    }
    else if (!dma_deallocate_p(m) && !scalar_entity)
        /* Except for the deallocation or if we have a scalar and then we have
           already created a pointer to it, the original array is referenced
           through pointer dereferencing: */
        dest = MakeUnaryCall(entity_intrinsic(DEREFERENCING_OPERATOR_NAME),
                entity_to_expression(to));
    else
        dest=entity_to_expression(to);


    switch(m) {
        case dma_deallocate:
            args = make_expression_list(dest);
            break;
        case dma_allocate:
            {
                expression sizeof_exp = get_sizeofexpression_for_region(reg_from);

                /* sizeof(element)*number elements of the array: */
                expression transfer_size = SizeOfDimensions(ld);
                transfer_size=MakeBinaryCall(
                        entity_intrinsic(MULTIPLY_OPERATOR_NAME),
                        sizeof_exp,
                        transfer_size);

                args = make_expression_list(dest, transfer_size);
            } break;
        case dma_load:
        case dma_store:
            /* Generate communication functions: */
            {
                //if(!scalar_entity) {
                /* Build the sizes of the array block to transfer: */
                list /*of expressions*/ transfer_sizes = NIL;
                FOREACH(DIMENSION,d,ld) {
                    expression transfer_size=
                        SizeOfDimension(d);
                    transfer_sizes=CONS(EXPRESSION,transfer_size,transfer_sizes);
                }
                transfer_sizes=gen_nreverse(transfer_sizes);

                /* Build the sizes of the array with element to transfer: */
                list/* of expressions*/ from_dims = NIL;
                list vardims = variable_to_dimensions(reg_from);
                FOREACH(DIMENSION,d, vardims) {
                    from_dims=CONS(EXPRESSION,SizeOfDimension(d),from_dims);
                }
                gen_full_free_list(vardims);
                from_dims=gen_nreverse(from_dims);

                /* Build the offsets of the array block to transfer: */
                list/* of expressions*/ offsets = NIL;
                FOREACH(EXPRESSION,e,lo)
                    offsets=CONS(EXPRESSION,e,offsets);
                offsets=gen_nreverse(offsets);
                /* Use a special transfert function for scalars instead of reusing
                   the 1D function. It may useful for example if it is implemented
                   as a FIFO at the hardware level: */
                //} else {
                //    /* If we have a scalar variable to transfert, generate
                //       synthetic transfer parameters: */
                //    /* 1 element to transfert */
                //    transfer_sizes = make_expression_list(int_to_expression(1));
                //    /* 1 dimension */
                //    from_dims = make_expression_list(int_to_expression(1));
                //    /* At the begining of the « array »: */
                //    offsets = make_expression_list(int_to_expression(0));
                //  }

                expression source = region_to_address(reg_from);
                /* Generate host and accel adresses: */
                args = CONS(EXPRESSION,source,CONS(EXPRESSION,dest,NIL));
                //if(dma_load_p(m))
                //    args=gen_nreverse(args);
                /* Output parameters in an order compatible with some C99
                   implementation of the runtime: size and block size first, so
                   that some arguments can be defined with them: */
                /* Insert offset: */
                args = gen_append(offsets, args);
                /* Insert the block size to transfert: */
                args = gen_append(transfer_sizes, args);
                /* Insert the array sizes: */
                args = gen_append(from_dims, args);
                /* Insert the element size expression: */
                expression sizeof_exp = get_sizeofexpression_for_region(reg_from);
                args = CONS(EXPRESSION,
                        sizeof_exp,
                        args);
            } break;
        default:
            pips_internal_error("should not happen");
    }
    return make_call(mcpy, args);
}

/* perform the convex hull between r0 and r1, and merge them if they have a common prefix
 * e.g. a[phi1].re[phi2] U_ex a[phi3].im[phi4] = a[phi1] U_may a[phi3]
 */
static
region extended_regions_must_convex_hull(region r0, region r1) {
    list i0 = reference_indices(region_any_reference(r0)),
         i1 = reference_indices(region_any_reference(r1));
    list n0 = NIL, n1=NIL;
    while(!ENDP(i0)&&!ENDP(i1)) {
        expression e0 = EXPRESSION(CAR(i0)),
                   e1 = EXPRESSION(CAR(i1));
        // stop there: we have found the common prefix
        if(expression_reference_p(e0) && entity_field_p(reference_variable(expression_reference(e0))) &&
                expression_reference_p(e1) && entity_field_p(reference_variable(expression_reference(e1))) ) {
            n0=gen_nreverse(n0);
            n1=gen_nreverse(n1);
            list t0 = reference_indices(region_any_reference(r0)),
                 t1 = reference_indices(region_any_reference(r1));
            reference_indices(region_any_reference(r0)) = n0;
            reference_indices(region_any_reference(r1)) = n1;
            region out = regions_must_convex_hull(r0, r1);
            region_approximation_tag(out)=is_approximation_may; // better pessimistic than nothing.
            reference_indices(region_any_reference(r0)) = t0;
            reference_indices(region_any_reference(r1)) = t1;
            gen_free_list(n0);
            gen_free_list(n1);
            return out;
        }
        // else keep on pushing
        n0=CONS(EXPRESSION,e0,n0);
        n1=CONS(EXPRESSION,e1,n1);
        POP(i0);
        POP(i1);
    }
    gen_free_list(n0);
    gen_free_list(n1);
    return regions_must_convex_hull(r0, r1);

}

/* Compute a call to a DMA function from the effects of a statement

   @param[in] stat is the statement we want to generate communication
   operations for
   @param[in] prefix is the prefix to be used for added variable.
   operations for

   @return a statement of the DMA transfers or statement_undefined if
   nothing is needed or if the dma function has been set
   to "" in the relevant property

   If this cannot be done, it throws a pips_user_error
   */
    static
statement effects_to_dma(statement stat,
        enum region_to_dma_switch s,
        hash_table e2e, expression * condition,
        bool fine_grain_analysis, const char* prefix,
        const char* suffix)
{
    /* if no dma is provided, skip the computation
     * it is used for scalope at least */
    if(empty_string_p(get_dma_name(s,dma1D)))
        return statement_undefined;

    /* work on a copy because we compute the rectangular hull in place
       and keep only store effects: we do not care for environment effects here (BC)
       */
    list l_eff_tmp = effects_store_effects(load_cumulated_rw_effects_list(stat));
    list rw_effects = gen_full_copy_list(l_eff_tmp);
    gen_free_list(l_eff_tmp); /* free the spine, as effects are shared with the database */
    transformer tr = transformer_range(load_statement_precondition(stat));

    /* ensure we only have a rectangular region
     * as a side effect, strided accesses are handled by region_rectangular_hull
     */
    for(list iter = rw_effects;!ENDP(iter);POP(iter)) {
        region *tmp = (region*)REFCAR(iter);
        region new = region_rectangular_hull(*tmp,true);
        //    free_effect(*tmp); SG: why does this lead to a segfault ?
        //    I find no sharing in region_rectangular_hull
        *tmp=new;
    }

    /* merge copy in and copy out for allocation
     * this is entity-based, we could do a better job if we were reference-based 
     */
    if (dma_allocate_p(s) || dma_deallocate_p(s)) {
        list out = NIL;
        set visited_entities = set_make(set_pointer);
        FOREACH(EFFECT,self,rw_effects) {
            entity svar = reference_variable(region_any_reference(self));
            if(!set_belong_p(visited_entities,svar)) {
                set_add_element(visited_entities, visited_entities, svar);
                FOREACH(EFFECT,other, rw_effects) {
                    if(self!=other) {
                        entity ovar = reference_variable(region_any_reference(other));
                        if(same_entity_p(svar,ovar)) {
                            effect etmp = extended_regions_must_convex_hull(self,other);
                            if(region_write_p(other))
                                region_action(etmp) = copy_action(region_action(other));
                            //free_effect(self);
                            self=etmp;
                        }
                    }
                }
                out=CONS(EFFECT,self,out);
            }
        }
        set_free(visited_entities);
        gen_free_list(rw_effects);
        rw_effects=gen_nreverse(out);
    }
    /* merge regions with same prefix but different suffixes (in presence of fields ...) */
    if (dma_load_p(s) || dma_store_p(s) || dma_allocate_p(s) ) {
        list out = NIL;
        set r_visited_entities = set_make(set_pointer);
        set w_visited_entities = set_make(set_pointer);
        FOREACH(EFFECT,self, rw_effects) {
            entity svar = reference_variable(region_any_reference(self));
            enum action_utype sa = action_tag(region_action(self));
            set this = is_action_read ==sa ? r_visited_entities : w_visited_entities;
            if(!set_belong_p(this ,svar)) {
                set_add_element(this, this, svar);
                FOREACH(EFFECT,other,rw_effects) {
                    if(self!=other) {
                        entity ovar = reference_variable(region_any_reference(other));
                        if(same_entity_p(svar,ovar) && (sa == action_tag(region_action(other))) ) {
                            effect etmp = extended_regions_must_convex_hull(self,other);
                            //free_effect(self);
                            self=etmp;
                        }
                    }
                }
                out=CONS(EFFECT,self,out);
            }
        }
        set_free(r_visited_entities);
        set_free(w_visited_entities);
        gen_free_list(rw_effects);
        rw_effects=gen_nreverse(out);
    }

    /* SG: to do: merge convex hulls when they refer to *all* fields of a region
     * to do this, according to BC, I should iterate over all regions,
     * detect fields and then iterate again over regions to find combinable regions
     * that way I would not generate needless read effects when all fields are accessed using the same pattern
     *
     * some more dev I am not willing to do right now :)
     */


    list effects = NIL;

    /* filter out relevant effects depending on operation mode */
    FOREACH(EFFECT,e,rw_effects) {
        if ((dma_load_p(s) || dma_allocate_p(s) || dma_deallocate_p(s))
                && action_read_p(effect_action(e)))
            effects=CONS(EFFECT,e,effects);
        else if ((dma_store_p(s)  || dma_allocate_p(s) || dma_deallocate_p(s))
                && action_write_p(effect_action(e)))
            effects=CONS(EFFECT,e,effects);
    }
    effects=gen_nreverse(effects);


    /* if we failed to provide a fine_grain_analysis, we can still rely on the definition region to over approximate the result
    */
    if(!fine_grain_analysis) {
        FOREACH(EFFECT,eff,rw_effects) {
            if(entity_pointer_p(reference_variable(region_any_reference(eff)))
                    && ! std_file_effect_p(eff)) {
                print_effect(eff);
                pips_user_error("pointers wreak havoc with isolate_statement\n");
            }
            descriptor d = effect_descriptor(eff);
            if(descriptor_convex_p(d)) {
                Psysteme sc_old = descriptor_convex(d);
                Psysteme sc_new = entity_declaration_sc(reference_variable(region_any_reference(eff)));
                sc_intersection(sc_new,sc_new,predicate_system(transformer_relation(tr)));
                sc_intersection(sc_old,sc_old,predicate_system(transformer_relation(tr)));
                sc_old=sc_normalize2(sc_old);
                sc_new=sc_normalize2(sc_new);
                if(!sc_equal_p(sc_old,sc_new)) {
                    sc_free(sc_old);
                    descriptor_convex(d)=sc_new;
                    effect_approximation_tag(eff)=is_approximation_may;
                }
            }
        }
    }

    /* handle the may approximations here: if the approximation is may,
     * we have to load the data, otherwise the store may store
     * irrelevant data
     */
    if (dma_load_p(s) || dma_allocate_p(s) || dma_deallocate_p(s)) {
        /* first step is to check for may-write effects */
        list may_write_effects = NIL;
        FOREACH(EFFECT,e,rw_effects) {
            if(approximation_may_p(effect_approximation(e)) &&
                    action_write_p(effect_action(e)) ) {
                effect fake = copy_effect(e);
                action_tag(effect_action(fake))=is_action_read;
                may_write_effects=CONS(EFFECT,fake,may_write_effects);
            }
        }
        may_write_effects=gen_nreverse(may_write_effects);
        /* then we will merge these effects with those
         * that were already gathered
         * because we are manipulating lists, it is not very efficient
         * but there should not be that many effects anyway
         */
        FOREACH(EFFECT,e_new,may_write_effects) {
            bool merged = false; // if we failed to merge e_new in effects, we just add it to the list */
            for(list iter=effects;!ENDP(iter);POP(iter)){
                effect * e_origin = (effect*)REFCAR(iter); // get a reference to change it in place if needed
                if(same_entity_p(
                            effect_any_entity(*e_origin),
                            effect_any_entity(e_new))) {
                    merged=true;
                    region tmp = extended_regions_must_convex_hull(*e_origin,e_new);
                    // there should be a free there, but it fails
                    *e_origin=tmp;
                }
            }
            /* no data was copy-in, add this effect */
            if(!merged) {
                effects=CONS(EFFECT,copy_effect(e_new),effects);
            }
        }
        gen_full_free_list(may_write_effects);
        effects=gen_nreverse(effects);
    }



    /* builds out transfer from gathered effects */
    list statements = NIL;
    FOREACH(EFFECT,eff,effects) {
        statement the_dma = statement_undefined;
        reference r = effect_any_reference(eff);
        entity re = reference_variable(r);

        if(entity_abstract_location_p(re)) {
            pips_user_error("We can't handle abstract locations here. Please try to "
                    "avoid using pointer or activate some pointer analyzes.\n");
        }


        struct dma_pair * val = (struct dma_pair *) hash_get(e2e, re);

        if( val == HASH_UNDEFINED_VALUE || (val->s != s) ) {
            if( !io_effect_p(eff) && !std_file_effect_p(eff) &&
                    (!entity_scalar_p(re) || get_bool_property("KERNEL_LOAD_STORE_SCALAR"))
              ) {
                list /*of dimensions*/ the_dims = NIL,
                     /*of expressions*/the_offsets = NIL;
                region_to_dimensions(eff,tr,&the_dims,&the_offsets,condition);

                entity eto;
                if(val == HASH_UNDEFINED_VALUE) {

                    /* initialized with NULL value */
                    expression init = int_to_expression(0);

                    /* Replace the reference to the array re to *eto: */
                    type re_type = ultimate_array_type(entity_type(re));
                    basic re_basic = basic_undefined;
                    pips_assert("the type of the considered effect is expected to be variable",
                            type_variable_p(re_type));
                    re_basic = variable_basic(type_variable(re_type));
                    if(basic_pointer_p(re_basic)) {
                        type pointed_type = basic_pointer(re_basic);
                        re_basic = variable_basic(type_variable(pointed_type));
                    }
                    entity renew = make_new_array_variable(get_current_module_entity(),copy_basic(re_basic),the_dims);
                    entity declaring_module =
                        get_current_module_entity();
                    // PIER Here we need to add a P4A variable prefix to the name to help
                    // p4a postprocessing
                    string str = strdup (concatenate (prefix,entity_user_name(re), suffix, NULL));
                    eto = make_temporary_pointer_to_array_entity_with_prefix(str,renew,declaring_module,init);
                    free (str);
                    AddLocalEntityToDeclarations(eto,get_current_module_entity(),stat);
                    isolate_patch_entities(stat,re,eto,the_offsets);

                    val=malloc(sizeof(*val));
                    val->new_ent=eto;
                    val->s=s;
                    hash_put(e2e,re,val);
                }
                else {
                    eto = val->new_ent;
                    val->s=s;/*to avoid duplicate*/
                }
                the_dma = instruction_to_statement(make_instruction_call(dimensions_to_dma(eff,eto,the_dims,the_offsets,s)));
                statements=CONS(STATEMENT,the_dma,statements);
            }
        }
    }
    gen_free_list(effects);
    gen_full_free_list(rw_effects);
    if (statements == NIL)
        return statement_undefined;
    else
        return make_block_statement(statements);
}

typedef struct {
    char * callee_module_name;
    list regions_to_extend;
    bool ok;
} param_t;

static bool do_check_isolate_statement_preconditions_on_call(call c, param_t* p) {
    if(same_string_p(entity_local_name(call_function(c)),p->callee_module_name)) {
        /* get parent statement */
        statement s = (statement)gen_get_ancestor(statement_domain,c);
        pips_assert("found ancestor",!statement_undefined_p(s));
        /* get associated regions */
        list regions = load_cumulated_rw_effects_list(s);
        /* verify the conditions : no complex offset handling */
        FOREACH(REGION,reg,regions) {
            reference ref = region_any_reference(reg);
            entity eref = reference_variable(ref);
            list indices = reference_indices(ref);
            size_t nbdims;
            type ut = ultimate_type(entity_type(eref));
            if(type_pointer_on_struct_variable_p(ut))
                nbdims = gen_length(variable_dimensions(type_variable(ut)));
            else if(pointer_type_p(ut))
                nbdims = gen_length(indices);
            else
                nbdims = gen_length(variable_dimensions(type_variable(ut)));
            if(nbdims>1) { // unhandled
                pips_user_warning("Trying to isolate a statement with a call that touches a multi-dimensional array `%s'\n", entity_user_name(eref));
                p->ok=false;
            }
            /* we are left with a scalar or an unidimensional array */
            else if(nbdims==1 && !ENDP(indices) ) { // see if we can approximate the region with an offset at zero
                expression exp_phi1 = EXPRESSION(CAR(indices));
                entity phi1 = expression_to_entity(exp_phi1);
                Psysteme sc_reg = sc_dup(region_system(reg));
                sc_transform_eg_in_ineg(sc_reg);
                Pcontrainte lower,upper;
                constraints_for_bounds(phi1,
                        &sc_inegalites(sc_reg), &lower, &upper);
                if( !CONTRAINTE_UNDEFINED_P(upper))
                {
                    expression eupper = constraints_to_loop_bound(upper,phi1,false,entity_intrinsic(DIVIDE_OPERATOR_NAME));
                    normalized nupper = NORMALIZE_EXPRESSION(eupper);
                    if(normalized_linear_p(nupper)) {
                        /* we can do it! leave it for later */
                        p->regions_to_extend = CONS(ENTITY, eref, p->regions_to_extend);
                    }
                    else {
                        pips_user_warning("Failed to normalized the upper bound for accesses to array `%s'\n",entity_user_name(eref));
                        p->ok=false;
                    }
                    free_expression(eupper);

                }
                else {
                    pips_user_warning("Failed to find an upper bound for accesses to array `%s'\n",entity_user_name(eref));
                    p->ok=false;
                }
                contrainte_rm(lower);
                contrainte_rm(upper);
                sc_rm(sc_reg);
            }
            if(!p->ok) {
                gen_recurse_stop(0);
                return false;
            }
        }

        // verify the call does not references globals
        list summary_regions = effects_to_list((effects)db_get_memory_resource(DBR_SUMMARY_REGIONS, p->callee_module_name, true));
        FOREACH(REGION, sr, summary_regions) {
            reference ref = region_any_reference(sr);
            entity eref = reference_variable(ref);
            if(!entity_local_variable_p(eref,get_current_module_entity()) && ! formal_parameter_p(eref) ) {
                /* The following test could be refined, but it is OK right now if we can
                   override it with the following property when generating code for GPU
                   for example. */
                if (!get_bool_property("ISOLATE_STATEMENT_EVEN_NON_LOCAL") &&
                        !io_effect_p(sr) && !std_file_effect_p(sr) ) {
                    pips_user_error("Cannot handle with some effects on non local variables in isolate_statement\n");
                    /* Should not return from previous exception anyway... */
                }

            }
        }
    }

    return true;
}

static bool do_isolate_statement_preconditions_satisified_p(statement s)
{
    callees c = compute_callees(s);
    list thecallees = callees_callees(c);
    /* we have to make sure every callee access no global variables and that
     * the associated regions has no offset
     */
    param_t p = { .ok=true, .regions_to_extend=NIL };
    FOREACH(STRING, callee_module_name, thecallees) {
        p.callee_module_name=callee_module_name;
        gen_context_recurse(s,&p,call_domain,do_check_isolate_statement_preconditions_on_call,gen_null);
        if(!p.ok) break;
    }
    free_callees(c);
    if(p.ok) { // this mean that all checks are ok, but some patching is needed
        // maybe we have registered some regions to extend to avoid interprocedural offset management?
        list regions = load_cumulated_rw_effects_list(s);
        FOREACH(ENTITY,e,p.regions_to_extend) {
            for(list iter=regions;!ENDP(iter);POP(iter)){ // change the region in place, thus the refcar
                region * reg = (region*)REFCAR(iter);
                reference ref = region_any_reference(*reg);
                list indices = reference_indices(ref);
                if(!ENDP(indices)) { // only entities with indices are considered
                    entity eref = reference_variable(ref);
                    if(same_entity_p(eref,e)) { // we got the right entity
                        expression exp_phi1 = EXPRESSION(CAR(indices));
                        entity phi1 = expression_to_entity(exp_phi1);
                        Psysteme sc_reg = sc_dup(region_system(*reg));
                        sc_transform_eg_in_ineg(sc_reg);
                        Pcontrainte lower,upper;
                        constraints_for_bounds(phi1,
                                &sc_inegalites(sc_reg), &lower, &upper);
                        if( !CONTRAINTE_UNDEFINED_P(upper))
                        {
                            expression eupper = constraints_to_loop_bound(upper,phi1,false,entity_intrinsic(DIVIDE_OPERATOR_NAME));
                            normalized nupper = NORMALIZE_EXPRESSION(eupper);
                            if(normalized_linear_p(nupper)) {
                                Psysteme sc =sc_new();
                                /* add a constraint 0 <= phi1 and the upperbound constraint*/
                                sc_add_phi_equation(&sc, int_to_expression(0), 1, false, false);
                                sc_add_phi_equation(&sc, eupper, 1, false, true);

                                bool must = false;
                                if(!CONTRAINTE_UNDEFINED_P(lower)) {
                                    expression elower = constraints_to_loop_bound(lower,phi1,true,entity_intrinsic(DIVIDE_OPERATOR_NAME));
                                    intptr_t p;
                                    must=(expression_integer_value(elower,&p)&&p==0);
                                    free_expression(elower);
                                }
                                /* duplicate the region */
                                region copy = make_region(copy_reference(ref),
                                        copy_action(region_action(*reg)),
                                        must?make_approximation_exact():make_approximation_may(), sc);
                                /* add it to current region */
                                region nreg = extended_regions_must_convex_hull(*reg,copy);
                                //free_effect(*reg);
                                //free_effect(copy);
                                *reg=nreg;
                            }
                            else pips_internal_error("This case should have been filtered out by `do_check_isolate_statement_preconditions_on_call'");
                        }
                        else pips_internal_error("This case should have been filtered out by `do_check_isolate_statement_preconditions_on_call'");
                    }
                }
            }
        }
        /* prune out read effects on pointers ...
         * the assumption is that reading a pointer is not that important and most certainly comes from passing a pointer as parameter to a function
         * if the pointer itself is not written, it should be ok.
         * Well this indeed very optimistic ...
         */
        list rdup = gen_copy_seq(regions);
        FOREACH(REGION,r,rdup) {
            if(region_read_p(r) ) {
                reference ref = region_any_reference(r);
                if(entity_pointer_p(reference_variable(ref)) && 
                        ENDP(reference_indices(ref))) {
                    gen_remove_once(&regions,r);
                }
            }
        }
        update_cumulated_rw_effects_list(s,regions);
        gen_free_list(rdup);
        gen_free_list(p.regions_to_extend);
    }
    return p.ok;
}


/* perform statement isolation on statement @p s
 * that is make sure that all access to variables in @p s
 * are made either on private variables or on new entities declared on a new memory space. The @p prefix is used as a prefix to new entities' name.
 */
void do_isolate_statement(statement s, const char* prefix, const char* suffix) {
    bool fine_grain_analysis = true;
    statement allocates, loads, stores, deallocates;
    /* this hash table holds an entity to (entity + tag ) binding */
    hash_table e2e ;
    if(!do_isolate_statement_preconditions_satisified_p(s)) {
        pips_user_warning("isolated statement has callees, transfers will be approximated\n");
        fine_grain_analysis = false; // FIXME : This could be true most of the time, especially in Par4All !
    }
    e2e = hash_table_make(hash_pointer,HASH_DEFAULT_SIZE);
    expression condition = expression_undefined;
    allocates = effects_to_dma(s,dma_allocate,e2e,&condition,
            fine_grain_analysis,prefix,suffix);
    loads = effects_to_dma(s,dma_load,e2e,NULL,fine_grain_analysis,prefix,
            suffix);
    stores = effects_to_dma(s,dma_store,e2e,NULL,fine_grain_analysis,prefix,
            suffix);
    deallocates = effects_to_dma(s,dma_deallocate,e2e,NULL,fine_grain_analysis,
            prefix,suffix);
    HASH_MAP(k,v,free(v),e2e);
    hash_table_free(e2e);

    /* Add the calls now if needed, in the correct order: */
    if (loads != statement_undefined)
        insert_statement(s, loads,true);
    if (stores != statement_undefined)
        insert_statement(s, stores,false);
    if (deallocates != statement_undefined)
        insert_statement(s, deallocates,false);
    if (allocates != statement_undefined)
        insert_statement(s, allocates,true);
    /* guard the whole block by according conditions */
    if(!expression_undefined_p(condition)) {

        /* prends ton couteau suisse et viens jouer avec moi dans pips */
        pips_assert("statement is a block",statement_block_p(s));
        for(list prev=NIL,iter=statement_block(s);!ENDP(iter);POP(iter)) {
            if(declaration_statement_p(STATEMENT(CAR(iter)))) prev=iter;
            else {
                pips_assert("there should be at least one declaration inserted by isolate_statement\n",!ENDP(prev));
                statement cond = 
                    instruction_to_statement(
                            make_instruction_test(
                                make_test(
                                    condition,
                                    make_block_statement(iter),
                                    make_empty_statement()
                                    )
                                )
                            );
                CDR(prev)=CONS(STATEMENT,cond,NIL);
                break;
            }
        }
    }
}

typedef struct {
    entity old;
    entity new;
    list offsets;
} isolate_param;

/** 
 * replace reference @p r on entity @p p->old  by a reference on entity @p p->new with offsets @p p->offsets
 */
static void isolate_patch_reference(reference r, isolate_param * p)
{
    if(same_entity_p(reference_variable(r),p->old))
    {
        list offsets = p->offsets;
        list indices = reference_indices(r);
        FOREACH(EXPRESSION,index,indices)
        {
            if(!ENDP(offsets)) {
                expression offset = EXPRESSION(CAR(offsets));
                if(!expression_reference_p(offset) || !entity_field_p(reference_variable(expression_reference(offset)))){
                    update_expression_syntax(index,
                            make_syntax_call(
                                make_call(
                                    entity_intrinsic(MINUS_OPERATOR_NAME),
                                    make_expression_list(
                                        copy_expression(index),
                                        copy_expression(offset)
                                        )
                                    )
                                )
                            );
                }
                POP(offsets);
            }
        }
        /* build up the replacement */
        syntax syn = 
            make_syntax_call(
                    make_call(
                        entity_intrinsic(DEREFERENCING_OPERATOR_NAME),
                        CONS(EXPRESSION,entity_to_expression(p->new),NIL)
                        )
                    );

        /* it is illegal to create a subscript without indices
         * quoting RK, at the airport back from SC 2010 */
        syntax snew = ENDP(indices) ?
            syn:
            make_syntax_subscript(
                    make_subscript(syntax_to_expression(syn),indices)
                    );
        expression parent = (expression)gen_get_ancestor(expression_domain,r);
        expression_syntax(parent)=syntax_undefined;
        update_expression_syntax(parent,snew);

    }
}

/** 
 * run isolate_patch_entities on all declared entities from @p s
 */
static void isolate_patch_statement(statement s, isolate_param *p)
{
    FOREACH(ENTITY,e,statement_declarations(s))
    {
        if(!value_undefined_p(entity_initial(e)))
            isolate_patch_entities(entity_initial(e),p->old,p->new,p->offsets);
    }
}

/** 
 * replace all references on entity @p old by references on entity @p new and adds offset @p offsets to its indices
 */
void isolate_patch_entities(void * where,entity old, entity new,list offsets)
{
    isolate_param p = { old,new,offsets };
    gen_context_multi_recurse(where,&p,
            reference_domain,gen_true,isolate_patch_reference,
            statement_domain,gen_true,isolate_patch_statement,
            0);
}


/* replaces expression @p e by its upper or lower bound under preconditions @p tr
 * @p is_upper is used to choose among lower and upperbound*/
static void bounds_of_expression(expression e, transformer tr,bool is_upper)
{
    intptr_t lbound, ubound;
    if(precondition_minmax_of_expression(e,tr,&lbound,&ubound))
    {
        free_syntax(expression_syntax(e));
        free_normalized(expression_normalized(e));
        expression new = int_to_expression(is_upper ? ubound : lbound);
        expression_syntax(e)=expression_syntax(new);
        expression_normalized(e)=expression_normalized(new);
        expression_syntax(new)=syntax_undefined;
        expression_normalized(new)=normalized_undefined;
        free_expression(new);
    }
}

/* replaces expression @p e by its upperbound under preconditions @p tr*/
static void upperbound_of_expression(expression e, transformer tr)
{
    bounds_of_expression(e,tr,true);
}

/* replaces expression @p e by its lowerbound under preconditions @p tr*/
static void lowerbound_of_expression(expression e, transformer tr)
{
    bounds_of_expression(e,tr,false);
}



/**
 * generate a list of dimensions @p dims and of offsets @p from a region @p r
 * for example if r = a[phi0,phi1] 0<=phi0<=2 and 1<=phi1<=4
 * we get dims = ( (0,3), (0,4) )
 * and offsets = ( 0 , 1 )
 * if @p exact is set to false, we are allowed to give an upper bound to the dimensions
 *
 * if at least one of the resulting dimension can be 0 (according to preconditions)
 * @p dimension_may_be_null is set to true
 *
 * @return false if we were enable to gather enough informations
 */
bool region_to_minimal_dimensions(region r, transformer tr, list * dims, list *offsets,bool exact, expression *condition)
{
    pips_assert("empty parameters\n",ENDP(*dims)&&ENDP(*offsets));
    reference ref = region_any_reference(r);
    bool fortran_p = fortran_module_p(get_current_module_entity());
    for(list iter = reference_indices(ref);!ENDP(iter); POP(iter))
    {
        expression index = EXPRESSION(CAR(iter));
        Variable phi = expression_to_entity(index);
        if(variable_phi_p((entity)phi)) {
            Psysteme sc = sc_dup(region_system(r));
            sc_transform_eg_in_ineg(sc);
            Pcontrainte lower,upper;
            constraints_for_bounds(phi, &sc_inegalites(sc), &lower, &upper);
            if( !CONTRAINTE_UNDEFINED_P(lower) && !CONTRAINTE_UNDEFINED_P(upper))
            {
                /* this is a constant : the dimension is 1 and the offset is the bound */
                if(bounds_equal_p(phi,lower,upper))
                {
#if 1
                    expression bound = constraints_to_loop_bound(lower,phi,true,entity_intrinsic(DIVIDE_OPERATOR_NAME));
                    if (fortran_p) {
                        // in fortran remove -1 to the bound since index 1 is
                        // offset 0
                        bound = add_integer_to_expression (bound, -1);
                    }
                    *dims=CONS(DIMENSION,make_dimension(int_to_expression(0),int_to_expression(0)),*dims);
                    *offsets=CONS(EXPRESSION,bound,*offsets);
#endif
                }
                /* this is a range : the dimension is eupper-elower +1 and the offset is elower */
                else
                {

                    expression elower = constraints_to_loop_bound(lower,phi,true,entity_intrinsic(DIVIDE_OPERATOR_NAME));
                    expression eupper = constraints_to_loop_bound(upper,phi,false,entity_intrinsic(DIVIDE_OPERATOR_NAME));
                    simplify_minmax_expression(elower,tr);
                    simplify_minmax_expression(eupper,tr);
                    expression offset = copy_expression(elower);
                    if (fortran_p) {
                        // in fortran remove -1 to the offset since index 1 is
                        // offset 0
                        offset = add_integer_to_expression (offset, -1);
                    }

                    bool compute_upperbound_p = 
                        !exact && (expression_minmax_p(elower)||expression_minmax_p(eupper));
                    expression dim = make_op_exp(MINUS_OPERATOR_NAME,eupper,elower);
                    if(compute_upperbound_p)
                        upperbound_of_expression(dim,tr);

                    /* sg : check if lower bound can be 0, in that case issue a ward */
                    if(condition!=0) {
                        expression lowerbound = copy_expression(dim);
                        lowerbound_of_expression(lowerbound,tr);
                        intptr_t lowerbound_value;
                        if(!expression_integer_value(lowerbound,&lowerbound_value) ||
                                lowerbound_value<=0) {
                            expression thetest = 
                                MakeBinaryCall(
                                        entity_intrinsic(GREATER_THAN_OPERATOR_NAME),
                                        copy_expression(dim),
                                        int_to_expression(0)
                                        );
                            if(expression_undefined_p(*condition))
                                *condition=thetest;
                            else
                                *condition=MakeBinaryCall(
                                        entity_intrinsic(AND_OPERATOR_NAME),
                                        *condition,
                                        thetest
                                        );
                        }
                    }

                    *dims=CONS(DIMENSION,
                            make_dimension(
                                int_to_expression(0),
                                dim
                                ),*dims);
                    *offsets=CONS(EXPRESSION,offset,*offsets);
                }
            }
            else {
                pips_user_warning("failed to analyse region\n");
                sc_free(sc);
                /* reset state */
                gen_full_free_list(*dims); *dims=NIL;
                gen_full_free_list(*offsets); *offsets=NIL;
                return false;
            }
            sc_free(sc);
        }
        /* index is a field ... */
        else { /* and the last field, store it as an extra dimension */
#if 0
            *dims=CONS(DIMENSION,
                    make_dimension(
                        int_to_expression(0),
                        int_to_expression(0)
                        ),*dims);
            *offsets=CONS(EXPRESSION,copy_expression(index),*offsets);
#else
            break;
#endif
        }
    }
    *dims=gen_nreverse(*dims);
    *offsets=gen_nreverse(*offsets);
    return true;
}

/** 
 * 
 * @return region from @p regions on entity @p e
 */
region find_region_on_entity(entity e,list regions)
{
    FOREACH(REGION,r,regions)
        if(same_entity_p(e,reference_variable(region_any_reference(r)))) return r;
    return region_undefined;
}


/** 
 * @return a range suitable for iteration over all the elements of dimension @p d
 */
range dimension_to_range(dimension d)
{
    return make_range(
            copy_expression(dimension_lower(d)),
            copy_expression(dimension_upper(d)),
            int_to_expression(1));
}

/** 
 * @return a statement holding the loop necessary to initialize @p new from @p old,
 * knowing the dimension of the isolated entity @p dimensions and its offsets @p offsets and the direction of the transfer @p t
 */

bool
isolate_statement(const char* module_name)
{
    /* init stuff */
    set_current_module_entity(module_name_to_entity( module_name ));
    set_current_module_statement((statement) db_get_memory_resource(DBR_CODE, module_name, true) );
    set_cumulated_rw_effects((statement_effects)db_get_memory_resource(DBR_REGIONS, module_name, true));
    module_to_value_mappings(get_current_module_entity());
    set_precondition_map( (statement_mapping) db_get_memory_resource(DBR_PRECONDITIONS, module_name, true) );


    /* get user input */
    const char* stmt_label=get_string_property("ISOLATE_STATEMENT_LABEL");
    statement statement_to_isolate;
    if(!empty_string_p(stmt_label)) {
      statement_to_isolate = find_statement_from_label_name(get_current_module_statement(),get_current_module_name(),stmt_label);
    } else {
      statement_to_isolate = get_current_module_statement();
    }
    /* and proceed */
    if(statement_undefined_p(statement_to_isolate))
        pips_user_error("statement labeled '%s' not found\n",stmt_label);
    else
    {
        const char* prefix =  get_string_property ("ISOLATE_STATEMENT_VAR_PREFIX");
        const char* suffix =  get_string_property ("ISOLATE_STATEMENT_VAR_SUFFIX");
        pips_debug (5, "isolate_statement prefix : %s\n", prefix);
        pips_debug (5, "isolate_statement suffix : %s\n", suffix);
        do_isolate_statement(statement_to_isolate, prefix, suffix);
    }



    /* validate */
    module_reorder(get_current_module_statement());
    DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name,get_current_module_statement());
    DB_PUT_MEMORY_RESOURCE(DBR_CALLEES, module_name, compute_callees(get_current_module_statement()));

    reset_current_module_entity();
    reset_current_module_statement();
    reset_cumulated_rw_effects();
    reset_precondition_map();
    free_value_mappings();

    return true;
}

