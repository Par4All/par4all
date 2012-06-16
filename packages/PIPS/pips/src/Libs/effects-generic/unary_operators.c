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
/* package generic effects :  Be'atrice Creusillet 5/97
 *
 * File: unary_operators.c
 * ~~~~~~~~~~~~~~~~~~~~~~~~
 *
 * This File contains generic unary operators for effects and lists of effects.
 *
 */

#include <stdio.h>
#include <string.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "ri-util.h"
#include "text-util.h"
#include "effects-util.h"
#include "misc.h"

#include "effects-convex.h"
#include "effects-generic.h"

void 
effects_map(list l_eff, void (*apply)(effect))
{
    MAP(EFFECT, eff, {apply(eff);}, l_eff);      
}

list
effects_to_effects_map(list l_eff, effect (*pure_apply)(effect))
{
    list l_new = NIL;
    MAP(EFFECT, eff,
	l_new = CONS(EFFECT, pure_apply(eff), l_new),
	l_eff);
    return gen_nreverse(l_new);
}

void
effects_filter_map(list l_eff, bool (*filter)(effect), void (*apply)(effect))
{
    MAP(EFFECT, eff, {if (filter(eff)) apply(eff);}, l_eff);      
}

list
effects_to_effects_filter_map(list l_eff , bool (*filter)(effect),
			      effect (*pure_apply)(effect))
{
    list l_new = NIL;
    MAP(EFFECT, eff,
	if (filter(eff)) l_new = CONS(EFFECT, pure_apply(eff), l_new),
	l_eff);
    return gen_nreverse(l_new);
}

list 
effects_add_effect(list l_eff, effect eff)
{
    return gen_nconc(l_eff, CONS(EFFECT, eff, NIL));
}

list
effects_read_effects(list l_eff)
{
    list l_new = NIL;
    MAP(EFFECT, eff,
  if (effect_read_p(eff)) l_new = CONS(EFFECT, eff, l_new),
  l_eff);
    return gen_nreverse(l_new);
}

list
effects_store_effects(list l_eff)
{
    list l_new = NIL;
    MAP(EFFECT, eff,
  if (store_effect_p(eff)) l_new = CONS(EFFECT, eff, l_new),
  l_eff);
    return gen_nreverse(l_new);
}

list
effects_write_effects(list l_eff)
{
    list l_new = NIL;
    MAP(EFFECT, eff,
	if (effect_write_p(eff)) l_new = CONS(EFFECT, eff, l_new),
	l_eff);
    return gen_nreverse(l_new);
}

/* At least one of the effects in l_eff is a write */
bool effects_write_at_least_once_p(list l_eff)
{
  bool write_once_p = false;

  FOREACH(EFFECT, eff, l_eff) {
    if (effect_write_p(eff)) {
      write_once_p = true;
      break;
    }
  }
  return write_once_p;
}

list
effects_read_effects_dup(list l_eff)
{
    list l_new = NIL;
    MAP(EFFECT, eff,
	if (effect_read_p(eff))
           l_new = CONS(EFFECT, (*effect_dup_func)(eff), l_new),
	l_eff);
    return gen_nreverse(l_new);
}

list
effects_write_effects_dup(list l_eff)
{
    list l_new = NIL;
    MAP(EFFECT, eff,
	if (effect_write_p(eff))
  	   l_new = CONS(EFFECT, (*effect_dup_func)(eff), l_new),
	l_eff);
    return gen_nreverse(l_new);
}

effect
effect_nop(effect eff)
{
    return eff;
}

list
effects_nop(list l_eff)
{
    return l_eff;
}

void
effect_to_may_effect(effect eff)
{
    effect_approximation_tag(eff) = is_approximation_may;
}

void
effects_to_may_effects(list l_eff)
{
    effects_map(l_eff, effect_to_may_effect);
}

void
effect_to_must_effect(effect eff)
{
    effect_approximation_tag(eff) = is_approximation_exact;
}

void
effects_to_must_effects(list l_eff)
{
    effects_map(l_eff, effect_to_must_effect);
}

void
effect_to_write_effect(effect eff)
{
  effect_action_tag(eff) = is_action_write;
/*   if (effect_read_p(eff)) */
/*     { */
/*       action ac = make_action_read(copy_action_kind(action_read(effect_action(eff)))); */
/*       free_action(effect_action(eff)); */
/*       effect_action(eff) = ac; */
/*     } */
}

void
effects_to_write_effects(list l_eff)
{
    effects_map(l_eff, effect_to_write_effect);
}

void
effect_to_read_effect(effect eff)
{
  effect_action_tag(eff) = is_action_read;
/*   if (effect_write_p(eff)) */
/*     { */
/*       action ac = make_action_read(copy_action_kind(action_write(effect_action(eff)))); */
/*       free_action(effect_action(eff)); */
/*       effect_action(eff) = ac; */
/*     } */
}

void
effects_to_read_effects(list l_eff)
{
    effects_map(l_eff, effect_to_read_effect);
}
void
array_effects_to_may_effects(list l_eff)
{
    FOREACH(EFFECT, eff, l_eff)
	{
	    if (!effect_scalar_p(eff))
            effect_to_may_effect(eff);
	}
}

/* returned list as no sharing with parameters */
list
effects_dup_without_variables(list l_eff, list l_var)
{
    list l_res = NIL;
    
    FOREACH(EFFECT, eff,l_eff)
    {
        if (gen_find_eq(effect_entity(eff), l_var) == entity_undefined)
        {
            l_res = CONS(EFFECT, (*effect_dup_func)(eff), l_res);
        }
        else
            pips_debug(7, "Effect on variable %s removed\n",
                    entity_local_name(effect_entity(eff)));
    }
    return gen_nreverse(l_res);
}


list
effects_dup(list l_eff)
{
    list l_new = NIL;
    list ec = list_undefined;

    ifdebug(8) {
      effects e = make_effects(l_eff);
      pips_assert("input effects are consistent", effects_consistent_p(e));
      effects_effects(e) = NIL;
      free_effects(e);
    }

    for(ec = l_eff; !ENDP(ec); POP(ec)) {
      effect eff = EFFECT(CAR(ec));

      /* build last to first */
      l_new = CONS(EFFECT, (*effect_dup_func)(eff), l_new);
    }

    /* and the order is reversed */
    l_new =  gen_nreverse(l_new);

    ifdebug(8) {
      effects e = make_effects(l_new);
      pips_assert("input effects are consistent", effects_consistent_p(e));
      effects_effects(e) = NIL;
      free_effects(e);
    }

    return l_new;
}

void
effect_free(effect eff)
{
    (*effect_free_func)(eff);
}

void
effects_free(list l_eff)
{
    MAP(EFFECT, eff,
	{(*effect_free_func)(eff);},
	l_eff);
    gen_free_list(l_eff);
}

/* list effect_to_nil_list(effect eff)
 * input    : an effect
 * output   : an empty list of effects
 * modifies : nothing
 * comment  : 
 */
list effect_to_nil_list(effect eff __attribute__((__unused__)))
{
    return(NIL);
}

/** frees the input effect and returns a NIL list
 */
list effect_to_nil_list_and_free(effect eff)
{
  effect_free(eff);
  return(NIL);
}

/* list effects_to_nil_list(eff)
 * input    : an effect
 * output   : an empty list of effects
 * modifies : nothing
 * comment  : 	
 */
list effects_to_nil_list(effect eff1  __attribute__((__unused__)), effect eff2 __attribute__((__unused__))   )
{
    return(NIL);
}

list effect_to_list(effect eff)
{
    return(CONS(EFFECT,eff,NIL));
}

list effect_to_may_effect_list(effect eff)
{
    effect_approximation_tag(eff) = is_approximation_may;
    return(CONS(EFFECT,eff,NIL));
}

list effects_to_written_scalar_entities(list l_eff)
{
  list l_ent = NIL;
  FOREACH(EFFECT, eff, l_eff)
    {
      if (store_effect_p(eff) && effect_write_p(eff) && effect_scalar_p(eff))
	l_ent = CONS(ENTITY, effect_entity(eff), l_ent);
    }
  return l_ent;
}
/***********************************************************************/
/* UNDEFINED UNARY OPERATORS                                           */
/***********************************************************************/

/* Composition with transformers */

list
effects_undefined_composition_with_transformer(list l_eff __attribute__((__unused__)), transformer trans __attribute__((__unused__)))
{
    return list_undefined;
}


list effects_composition_with_transformer_nop(list l_eff,
					      transformer trans __attribute__((__unused__)))
{
  return l_eff;
}




/* Composition with preconditions */

list
effects_undefined_composition_with_preconditions(list l_eff __attribute__((__unused__)), transformer trans __attribute__((__unused__)))
{
    return list_undefined;
}

list
effects_composition_with_preconditions_nop(list l_eff, transformer trans __attribute__((__unused__)))
{
  return l_eff;
}

/* Union over a range */

descriptor
loop_undefined_descriptor_make(loop l __attribute__((__unused__)))
{
    return descriptor_undefined;
}

list 
effects_undefined_union_over_range(
    list l_eff __attribute__((__unused__)), entity index __attribute__((__unused__)), range r __attribute__((__unused__)), descriptor d __attribute__((__unused__)))
{
    return list_undefined;
}

list effects_union_over_range_nop(list l_eff,
				  entity index __attribute__((__unused__)),
				  range r __attribute__((__unused__)),
				  descriptor d __attribute__((__unused__)))
{
  return l_eff;
}


list
effects_undefined_descriptors_variable_change(list l_eff __attribute__((__unused__)),
					      entity orig_ent __attribute__((__unused__)),
					      entity new_ent __attribute__((__unused__)))
{
    return list_undefined;
}

list
effects_descriptors_variable_change_nop(list l_eff, entity orig_ent __attribute__((__unused__)),
					      entity new_ent __attribute__((__unused__)))
{
    return l_eff;
}


descriptor
effects_undefined_vector_to_descriptor(Pvecteur v __attribute__((__unused__)))
{
    return descriptor_undefined;
}

list 
effects_undefined_loop_normalize(list l_eff __attribute__((__unused__)),
				 entity index __attribute__((__unused__)),
				 range r __attribute__((__unused__)),
				 entity *new_index, 
				 descriptor range_descriptor __attribute__((__unused__)),
				 bool descriptor_update_p __attribute__((__unused__)))
{
  *new_index = entity_undefined;
  return list_undefined; 
}

list 
effects_loop_normalize_nop(list l_eff, 
			   entity index __attribute__((__unused__)), 
			   range r __attribute__((__unused__)),
			   entity *new_index __attribute__((__unused__)), 
			   descriptor range_descriptor __attribute__((__unused__)),
			   bool descriptor_update_p __attribute__((__unused__)))
{
    return l_eff; 
}

list /* of nothing */
db_get_empty_list(string name)
{
    pips_debug(5, "getting nothing for %s\n", name);
    return NIL;
}


/* adding special dimensions */

/* void effect_add_dereferencing_dimension(effect eff)
 * input    : an effect
 * output   : nothing
 * modifies : the effect eff.
 * comment  : adds a last dimension  to represent an effect on the memory
 *            location pointed by the reference of the initial effect.
 *            also modifies the descriptor if there is one
 */
void effect_add_dereferencing_dimension(effect eff)
{

  if (!entity_heap_location_p(effect_entity(eff))
      && !entity_flow_or_context_sentitive_heap_location_p(effect_entity(eff))
      && effect_abstract_location_p(eff))
    {

      pips_debug(8, "abstract location \n");
      cell eff_c = effect_cell(eff);
      if (!anywhere_effect_p(eff))
	{
	  /* change for an anywhere effect. More work could be done here
	     in case of a typed abstract location
	  */
	  entity anywhere_ent = entity_all_locations();
	  if (cell_preference_p(eff_c))
	    {
	      /* it's a preference : we change for a reference cell */
	      free_cell(eff_c);
	      effect_cell(eff) = make_cell_reference(make_reference(anywhere_ent, NIL));
	    }
	  else
	    {
	      reference_variable(cell_reference(eff_c)) = anywhere_ent;
	    }
	}
    }
  else
    {
      expression deref_exp = int_to_expression(0);
      pips_debug(8, "heap or concrete location \n");

      (*effect_add_expression_dimension_func)(eff, deref_exp);
      free_expression(deref_exp);
    }
  return;
}

static expression field_entity_to_expression(entity f)
{
  reference r = make_reference(f, NIL);
  expression e;
  syntax s = make_syntax(is_syntax_reference, r);
  e = make_expression(s, make_normalized(is_normalized_complex, UU));
  return e;
}

/* void effect_add_dereferencingfield_dimension(effect eff, int rank)
 * input    : an effect
 * output   : nothing
 * modifies : the effect eff.
 * comment  : adds a last dimension to represent an effect on the field
 *            of rank "rank" of the actual reference represented by the
 *            effect reference.	The dimension is added at the end of the
 *            effect reference dimensions.Also modifies the descriptor
 *            if the representation includes one.
 */

/**
   @brief adds a last dimension to the effect reference, which is a
          reference expression to the field entity.
   @param effect the input effect, whose reference is modified
   @param field the input field entity
 */
void effect_add_field_dimension(effect eff, entity field)
{
  cell eff_c = effect_cell(eff);

  pips_debug_effect(8, "begin with effect :\n", eff);
  pips_debug(8, "and field: %s\n", entity_name(field));

  if (!entity_flow_or_context_sentitive_heap_location_p(effect_entity(eff))
			      && effect_abstract_location_p(eff))
    {
      if (!anywhere_effect_p(eff))
	{
	  /* change for an anywhere effect. More work could be done here
	     in case of a typed abstract location
	  */
	  entity anywhere_ent = entity_all_locations();
	  if (cell_preference_p(eff_c))
	    {
	      /* it's a preference : we change for a reference cell */
	      free_cell(eff_c);
	      effect_cell(eff) = make_cell_reference(make_reference(anywhere_ent, NIL));
	    }
	  else
	    {
	      reference_variable(cell_reference(eff_c)) = anywhere_ent;
	    }
	}
    }
  else
    {
      reference ref;
      if (cell_preference_p(eff_c))
	{
	  /* it's a preference : we change for a reference cell */
	  pips_debug(8, "It's a preference\n");
	  ref = copy_reference(preference_reference(cell_preference(eff_c)));
	  free_cell(eff_c);
	  effect_cell(eff) = make_cell_reference(ref);
	}
      else
	{
	  /* it's a reference : let'us modify it */
	  ref = cell_reference(eff_c);
	}

      reference_indices(ref) = gen_nconc(reference_indices(ref),
					 CONS(EXPRESSION,
					      field_entity_to_expression(field),
					      NIL));
    }
  pips_debug_effect(8, "end with effect :\n",eff);

  return;
}

/***********************************************************************/
/* FILTERING DECLARATIONS                                              */
/***********************************************************************/

/** filter the input effects using the input declaration - matching
    effects with no dereferencements are skipped - effects with
    dereferencements are translated using the declaration initial
    value if it exists, or are translated to an abstract location
    effect (currently anywhere) otherwise.

    @param l_eff is the input effect list, it is freed by the function
    to avoid copying the potentially numerous effects which are not
    concerned by the declaration.

    @param decl is an entity.

    usage: l_new_eff = filter_effects_with_declaration(l_eff, decl)
 */
list filter_effects_with_declaration(list l_eff, entity decl)
{
  list l_res = NIL;
  storage decl_s = entity_storage(decl);

  ifdebug(8)
    {
      type ct = entity_basic_concrete_type(decl);
      pips_debug(8, "dealing with entity : %s with type %s\n",
		 entity_local_name(decl),words_to_string(words_type(ct,NIL,false)));
    }

  if (storage_ram_p(decl_s)
      /* static variable declaration has no effect, even in case of initialization. */
      && !static_area_p(ram_section(storage_ram(decl_s)))
      && type_variable_p(entity_type(decl)))
    {
      value v_init = entity_initial(decl);
      expression exp_init = expression_undefined;
      if(value_expression_p(v_init))
	exp_init = value_expression(v_init);

      // We must first eliminate effects on the declared variable
      // except if it is a static or extern variable.
      // or use the initial value to translate them to the preceding memory state
      // We should take care of the transformer too for convex effects.
      // But which transformer ? Is the statement transfomer OK?
      // or do we need to use the transformer for each variable initialization ?
      // The last statement is probably the truth, but this is not what is currently implemented

      FOREACH(EFFECT, eff, l_eff)
	{
	  reference eff_ref = effect_any_reference(eff);
	  entity eff_ent = reference_variable(eff_ref);

	  pips_debug_effect(8,"dealing_with_effect: \n", eff);

	  if (eff_ent == decl)
	    {
	      pips_debug(8, "same entity\n");
	      if(ENDP(reference_indices(eff_ref)))
		{
		  // effect on the variable itself: no need to keep it
		  free_effect(eff);
		}
	      else
		/* here, it is a store effect */
		{
		  bool exact_p;
		  // no need to keep the effect if there is no pointer in the path of the effect
		  // or if it's a FILE* - well the latter is a hack, but with constant path
		  // effects this should not happen - BC
		  if (!effect_reference_contains_pointer_dimension_p(eff_ref, &exact_p)
		      || FILE_star_effect_reference_p(eff_ref))
		    {
		      free_effect(eff);
		    }
		  else
		    {
		      if(!expression_undefined_p(exp_init)) // there is an inital value
			{
			  // let us re-use an existing method even if it's not the fastest method
			  // interprocedural translation and intra-procedural
			  // propagation will have to be re-packaged later
			  list l_tmp = CONS(EFFECT, eff, NIL);
			  list l_res_tmp;

			  if(c_effects_on_formal_parameter_backward_translation_func
			     == c_convex_effects_on_formal_parameter_backward_translation)
			    {
			      Psysteme sc = sc_new();
			      sc_creer_base(sc);
			      set_translation_context_sc(sc);
			    }

			  /* beware of casts : do not take them into account for the moment */
			  syntax s_init = expression_syntax(exp_init);
			  if (syntax_cast_p(s_init))
			    exp_init = cast_expression(syntax_cast(s_init));
			  l_res_tmp = (*c_effects_on_formal_parameter_backward_translation_func)
			    (l_tmp, exp_init, transformer_undefined);

			  if(c_effects_on_formal_parameter_backward_translation_func
			     == c_convex_effects_on_formal_parameter_backward_translation)
			    {
			      reset_translation_context_sc();
			    }

			  if (!exact_p) effects_to_may_effects(l_res_tmp);

			  l_res = (*effects_union_op)
			    (l_res_tmp, l_res, effects_same_action_p);
			  gen_full_free_list(l_tmp); /* also free the input effect */
			}
		      else
			{
			  pips_debug(8, "there is no inital_value\n");
			  if (get_constant_paths_p())
			    {
			      pips_debug(8, "-> anywhere effect \n");
			      list l_tmp = gen_nconc(effect_to_list(make_anywhere_effect(copy_action(effect_action(eff)))), l_res);
			      l_res = clean_anywhere_effects(l_tmp);
			      gen_full_free_list(l_tmp);
			    }
			  free_effect(eff);

			}
		    }
		} /* if( !ENP(reference_indices(eff_ref))) */
	    }
	  else
	    {
	      // keep the effect if it's an effect on another entity
	      l_res = CONS(EFFECT, eff, l_res);
	    }

	} /* FOREACH */
      gen_free_list(l_eff);
      l_res = gen_nreverse(l_res); // we try to preserve the order of the input list

    }
  return l_res;
}

/***********************************************************************/
/*                                               */
/***********************************************************************/

/**
  @input eff is an input effect describing a memory path
  @return a list of effects corresponding to effects on eff cell prefix pointer paths
*/
list effect_intermediary_pointer_paths_effect(effect eff)
{
  pips_debug_effect(5, "input effect :", eff);
  list l_res = NIL;
  reference ref = effect_any_reference(eff);
  entity e = reference_variable(ref);
  descriptor d = effect_descriptor(eff);
  list ref_inds = reference_indices(ref);
  int nb_phi_init = (int) gen_length(ref_inds);
  reference tmp_ref = make_reference(e, NIL);
  type t = entity_basic_concrete_type(e);
  bool finished = false;

  if (entity_abstract_location_p(e))
    {
      if (anywhere_effect_p(eff)
	  || null_pointer_value_cell_p(effect_cell(eff))
	  || undefined_pointer_value_cell_p(effect_cell(eff)))
	return CONS(EFFECT, copy_effect(eff), NIL);
    }

  while (!finished && !ENDP(ref_inds))
    {
      switch (type_tag(t))
	{

	case is_type_variable:
	  {
	    pips_debug(5," variable case\n");
	    basic b = variable_basic(type_variable(t));
	    size_t nb_dim = gen_length(variable_dimensions(type_variable(t)));

	    /* add to tmp_ref as many indices from ref as nb_dim */
	    for(size_t i = 0; i< nb_dim; i++, POP(ref_inds))
	      {
		reference_indices(tmp_ref) =
		  gen_nconc(reference_indices(tmp_ref),
			    CONS(EXPRESSION,
				 copy_expression(EXPRESSION(CAR(ref_inds))),
				 NIL));
	      }

	    if (basic_pointer_p(b))
	      {
		pips_debug(5," pointer basic\n");
		if (!ENDP(ref_inds))
		  {
		    pips_debug(5,"and ref_inds is not empty\n");
		    int nb_phi_tmp_eff = (int) gen_length(reference_indices(tmp_ref));
		    effect tmp_eff = effect_undefined;
		    if (descriptor_convex_p(d))
		      {
			if (nb_phi_tmp_eff == 0)
			  {
			    tmp_eff =
			      make_effect(make_cell_reference(copy_reference(tmp_ref)),
					  copy_action(effect_action(eff)),
					  copy_approximation(effect_approximation(eff)),
					  make_descriptor_convex(sc_new()));
			  }
			else
			  {
			    tmp_eff =
			      make_effect(make_cell_reference(copy_reference(tmp_ref)),
					  copy_action(effect_action(eff)),
					  copy_approximation(effect_approximation(eff)),
					  copy_descriptor(d));
			    for(int nphi = nb_phi_tmp_eff; nphi <= nb_phi_init; nphi++)
			      {
				extern void convex_region_descriptor_remove_ith_dimension(effect, int);
				convex_region_descriptor_remove_ith_dimension(tmp_eff, nphi);
			      }
			  }
		      }
		    else if (!descriptor_none_p(d))
		      {
			pips_internal_error("invalid effect descriptor kind\n");
		      }
		    else
		      {
			tmp_eff =
			      make_effect(make_cell_reference(copy_reference(tmp_ref)),
					  copy_action(effect_action(eff)),
					  copy_approximation(effect_approximation(eff)),
					  make_descriptor_none());
		      }

		    l_res = CONS(EFFECT, tmp_eff, l_res);
		    reference_indices(tmp_ref) =
		      gen_nconc(reference_indices(tmp_ref),
				CONS(EXPRESSION,
				     copy_expression(EXPRESSION(CAR(ref_inds))),
				     NIL));
		    POP(ref_inds);

		    type new_t = copy_type(basic_pointer(b));
		    /* free_type(t);*/
		    t = new_t;
		  }
		else
		  finished = true;
	      }
	    else if (basic_derived_p(b))
	      {
		pips_debug(5,"derived basic\n");
		type new_t = entity_basic_concrete_type(basic_derived(b));
		t = new_t;
	      }
	    else
	      finished = true;
	  }
	  break;
	case is_type_struct:
	case is_type_union:
	case is_type_enum:
	  {
	    pips_debug(5,"struct union or enum type\n");

	    /* add next index */
	    expression field_exp = EXPRESSION(CAR(ref_inds));
	    reference_indices(tmp_ref) =
	      gen_nconc(reference_indices(tmp_ref),
			CONS(EXPRESSION,
			     copy_expression(field_exp),
			     NIL));
	    POP(ref_inds);
	    entity field_ent = expression_to_entity(field_exp);
	    pips_assert("expression is a field entity\n", !entity_undefined_p(field_ent));
	    type new_t = entity_basic_concrete_type(field_ent);
	    t = new_t;
	  }
	  break;
	default:
	    pips_internal_error("unexpected type tag");

	}
    }
  return l_res;
}
