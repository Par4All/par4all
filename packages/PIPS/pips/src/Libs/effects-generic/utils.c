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
 * File: utils.c
 * ~~~~~~~~~~~~~~~~~
 *
 * This File contains various useful functions, some of which should be moved
 * elsewhere.
 *
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "ri-util.h"
#include "effects-util.h"
#include "misc.h"
#include "text-util.h"
#include "properties.h"
#include "preprocessor.h"

#include "effects-generic.h"


/********************************************************************* MISC */


/* Statement stack to walk on control flow representation */
DEFINE_GLOBAL_STACK(effects_private_current_stmt, statement)

/* Context stack to keep current context when walking on expressions */
DEFINE_GLOBAL_STACK(effects_private_current_context, transformer)

bool effects_private_current_context_stack_initialized_p()
{
    return (effects_private_current_context_stack != stack_undefined);
}
bool effects_private_current_stmt_stack_initialized_p()
{
    return (effects_private_current_stmt_stack != stack_undefined);
}

bool normalizable_and_linear_loop_p(entity index, range l_range)
{
    Value incr = VALUE_ZERO;
    normalized nub, nlb;
    expression e_incr = range_increment(l_range);
    normalized n;
    bool result = true;

    /* Is the loop index an integer variable */
    if (! entity_integer_scalar_p(index))
    {
	pips_user_warning("non integer scalar loop index %s.\n",
			  entity_local_name(index));
	result = false;
    }
    else
    {
	/* Is the loop increment numerically known ? */
	n = NORMALIZE_EXPRESSION(e_incr);
	if(normalized_linear_p(n))
	{
	    Pvecteur v_incr = normalized_linear(n);
	    if(vect_constant_p(v_incr))
		incr = vect_coeff(TCST, v_incr);
	}

	nub = NORMALIZE_EXPRESSION(range_upper(l_range));
	nlb = NORMALIZE_EXPRESSION(range_lower(l_range));

	result = value_notzero_p(incr) && normalized_linear_p(nub)
	    && normalized_linear_p(nlb);
    }

    return(result);
}

transformer
transformer_remove_variable_and_dup(transformer orig_trans, entity ent)
{
    transformer res_trans = transformer_undefined;

    if (orig_trans != transformer_undefined)
    {
	res_trans = copy_transformer(orig_trans);
	gen_remove(&transformer_arguments(res_trans), ent);
    }
    return(res_trans);
}


/**************************************** DESCRIPTORS (should not be there) */

static bool descriptor_range_p = false;

void
set_descriptor_range_p(bool b)
{
    descriptor_range_p = b;
}

bool
get_descriptor_range_p(void)
{
    return(descriptor_range_p);
}

descriptor
descriptor_inequality_add(descriptor d, Pvecteur v)
{
    if (!VECTEUR_UNDEFINED_P(v))
    {
	Psysteme sc = descriptor_convex(d);
	Pcontrainte contrainte = contrainte_make(v);
	sc_add_inegalite(sc, contrainte);
	sc->base = BASE_NULLE;
	sc_creer_base(sc);
	descriptor_convex_(d) = newgen_Psysteme(sc);
    }
    return d;
}

transformer
descriptor_to_context(descriptor d)
{
    transformer context;
    if (descriptor_none_p(d))
	context = transformer_undefined;
    else
    {
	Psysteme sc = sc_dup(descriptor_convex(d));
	context = make_transformer(NIL, make_predicate(sc));
    }
    return(context);
}

void
descriptor_variable_rename(descriptor d, entity old_ent, entity new_ent)
{
    if (descriptor_convex_p(d))
    {
	sc_variable_rename(descriptor_convex(d),
			   (Variable) old_ent,
			   (Variable) new_ent);
    }
}

descriptor
descriptor_append(descriptor d1, descriptor d2)
{
    if (descriptor_convex_p(d1) && descriptor_convex_p(d2))
    {
	Psysteme
	    sc1 = descriptor_convex(d1),
	    sc2 = descriptor_convex(d2);

	sc1 = sc_safe_append(sc1, sc2);
	descriptor_convex_(d1) = newgen_Psysteme(sc1);
    }
    else
	d1 = descriptor_undefined;
    return d1;
}

/***********************************************************************/
/*                                                                     */
/***********************************************************************/

transformer
load_undefined_context(statement s __attribute__ ((__unused__)) )
{
    return transformer_undefined;
}

transformer
load_undefined_transformer(statement s __attribute__ ((__unused__)) )
{
    return transformer_undefined;
}

bool
empty_context_test_false(transformer context __attribute__ ((__unused__)) )
{
    return false;
}

void
effects_computation_no_init(const char *module_name __attribute__ ((__unused__)) )
{
    return;
}

void
effects_computation_no_reset(const char *module_name __attribute__ ((__unused__)) )
{
    return;
}


/***********************************************************************/
/* FORMER effects/utils.c                                              */
/***********************************************************************/


string vect_debug_entity_name(e)
entity e;
{
    return((e == (entity) TCST) ? "TCST" : entity_name(e));
}


/* check that *some* read or write effects are on integer variables
 *
 * FI: this is almost always true because of array subscript expressions
 */
/*
  This function might be dangerous in case of abstract locations and
  with complex memory paths involving dereferencements, struct fields,
  and array indices. BC.
 */
bool some_integer_scalar_read_or_write_effects_p(cons * fx) {
  bool r_or_w_p = false;
  FOREACH(EFFECT, ef,fx) {
    entity e = reference_variable(effect_any_reference(ef));
	  if( store_effect_p(ef) && integer_scalar_entity_p(e)) {
	    r_or_w_p = true;
	    break;
	  }
  }
  return r_or_w_p;
}




/* Return true if a statement has an I/O effect in the effects
   list. */
bool statement_io_effect_p(statement s)
{
   bool io_effect_found = false;
   list effects_list = load_proper_rw_effects_list(s);

   /* If there is an I/O effects, the following entity should
      exist. If it does not exist, statement_io_effect_p() will return
      false anyway. */
   entity private_io_entity =
      FindEntity(IO_EFFECTS_PACKAGE_NAME,
			    IO_EFFECTS_ARRAY_NAME);

   MAP(EFFECT, an_effect,
       {
          reference a_reference = effect_any_reference(an_effect);
          entity a_touched_variable =
             reference_variable(a_reference);

          if (a_touched_variable == private_io_entity) {
             io_effect_found = true;
             break;
          }
       },
       effects_list);

   return io_effect_found;
}

/* Return true if the statement has a write effect on at least one of
   the argument (formal parameter) of the module and if the argument
   passing mode is by reference. Note that the return variable of a
   function is also considered here as a formal parameter. */
bool statement_has_a_formal_argument_write_effect_p(statement s)
{
   bool write_effect_on_a_module_argument_found = false;
   entity module = get_current_module_entity();
   list effects_list = load_proper_rw_effects_list(s);
   /* it might be better to check the parameter passing mode itself,
      via the module type */
   bool fortran_p = fortran_module_p(module);

   FOREACH(EFFECT, an_effect, effects_list) {
     entity a_variable = reference_variable(effect_any_reference(an_effect));
     bool formal_p = variable_is_a_module_formal_parameter_p(a_variable,
							module);
     bool return_variable_p = variable_return_p(a_variable);

     if (action_write_p(effect_action(an_effect))
	 && (return_variable_p
	     || (formal_p && fortran_p)
	     )
	 ) {
       write_effect_on_a_module_argument_found = true;
       break;
     }
   }       ;

   return write_effect_on_a_module_argument_found;

}



list /* of effect */ make_effects_for_array_declarations(list refs)
{
  list leff = NIL;
  effect eff;
  MAPL(l1,
  {
    reference ref = REFERENCE(CAR(l1));
    /* FI: in this context, I assume that eff is never returned undefined */
    /* FI: memory leak for action? */
    eff = (*reference_to_effect_func)(ref, make_action_read_memory(), true);
     if(effect_undefined_p(eff)) {
       pips_debug(8, "Reference to \"%s\" ignored\n",
		  entity_name(reference_variable(ref)));
     }
     else
       leff= CONS(EFFECT,eff,leff);
  },refs);

  gen_free_list(refs);
  return leff;
}





list summary_effects_from_declaration(const char *module_name __attribute__ ((unused)))
{
  list sel = NIL;
  //entity mod = module_name_to_entity(module_name);
  //list decls = code_declarations(value_code(entity_initial(mod)));
  list decls = current_module_declarations();
  list refs = list_undefined;

  refs = extract_references_from_declarations(decls);

  ifdebug(8) {
    pips_debug(8, "References from declarations:\n");
    MAP(REFERENCE, r, {
      pips_debug(8, "Reference for variable \"%s\"\n",
		 entity_name(reference_variable(r)));
      print_reference(r);
      fprintf(stderr, "\n");
    }, refs);
  }

  sel = make_effects_for_array_declarations(refs);

  return sel;
}

/* Debugging functions (to be augmented for GAPs) */
void dump_cell(cell c)
{
  fprintf(stderr, "Cell %p = (cell_tag=%u, reference=%p)\n", c, cell_tag(c),
	  cell_preference_p(c)? preference_reference(cell_preference(c)):cell_reference(c));
}

void dump_effect(effect e)
{
  cell c = effect_cell(e);
  action ac = effect_action(e);
  action_kind ak = action_read_p(ac)? action_read(ac):
    action_write(ac);
  approximation ap = effect_approximation(e);
  descriptor d = effect_descriptor(e);

  effect_consistent_p(e);
  fprintf(stderr, "Effect %p = (domain=%td, cell=%p, action=%p,"
	  " action_kind=%p, approximation=%p, descriptor=%p\n",
	  e, effect_domain_number(e), c, ac, ak, ap, d);
  dump_cell(c);
}

void dump_effects(list le)
{
  int i = 1;
  FOREACH(EFFECT, e, le) {
      fprintf(stderr, "%d ", i++);
      dump_effect(e);
    }
}

/* Check if a reference appears more than once in the effect list. If
   persistant_p is true, do not go thru persistant arcs. Else, use all
   references. */
bool effects_reference_sharing_p(list el, bool persistant_p) {
  bool sharing_p = false;
  list srl = NIL; /* store reference list */
  list erl = NIL; /* environment reference list */
  list tdrl = NIL; /* type declaration reference list */

  //list ce = list_undefined; /* current effect */
  //for (ce = el; !ENDP(ce); POP(ce)) {
  //effect e = EFFECT(CAR(ce));
  FOREACH(EFFECT,e,el) {
    cell c = effect_cell(e);
    reference r = reference_undefined;

    pips_assert("effect e is consistent", effect_consistent_p(e));

    if(persistant_p) {
      if(cell_reference_p(c))
        r = cell_reference(c);
    } else
      r = effect_any_reference(e);

    if(!reference_undefined_p(r)) {
      /* FI: I though about parametrizing thru a list, but this
       requires to conditional affectation, before and after each
       loop body. Hence the cut-and-paste. */
      if(store_effect_p(e)) {
        if(gen_in_list_p((void *)r, srl)) {
          fprintf(stderr, "this effect shares its reference with "
            "another effect in list srl\n");
          (*effect_prettyprint_func)(e);
          sharing_p = true;
          break;
        } else {
          srl = CONS(REFERENCE, r, srl);
        }
      } else if(environment_effect_p(e)) {
        if(gen_in_list_p((void *)r, erl)) {
          fprintf(stderr, "this effect shares its reference with "
            "another effect in list srl\n");
          (*effect_prettyprint_func)(e);
          sharing_p = true;
          break;
        } else {
          erl = CONS(REFERENCE, r, erl);
        }
      } else if(type_declaration_effect_p(e)) {
        if(gen_in_list_p((void *)r, tdrl)) {
          fprintf(stderr, "this effect shares its reference with "
            "another effect in list srl\n");
          (*effect_prettyprint_func)(e);
          sharing_p = true;
          break;
        } else {
          tdrl = CONS(REFERENCE, r, tdrl);
        }
      } else {
      }
    }
  }
  return sharing_p;
}

/************************ anywhere effects ********************/

/**
 @return a new anywhere effect.
 @param act is an action tag

 Allocate a new anywhere effect using generic function
 reference_to_effect_func, and the anywhere entity on demand
 which may not be best if we want to express it's aliasing with all
 module areas. In the later case, the anywhere entity should be
 generated by bootstrap and be updated each time new areas are
 declared by the parsers. I do not use a persistant anywhere
 reference to avoid trouble with convex-effect nypassing of the
 persistant pointer. (re-used from original non-generic function
 anywhere_effect.)

 Action a is integrated in the new effect (aliasing).
 
 FI: the type should always be passed and ignored according to
 ALIAS_ACROSS_TYPES
 */
effect make_anywhere_effect(action act)
{

  entity anywhere_ent = entity_all_locations();
  effect anywhere_eff = effect_undefined;

  anywhere_eff = (*reference_to_effect_func)
    (make_reference(anywhere_ent, NIL),
     //     copy_action(act), false);
     act, false);
  effect_to_may_effect(anywhere_eff);
  return anywhere_eff;
}

effect make_anywhere_write_memory_effect()
{

  return make_anywhere_effect(make_action_write_memory());
}

effect make_anywhere_read_memory_effect()
{

  return make_anywhere_effect(make_action_read_memory());
}

list make_anywhere_read_write_memory_effects()
{
  list l = NIL;
  l = CONS(EFFECT, make_anywhere_write_memory_effect(), l);
  l = CONS(EFFECT, make_anywhere_read_memory_effect(), l);
  return l;
}


/**
   remove duplicate anywhere effects and keep anywhere effects and
   effects not combinable with anywhere effects.

   @param l_eff is a list of effects
   @return a new list with no sharing with the initial effect list.

 */
list clean_anywhere_effects(list l_eff)
{
  list l_tmp;
  list l_res;
  bool anywhere_w_p = false;
  bool anywhere_r_p = false;

  l_tmp = l_eff;
  while ((!anywhere_w_p || !anywhere_r_p) && !ENDP(l_tmp))
    {
      effect eff = EFFECT(CAR(l_tmp));
      if (anywhere_effect_p(eff))
	{
	  anywhere_w_p = anywhere_w_p || effect_write_p(eff);
	  anywhere_r_p = anywhere_r_p || effect_read_p(eff);
	}

      POP(l_tmp);
    }

  l_res = NIL;

  if (anywhere_r_p)
    l_res = gen_nconc(l_res,
		      CONS(EFFECT, make_anywhere_effect(make_action_read_memory()),
			   NIL));
  if (anywhere_w_p)
    l_res = gen_nconc(l_res,
		      CONS(EFFECT, make_anywhere_effect(make_action_write_memory()),
			   NIL));


  l_tmp = l_eff;
  while (!ENDP(l_tmp))
    {
      effect eff = EFFECT(CAR(l_tmp));
      pips_debug_effect(4, "considering effect:", eff);

      if (malloc_effect_p(eff) || io_effect_p(eff) ||
	  (!get_bool_property("USER_EFFECTS_ON_STD_FILES") && std_file_effect_p(eff)) ||
	  (effect_write_p(eff) && !anywhere_w_p) ||
	  (effect_read_p(eff) && !anywhere_r_p))
	{
	  pips_debug(4, "added\n");
	  l_res = gen_nconc(l_res, CONS(EFFECT, (*effect_dup_func)(eff), NIL));
	}
      POP(l_tmp);
    }

  return l_res;
}


/************************ effects on special pointer values ********************/

/*
  The semantics of the resulting effects is not well defined...
  These effects should be used with care, as intermediaries.
 */

/**
 @return a new effect on null_pointer_value.
 @param act is an action tag

 Allocate a new anywhere effect using generic function
 reference_to_effect_func, and the anywhere entity on demand
 which may not be best if we want to express it's aliasing with all
 module areas. In the later case, the anywhere entity should be
 generated by bootstrap and be updated each time new areas are
 declared by the parsers. I do not use a persistant anywhere
 reference to avoid trouble with convex-effect nypassing of the
 persistant pointer. (re-used from original non-generic function
 anywhere_effect.)

   Action a is integrated in the new effect (aliasing).
 */
effect make_null_pointer_value_effect(action act)
{

  entity null_ent = null_pointer_value_entity();
  effect null_eff = effect_undefined;

  null_eff = (*reference_to_effect_func)
    (make_reference(null_ent, NIL),
     //     copy_action(act), false);
     act, false);
  return null_eff;
}

bool null_pointer_value_effect_p(effect eff)
{
  return(null_pointer_value_entity_p(effect_entity(eff)));
}

/**
 @return a new effect on undefined_pointer_value.
 @param act is an action tag

 Allocate a new effect on undefined pointer value using generic function
 reference_to_effect_func, and the anywhere entity on demand
 which may not be best if we want to express it's aliasing with all
 module areas. In the later case, the anywhere entity should be
 generated by bootstrap and be updated each time new areas are
 declared by the parsers. I do not use a persistant anywhere
 reference to avoid trouble with convex-effect nypassing of the
 persistant pointer. (re-used from original non-generic function
 anywhere_effect.)

   Action a is integrated in the new effect (aliasing).
 */
effect make_undefined_pointer_value_effect(action act)
{

  entity undefined_ent = undefined_pointer_value_entity();
  effect undefined_eff = effect_undefined;

  undefined_eff = (*reference_to_effect_func)
    (make_reference(undefined_ent, NIL),
     //     copy_action(act), false);
     act, false);
  return undefined_eff;
}

bool undefined_pointer_value_effect_p(effect eff)
{
  return(undefined_pointer_value_entity_p(effect_entity(eff)));
}


/********************** Effects on all accessible paths  ***************/

/**
  @param eff_write a write effect
  @param is the action of the generated effects :
         'r' for read, 'w' for write, and 'x' for read and write.
  @return a list of effects. beware : eff_write is included in the list.

 */
list effect_to_effects_with_given_tag(effect eff, tag act)
{
  list l_res = NIL;
  effect eff_read = effect_undefined;
  effect eff_write = effect_undefined;

  pips_assert("effect is defined \n", !effect_undefined_p(eff));

  if (act == 'x')
    {
      eff_write = eff;
      effect_action_tag(eff_write) = is_action_write;
      eff_read = (*effect_dup_func)(eff_write);
      effect_action_tag(eff_read) = is_action_read;
    }
  else if (act == 'r')
    {

      eff_read = eff;
      effect_action_tag(eff_read) = is_action_read;
      eff_write = effect_undefined;
    }
  else
    {
      eff_read = effect_undefined;
      eff_write = eff;
      effect_action_tag(eff_write) = is_action_write;
    }

  ifdebug(8)
    {
      pips_debug(8, "adding effects to l_res : \n");
      if(!effect_undefined_p(eff_write))
	(*effect_prettyprint_func)(eff_write);
      if(!effect_undefined_p(eff_read))
	(*effect_prettyprint_func)(eff_read);
    }

  if(!effect_undefined_p(eff_write))
    l_res = gen_nconc(l_res, CONS(EFFECT, eff_write, NIL));
  if(!effect_undefined_p(eff_read))
    l_res = gen_nconc(l_res, CONS(EFFECT, eff_read, NIL));

  return l_res;
}

/**
   @param eff is an effect whose reference is the beginning access path.
          it is not modified or re-used.
   @param eff_type is the type of the object represented by the effect
          access path. This avoids computing it at each step.
   @param act is the action of the generated effects :
              'r' for read, 'w' for write, and 'x' for read and write.
   @param level represents the maximum number of dereferencing dimensions
          in the resulting effects.
   @param pointer_only must be set to true to only generate paths to pointers.
   @return a list of effects on all the accessible paths from eff reference.
 */
list generic_effect_generate_all_accessible_paths_effects_with_level(effect eff,
								     type eff_type,
								     tag act,
								     bool add_eff,
								     int level,
								     bool pointers_only)
{
  list l_res = NIL;
  pips_assert("the effect must be defined\n", !effect_undefined_p(eff));
  pips_debug_effect(6, "input effect:", eff);
  pips_debug(6, "input type: %s (%s)\n",
	     words_to_string(words_type(eff_type, NIL, false)),
	     type_to_string(eff_type));
  pips_debug(6, "add_eff is %s\n", add_eff? "true": "false");
  if (type_with_const_qualifier_p(eff_type))
    {
      pips_debug(6, "const qualifier\n");
      if (act == 'w')
	return NIL;
      else if (act == 'x')
	act = 'r';
    }


  if (FILE_star_type_p(eff_type))
    {
      /* there is no other accessible path */
      pips_debug(6, "FILE star path -> returning NIL or the path itself \n");
      if (add_eff)
	l_res = effect_to_list(eff);
    }
  else if (anywhere_effect_p(eff) || entity_null_locations_p(effect_entity(eff))
	   || abstract_pointer_value_cell_p(effect_cell(eff)))
    {
      /* there is no other accessible path */
      pips_debug(6, "anywhere effect -> returning NIL \n");

    }
  else
    {
      effect eff_write = effect_undefined;

      /* this may lead to memory leak if no different access path is
	 reachable */
      eff_write = (*effect_dup_func)(eff);

      pips_debug(6, "level is %d\n", level);
      pips_debug_effect(6, "considering effect : \n", eff);

      switch (type_tag(eff_type))
	{
	case is_type_variable :
	  {
	    variable v = type_variable(eff_type);
	    basic b = variable_basic(v);
	    bool add_array_dims = false;

	    pips_debug(8, "variable case, of dimension %d\n",
		       (int) gen_length(variable_dimensions(v)));

	    /* we first add the array dimensions if any */
	    FOREACH(DIMENSION, c_t_dim,
		    variable_dimensions(v))
	      {
		(*effect_add_expression_dimension_func)
		  (eff_write, make_unbounded_expression());
		add_array_dims = true;
	      }

	    /* if the basic if an end basic, add the path if add_eff is true
	       or if there has been array dimensions added to the original input path */
	    if(basic_int_p(b) ||
	       basic_float_p(b) ||
	       basic_logical_p(b) ||
	       basic_overloaded_p(b) ||
	       basic_complex_p(b) || basic_bit_p(b) || basic_string_p(b)) /* should I had basic_string_p here or make a special case?*/
	      {
		pips_debug(6, "end basic case\n");
		if ((add_array_dims || add_eff) && !pointers_only)
		  l_res = gen_nconc
		    (l_res,
		     effect_to_effects_with_given_tag(eff_write,act));
	      }
	    /* If the basic is a pointer type, we must add an effect
	       with a supplementary dimension, and then recurse
               on the pointed type.
	    */
	    else if(basic_pointer_p(b))
	      {
		if (add_array_dims || add_eff)
		  l_res = gen_nconc
		    (l_res,
		     effect_to_effects_with_given_tag(eff_write,act));
		if (level > 0)
		  {
		    pips_debug(8, "pointer case, \n");

		    eff_write = (*effect_dup_func)(eff_write);
		    (*effect_add_expression_dimension_func)
		      (eff_write, make_unbounded_expression());

		    /*l_res = gen_nconc
		      (l_res,
		      effect_to_effects_with_given_tag(eff_write,act));*/

		    l_res = gen_nconc
		      (l_res,
		       generic_effect_generate_all_accessible_paths_effects_with_level
		       (eff_write,  basic_pointer(b), act, /*false*/ true, level - 1, pointers_only));
		  }
		else
		  {
		    pips_debug(8, "pointer case with level == 0 -> no additional dimension\n");
		  }
	      }
	    else if (basic_derived_p(b))
	      {
		if (!type_enum_p(entity_type(basic_derived(b))))
		  {
		    pips_debug(8, "struct or union case\n");
		    list l_fields = type_fields(entity_type(basic_derived(b)));
		    FOREACH(ENTITY, f, l_fields)
		      {
			type current_type = entity_basic_concrete_type(f);
			effect current_eff = (*effect_dup_func)(eff_write);

			// we add the field index
			effect_add_field_dimension(current_eff, f);

			// and call ourselves recursively
			l_res = gen_nconc
			  (l_res,
			   generic_effect_generate_all_accessible_paths_effects_with_level
			   (current_eff,  current_type, act, true, level, pointers_only));
		      }
		  }
	      }
	    else if (!basic_typedef_p(b))
	      {

		if (!pointers_only && (add_array_dims || add_eff))
		  l_res = gen_nconc
		    (l_res,
		     effect_to_effects_with_given_tag(eff_write,act));
	      }
	    else
	      {
		pips_internal_error("unexpected typedef basic");
	      }

	    break;
	  }
	case is_type_void:
	  {
	    pips_debug(8, "void case\n");
	    if (add_eff)
	      l_res = CONS(EFFECT, eff, NIL);
	    break;
	  }
	case is_type_functional:
	  pips_debug(8, "functional case\n");
	  pips_user_warning("possible effect through indirect call (type is: %s(%s)) -> returning anywhere\n",
			    words_to_string(words_type(eff_type, NIL, false)),
			    type_to_string(eff_type));
	  pips_debug_effect(0, "", eff);
	  l_res = make_anywhere_read_write_memory_effects();
	  break;
	case is_type_struct:
	case is_type_union:
	case is_type_enum:
	  pips_debug(8, "agregate type case\n");
	  pips_internal_error("aggregate type not handeld yet\n");
	  break;

	case is_type_area:
	case is_type_statement:
	case is_type_varargs:
	case is_type_unknown:
	  pips_internal_error("unexpected type in this context \n");
	  break;
	default:
	  {
	    pips_internal_error("unknown type tag\n");
	  }
	} /*switch */

    } /* else */

  pips_debug_effects(8, "output effects:\n", l_res);

  return(l_res);
}

/**
   @param eff is an effect whose reference is the beginning access path.
          it is not modified or re-used.
   @param eff_type is the type of the object represented by the effect
          access path. This avoids computing it at each step.
   @param act is the action of the generated effects :
              'r' for read, 'w' for write, and 'x' for read and write.
   @return a list of effects on all the accessible paths from eff reference.
 */
list generic_effect_generate_all_accessible_paths_effects(effect eff,
							  type eff_type,
							  tag act)
{
  return generic_effect_generate_all_accessible_paths_effects_with_level(eff,
									 eff_type,
									 act,
									 false,
									 10, /* to avoid too long paths until GAPS are handled */
									 false);
}

/******************************************************************/


/**
 NOT YET IMPLEMENTED FOR VARARGS AND FUNCTIONAL TYPES.

 @param eff is an effect
 @return true if the effect reference maybe an access path to a pointer
*/
static bool r_effect_pointer_type_p(effect eff, list l_ind, type ct)
{
  bool p = false, finished = false;

  pips_debug(7, "begin with type %s\n and number of indices : %d\n",
	     words_to_string(words_type(ct,NIL,false)),
	     (int) gen_length(l_ind));
  while (!finished)
    {
      switch (type_tag(ct))
	{
	case is_type_variable :
	  {
	    variable v = type_variable(ct);
	    basic b = variable_basic(v);
	    list l_dim = variable_dimensions(v);

	    pips_debug(8, "variable case, of basic %s, of dimension %d\n",
		       basic_to_string(b),
		       (int) gen_length(variable_dimensions(v)));

	    while (!ENDP(l_dim) && !ENDP(l_ind))
	      {
		POP(l_dim);
		POP(l_ind);
	      }

	    if(ENDP(l_ind) && ENDP(l_dim))
	      {
	      if(basic_pointer_p(b))
		{
		  p = true;
		  finished = true;
		}
	      else
		finished = true;
	      }
	    else if (ENDP(l_dim)) /* && !ENDP(l_ind) by construction */
	      {
		pips_assert("the current basic should be a pointer or a derived\n",
			    basic_pointer_p(b) || basic_derived_p(b));

		if (basic_pointer_p(b))
		  {
		    ct = basic_pointer(b);
		    POP(l_ind);
		  }
		else /* b is a derived */
		  {
		    ct = entity_type(basic_derived(b));
		    p = r_effect_pointer_type_p(eff, l_ind, ct);
		    finished = true;
		  }

	      }
	    else /* ENDP(l_ind) but !ENDP(l_dim) */
	      {
		finished = true;
	      }

	    break;
	  }
	case is_type_struct:
	case is_type_union:
	case is_type_enum:
	  {
	    list l_ent = type_fields(ct);
	    expression field_exp = EXPRESSION(CAR(l_ind));
	    entity field = entity_undefined;

	    pips_debug(7, "field case, with field expression : %s \n",
		       words_to_string(words_expression(field_exp,NIL)));

	    /* If the field is known, we only look at the corresponding type.
	       If not, we have to recursively look at each field
	    */
	    if (!unbounded_expression_p(field_exp))
	      {
		pips_assert("the field expression must be a reference\n",
			    expression_reference_p(field_exp));
		field = expression_variable(field_exp);
		if (variable_phi_p(field))
		  field = entity_undefined;
	      }

	    if (!entity_undefined_p(field))
	      {
		/* the current type is the type of the field */
		ct = entity_basic_concrete_type(field);
		p = r_effect_pointer_type_p(eff, CDR(l_ind), ct);
		/* free_type(ct); */
		ct = type_undefined;
		finished = true;
	      }
	    else
	      /* look at each field until a pointer is found*/
	      {
		while (!ENDP(l_ent) && p)
		  {
		    type new_ct = entity_basic_concrete_type(ENTITY(CAR(l_ent)));
		    p = r_effect_pointer_type_p(eff, CDR(l_ind),
						new_ct);
		    POP(l_ent);
		  }
		finished = true;
	      }
	    break;
	  }
	default:
	  {
	    pips_internal_error("case not handled yet");
	  }
	} /*switch */

    }/*while */
  pips_debug(8, "end with p = %s\n", p== false ? "false" : "true");
  return p;

}


/**
 NOT YET IMPLEMENTED FOR VARARGS AND FUNCTIONAL TYPES.

 @param eff is an effect
 @return true if the effect reference maybe an access path to a pointer
*/
bool effect_pointer_type_p(effect eff)
{
  bool p = false;
  reference ref = effect_any_reference(eff);
  list l_ind = reference_indices(ref);
  entity ent = reference_variable(ref);
  type t = entity_basic_concrete_type(ent);

  pips_debug(8, "begin with effect reference %s\n",
	     words_to_string(words_reference(ref,NIL)));
  if (entity_abstract_location_p(ent))
    p = true;
  else
    p = r_effect_pointer_type_p(eff, l_ind, t);

  pips_debug(8, "end with p = %s\n", p== false ? "false" : "true");
  return p;

}



type simple_effect_reference_type(reference ref)
{
  type bct = entity_basic_concrete_type(reference_variable(ref));
  type ct; /* current_type */

  list l_inds = reference_indices(ref);

  type t = type_undefined; /* result */
  bool finished = false;

  pips_debug(8, "beginning with reference : %s\n", words_to_string(words_reference(ref,NIL)));

  ct = bct;
  while (! finished)
    {
      basic cb = variable_basic(type_variable(ct)); /* current basic */
      list cd = variable_dimensions(type_variable(ct)); /* current type dimensions */

      while(!ENDP(cd) && !ENDP(l_inds))
	{
	  pips_debug(8, "poping one array dimension \n");
	  POP(cd);
	  POP(l_inds);
	}

      if(ENDP(l_inds))
	{
	  pips_debug(8, "end of reference indices, generating type\n");
	  t = make_type(is_type_variable,
			make_variable(copy_basic(cb),
				      gen_full_copy_list(cd),
				      NIL));
	  finished = true;
	}
      else /* ENDP (cd) && ! ENDP(l_inds) */
	{
	  switch (basic_tag(cb))
	    {
	    case is_basic_pointer:
	      /* in an effect reference there is always an index for a pointer */
	      pips_debug(8, "poping pointer dimension\n");
	      POP(l_inds);
	      ct = basic_pointer(cb);
	      break;
	    case is_basic_derived:
	      {
		/* we must know which field it is, else return an undefined type */
		expression field_exp = EXPRESSION(CAR(l_inds));
		entity field = entity_undefined;
		pips_debug(8, "field dimension : %s\n",
			   words_to_string(words_expression(field_exp,NIL)));

		if (!unbounded_expression_p(field_exp))
		  {
		    pips_assert("the field expression must be a reference\n",
				expression_reference_p(field_exp));
		    field = expression_variable(field_exp);
		    if (variable_phi_p(field))
		      field = entity_undefined;
		  }

		if (!entity_undefined_p(field))
		  {
		    pips_debug(8, "known field, poping field dimension\n");
		    bct = entity_basic_concrete_type(field);
		    ct = bct;
		    POP(l_inds);
		  }
		else
		  {
		    pips_debug(8, "unknown field, returning type_undefined\n");
		    t = type_undefined;
		    finished = true;
		  }
	      }
	      break;
	    case is_basic_int:
	    case is_basic_float:
	    case is_basic_logical:
	    case is_basic_complex:
	    case is_basic_string:
	    case is_basic_bit:
	    case is_basic_overloaded:
	      pips_internal_error("fundamental basic not expected here ");
	      break;
	    case is_basic_typedef:
	      pips_internal_error("typedef not expected here ");
	    } /* switch (basic_tag(cb)) */
	}

    } /* while (!finished) */


  pips_debug(6, "returns with %s\n", words_to_string(words_type(t,NIL,false)));
  return t;

}



bool regions_weakly_consistent_p(list rl)
{
  FOREACH(EFFECT, r, rl) {
    descriptor rd = effect_descriptor(r);

    if(descriptor_convex_p(rd)) {
      Psysteme rsc = descriptor_convex(rd);

      pips_assert("rsc is weakly consistent", sc_weak_consistent_p(rsc));
    }
  }
  return true;
}

bool region_weakly_consistent_p(effect r)
{
  descriptor rd = effect_descriptor(r);

  if(descriptor_convex_p(rd)) {
    Psysteme rsc = descriptor_convex(rd);

    pips_assert("rsc is weakly consistent", sc_weak_consistent_p(rsc));
  }

  return true;
}

/**
   Effects are not copied but a new list is built.
 */
list statement_modified_pointers_effects_list(statement s)
{
  list l_cumu_eff = load_rw_effects_list(s);
  list l_res = NIL;
  bool anywhere_p = false;

  ifdebug(6){
	 pips_debug(6, " effects before selection: \n");
	 (*effects_prettyprint_func)(l_cumu_eff);
       }

  FOREACH(EFFECT, eff, l_cumu_eff)
    {
      if (!anywhere_p && effect_write_p(eff))
	{
	  if (anywhere_effect_p(eff))
	    anywhere_p = true;
	  else if (effect_pointer_type_p(eff))
	    l_res = gen_nconc(l_res, CONS(EFFECT, eff, NIL));
	}
    }
  if (anywhere_p)
    {
      gen_free_list(l_res);
      l_res = CONS(EFFECT, make_anywhere_effect(make_action_write_memory()), NIL);
    }

  ifdebug(6){
	 pips_debug(6, " effects after selection: \n");
	 (*effects_prettyprint_func)(l_res);
       }


  return l_res;
}

/******************************************************************/


static bool effects_reference_indices_may_equal_p(expression ind1, expression ind2)
{
  if (unbounded_expression_p(ind1) || unbounded_expression_p(ind2))
    return true;
  else
    return same_expression_p(ind1, ind2);
}

/**
   This function should be instanciated differently for simple and convex
   effects : much more work should be done for convex effects.

   @return true if the effects have comparable access paths
               in which case result is set to
	          0 if the effects paths may be equal
		  1 if eff1 access path may lead to eff2 access path
		  -1 if eff2 access path may lead to eff1 access path
           false otherwise.
*/
static bool effects_access_paths_comparable_p(effect eff1, effect eff2,
int *result)
{
  bool comparable_p = true; /* assume they are compable */
  reference ref1 = effect_any_reference(eff1);
  reference ref2 = effect_any_reference(eff2);
  list linds1 = reference_indices(ref1);
  list linds2 = reference_indices(ref2);

  pips_debug_effect(8, "begin\neff1 = \n", eff1);
  pips_debug_effect(8, "begin\neff2 = \n", eff2);

  /* to be comparable, they must have the same entity */
  comparable_p = same_entity_p(reference_variable(ref1),
			       reference_variable(ref2));

  while( comparable_p && !ENDP(linds1) && !ENDP(linds2))
    {
      if (!effects_reference_indices_may_equal_p(EXPRESSION(CAR(linds1)),
						EXPRESSION(CAR(linds2))))
	comparable_p = false;

      POP(linds1);
      POP(linds2);
    }

  if (comparable_p)
    {
      *result = (int) (gen_length(linds2) - gen_length(linds1)) ;
      if (*result != 0) *result = *result / abs(*result);
    }

  pips_debug(8, "end with comparable_p = %s and *result = %d",
	     comparable_p ? "true" : "false", *result);

  return comparable_p;
}


/* do not reuse l_eff after calling this function
 */
list generic_effects_store_update(list l_eff, statement s, bool backward_p)
{

   transformer t; /* transformer of statement s */
   list l_eff_pointers;
   list l_res = NIL;
   bool anywhere_w_p = false;
   bool anywhere_r_p = false;

   pips_debug(5, "begin\n");

   debug_on("SEMANTICS_DEBUG_LEVEL");
   t = (*load_completed_transformer_func)(s);
   debug_off();

   if (l_eff !=NIL)
     {
       /* first change the store of the descriptor */
       if (backward_p)
	 l_eff = (*effects_transformer_composition_op)(l_eff, t);
       else
	 l_eff =  (*effects_transformer_inverse_composition_op)(l_eff, t);

       ifdebug(5){
	 pips_debug(5, " effects after composition with transformer: \n");
	 (*effects_prettyprint_func)(l_eff);
       }

       if (get_bool_property("EFFECTS_POINTER_MODIFICATION_CHECKING"))
	 {
	   /* then change the effects references if some pointer is modified */
	   /* backward_p is not used here because we lack points_to information
	      and we thus generate anywhere effects
	   */
	   l_eff_pointers = statement_modified_pointers_effects_list(s);

	   while( !ENDP(l_eff) &&
		  ! (anywhere_w_p && anywhere_r_p))
	     {
	       list l_eff_p_tmp = l_eff_pointers;
	       effect eff = EFFECT(CAR(l_eff));
	       bool eff_w_p = effect_write_p(eff);
	       bool found = false;



	       while( !ENDP(l_eff_p_tmp) &&
		      !((eff_w_p && anywhere_w_p) || (!eff_w_p && anywhere_r_p)))
		 {
		   effect eff_p = EFFECT(CAR(l_eff_p_tmp));
		   effect new_eff = effect_undefined;
		   int comp_res = 0;

		   if(effects_access_paths_comparable_p(eff, eff_p, &comp_res)
		      && comp_res <=0 )
		     {
		       new_eff = make_anywhere_effect(copy_action(effect_action(eff)));
		       l_res = gen_nconc(l_res, CONS(EFFECT, new_eff, NIL));
		       found = true;
		       if (eff_w_p)
			 anywhere_w_p = true;
		       else
			 anywhere_r_p = true;

		     } /*  if(effects_access_paths_comparable_p) */

		   POP(l_eff_p_tmp);
		 } /* while( !ENDP(l_eff_p_tmp))*/

	       /* if we have found no modifiying pointer, we keep the effect */
	       if (!found)
		 {
		   /* is the copy necessary ?
            * sg: yes, to be consistent with other branches of the test */
		   l_res = gen_nconc(l_res, CONS(EFFECT,(*effect_dup_func)(eff) , NIL));

		 }

	       POP(l_eff);

	     } /* while( !ENDP(l_eff)) */

	   ifdebug(5){
	     pips_debug(5, " effects after composition with pointer effects: \n");
	     (*effects_prettyprint_func)(l_res);
	   }

	 } /* if (get_bool_property("EFFECTS_POINTER_MODIFICATION_CHECKING"))*/
       else
	 l_res = l_eff;
     } /* if (l_eff !=NIL) */

   free_transformer(t);

   return l_res;
}

/************ CONVERSION TO CONSTANT PATH EFFECTS ***********/




/**
   @param l_pointer_eff is a list of effects that may involve access
   paths dereferencing pointers.

   @return a list of effects with no access paths dereferencing pointers.

   Two algorithms are currently used, depending on the value returned
   by get_use_points_to.

   If true, when there is an effect reference with a dereferencing
   dimension, eval_cell_with_points_to is called to find an equivalent
   constant path using points-to.

   If false, effect references with a dereferencing dimension are
   systematically replaced by anywhere effects.
 */
list pointer_effects_to_constant_path_effects(list l_pointer_eff)
{
  list le = NIL;

  pips_debug_effects(8, "input effects : \n", l_pointer_eff);

  FOREACH(EFFECT, eff, l_pointer_eff)
    {

     pips_debug_effect(8, "current effect : \n", eff);
   
      if(store_effect_p(eff))
	{
	  //bool exact_p;
	  //reference ref = effect_any_reference(eff);

	  pips_debug(8, "store effect\n");

	  if (io_effect_p(eff)
	      || malloc_effect_p(eff)
	      || (!get_bool_property("USER_EFFECTS_ON_STD_FILES")
		  && std_file_effect_p(eff)))
	    {
	      pips_debug(8, "special effect \n");
	      le = CONS(EFFECT, (*effect_dup_func)(eff), le);
	    }
	  else
	    {
	      list l_const = (*effect_to_constant_path_effects_func)(eff);
	      pips_debug(8,"computing the union\n");
	      pips_debug_effects(8, "l_const before union: \n", le);
	      pips_debug_effects(8, "le before union: \n", le);
	      le = (*effects_union_op)(l_const, le, effects_scalars_and_same_action_p);
	      pips_debug_effects(8, "effects after union: \n", le);
	    }
	}
      else
	{
	  pips_debug(8, "non store effect\n");
	  le = CONS(EFFECT, (*effect_dup_func)(eff), le);
	}
    }

  pips_debug_effects(8, "ouput effects : \n", le);

  return le;
}


list effect_to_constant_path_effects_with_no_pointer_information(effect eff)
{
  list le = NIL;
  bool exact_p;
  reference ref = effect_any_reference(eff);

  if (effect_reference_dereferencing_p(ref, &exact_p))
    {
      pips_debug(8, "dereferencing case \n");
      le = CONS(EFFECT, make_anywhere_effect(copy_action(effect_action(eff))), le);
    }
  else
    le = CONS(EFFECT, (*effect_dup_func)(eff), le);

  return le;
}


