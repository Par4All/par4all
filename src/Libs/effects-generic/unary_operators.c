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
#include "ri-util.h"
#include "misc.h"

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
effects_write_effects(list l_eff)
{
    list l_new = NIL;
    MAP(EFFECT, eff,
	if (effect_write_p(eff)) l_new = CONS(EFFECT, eff, l_new),
	l_eff);
    return gen_nreverse(l_new);
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
effect_to_write_effect(effect eff)
{
    effect_action_tag(eff) = is_action_write;
}

void
effects_to_write_effects(list l_eff)
{
    effects_map(l_eff, effect_to_write_effect);
}

void
array_effects_to_may_effects(list l_eff)
{
    MAP(EFFECT, eff, 
	{
	    if (!effect_scalar_p(eff))
		effect_to_may_effect(eff);
	}, 
	l_eff);      

}

list
effects_dup_without_variables(list l_eff, list l_var)
{
    list l_res = NIL;
    
    MAP(EFFECT, eff,
    {
	if (gen_find_eq(effect_entity(eff), l_var) == entity_undefined)
	{
	  l_res = CONS(EFFECT, (*effect_dup_func)(eff), l_res);
	}
        else
	    pips_debug(7, "Effect on variable %s removed\n",
		       entity_local_name(effect_entity(eff)));
    }, l_eff);
    return gen_nreverse(l_res);
}

effect 
effect_dup(effect eff)
{
    return((*effect_dup_func)(eff));
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


/***********************************************************************/
/* UNDEFINED UNARY OPERATORS                                           */
/***********************************************************************/

/* Composition with transformers */

list
effects_undefined_composition_with_transformer(list l_eff __attribute__((__unused__)), transformer trans __attribute__((__unused__)))
{
    return list_undefined;
}

list effects_composition_with_effect_transformer(list l_eff,
						 transformer trans __attribute__((__unused__)))
{
  /* FI: used to be nop and wrong information is now preserved
     intraprocedurally with loops, maybe because I modified simple
     effects; since we do not have transformers, we use instead the
     effects themselves, which could be transformed into a
     transformer... 

     The effects are supposed to be ordered. A write effect must
     appears before another effect to require an update.
*/
  list l1 = list_undefined;
  list l2 = list_undefined;
  extern string words_to_string(list);

  ifdebug(8) {
    pips_debug(8, "Begin: %zd effects before composition:\n", gen_length(l_eff));
    MAP(EFFECT, eff, {
	reference r = effect_any_reference(eff);
	pips_debug(8, "%p: %s\n", eff, words_to_string(effect_words_reference_with_addressing_as_it_is(r, addressing_tag(effect_addressing(eff)))));
	pips_assert("Effect eff is consitent", effect_consistent_p(eff));
      },  l_eff);
  }

  for(l1= l_eff; !ENDP(l1); POP(l1)) {
    effect e1 = EFFECT(CAR(l1));
    for(l2 = CDR(l1); !ENDP(l2); POP(l2)) {
      effect e2 = EFFECT(CAR(l2));

      ifdebug(1) {
	pips_assert("Effect e1 is consitent", effect_consistent_p(e1));
	pips_assert("Effect e2 is consitent", effect_consistent_p(e2));
      }

      ifdebug(8) {
	reference r1 = effect_any_reference(e1);
	reference r2 = effect_any_reference(e2);
	(void) fprintf(stderr, "e1 %p: %s (%s)\n", e1, words_to_string(effect_words_reference_with_addressing_as_it_is(r1, addressing_tag(effect_addressing(e1)))),
		       action_to_string(effect_action(e1)));
	(void) fprintf(stderr, "e2 %p: %s (%s)\n", e2,
		       words_to_string(effect_words_reference_with_addressing_as_it_is(r2, addressing_tag(effect_addressing(e2)))),
		       action_to_string(effect_action(e2)));
      }

      e2 = effect_interference(e2, e1);

      ifdebug(8) {
	reference r2 = effect_any_reference(e2);
	tag ad2 = addressing_tag(effect_addressing(e2));
	(void) fprintf(stderr, "resulting e2 %p: %s (%s)\n", e2,
		       words_to_string(effect_words_reference_with_addressing_as_it_is(r2, ad2)),
		       action_to_string(effect_action(e2)));
	pips_assert("New effect e2 is consitent", effect_consistent_p(e2));
      }

      EFFECT_(CAR(l2)) = e2;
    }
  }

  ifdebug(8) {
    pips_debug(8, "End: %zd effects after composition:\n", gen_length(l_eff));
    MAP(EFFECT, eff, {
	reference r = effect_any_reference(eff);
	pips_debug(8, "%p: %s\n", eff, words_to_string(effect_words_reference_with_addressing_as_it_is(r, addressing_tag(effect_addressing(eff)))));
      },  l_eff);
  }

  /* FI: Not generic. */
  l_eff = proper_effects_combine(l_eff, FALSE);

  ifdebug(8) {
    pips_debug(8, "End: %zd effects after composition:\n", gen_length(l_eff));
    MAP(EFFECT, eff, {
	reference r = effect_any_reference(eff);
	pips_debug(8, "%p: %s\n", eff, words_to_string(effect_words_reference_with_addressing_as_it_is(r, addressing_tag(effect_addressing(eff)))));
      },  l_eff);
  }

  return l_eff;
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
