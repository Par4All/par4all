/* copies an effect with no subcript expression */
#define make_sdfi_effect(e) \
 (reference_indices(effect_reference(e)) == NIL) ? \
  make_simple_effect(make_reference(reference_variable(effect_reference(e)), NIL),\
     make_action(action_tag(effect_action(e)), UU), \
     make_approximation(approximation_tag(effect_approximation(e)), UU)) : \
  make_simple_effect(make_reference(reference_variable(effect_reference(e)), NIL), \
	      make_action(action_tag(effect_action(e)), UU), \
	      make_approximation(is_approximation_may, UU))
