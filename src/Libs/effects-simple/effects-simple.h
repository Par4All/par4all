/* header file built by cproto */
#ifndef effects_simple_header_included
#define effects_simple_header_included
/* copies an effect with no subcript expression */
#define make_sdfi_effect(e) \
 (reference_indices(effect_reference(e)) == NIL) ? \
  make_simple_effect(make_reference(reference_variable(effect_reference(e)), NIL),\
     make_action(action_tag(effect_action(e)), UU), \
     make_approximation(approximation_tag(effect_approximation(e)), UU)) : \
  make_simple_effect(make_reference(reference_variable(effect_reference(e)), NIL), \
	      make_action(action_tag(effect_action(e)), UU), \
	      make_approximation(is_approximation_may, UU))


 
/* binary_operators.c */
extern list ReferenceUnion(list /*l1*/, list /*l2*/, boolean (* /*union_combinable_p*/)(effect, effect));
extern list ReferenceTestUnion(list /*l1*/, list /*l2*/, boolean (* /*union_combinable_p*/)(effect, effect));
extern list EffectsMayUnion(list /*l1*/, list /*l2*/, boolean (* /*union_combinable_p*/)(effect, effect));
extern list EffectsMustUnion(list /*l1*/, list /*l2*/, boolean (* /*union_combinable_p*/)(effect, effect));
extern list effects_may_union(effect /*eff1*/, effect /*eff2*/);
extern list effects_must_union(effect /*eff1*/, effect /*eff2*/);
extern effect effect_may_union(effect /*eff1*/, effect /*eff2*/);
extern effect effect_must_union(effect /*eff1*/, effect /*eff2*/);
extern list EffectsSupDifference(list /*l1*/, list /*l2*/, boolean (* /*difference_combinable_p*/)(effect, effect));
extern list EffectsInfDifference(list /*l1*/, list /*l2*/, boolean (* /*difference_combinable_p*/)(effect, effect));
extern effect proper_to_summary_simple_effect(effect /*eff*/);
/* interface.c */
extern bool cumulated_references(string /*module_name*/);
extern bool proper_references(string /*module_name*/);
extern bool proper_effects(string /*module_name*/);
extern bool summary_effects(string /*module_name*/);
extern bool cumulated_effects(string /*module_name*/);
extern bool in_summary_effects(string /*module_name*/);
extern bool out_summary_effects(string /*module_name*/);
extern bool in_effects(string /*module_name*/);
extern bool out_effects(string /*module_name*/);
extern bool print_code_proper_effects(string /*module_name*/);
extern bool print_code_cumulated_effects(string /*module_name*/);
extern bool print_code_proper_references(string /*module_name*/);
extern bool print_code_cumulated_references(string /*module_name*/);
extern bool print_code_in_effects(string /*module_name*/);
extern bool print_code_out_effects(string /*module_name*/);
extern bool print_source_proper_effects(string /*module_name*/);
extern bool print_source_cumulated_effects(string /*module_name*/);
extern bool print_source_in_effects(string /*module_name*/);
extern bool print_source_out_effects(string /*module_name*/);
extern text get_text_proper_effects(string /*module_name*/);
extern text get_text_cumulated_effects(string /*module_name*/);
extern list proper_effects_of_expression(expression /*e*/);
extern list expression_to_proper_effects(expression /*e*/);
extern list proper_effects_of_range(range /*r*/);
extern void rproper_effects_of_statement(statement /*s*/);
extern void rcumulated_effects_of_statement(statement /*s*/);
extern list statement_to_effects(statement /*s*/);
extern bool full_simple_proper_effects(string /*module_name*/, statement /*current*/);
extern bool simple_cumulated_effects(string /*module_name*/, statement /*current*/);
/* methods.c */
extern void set_methods_for_proper_references(void);
extern void set_methods_for_cumulated_references(void);
extern void set_methods_for_proper_simple_effects(void);
extern void set_methods_for_simple_effects(void);
extern void set_methods_for_rw_effects_prettyprint(string /*module_name*/);
extern void set_methods_for_inout_effects_prettyprint(string /*module_name*/);
extern void reset_methods_for_effects_prettyprint(string /*module_name*/);
/* interprocedural.c */
extern list simple_effects_backward_translation(entity /*func*/, list /*real_args*/, list /*l_eff*/, transformer /*context*/);
extern list effects_dynamic_elim(list /*l_eff*/);
extern list summary_effect_to_proper_effect(call /*c*/, effect /*e*/);
extern list summary_to_proper_effects(entity /*func*/, list /*args*/, list /*func_sdfi*/);
extern list simple_effects_forward_translation(entity /*callee*/, list /*real_args*/, list /*l_eff*/, transformer /*context*/);
/* prettyprint.c */
extern int compare_effect_reference(effect */*e1*/, effect */*e2*/);
extern int compare_effect_reference_in_common(effect */*e1*/, effect */*e2*/);
extern text simple_rw_effects_to_text(list /*l*/);
extern text simple_inout_effects_to_text(list /*l*/);
extern string effect_to_string(effect /*eff*/);
extern list words_effect(effect /*obj*/);
extern void print_effects(list /*pc*/);
/* unary_operators.c */
extern effect reference_effect_dup(effect /*eff*/);
extern void reference_effect_free(effect /*eff*/);
extern effect reference_to_simple_effect(reference /*ref*/, action /*ac*/);
extern effect simple_effect_dup(effect /*eff*/);
extern void simple_effect_free(effect /*eff*/);
extern effect reference_to_reference_effect(reference /*ref*/, action /*ac*/);
extern list simple_effects_union_over_range(list /*l_eff*/, entity /*i*/, range /*r*/, descriptor /*d*/);
extern list effect_to_may_sdfi_list(effect /*eff*/);
extern list effect_to_sdfi_list(effect /*eff*/);
extern void simple_effects_descriptor_normalize(list /*l_eff*/);
#endif /* effects_simple_header_included */

/***********written by Dat*************/
extern text my_get_text_proper_effects(string module_name);
/**************************************/


