/* header file built by cproto */
#ifndef effects_generic_header_included
#define effects_generic_header_included
/* $Id$
 */


/* some useful SHORTHANDS for EFFECT:
 */
#define effect_entity(e) reference_variable(effect_reference(e))
#define effect_action_tag(eff) action_tag(effect_action(eff))
#define effect_approximation_tag(eff) \
	approximation_tag(effect_approximation(eff))

#define effect_scalar_p(eff) entity_scalar_p(effect_entity(eff))
#define effect_read_p(eff) (action_tag(effect_action(eff))==is_action_read)
#define effect_write_p(eff) (action_tag(effect_action(eff))==is_action_write)
#define effect_may_p(eff) \
        (approximation_tag(effect_approximation(eff)) == is_approximation_may)
#define effect_must_p(eff) \
        (approximation_tag(effect_approximation(eff)) == is_approximation_must)
#define effect_exact_p(eff) \
        (approximation_tag(effect_approximation(eff)) ==is_approximation_exact)


/* some string constants for prettyprints...
 */
#define ACTION_UNDEFINED 	string_undefined
#define ACTION_READ 		"R"
#define ACTION_WRITE 		"W"
#define ACTION_IN    		"IN"
#define ACTION_OUT		"OUT"
#define ACTION_COPYIN		"COPYIN"
#define ACTION_COPYOUT		"COPYOUT"
#define ACTION_PRIVATE		"PRIVATE"


/* prettyprint function types:
 */
#include "text.h" /* hum... */
typedef text (*generic_text_function)(list /* of effect */);
typedef void (*generic_prettyprint_function)(list /* of effect */);
typedef void (*generic_attachment_function)(text);

 
/* for db_* functions 
 */
#define DB_GET_SE(name, NAME)				\
static statement_effects db_get_##name(char * modname)	\
{ return (statement_effects)				\
  db_get_memory_resource(DBR_##NAME, modname, TRUE);}

#define DB_GET_LS(name, NAME)				\
static list db_get_##name(char * modname)		\
{ return effects_to_list((effects)			\
  db_get_memory_resource(DBR_##NAME, modname, TRUE));}

#define DB_PUT_SE(name, NAME)						\
static void db_put_##name(char * modname, statement_effects se)		\
{ DB_PUT_MEMORY_RESOURCE(DBR_##NAME, modname, (char*) se);}

#define DB_PUT_LS(name, NAME)				\
static void db_put_##name(char * modname, list l)	\
{DB_PUT_MEMORY_RESOURCE(DBR_##NAME,modname,(char*)list_to_effects(l));}

#define DB_NOPUT_SE(name)\
static void db_put_##name(char *m, statement_effects se) \
{ free_statement_effects(se); return; }

#define DB_NOPUT_LS(name)\
static void db_put_##name(char *m, list l) \
{ gen_full_free_list(l); return;}

#define DB_GETPUT_SE(name, NAME) DB_GET_SE(name, NAME) DB_PUT_SE(name, NAME)
#define DB_GETNOPUT_SE(name, NAME) DB_GET_SE(name, NAME) DB_NOPUT_SE(name)
#define DB_GETPUT_LS(name, NAME) DB_GET_LS(name, NAME) DB_PUT_LS(name, NAME)
#define DB_GETNOPUT_LS(name, NAME) DB_GET_LS(name, NAME)DB_NOPUT_LS(name)


/* For COMPATIBILITY purpose only - DO NOT USE anymore
 */
#define effect_variable(e) reference_variable(effect_reference(e))

/* end of effects-generic-local.h
 */
/* proper_effects_engine.c */
extern void set_contracted_proper_effects(bool /*b*/);
extern void proper_effects_error_handler(void);
extern list generic_proper_effects_of_range(range /*r*/);
extern list generic_proper_effects_of_lhs(reference /*ref*/);
extern list generic_proper_effects_of_reference(reference /*ref*/);
extern list generic_proper_effects_of_syntax(syntax /*s*/);
extern list generic_proper_effects_of_expression(expression /*e*/);
extern list generic_proper_effects_of_expressions(list /*exprs*/);
extern list generic_r_proper_effects_of_call(call /*c*/);
extern void proper_effects_of_module_statement(statement /*module_stat*/);
extern bool proper_effects_engine(char */*module_name*/);
extern void expression_proper_effects_engine(string /*module_name*/, statement /*current*/);
/* rw_effects_engine.c */
extern void set_contracted_rw_effects(bool /*b*/);
extern bool summary_rw_effects_engine(string /*module_name*/);
extern void rw_effects_of_module_statement(statement /*module_stat*/);
extern bool rw_effects_engine(char */*module_name*/);
/* in_effects_engine.c */
extern bool summary_in_effects_engine(char */*module_name*/);
extern bool in_effects_engine(char */*module_name*/);
/* out_effects_engine.c */
extern void reset_out_summary_effects_list(void);
extern void update_out_summary_effects_list(list /*l_out*/);
extern list get_out_summary_effects_list(void);
extern bool summary_out_effects_engine(char */*module_name*/);
extern bool out_effects_engine(char */*module_name*/);
/* mappings.c */
extern bool proper_references_undefined_p(void);
extern void reset_proper_references(void);
extern void error_reset_proper_references(void);
extern void set_proper_references(statement_effects /*o*/);
extern statement_effects get_proper_references(void);
extern void init_proper_references(void);
extern void close_proper_references(void);
extern void store_proper_references(statement /*k*/, effects /*v*/);
extern void update_proper_references(statement /*k*/, effects /*v*/);
extern effects load_proper_references(statement /*k*/);
extern effects delete_proper_references(statement /*k*/);
extern bool bound_proper_references_p(statement /*k*/);
extern void store_or_update_proper_references(statement /*k*/, effects /*v*/);
extern bool cumulated_references_undefined_p(void);
extern void reset_cumulated_references(void);
extern void error_reset_cumulated_references(void);
extern void set_cumulated_references(statement_effects /*o*/);
extern statement_effects get_cumulated_references(void);
extern void init_cumulated_references(void);
extern void close_cumulated_references(void);
extern void store_cumulated_references(statement /*k*/, effects /*v*/);
extern void update_cumulated_references(statement /*k*/, effects /*v*/);
extern effects load_cumulated_references(statement /*k*/);
extern effects delete_cumulated_references(statement /*k*/);
extern bool bound_cumulated_references_p(statement /*k*/);
extern void store_or_update_cumulated_references(statement /*k*/, effects /*v*/);
extern bool proper_rw_effects_undefined_p(void);
extern void reset_proper_rw_effects(void);
extern void error_reset_proper_rw_effects(void);
extern void set_proper_rw_effects(statement_effects /*o*/);
extern statement_effects get_proper_rw_effects(void);
extern void init_proper_rw_effects(void);
extern void close_proper_rw_effects(void);
extern void store_proper_rw_effects(statement /*k*/, effects /*v*/);
extern void update_proper_rw_effects(statement /*k*/, effects /*v*/);
extern effects load_proper_rw_effects(statement /*k*/);
extern effects delete_proper_rw_effects(statement /*k*/);
extern bool bound_proper_rw_effects_p(statement /*k*/);
extern void store_or_update_proper_rw_effects(statement /*k*/, effects /*v*/);
extern bool rw_effects_undefined_p(void);
extern void reset_rw_effects(void);
extern void error_reset_rw_effects(void);
extern void set_rw_effects(statement_effects /*o*/);
extern statement_effects get_rw_effects(void);
extern void init_rw_effects(void);
extern void close_rw_effects(void);
extern void store_rw_effects(statement /*k*/, effects /*v*/);
extern void update_rw_effects(statement /*k*/, effects /*v*/);
extern effects load_rw_effects(statement /*k*/);
extern effects delete_rw_effects(statement /*k*/);
extern bool bound_rw_effects_p(statement /*k*/);
extern void store_or_update_rw_effects(statement /*k*/, effects /*v*/);
extern bool invariant_rw_effects_undefined_p(void);
extern void reset_invariant_rw_effects(void);
extern void error_reset_invariant_rw_effects(void);
extern void set_invariant_rw_effects(statement_effects /*o*/);
extern statement_effects get_invariant_rw_effects(void);
extern void init_invariant_rw_effects(void);
extern void close_invariant_rw_effects(void);
extern void store_invariant_rw_effects(statement /*k*/, effects /*v*/);
extern void update_invariant_rw_effects(statement /*k*/, effects /*v*/);
extern effects load_invariant_rw_effects(statement /*k*/);
extern effects delete_invariant_rw_effects(statement /*k*/);
extern bool bound_invariant_rw_effects_p(statement /*k*/);
extern void store_or_update_invariant_rw_effects(statement /*k*/, effects /*v*/);
extern bool cumulated_rw_effects_undefined_p(void);
extern void reset_cumulated_rw_effects(void);
extern void error_reset_cumulated_rw_effects(void);
extern void set_cumulated_rw_effects(statement_effects /*o*/);
extern statement_effects get_cumulated_rw_effects(void);
extern void init_cumulated_rw_effects(void);
extern void close_cumulated_rw_effects(void);
extern void store_cumulated_rw_effects(statement /*k*/, effects /*v*/);
extern void update_cumulated_rw_effects(statement /*k*/, effects /*v*/);
extern effects load_cumulated_rw_effects(statement /*k*/);
extern effects delete_cumulated_rw_effects(statement /*k*/);
extern bool bound_cumulated_rw_effects_p(statement /*k*/);
extern void store_or_update_cumulated_rw_effects(statement /*k*/, effects /*v*/);
extern bool expr_prw_effects_undefined_p(void);
extern void reset_expr_prw_effects(void);
extern void error_reset_expr_prw_effects(void);
extern void set_expr_prw_effects(persistant_expression_to_effects /*o*/);
extern persistant_expression_to_effects get_expr_prw_effects(void);
extern void init_expr_prw_effects(void);
extern void close_expr_prw_effects(void);
extern void store_expr_prw_effects(expression /*k*/, effects /*v*/);
extern void update_expr_prw_effects(expression /*k*/, effects /*v*/);
extern effects load_expr_prw_effects(expression /*k*/);
extern effects delete_expr_prw_effects(expression /*k*/);
extern bool bound_expr_prw_effects_p(expression /*k*/);
extern void store_or_update_expr_prw_effects(expression /*k*/, effects /*v*/);
extern list load_proper_rw_effects_list(statement /*s*/);
extern void store_proper_rw_effects_list(statement /*s*/, list /*l_eff*/);
extern void update_proper_rw_effects_list(statement /*s*/, list /*l_eff*/);
extern list load_rw_effects_list(statement /*s*/);
extern void store_rw_effects_list(statement /*s*/, list /*l_eff*/);
extern void update_rw_effects_list(statement /*s*/, list /*l_eff*/);
extern list load_invariant_rw_effects_list(statement /*s*/);
extern void store_invariant_rw_effects_list(statement /*s*/, list /*l_eff*/);
extern void update_invariant_rw_effects_list(statement /*s*/, list /*l_eff*/);
extern list load_cumulated_rw_effects_list(statement /*s*/);
extern void store_cumulated_rw_effects_list(statement /*s*/, list /*l_eff*/);
extern void update_cumulated_rw_effects_list(statement /*s*/, list /*l_eff*/);
extern bool in_effects_undefined_p(void);
extern void reset_in_effects(void);
extern void error_reset_in_effects(void);
extern void set_in_effects(statement_effects /*o*/);
extern statement_effects get_in_effects(void);
extern void init_in_effects(void);
extern void close_in_effects(void);
extern void store_in_effects(statement /*k*/, effects /*v*/);
extern void update_in_effects(statement /*k*/, effects /*v*/);
extern effects load_in_effects(statement /*k*/);
extern effects delete_in_effects(statement /*k*/);
extern bool bound_in_effects_p(statement /*k*/);
extern void store_or_update_in_effects(statement /*k*/, effects /*v*/);
extern bool cumulated_in_effects_undefined_p(void);
extern void reset_cumulated_in_effects(void);
extern void error_reset_cumulated_in_effects(void);
extern void set_cumulated_in_effects(statement_effects /*o*/);
extern statement_effects get_cumulated_in_effects(void);
extern void init_cumulated_in_effects(void);
extern void close_cumulated_in_effects(void);
extern void store_cumulated_in_effects(statement /*k*/, effects /*v*/);
extern void update_cumulated_in_effects(statement /*k*/, effects /*v*/);
extern effects load_cumulated_in_effects(statement /*k*/);
extern effects delete_cumulated_in_effects(statement /*k*/);
extern bool bound_cumulated_in_effects_p(statement /*k*/);
extern void store_or_update_cumulated_in_effects(statement /*k*/, effects /*v*/);
extern bool invariant_in_effects_undefined_p(void);
extern void reset_invariant_in_effects(void);
extern void error_reset_invariant_in_effects(void);
extern void set_invariant_in_effects(statement_effects /*o*/);
extern statement_effects get_invariant_in_effects(void);
extern void init_invariant_in_effects(void);
extern void close_invariant_in_effects(void);
extern void store_invariant_in_effects(statement /*k*/, effects /*v*/);
extern void update_invariant_in_effects(statement /*k*/, effects /*v*/);
extern effects load_invariant_in_effects(statement /*k*/);
extern effects delete_invariant_in_effects(statement /*k*/);
extern bool bound_invariant_in_effects_p(statement /*k*/);
extern void store_or_update_invariant_in_effects(statement /*k*/, effects /*v*/);
extern bool out_effects_undefined_p(void);
extern void reset_out_effects(void);
extern void error_reset_out_effects(void);
extern void set_out_effects(statement_effects /*o*/);
extern statement_effects get_out_effects(void);
extern void init_out_effects(void);
extern void close_out_effects(void);
extern void store_out_effects(statement /*k*/, effects /*v*/);
extern void update_out_effects(statement /*k*/, effects /*v*/);
extern effects load_out_effects(statement /*k*/);
extern effects delete_out_effects(statement /*k*/);
extern bool bound_out_effects_p(statement /*k*/);
extern void store_or_update_out_effects(statement /*k*/, effects /*v*/);
extern list load_in_effects_list(statement /*s*/);
extern void store_in_effects_list(statement /*s*/, list /*l_eff*/);
extern void update_in_effects_list(statement /*s*/, list /*l_eff*/);
extern list load_cumulated_in_effects_list(statement /*s*/);
extern void store_cumulated_in_effects_list(statement /*s*/, list /*l_eff*/);
extern void update_cummulated_in_effects_list(statement /*s*/, list /*l_eff*/);
extern list load_invariant_in_effects_list(statement /*s*/);
extern void store_invariant_in_effects_list(statement /*s*/, list /*l_eff*/);
extern void update_invariant_in_effects_list(statement /*s*/, list /*l_eff*/);
extern list load_out_effects_list(statement /*s*/);
extern void store_out_effects_list(statement /*s*/, list /*l_eff*/);
extern void update_out_effects_list(statement /*s*/, list /*l_eff*/);
extern list load_statement_local_regions(statement /*s*/);
extern void store_statement_local_regions(statement /*s*/, list /*t*/);
extern void update_statement_local_regions(statement /*s*/, list /*t*/);
extern list load_statement_inv_regions(statement /*s*/);
extern void store_statement_inv_regions(statement /*s*/, list /*t*/);
extern void update_statement_inv_regions(statement /*s*/, list /*t*/);
extern list load_statement_proper_regions(statement /*s*/);
extern void store_statement_proper_regions(statement /*s*/, list /*t*/);
extern list load_statement_in_regions(statement /*s*/);
extern void store_statement_in_regions(statement /*s*/, list /*t*/);
extern list load_statement_inv_in_regions(statement /*s*/);
extern void store_statement_inv_in_regions(statement /*s*/, list /*t*/);
extern void update_statement_inv_in_regions(statement /*s*/, list /*t*/);
extern list load_statement_cumulated_in_regions(statement /*s*/);
extern void store_statement_cumulated_in_regions(statement /*s*/, list /*t*/);
extern list load_statement_out_regions(statement /*s*/);
extern void store_statement_out_regions(statement /*s*/, list /*t*/);
/* unary_operators.c */
extern void effects_map(list /*l_eff*/, void (* /*apply*/)(effect));
extern list effects_to_effects_map(list /*l_eff*/, effect (* /*pure_apply*/)(effect));
extern void effects_filter_map(list /*l_eff*/, bool (* /*filter*/)(effect), void (* /*apply*/)(effect));
extern list effects_to_effects_filter_map(list /*l_eff*/, bool (* /*filter*/)(effect), effect (* /*pure_apply*/)(effect));
extern list effects_add_effect(list /*l_eff*/, effect /*eff*/);
extern list effects_read_effects(list /*l_eff*/);
extern list effects_write_effects(list /*l_eff*/);
extern list effects_read_effects_dup(list /*l_eff*/);
extern list effects_write_effects_dup(list /*l_eff*/);
extern effect effect_nop(effect /*eff*/);
extern list effects_nop(list /*l_eff*/);
extern void effect_to_may_effect(effect /*eff*/);
extern void effects_to_may_effects(list /*l_eff*/);
extern void effect_to_write_effect(effect /*eff*/);
extern void effects_to_write_effects(list /*l_eff*/);
extern void array_effects_to_may_effects(list /*l_eff*/);
extern list effects_dup_without_variables(list /*l_eff*/, list /*l_var*/);
extern effect effect_dup(effect /*eff*/);
extern list effects_dup(list /*l_eff*/);
extern void effect_free(effect /*eff*/);
extern void effects_free(list /*l_eff*/);
extern list effect_to_nil_list(effect /*eff*/);
extern list effects_to_nil_list(effect /*eff1*/, effect /*eff2*/);
extern list effect_to_list(effect /*eff*/);
extern list effect_to_may_effect_list(effect /*eff*/);
extern list effects_undefined_composition_with_transformer(list /*l_eff*/, transformer /*trans*/);
extern list effects_composition_with_transformer_nop(list /*l_eff*/, transformer /*trans*/);
extern list effects_undefined_composition_with_preconditions(list /*l_eff*/, transformer /*trans*/);
extern list effects_composition_with_preconditions_nop(list /*l_eff*/, transformer /*trans*/);
extern descriptor loop_undefined_descriptor_make(loop /*l*/);
extern list effects_undefined_union_over_range(list /*l_eff*/, entity /*index*/, range /*r*/, descriptor /*d*/);
extern list effects_union_over_range_nop(list /*l_eff*/, entity /*index*/, range /*r*/, descriptor /*d*/);
extern list effects_undefined_descriptors_variable_change(list /*l_eff*/, entity /*orig_ent*/, entity /*new_ent*/);
extern list effects_descriptors_variable_change_nop(list /*l_eff*/, entity /*orig_ent*/, entity /*new_ent*/);
extern descriptor effects_undefined_vector_to_descriptor(Pvecteur /*v*/);
extern list effects_undefined_loop_normalize(list /*l_eff*/, entity /*index*/, range /*r*/, entity */*new_index*/, descriptor /*range_descriptor*/, bool /*descriptor_update_p*/);
extern list effects_loop_normalize_nop(list /*l_eff*/, entity /*index*/, range /*r*/, entity */*new_index*/, descriptor /*range_descriptor*/, bool /*descriptor_update_p*/);
extern list db_get_empty_list(string /*name*/);
/* binary_operators.c */
extern list list_of_effects_generic_binary_op(list /*l1*/, list /*l2*/, bool (* /*r1_r2_combinable_p*/)(effect, effect), list (* /*r1_r2_binary_op*/)(effect, effect), list (* /*r1_unary_op*/)(effect), list (* /*r2_unary_op*/)(effect));
extern list proper_to_summary_effects(list /*l_effects*/);
extern list proper_effects_contract(list /*l_effects*/);
extern list proper_effects_combine(list /*l_effects*/, bool /*scalars_only_p*/);
extern bool combinable_effects_p(effect /*eff1*/, effect /*eff2*/);
extern bool effects_same_action_p(effect /*eff1*/, effect /*eff2*/);
extern bool effects_same_variable_p(effect /*eff1*/, effect /*eff2*/);
extern bool r_r_combinable_p(effect /*eff1*/, effect /*eff2*/);
extern bool w_w_combinable_p(effect /*eff1*/, effect /*eff2*/);
extern bool r_w_combinable_p(effect /*eff1*/, effect /*eff2*/);
extern bool w_r_combinable_p(effect /*eff1*/, effect /*eff2*/);
extern list effects_undefined_binary_operator(list /*l1*/, list /*l2*/, bool (* /*effects_combinable_p*/)(effect, effect));
extern list effects_entities_intersection(list /*l1*/, list /*l2*/, bool (* /*intersection_combinable_p*/)(effect, effect));
extern list effects_entities_inf_difference(list /*l1*/, list /*l2*/, bool (* /*difference_combinable_p*/)(effect, effect));
/* utils.c */
extern void (*effects_computation_init_func)(string);
extern void (*effects_computation_reset_func)(string);
extern effect (*effect_dup_func)(effect eff);
extern void (*effect_free_func)(effect eff);
extern effect (*reference_to_effect_func)(reference, action);
extern effect (*effect_union_op)(effect, effect);
extern list (*effects_union_op)(list, list, bool (*eff1_eff2_combinable_p)(effect, effect));
extern list (*effects_test_union_op)(list, list, bool (*eff1_eff2_combinable_p)(effect, effect));
extern list (*effects_intersection_op)(list, list, bool (*eff1_eff2_combinable_p)(effect, effect));
extern list (*effects_sup_difference_op)(list, list, bool (*eff1_eff2_combinable_p)(effect, effect));
extern list (*effects_inf_difference_op)(list, list, bool (*eff1_eff2_combinable_p)(effect, effect));
extern list (*effects_transformer_composition_op)(list, transformer);
extern list (*effects_transformer_inverse_composition_op)(list, transformer);
extern list (*effects_precondition_composition_op)(list, transformer);
extern list (*effects_descriptors_variable_change_func)(list, entity, entity);
extern descriptor (*loop_descriptor_make_func)(loop);
extern list (*effects_loop_normalize_func)(list, entity, range, entity *, descriptor, bool);
extern list (*effects_union_over_range_op)(list, entity, range, descriptor);
extern descriptor (*vector_to_descriptor_func)(Pvecteur);
extern list (*effects_backward_translation_op)(entity, list, list, transformer);
extern list (*effects_forward_translation_op)(entity, list, list, transformer);
extern list (*effects_local_to_global_translation_op)(list);
extern transformer (*load_context_func)(statement);
extern transformer (*load_transformer_func)(statement);
extern bool (*empty_context_test)(transformer);
extern effect (*proper_to_summary_effect_func)(effect);
extern void (*effects_descriptor_normalize_func)(list);
extern statement_effects (*db_get_proper_rw_effects_func)(char *);
extern void (*db_put_proper_rw_effects_func)(char *, statement_effects);
extern statement_effects (*db_get_invariant_rw_effects_func)(char *);
extern void (*db_put_invariant_rw_effects_func)(char *, statement_effects);
extern statement_effects (*db_get_rw_effects_func)(char *);
extern void (*db_put_rw_effects_func)(char *, statement_effects);
extern list (*db_get_summary_rw_effects_func)(char *);
extern void (*db_put_summary_rw_effects_func)(char *, list);
extern statement_effects (*db_get_in_effects_func)(char *);
extern void (*db_put_in_effects_func)(char *, statement_effects);
extern statement_effects (*db_get_cumulated_in_effects_func)(char *);
extern void (*db_put_cumulated_in_effects_func)(char *, statement_effects);
extern statement_effects (*db_get_invariant_in_effects_func)(char *);
extern void (*db_put_invariant_in_effects_func)(char *, statement_effects);
extern list (*db_get_summary_in_effects_func)(char *);
extern void (*db_put_summary_in_effects_func)(char *, list);
extern list (*db_get_summary_out_effects_func)(char *);
extern void (*db_put_summary_out_effects_func)(char *, list);
extern statement_effects (*db_get_out_effects_func)(char *);
extern void (*db_put_out_effects_func)(char *, statement_effects);
extern void (*effects_prettyprint_func)(list);
extern text (*effects_to_text_func)(list);
extern void (*attach_effects_decoration_to_text_func)(text);
extern void generic_effects_reset_all_methods(void);
extern void make_effects_private_current_stmt_stack(void);
extern void free_effects_private_current_stmt_stack(void);
extern stack get_effects_private_current_stmt_stack(void);
extern void set_effects_private_current_stmt_stack(stack /*s*/);
extern void reset_effects_private_current_stmt_stack(void);
extern void effects_private_current_stmt_push(statement /*i*/);
extern bool effects_private_current_stmt_filter(statement /*i*/);
extern void effects_private_current_stmt_rewrite(statement /*i*/);
extern statement effects_private_current_stmt_replace(statement /*i*/);
extern statement effects_private_current_stmt_pop(void);
extern statement effects_private_current_stmt_head(void);
extern bool effects_private_current_stmt_empty_p(void);
extern int effects_private_current_stmt_size(void);
extern void error_reset_effects_private_current_stmt_stack(void);
extern void make_effects_private_current_context_stack(void);
extern void free_effects_private_current_context_stack(void);
extern stack get_effects_private_current_context_stack(void);
extern void set_effects_private_current_context_stack(stack /*s*/);
extern void reset_effects_private_current_context_stack(void);
extern void effects_private_current_context_push(transformer /*i*/);
extern bool effects_private_current_context_filter(transformer /*i*/);
extern void effects_private_current_context_rewrite(transformer /*i*/);
extern transformer effects_private_current_context_replace(transformer /*i*/);
extern transformer effects_private_current_context_pop(void);
extern transformer effects_private_current_context_head(void);
extern bool effects_private_current_context_empty_p(void);
extern int effects_private_current_context_size(void);
extern void error_reset_effects_private_current_context_stack(void);
extern bool effects_private_current_context_stack_initialized_p(void);
extern bool normalizable_and_linear_loop_p(entity /*index*/, range /*l_range*/);
extern transformer transformer_remove_variable_and_dup(transformer /*orig_trans*/, entity /*ent*/);
extern tag approximation_and(tag /*t1*/, tag /*t2*/);
extern tag approximation_or(tag /*t1*/, tag /*t2*/);
extern void set_descriptor_range_p(bool /*b*/);
extern bool get_descriptor_range_p(void);
extern descriptor descriptor_inequality_add(descriptor /*d*/, Pvecteur /*v*/);
extern transformer descriptor_to_context(descriptor /*d*/);
extern void descriptor_variable_rename(descriptor /*d*/, entity /*old_ent*/, entity /*new_ent*/);
extern descriptor descriptor_append(descriptor /*d1*/, descriptor /*d2*/);
extern transformer load_undefined_context(statement /*s*/);
extern transformer load_undefined_transformer(statement /*s*/);
extern bool empty_context_test_false(transformer /*context*/);
extern void effects_computation_no_init(string /*module_name*/);
extern void effects_computation_no_reset(string /*module_name*/);
extern string vect_debug_entity_name(entity /*e*/);
extern bool integer_scalar_read_effects_p(cons */*fx*/);
extern bool some_integer_scalar_read_or_write_effects_p(cons */*fx*/);
extern bool effects_write_entity_p(cons */*fx*/, entity /*e*/);
extern bool effects_read_or_write_entity_p(cons */*fx*/, entity /*e*/);
extern entity effects_conflict_with_entity(cons */*fx*/, entity /*e*/);
extern list effects_conflict_with_entities(cons */*fx*/, entity /*e*/);
extern bool io_effect_entity_p(entity /*e*/);
extern bool statement_io_effect_p(statement /*s*/);
extern bool statement_has_a_formal_argument_write_effect_p(statement /*s*/);
extern list make_effects_for_array_declarations(list /*refs*/);
extern list extract_references_from_declarations(list /*decls*/);
extern list summary_effects_from_declaration(string /*module_name*/);
/* prettyprint.c */
extern void set_action_interpretation(string /*r*/, string /*w*/);
extern void reset_action_interpretation(void);
extern string action_interpretation(int tag);
extern void set_is_user_view_p(bool /*user_view_p*/);
extern void set_prettyprint_with_attachments(bool /*attachments_p*/);
extern void reset_generic_prettyprints(void);
extern void set_a_generic_prettyprint(string /*resource_name*/, bool /*is_a_summary*/, gen_chunk */*res*/, generic_text_function /*tf*/, generic_prettyprint_function /*tp*/, generic_attachment_function /*ta*/);
extern void add_a_generic_prettyprint(string /*resource_name*/, bool /*is_a_summary*/, generic_text_function /*tf*/, generic_prettyprint_function /*tp*/, generic_attachment_function /*ta*/);
extern bool print_source_or_code_effects_engine(string /*module_name*/, string /*file_suffix*/);
extern list effect_words_reference(reference /*obj*/);
extern text get_any_effect_type_text(string /*module_name*/, string /*resource_name*/, string /*summary_resource_name*/, bool /*give_code_p*/);
extern bool print_source_or_code_with_any_effects_engine(string /*module_name*/, string /*resource_name*/, string /*summary_resource_name*/, string /*file_suffix*/);
/* intrinsics.c */
extern list generic_proper_effects_of_intrinsic(entity /*e*/, list /*args*/);
#endif /* effects_generic_header_included */
