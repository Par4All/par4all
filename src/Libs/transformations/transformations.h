/* header file built by cproto */
#ifndef transformations_header_included
#define transformations_header_included
#define SIGN_EQ(a,b) ((((a)>0 && (b)>0) || ((a)<0 && (b)<0)) ? TRUE : FALSE)
#define FORTRAN_DIV(n,d) (SIGN_EQ((n),(d)) ? ABS(n)/ABS(d) : -(ABS(n)/ABS(d)))
#define FORTRAN_MOD(n,m) (SIGN_EQ((n),(m)) ? ABS(n)%ABS(m) : -(ABS(n)%ABS(m)))

/* What is returned by dead_test_filter : */
enum dead_test { nothing_about_test, then_is_dead, else_is_dead };
typedef enum dead_test dead_test;
/* replace.c */
extern bool simple_ref_eq_p(reference /*r1*/, reference /*r2*/);
extern void StatementReplaceReference(statement /*s*/, reference /*ref*/, expression /*next*/);
extern void ExpressionReplaceReference(expression /*e*/, reference /*ref*/, expression /*next*/);
extern void RangeReplaceReference(range /*r*/, reference /*ref*/, expression /*next*/);
extern void CallReplaceReference(call /*c*/, reference /*ref*/, expression /*next*/);
extern void ReplaceReference(char */*mod_name*/, reference /*ref*/, expression /*next_expr*/);
/* loop_unroll.c */
extern expression make_ref_expr(entity /*ent*/, cons */*args*/);
extern entity find_final_statement_label(statement /*s*/);
extern void loop_unroll(statement /*loop_statement*/, int /*rate*/);
extern void full_loop_unroll(statement /*loop_statement*/);
extern bool recursiv_loop_unroll(statement /*stmt*/, entity /*lb_ent*/, int /*rate*/);
extern bool unroll(char */*mod_name*/);
extern bool find_loop_and_fully_unroll(statement /*s*/);
extern bool full_unroll(char */*mod_name*/);
extern bool find_unroll_pragma_and_fully_unroll(statement /*s*/);
extern bool full_unroll_pragma(char */*mod_name*/);
/* prettyprintcray.c */
extern bool same_entity_name_p(entity /*e1*/, entity /*e2*/);
extern bool entity_in_list(entity /*ent*/, cons */*ent_l*/);
extern list concat_new_entities(list /*l1*/, list /*l2*/);
extern list real_loop_locals(loop /*lp*/, effects /*cfx*/);
extern list all_enclosed_scope_variables(statement /*stmt*/);
extern text text_microtasked_loop(entity /*module*/, int /*margin*/, statement /*lp_stt*/);
extern text text_vectorized_loop(entity /*module*/, int /*margin*/, statement /*lp_stt*/);
extern text text_cray(entity /*module*/, int /*margin*/, statement /*stat*/);
extern bool print_parallelizedcray_code(char */*mod_name*/);
/* strip_mine.c */
extern statement loop_strip_mine(statement /*loop_statement*/, int /*chunk_size*/, int /*chunk_number*/);
extern statement loop_chunk_size_and_strip_mine(cons */*lls*/);
extern bool strip_mine(char */*mod_name*/);
/* interactive_loop_transformation.c */
extern entity selected_label;
extern bool selected_loop_p(loop /*l*/);
extern bool interactive_loop_transformation(string /*module_name*/, statement (* /*loop_transformation*/)(void));
/* loop_interchange.c */
extern bool loop_interchange(string /*module_name*/);
/* interchange.c */
extern statement gener_DOSEQ(list /*lls*/, Pvecteur */*pvg*/, Pbase /*base_oldindex*/, Pbase /*base_newindex*/, Psysteme /*sc_newbase*/);
extern statement interchange(cons */*lls*/);
extern statement interchange_two_loops(list /*lls*/, int /*n1*/, int /*n2*/);
/* target.c */
extern int get_cache_line_size(void);
extern int get_processor_number(void);
extern int get_vector_register_length(void);
extern int get_vector_register_number(void);
extern int get_minimal_task_size(void);
/* nest_parallelization.c */
extern statement loop_preserve(statement /*s*/, int /*c*/);
extern statement loop_vectorize(statement /*s*/, int /*c*/);
extern statement tuned_loop_parallelize(statement /*s*/, int /*c*/);
extern statement tuned_loop_unroll(statement /*s*/, int /*c*/);
extern bool current_loop_index_p(reference /*r*/);
extern statement tuned_loop_strip_mine(statement /*s*/);
extern bool nest_parallelization(string /*module_name*/);
extern statement parallelization(list /*lls*/, bool (* /*loop_predicate*/)(void));
extern statement one_loop_parallelization(statement /*s*/);
extern reference reference_identity(reference /*r*/);
extern bool constant_array_reference_p(reference /*r*/);
extern statement loop_nest_parallelization(list /*lls*/);
extern statement mark_loop_as_parallel(list /*lls*/);
extern bool nth_loop_p(statement /*ls*/);
extern int numerical_loop_iteration_count(loop /*l*/);
extern Pvecteur estimate_loop_iteration_count(loop /*l*/);
extern Pvecteur estimate_range_count(range /*r*/);
extern bool contiguous_array_reference_p(reference /*r*/);
extern bool carried_dependence_p(statement /*s*/);
extern int look_for_references_in_statement(statement /*s*/, statement (* /*reference_transformation*/)(void), bool (* /*reference_predicate*/)(void));
extern int look_for_references_in_expression(expression /*e*/, statement (* /*reference_transformation*/)(void), bool (* /*reference_predicate*/)(void));
extern int look_for_references_in_range(range /*r*/, statement (* /*reference_transformation*/)(void), bool (* /*reference_predicate*/)(void));
extern int look_for_references_in_call(call /*c*/, statement (* /*reference_transformation*/)(void), bool (* /*reference_predicate*/)(void));
/* coarse_grain_parallelization.c */
extern void coarse_grain_parallelization_error_handler(void);
extern bool coarse_grain_parallelization(string /*module_name*/);
/* dead_code_elimination.c */
extern void suppress_dead_code_statement(statement /*mod_stmt*/);
extern bool suppress_dead_code(char */*mod_name*/);
extern bool statement_write_effect_p(statement /*s*/);
/* trivial_test_elimination.c */
extern void suppress_trivial_test_statement(statement /*mod_stmt*/);
extern bool suppress_trivial_test(char */*mod_name*/);
/* declaration_table_normalization.c */
extern void normalize_declaration_table_region(list /*lglob*/, list /*ldecls*/, Psysteme /*sum_prec*/);
extern bool normalize_declaration_table(char */*mod_name*/);
/* privatize.c */
extern bool is_implied_do_index(entity /*e*/, instruction /*ins*/);
extern bool privatize_module(char */*mod_name*/);
/* array_privatization.c */
extern bool private_effects_undefined_p(void);
extern void reset_private_effects(void);
extern void error_reset_private_effects(void);
extern void set_private_effects(statement_effects /*o*/);
extern statement_effects get_private_effects(void);
extern void init_private_effects(void);
extern void close_private_effects(void);
extern void store_private_effects(statement /*k*/, effects /*v*/);
extern void update_private_effects(statement /*k*/, effects /*v*/);
extern effects load_private_effects(statement /*k*/);
extern effects delete_private_effects(statement /*k*/);
extern bool bound_private_effects_p(statement /*k*/);
extern void store_or_update_private_effects(statement /*k*/, effects /*v*/);
extern bool copy_out_effects_undefined_p(void);
extern void reset_copy_out_effects(void);
extern void error_reset_copy_out_effects(void);
extern void set_copy_out_effects(statement_effects /*o*/);
extern statement_effects get_copy_out_effects(void);
extern void init_copy_out_effects(void);
extern void close_copy_out_effects(void);
extern void store_copy_out_effects(statement /*k*/, effects /*v*/);
extern void update_copy_out_effects(statement /*k*/, effects /*v*/);
extern effects load_copy_out_effects(statement /*k*/);
extern effects delete_copy_out_effects(statement /*k*/);
extern bool bound_copy_out_effects_p(statement /*k*/);
extern void store_or_update_copy_out_effects(statement /*k*/, effects /*v*/);
extern void array_privatization_error_handler(void);
extern bool array_privatizer(char */*module_name*/);
extern bool array_section_privatizer(char */*module_name*/);
extern bool print_code_privatized_regions(string /*module_name*/);
extern bool declarations_privatizer(char */*mod_name*/);
/* standardize_structure.c */
extern bool stf(char */*mod_name*/);
/* use_def_elimination.c */
extern void use_def_elimination_error_handler(void);
extern void set_control_statement_father(control /*c*/);
extern void use_def_elimination_on_a_statement(statement /*s*/);
extern bool use_def_elimination(char */*module_name*/);
/* loop_normalize.c */
extern bool loop_normalize(char */*mod_name*/);
extern list ln_of_loop(loop /*l*/, hash_table /*fst*/, list */*ell*/, list */*etl*/, list */*swfl*/, int */*Gcount_nlc*/);
extern list ln_of_statement(statement /*s*/, hash_table /*fst*/, list */*ell*/, list */*etl*/, list */*swfl*/, int */*Gcount_nlc*/);
extern void ln_of_unstructured(unstructured /*u*/, hash_table /*fst*/, list */*ell*/, list */*etl*/, list */*swfl*/, int */*Gcount_nlc*/);
/* declarations.c */
extern bool clean_declarations(string /*name*/);
/* clone.c */
extern void clone_error_handler(void);
extern bool clone_on_argument(string /*name*/);
extern bool clone(string /*name*/);
extern bool clone_substitute(string /*name*/);
/* transformation_test.c */
extern bool blind_loop_distribution(char */*mod_name*/);
extern bool transformation_test(char */*mod_name*/);
#endif /* transformations_header_included */
