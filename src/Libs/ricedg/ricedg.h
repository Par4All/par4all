/* 	$Id$	 */

/* header file built by cproto */
#ifndef ricedg_header_included
#define ricedg_header_included
#define INFAISABLE 0
#define FAISABLE 1

/* maximun number of nested loops */
#define MAXDEPTH 9
#define MAXSV 100


/*the variables for the statistics of test of dependence and parallelization */
extern int NbrArrayDepInit;
extern int NbrIndepFind;
extern int NbrAllEquals;
extern int NbrDepCnst;
extern int NbrDepExact;
extern int NbrDepInexactEq;
extern int NbrDepInexactFM;
extern int NbrDepInexactEFM;
extern int NbrScalDep;
extern int NbrIndexDep;
extern int deptype[5][3], constdep[5][3];
extern int NbrTestCnst;
extern int NbrTestGcd;
extern int NbrTestSimple; /* by sc_normalize() */
extern int NbrTestDiCnst;
extern int NbrTestProjEqDi;
extern int NbrTestProjFMDi;
extern int NbrTestProjEq;
extern int NbrTestProjFM;
extern int NbrTestDiVar;
extern int NbrProjFMTotal;
extern int NbrFMSystNonAug;
extern int FMComp[17]; 
extern boolean is_test_exact;
extern boolean is_test_inexact_eq;
extern boolean is_test_inexact_fm;
extern boolean is_dep_cnst;
extern boolean is_test_Di;
extern boolean Finds2s1;

extern int Nbrdo;

/* Definition for the dependance_verticies_p function
 */

#define FLOW_DEPENDANCE 1
#define ANTI_DEPENDANCE 2
#define OUTPUT_DEPENDANCE 4
#define INPUT_DEPENDANCE 8




/* util.c */
extern statement vertex_to_statement(vertex /*v*/);
extern int vertex_to_ordering(vertex /*v*/);
extern hash_table compute_ordering_to_dg_mapping(graph /*dependance_graph*/);
extern void prettyprint_dependence_graph(FILE */*fd*/, statement /*mod_stat*/, graph /*mod_graph*/);
extern void prettyprint_dependence_graph_view(FILE */*fd*/, statement /*mod_stat*/, graph /*mod_graph*/);
extern void print_vect_in_vertice_val(FILE */*fd*/, Pvecteur /*v*/, Pbase /*b*/);
extern void print_dependence_cone(FILE */*fd*/, Ptsg /*dc*/, Pbase /*basis*/);
extern Psysteme sc_restricted_to_variables_transitive_closure(Psysteme /*sc*/, Pbase /*variables*/);
/* contexts.c */
extern statement_mapping contexts_mapping_of_nest(statement /*stat*/);
/* testdep_util.c */
extern entity DiVars[9 ];
extern entity LiVars[9 ];
extern entity DsiVars[100 ];
extern entity MakeDiVar(int /*l*/);
extern entity GetDiVar(int /*l*/);
extern entity MakeLiVar(int /*l*/);
extern entity GetLiVar(int /*l*/);
extern entity MakeDsiVar(int /*l*/);
extern entity GetDsiVar(int /*l*/);
extern int DiVarLevel(entity /*e*/);
extern void sc_add_di(int /*l*/, entity /*e*/, Psysteme /*s*/, int /*li*/);
extern void sc_add_dsi(int /*l*/, entity /*e*/, Psysteme /*s*/);
extern int sc_proj_on_di(int /*cl*/, Psysteme /*s*/);
extern Pbase MakeDibaseinorder(int /*n*/);
extern int FindMaximumCommonLevel(cons */*n1*/, cons */*n2*/);
extern void ResetLoopCounter(void);
extern entity MakeLoopCounter(void);
extern int dep_type(action /*ac1*/, action /*ac2*/);
extern int sc_proj_optim_on_di_ofl(int /*cl*/, Psysteme */*psc*/);
extern boolean sc_faisabilite_optim(Psysteme /*sc*/);
extern Psysteme sc_projection_optim_along_vecteur_ofl(Psysteme /*sc*/, Pvecteur /*pv*/);
extern boolean sc_minmax_of_variable_optim(Psysteme /*ps*/, Variable /*var*/, Value */*pmin*/, Value */*pmax*/);
extern Psysteme sc_invers(Psysteme /*ps*/);
extern void vect_chg_var_sign(Pvecteur */*ppv*/, Variable /*var*/);
/* ricedg.c */
extern int NbrArrayDepInit;
extern int NbrIndepFind;
extern int NbrAllEquals;
extern int NbrDepCnst;
extern int NbrTestExact;
extern int NbrDepInexactEq;
extern int NbrDepInexactFM;
extern int NbrDepInexactEFM;
extern int NbrScalDep;
extern int NbrIndexDep;
extern int deptype[5][3];
extern int constdep[5][3];
extern int NbrTestCnst;
extern int NbrTestGcd;
extern int NbrTestSimple;
extern int NbrTestDiCnst;
extern int NbrTestProjEqDi;
extern int NbrTestProjFMDi;
extern int NbrTestProjEq;
extern int NbrTestProjFM;
extern int NbrTestDiVar;
extern int NbrProjFMTotal;
extern int NbrFMSystNonAug;
extern int FMComp[17];
extern boolean is_test_exact;
extern boolean is_test_inexact_eq;
extern boolean is_test_inexact_fm;
extern boolean is_dep_cnst;
extern boolean is_test_Di;
extern boolean Finds2s1;
extern int Nbrdo;
extern bool context_map_undefined_p(void);
extern void set_context_map(statement_mapping /*m*/);
extern statement_mapping get_context_map(void);
extern void reset_context_map(void);
extern void free_context_map(void);
extern void make_context_map(void);
extern Psysteme load_statement_context(statement /*s*/);
extern bool statement_context_undefined_p(statement /*s*/);
extern void store_statement_context(statement /*s*/, Psysteme /*t*/);
extern void update_statement_context(statement /*s*/, Psysteme /*t*/);
extern bool rice_fast_dependence_graph(char */*mod_name*/);
extern bool rice_full_dependence_graph(char */*mod_name*/);
extern bool rice_semantics_dependence_graph(char */*mod_name*/);
extern bool rice_regions_dependence_graph(char */*mod_name*/);
extern list TestCoupleOfReferences(list /*n1*/, Psysteme /*sc1*/, statement /*s1*/, effect /*ef1*/, reference /*r1*/, list /*n2*/, Psysteme /*sc2*/, statement /*s2*/, effect /*ef2*/, reference /*r2*/, list /*llv*/, Ptsg */*gs*/, list */*levelsop*/, Ptsg */*gsop*/);
extern void writeresult(char */*mod_name*/);
/* prettyprint.c */
extern bool print_whole_dependence_graph(string /*mod_name*/);
extern bool print_filtered_dependence_graph(string /*mod_name*/);
extern bool print_filtered_dependence_daVinci_graph(string /*mod_name*/);
extern bool print_effective_dependence_graph(string /*mod_name*/);
extern bool print_loop_carried_dependence_graph(string /*mod_name*/);
extern bool print_dependence_graph(string /*name*/);
extern bool print_chains_graph(string /*name*/);
/* quick_privatize.c */
extern void quick_privatize_graph(graph /*dep_graph*/);
/* trace.c */
extern vertex get_vertex_in_list(list /*in_l*/, string /*in_s*/);
extern void prettyprint_graph_text(FILE */*out_f*/, list /*l_of_vers*/);
extern void prettyprint_graph_daVinci(FILE */*out_f*/, list /*l_of_vers*/);
extern graph make_filtered_dependence_graph(graph /*mod_graph*/);
extern list make_filtered_dg_or_dvdg(statement /*mod_stat*/, graph /*mod_graph*/);
extern bool print_filtered_dg_or_dvdg(string /*mod_name*/, bool /*is_dv*/);
/* impact.c */
extern bool impact_check(string /*module_name*/);
#endif /* ricedg_header_included */
