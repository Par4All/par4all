
/* Dependence Graph computation for Allen & Kennedy algorithm
 *
 * Remi Triolet
 *
 * Modifications:
 *  - new option to use semantics analysis results (Francois Irigoin, 
 *    12 April 1991)
 *
 *  - compute the dependence cone, the statistics.
 *    (Yi-Qing, August 1991) 
 *
 *  - updated using DEFINE_CURRENT_MAPPING, BA, September 3, 1993
 *
 *  - dg_type introduced to replace dg_fast and dg_semantics; it is more
 *    general. (BC, August 1995).
 *
 *  - TestDependence split into different procedures for more readability.
 *    (BC, August 1995).
 *
 * Notes:
 *  - Many values seem to be assigned to StatementToContext and never be
 *    freed;
 *
 */
#include <stdio.h>
#include <values.h>
#include <string.h>

#include <setjmp.h>

#include "genC.h"
#include "text.h"
#include "ri.h"
#include "graph.h"
#include "dg.h"
#include "database.h"

#include "misc.h"
#include "text-util.h"

#include "ri-util.h" /* linear.h is included in */
#include "control.h"
#include "effects.h"
#include "pipsdbm.h"
#include "semantics.h"

#include "constants.h"
#include "properties.h"
#include "resources.h"

/* includes pour system generateur */
#include "ray_dte.h"
#include "sommet.h"
#include "sg.h"
#include "polyedre.h"

#include "ricedg.h" 


/* local variables */
/*the variables for the statistics of test of dependence and parallelization */
/* they should not be global ? FC.
 */
int NbrArrayDepInit=0;
int NbrIndepFind=0;
int NbrAllEquals=0;
int NbrDepCnst=0;
int NbrTestExact=0;
int NbrDepInexactEq=0;
int NbrDepInexactFM=0;
int NbrDepInexactEFM=0;
int NbrScalDep=0;
int NbrIndexDep=0;
int deptype[5][3], constdep[5][3];
int NbrTestCnst=0;
int NbrTestGcd=0;
int NbrTestSimple=0; /* by sc_normalize() */
int NbrTestDiCnst=0;
int NbrTestProjEqDi=0;
int NbrTestProjFMDi=0;
int NbrTestProjEq=0;
int NbrTestProjFM=0;
int NbrTestDiVar=0;
int NbrProjFMTotal=0;
int NbrFMSystNonAug=0;
int FMComp[17];   /*for counting the number of F-M complexity less than 16.
		     The complexity of one projection by F-M is multiply 
		     of the nbr. of inequations positive and the nbr. of 
		     inequations negatives who containe the variable
		     eliminated.*/ 
boolean is_test_exact = TRUE;
boolean is_test_inexact_eq = FALSE;
boolean is_test_inexact_fm = FALSE;
boolean is_dep_cnst = FALSE;
boolean is_test_Di;
boolean Finds2s1;


jmp_buf overflow_error; /* to deal with overflow errors occuring during the projection 
                         * of a Psysteme along a variable */


int Nbrdo;


/* instantiation of the dependence graph */
typedef dg_arc_label arc_label;
typedef dg_vertex_label vertex_label;

/* to map statements to execution contexts */
      /* Psysteme_undefined is not defined in sc.h; as Psysteme is external,
       * I define it here. BA, September 1993 
       */
#define Psysteme_undefined SC_UNDEFINED
DEFINE_CURRENT_MAPPING(context, Psysteme)


/* to map statements to enclosing loops */
/* DEFINE_CURRENT_MAPPING(loops, list) now defined in ri-util, BA, September 1993 */


/* the dependence graph being updated */
static graph dg;


static bool PRINT_RSDG = FALSE;


/* Different types of dependence tests:
 *  
 * switch dg_type:
 * 
 *      DG_FAST: no context constraints are added
 *      DG_FULL: use loop bounds as context
 *      DG_SEMANTICS : use preconditions as context
 *
 * The use of the variable dg_type allows to add more cases in the future.
 */
#define DG_FAST 1
#define DG_FULL 2
#define DG_SEMANTICS 3

static dg_type = DG_FAST;

    
/*********************************************************************************/
/* INTERFACE FUNCTIONS                                                           */
/*********************************************************************************/

static bool rice_dependence_graph(char */*mod_name*/);


bool rice_fast_dependence_graph(mod_name)
char *mod_name;
{
    dg_type = DG_FAST;
    return rice_dependence_graph(mod_name);
}

bool rice_full_dependence_graph(mod_name)
char *mod_name;
{
    dg_type = DG_FULL;
    return rice_dependence_graph(mod_name);
}

bool rice_semantics_dependence_graph(mod_name)
char *mod_name;
{
    dg_type = DG_SEMANTICS;
    return rice_dependence_graph(mod_name);
}



/*********************************************************************************/
/* STATIC FUNCTION DECLARATIONS                                                  */
/*********************************************************************************/


static void rdg_unstructured(unstructured /*u*/);
static void rdg_statement(statement /*stat*/);
static void rdg_loop(statement /*stat*/);
static void rice_update_dependence_graph(statement /*stat*/, set /*region*/);
static list TestCoupleOfEffects(statement /*s1*/, effect /*e1*/, statement /*s2*/,
				 effect /*e2*/, list /*llv*/, Ptsg */*gs*/, 
				 list */*levelsop*/, Ptsg */*gsop*/);
static list TestCoupleOfReferences(list /*n1*/, Psysteme /*sc1*/, 
				    statement /*s1*/, effect /*ef1*/, 
				    reference /*r1*/, list /*n2*/, 
				    Psysteme /*sc2*/, statement /*s2*/, 
				    effect /*ef2*/, reference /*r2*/, 
				    list /*llv*/, Ptsg */*gs*/, 
				    list */*levelsop*/, Ptsg */*gsop*/);
static list TestDependence(list /*n1*/, Psysteme /*sc1*/, statement /*s1*/, 
			    effect /*ef1*/, reference /*r1*/, list /*n2*/, 
			    Psysteme /*sc2*/, statement /*s2*/, effect /*ef2*/, 
			    reference /*r2*/, list /*llv*/, Ptsg */*gs*/, 
			    list */*levelsop*/, Ptsg */*gsop*/);
static boolean build_and_test_dependence_context(reference /*r1*/, reference /*r2*/,
						 Psysteme /*sc1*/, Psysteme /*sc2*/,
						 Psysteme */*psc_dep*/, 
						 list /*llv*/, 
						 list /*s2_enc_loops*/);
static boolean gcd_and_constant_dependence_test(reference /*r1*/, reference /*r2*/,
						list /*llv*/, list /*s2_enc_loops*/,
						Psysteme */*psc_dep*/);
static void dependence_system_add_lci_and_di(Psysteme */*psc_dep*/,
					     list /*s1_enc_loops*/,
					     Pvecteur */*p_DiIncNonCons*/);
static list TestDiVariables(Psysteme /*ps*/, int /*cl*/, statement /*s1*/, 
			     effect /*ef1*/, statement /*s2*/, effect /*ef2*/);
static Ptsg dependence_cone_positive(Psysteme /*dep_sc*/);
static void quick_privatize_graph(graph /*dep_graph*/);
static bool quick_privatize_loop(statement /*stat*/, list /*successors*/);
static bool quick_privatize_statement_pair(statement /*s1*/, statement /*s2*/, 
					   list /*conflicts*/);
static list loop_variant_list(statement /*stat*/);
static boolean TestDiCnst(Psysteme /*ps*/, int /*cl*/, statement /*s1*/, 
			  effect /*ef1*/, statement /*s2*/, effect /*ef2*/);
static void writeresult(char */*mod_name*/);



/*********************************************************************************/
/* WALK THROUGH THE DEPENDENCE GRAPH                                             */
/*********************************************************************************/

/* The supplementary call to init_ordering_to_statement should be 
   avoided if ordering.c were more clever. */
static bool rice_dependence_graph(mod_name)
char *mod_name;
{
    FILE *fp;

    statement mod_stat;
    int i,j;
    graph chains;
    string dg_name;
    entity module = local_name_to_top_level_entity(mod_name);

    debug_on("RICEDG_DEBUG_LEVEL");
    debug(1,"rice_dependence_graph", 
	  "Computing Rice dependence graph for %s\n", mod_name);

    set_current_module_entity(module);

    set_current_module_statement( (statement)
	db_get_memory_resource(DBR_CODE, mod_name, TRUE) );
    mod_stat = get_current_module_statement();
   
    initialize_ordering_to_statement(mod_stat);

    chains = (graph)
	db_get_memory_resource(DBR_CHAINS, mod_name, TRUE);

    ifdebug(1) {
	gen_consistent_p (chains);
    }
    
    dg = gen_copy_tree (chains);

    debug(8,"rice_dependence_graph","original graph\n");
    ifdebug(8) {  	
	print_graph(stderr, mod_stat, dg);
    }
    
    if(dg_type == DG_SEMANTICS)
	set_precondition_map( (statement_mapping)
	    db_get_memory_resource(DBR_PRECONDITIONS, mod_name, TRUE) );

    set_cumulated_effects_map( effectsmap_to_listmap((statement_mapping) 
	db_get_memory_resource(DBR_CUMULATED_EFFECTS, mod_name, TRUE)) );

    debug(1, "rice_dependence_graph", "finding enclosing loops ...\n");

    set_enclosing_loops_map( loops_mapping_of_statement(mod_stat));
    ifdebug(3) {
	printf("\nThe number of DOs :\n");
	printf(" Nbrdo=%d",Nbrdo);
    }
   
    debug_on("QUICK_PRIVATIZER_DEBUG_LEVEL");
    quick_privatize_graph(dg);
    debug_off();

    for (i=0;i<=4;i++){
	for(j=0;j<=2;j++){
	    deptype[i][j] = 0;
	    constdep[i][j] = 0;
	}
    }
	    
    /* walk thru mod_stat to find well structured loops.
       update dependence graph for these loops. */
    rdg_statement(mod_stat);
  
    ifdebug(3) {
	printf("\nThe results of statistique of test of dependence are:\n");
	printf("NbrArrayDepInit = %d\n",NbrArrayDepInit); 
	printf("NbrIndepFind = %d\n",NbrIndepFind); 
	printf("NbrAllEquals = %d\n",NbrAllEquals);
	printf("NbrDepCnst = %d\n",NbrDepCnst);
	printf("NbrTestExact= %d\n",NbrTestExact);
	printf("NbrDepInexactEq= %d\n",NbrDepInexactEq);
	printf("NbrDepInexactFM= %d\n",NbrDepInexactFM);
	printf("NbrDepInexactEFM= %d\n",NbrDepInexactEFM);
	printf("NbrScalDep = %d\n",NbrScalDep);
	printf("NbrIndexDep = %d\n",NbrIndexDep);
	printf("deptype[][]");
	for (i=0;i<=4;i++)
	    for(j=0;j<=2;j++)
		printf("%d  ", deptype[i][j]);
	printf("\nconstdep[][]");
	for (i=0;i<=4;i++)
	    for(j=0;j<=2;j++)
		printf("%d  ", constdep[i][j]);
	printf("\nNbrTestCnst = %d\n",NbrTestCnst); 
	printf("NbrTestGcd = %d\n",NbrTestGcd);
	printf("NbrTestSimple = %d\n",NbrTestSimple);
	printf("NbrTestDiCnst = %d\n",NbrTestDiCnst);
	printf("NbrTestProjEqDi = %d\n",NbrTestProjEqDi);
	printf("NbrTestProjFMDi = %d\n",NbrTestProjFMDi);
	printf("NbrTestProjEq = %d\n",NbrTestProjEq);
	printf("NbrTestProjFM = %d\n",NbrTestProjFM);
	printf("NbrTestDiVar = %d\n",NbrTestDiVar);
	printf("NbrProjFMTotal = %d\n",NbrProjFMTotal);
	printf("NbrFMSystNonAug = %d\n",NbrFMSystNonAug);
	printf("FMComp[]");
	for (i=0;i<17;i++)
	    printf("%d ", FMComp[i]);
    }
    /* write the result to the file correspondant in the order of :
       module,NbrArrayDeptInit,NbrIndeptFind,NbrArrayDepInit,NbrScalDep,
       NbrIndexDep,deptype[5][3]*/
    if (get_bool_property(RICEDG_PROVIDE_STATISTICS))
	writeresult(mod_name);

    ifdebug(1) {
	fprintf(stderr, "updated graph\n");
	print_graph(stderr, mod_stat, dg);
    }

    /* FI: this is not a proper way to do it */
    if (get_bool_property("PRINT_DEPENDENCE_GRAPH") || PRINT_RSDG) {
	dg_name = strdup(concatenate(db_get_current_workspace_directory(), 
			             "/", mod_name, ".dg", NULL));
	fp = safe_fopen(dg_name, "w");
	print_graph(fp, mod_stat, dg);
	safe_fclose(fp,dg_name);
    }
    

    debug_off();

    DB_PUT_MEMORY_RESOURCE(DBR_DG, strdup(mod_name), (char*) dg);

    reset_current_module_entity();
    reset_current_module_statement();
    reset_precondition_map();
    reset_cumulated_effects_map();
    reset_enclosing_loops_map();

    return TRUE;
}





static void rdg_unstructured(u)
unstructured u ;
{
    list blocs = NIL ;

    CONTROL_MAP(c, {
	rdg_statement(control_statement(c));
    }, unstructured_control(u), blocs);

    gen_free_list( blocs );
}



static void rdg_statement(stat)
statement stat;
{
    instruction istat = statement_instruction(stat);

    switch (instruction_tag(istat)) {

      case is_instruction_block:
	MAPL(pc, {
	    rdg_statement(STATEMENT(CAR(pc)));
	}, instruction_block(istat));
	break;

      case is_instruction_test:
	rdg_statement(test_true(instruction_test(istat)));
	rdg_statement(test_false(instruction_test(istat)));
	break;

      case is_instruction_loop:
	rdg_loop(stat);
	break;

      case is_instruction_goto: 
      case is_instruction_call: 
	break;

      case is_instruction_unstructured:
	rdg_unstructured(instruction_unstructured(istat));
	break;

      default:
	pips_error("rdg_statement", "case default reached\n");
    }
}



static void rdg_loop(stat)
statement stat;
{
    set region;

    if (get_bool_property("COMPUTE_ALL_DEPENDENCES")) 
    {
	region = region_of_loop(stat);
	ifdebug(7)  
	{
	    fprintf(stderr, "[rdg_loop] applied on region:\n");
	    print_statement_set(stderr, region);
	}
	rice_update_dependence_graph(stat, region);
    }
    else 
    {
	if ((region = distributable_loop(stat)) == set_undefined) 
	{
	    instruction i = statement_instruction(stat) ;
	    
	    ifdebug(1) 
	    {
		fprintf( stderr, "[rdg_loop] skipping loop %d (but recursing)\n", 
			statement_number(stat));
	    }
	    rdg_statement(loop_body(instruction_loop(i))) ;
	}
	else 
	{
	    rice_update_dependence_graph(stat, region);
	}
    }
}



/*********************************************************************************/
/* UPDATING OF DEPENDENCE GRAPH                                                  */
/*********************************************************************************/

static void rice_update_dependence_graph(stat, region)
statement stat;
set region;
{
    list pv1, ps, pss;
    list llv, llv1;

    pips_assert("rice_update_dependence_graph", statement_loop_p(stat));

    debug(1, "rice_update_dependence_graph", "updating dependence graph\n");
    
    if(dg_type == DG_FULL) {
	debug(1, "rice_update_dependence_graph", 
	      "computing execution contexts\n");
	set_context_map( contexts_mapping_of_nest(stat) );
    }

    llv = loop_variant_list(stat);

    ifdebug(6) {
	fprintf(stderr,"The list of loop variants is :\n");
	MAP(ENTITY, e, fprintf(stderr," %s", entity_local_name(e)), llv);
	fprintf(stderr,"\n");
    }

    for (pv1 = graph_vertices(dg); pv1 != NIL; pv1 = CDR(pv1)) 
    {
	vertex v1 = VERTEX(CAR(pv1));
	dg_vertex_label dvl1 = (dg_vertex_label) vertex_vertex_label(v1);
	statement s1 = vertex_to_statement(v1);

	if (! set_belong_p(region, (char *) s1)) continue;

	dg_vertex_label_sccflags(dvl1) = make_sccflags(scc_undefined, 0, 0, 0);

	ps = vertex_successors(v1);
	pss = NIL; 
	while (ps != NIL) 
	{
	    successor su = SUCCESSOR(CAR(ps));
	    vertex v2 = successor_vertex(su);
	    statement s2 = vertex_to_statement(v2);
	    dg_arc_label dal = (dg_arc_label) successor_arc_label(su);
	    list true_conflicts = NIL;
	    list pc, pchead;
	    
	    if (! set_belong_p(region, (char *) s2)) 
	    {
		pss = ps;
		ps = CDR(ps);
		continue;
	    }
	    
	    pc = dg_arc_label_conflicts(dal);
	    pchead = pc;
	    while (pc !=NIL) 
	    {
		conflict c = CONFLICT(CAR(pc)) ;
		effect e1 = conflict_source(c);
		effect e2 = conflict_sink(c);
			
		   
		ifdebug(4) {
		    fprintf(stderr, "dep %02d (", statement_number(s1));
		    print_words(stderr, words_effect(e1));
		    fprintf(stderr, ") --> %02d (", statement_number(s2));
		    print_words(stderr, words_effect(e2));
		    fprintf(stderr, ") \n");
		}
		
		if (conflict_cone(c) != cone_undefined) 
		{
		    /* This conflict cone has been updated. */
		    ifdebug(4) { 
			fprintf(stderr, " \nThis dependence has been computed.\n");
		    }
		    true_conflicts = gen_nconc(true_conflicts, 
					       CONS(CONFLICT, c, NIL));
		}
		else  /*Compute this conflit and it's opposite*/
		{
		    list ps2su = NIL, ps2sus = NIL, pcs2s1 = NIL, pchead1 = NIL;
		    successor s2su = successor_undefined;
		    vertex v1bis;
		    statement s1bis;
		    dg_arc_label dals2s1 = dg_arc_label_undefined;
		    conflict cs2s1 = conflict_undefined;
		    effect e1bis = effect_undefined, e2bis = effect_undefined;
		    list levels = list_undefined;
		    list levelsop = list_undefined;
		    Ptsg gs = SG_UNDEFINED;
		    Ptsg gsop = SG_UNDEFINED;

		    Finds2s1 = FALSE;

		    /*looking for the opposite dependence from (s2,e2) to 
		      (s1,e1) */
		    
		    if (!((s1==s2) && (action_write_p(effect_action(e1)))
			  && (action_write_p(effect_action(e2)))) )
             /* && (reference_indices(effect_reference(e1))) != NIL) */
		    {
			debug (4, "rice_update_dependence_graph", 
			       "looking for the opposite dependence");  
			
			ps2su = vertex_successors(v2); 
			ps2sus = NIL;
			while (ps2su !=NIL && !Finds2s1) 
			{
			    s2su = SUCCESSOR(CAR(ps2su));
			    v1bis = successor_vertex(s2su);
			    s1bis = vertex_to_statement(v1bis);
			    if (s1bis != s1) 
			    {
				ps2sus = ps2su;
				ps2su = CDR(ps2su);
				continue;
			    }
			    else 
			    {
				dals2s1 = (dg_arc_label) successor_arc_label(s2su); 
				pcs2s1 = dg_arc_label_conflicts(dals2s1);
				pchead1 = pcs2s1;
				while ((pcs2s1!=NIL) && !Finds2s1) 
				{
				    cs2s1 =  CONFLICT(CAR(pcs2s1));   
				    e1bis = conflict_source(cs2s1);
				    e2bis = conflict_sink(cs2s1);
				    if (e1bis==e2 && e2bis==e1) 
				    {
					Finds2s1 = TRUE;
					continue;
				    }
				    else 
				    {
					pcs2s1 = CDR(pcs2s1);
					continue;
				    }
				}
				if (!Finds2s1) 
				{
				    ps2sus = ps2su;
				    ps2su = CDR(ps2su);
				}
				continue;
			    }
			}
			
			/* if (!Finds2s1) pips_error("rice_update_dependence_graph",  
			   "Expected opposite dependence are not found"); */
			
			if (Finds2s1) 
			{
			    ifdebug(4)  
			    {
				fprintf(stderr, "\n dep %02d (", 
					statement_number(s2));
				print_words(stderr, words_effect(e1bis));
				fprintf(stderr, ") --> %02d (", 
					statement_number(s1));
				print_words(stderr, words_effect(e2bis));
				fprintf(stderr, ") \n");
			    }
			}
			
		    }

		    llv1 = gen_copy_seq(llv);
		
		    levels = TestCoupleOfEffects(s1, e1, s2, e2,llv1,&gs,
						 &levelsop, &gsop);
		    
		    /* updating DG for the dependence (s1,e1)-->(s2,e2)*/
		    if (levels == NIL) 
		    {
			debug(4,"", "\nThe dependence (s1,e1)-->(s2,e2)"
			      " must be removed. \n");
			
			conflict_source(c) = effect_undefined;
			conflict_sink(c) = effect_undefined;
			gen_free(c);
		    }
		    else 
		    {
			debug(4,"", 
			      "\nUpdating the dependence (s1,e1)-->(s2,e2)\n");  

			if (!SG_UNDEFINED_P(gs))
			    conflict_cone(c) = make_cone(levels,gs);
			else
			    conflict_cone(c) = make_cone(levels,SG_UNDEFINED);
			true_conflicts = gen_nconc(true_conflicts, 
						   CONS(CONFLICT, c, NIL));
		    } 

		    if (Finds2s1) 
		    {
			/* updating DG for the dependence (s2,e2)-->(s1,e1)*/
			if (levelsop == NIL) {
			    debug(4,"","\nThe dependence (s2,e2)-->(s1,e1)"
				  " must be removed.\n");
			    
			    conflict_source(cs2s1) = effect_undefined;
			    conflict_sink(cs2s1) = effect_undefined;
			    /*gen_free(cs2s1);*/
			    
			    if(pchead == pchead1) 
				/* They are in the same conflicts list */
				gen_remove(&pchead, cs2s1);
			    else 
			    {
				gen_remove(&pchead1, cs2s1);
				if (pchead1 != NIL) 
				{
				    dg_arc_label_conflicts(dals2s1) = pchead1;
				    successor_arc_label(s2su) = 
					dals2s1;
				}
				else 
				{ 
				    /* This successor has only one 
				       conflict that has been killed.*/  
				    successor_vertex(s2su) = vertex_undefined;
				    successor_arc_label(s2su) = (char *) 
					dg_arc_label_undefined;
				    gen_free(s2su);
				    ps2su = CDR(ps2su);
				    if (ps2sus == NIL) {
					vertex_successors(v2) = ps2su;
				    }
				    else 
				    {
					CDR(ps2sus) = ps2su;
				    }
				}
			    }
			}
			
			else {
			    debug(4, "",
				  "\nUpdating the dependence (s2,e2)-->(s1,e1)\n");

			    if (!SG_UNDEFINED_P(gsop))
				conflict_cone(cs2s1) = make_cone(levelsop,gsop);
			    else 
				conflict_cone(cs2s1) = 
				    make_cone(levelsop,SG_UNDEFINED);	
			}  
		    }  
		}
		pc = CDR(pc);
	    }
	  
	   /* gen_free_list(dg_arc_label_conflicts(dal));*/
	    
	    if (true_conflicts != NIL) 
	    {
		dg_arc_label_conflicts(dal) = true_conflicts;
		pss = ps;
		ps = CDR(ps);
	    }
	    else 
	    {
		successor_vertex(su) = vertex_undefined;
		successor_arc_label(su) = (char *) dg_arc_label_undefined;
		gen_free(su);
		ps = CDR(ps);
		if (pss == NIL) 
		    vertex_successors(v1) = ps;
		else 
		    CDR(pss) = ps;
	    }
	    /*ifdebug(4){ print_graph(stderr, mod_stat, dg); }*/
	}
    }

    ifdebug(8) 
    {
	fprintf(stderr, "rice_update_dependence_graph] updated graph\n");
	print_statement_set(stderr, region);
	print_graph(stderr, get_current_module_statement(), dg);
    }

    reset_context_map();
}


/*********************************************************************************/
/* DEPENDENCE TEST                                                               */
/*********************************************************************************/



static list TestCoupleOfEffects(s1, e1, s2, e2,llv,gs,levelsop,gsop)
statement s1, s2;
effect e1, e2;
list llv; 
list *levelsop;
Ptsg *gs,*gsop;
{
    list n1 = load_statement_enclosing_loops(s1);
    Psysteme sc1 = SC_UNDEFINED;
    reference r1 = effect_reference( e1 ) ;

    list n2 = load_statement_enclosing_loops(s2);
    Psysteme sc2 = SC_UNDEFINED;
    reference r2 = effect_reference(e2) ;
    

    switch (dg_type)
    {
    case DG_FAST :
    {
	/* use region information if some is available */
	transformer t1 = effect_context(e1);
	transformer t2 = effect_context(e2);
	
	if(t1 != transformer_undefined) {
	    sc1 = (Psysteme) predicate_system(transformer_relation(t1));
	}
	else {
	    sc1 = SC_UNDEFINED;
	}
	
	if(t2 != transformer_undefined) {
	    sc2 = (Psysteme) predicate_system(transformer_relation(t2));
	}
	else {
	    sc2 = SC_UNDEFINED;
	}
	break;
    }

    case DG_FULL :
    {
	sc1 = load_statement_context(s1);
	sc2 = load_statement_context(s2);
	break;
    }

    case DG_SEMANTICS :
    {
	/* This is not correct because loop bounds should be frozen on
	   loop entry; we assume variables used in loop bounds are not
	       too often modified by the loop body */
	transformer t1 = load_statement_precondition(s1);
	transformer t2 = load_statement_precondition(s2);
	
	sc1 = (Psysteme) predicate_system(transformer_relation(t1));
	sc2 = (Psysteme) predicate_system(transformer_relation(t2));
	break;
    }
	
    default :
	pips_error ("TestCoupleOfEffects",
		    "Unknown dependence test %d\n", dg_type);
	break;
    }


    return(TestCoupleOfReferences(n1, sc1, s1, e1, r1, n2, sc2, s2, e2, 
				  r2, llv, gs, levelsop, gsop));
}



/* 
this function checks if two references have memory locations in common.

the problem is obvious for references to the same scalar variable.

the problem is also obvious if the references are not of the same
variable, except if the two variables have memory locations in common,
in which case we assume there are common locations.

when both references are of the same array variable, the function
TestDependence is called.
*/

static list TestCoupleOfReferences(n1, sc1, s1, ef1, r1, n2, sc2, s2, ef2, r2,llv,gs,levelsop, gsop)
list n1, n2;
Psysteme sc1, sc2;
statement s1, s2;
effect ef1, ef2;
reference r1, r2;
list llv;
list *levelsop ;
Ptsg *gs, *gsop;
/*boolean Finds2s1;*/
{
    int i, cl, dims, ty;
    list levels = NIL, levels1 = NIL;

    entity e1 = reference_variable(r1), 
    e2 = reference_variable(r2);

    list b1 = reference_indices(r1),
         b2 = reference_indices(r2);


    if(e1 != e2) 
    {
	fprintf(stderr, "dep %02d (", statement_number(s1));
	print_words(stderr, words_effect(ef1));
	fprintf(stderr, ") --> %02d (", statement_number(s2));
	print_words(stderr, words_effect(ef2));
	fprintf(stderr, ") \n");
	user_warning("TestCoupleOfReferences",
		     "Dependence between differents variables: "
		     "%s and %s\nDependence assumed\n",
		     entity_local_name(e1), entity_local_name(e2));
	
    }
	
    if (e1 == e2 && !entity_scalar_p(e1) && !entity_scalar_p(e2)) 
    {
	if (get_bool_property("RICEDG_STATISTICS_ALL_ARRAYS")) 
	{
	    NbrArrayDepInit++;
	}
	else 
	{
	    if (b1 != NIL && b2 != NIL) 
		NbrArrayDepInit++; 
	}
	    
	if (b1 == NIL || b2 == NIL) 
	{
	    /* A(*) reference appears in the dependence graph */ 
	    cl = FindMaximumCommonLevel(n1, n2);

	    for (i = 1; i <= cl; i++) 
		levels = gen_nconc(levels, CONS(INT, i, NIL));

	    if (statement_possible_less_p(s1, s2)) 
		levels = gen_nconc(levels, CONS(INT, cl+1, NIL));

	    if (Finds2s1) {
		for (i = 1; i <= cl; i++) 
		    levels1 = gen_nconc(levels1, CONS(INT, i, NIL));
		
		if (statement_possible_less_p(s2, s1)) 
		    levels1 = gen_nconc(levels1, CONS(INT, cl+1, NIL));
		
		*levelsop = levels1;
	    }
	}

	else 
	{
	    levels = TestDependence(n1, sc1, s1, ef1, r1, n2,sc2,s2, ef2,
				    r2,llv,gs,levelsop,gsop); 
	}

	if (get_bool_property("RICEDG_STATISTICS_ALL_ARRAYS") ||  
	    (b1 != NIL && b2 != NIL) ) 
	{
	    if (levels != NIL)  
	    {	
		/* test the dependence type, constant dependence?  */
		dims = gen_length(b1);
		ty = dep_type(effect_action(ef1),effect_action(ef2));
		deptype[dims][ty]++;
		if(is_dep_cnst)
		{
		    NbrDepCnst++;
		    constdep[dims][ty]++;
		}
	    }
		
	    if (*levelsop != NIL) 
	    {				
		/* test the dependence type, constant dependence?
		   exact dependence? */
		dims = gen_length(b1);
		ty = dep_type(effect_action(ef2),effect_action(ef1));
		deptype[dims][ty]++;
		if(is_dep_cnst)
		{
		    NbrDepCnst++;
		    constdep[dims][ty]++;
		}
	    }
	    
	    if(levels != NIL || *levelsop != NIL) 
	    {
		if(is_test_exact) NbrTestExact++;
		else 
		{
		    if (is_test_inexact_eq) 
		    {
			if (is_test_inexact_fm) NbrDepInexactEFM++;
			else NbrDepInexactEq++;
		    }
		    else NbrDepInexactFM++;
		}		
	    }
	       
	    ifdebug(6) 
	    {
		if(is_test_exact) 
		    fprintf(stderr, "\nThis test is exact! \n");
		else 
		    if (is_test_inexact_eq) 
			fprintf(stderr, "\nTest not exact : "
				"non-exact elimination of equation!");
		    else
			fprintf(stderr, "\nTest not exact : "
				"non-exact elimination of F-M!");
	    }
	}
	
	return(levels);
    } 
	
    else 
    { /* the case of scalar variables and equivalenced arrays */
	cl = FindMaximumCommonLevel(n1, n2);
	
	for (i = 1; i <= cl; i++) 
	    levels = gen_nconc(levels, CONS(INT, i, NIL));
	
	if (statement_possible_less_p(s1, s2)) 
	    levels = gen_nconc(levels, CONS(INT, cl+1, NIL));
	
	if ((instruction_loop_p(statement_instruction(s1)))|| 
	    instruction_loop_p(statement_instruction(s2)))
	    NbrIndexDep ++;
	else 
	{/*scalar variable dependence */
	    NbrScalDep ++;
	    ty = dep_type(effect_action(ef1),effect_action(ef2));
	    deptype[0][ty]++;
	}
	
	if (Finds2s1) 
	{
	    for (i = 1; i <= cl; i++) 
		levels1 = gen_nconc(levels1, CONS(INT, i, NIL));
	    
	    if (statement_possible_less_p(s2, s1)) 
		levels1 = gen_nconc(levels1, CONS(INT, cl+1, NIL));
	    
	    *levelsop = levels1;
	    if ((instruction_loop_p(statement_instruction(s1)))|| 
		instruction_loop_p(statement_instruction(s2)))
		NbrIndexDep ++;
	    else 
	    {/*case of scalar variable dependence */
		NbrScalDep ++;
		ty = dep_type(effect_action(ef2),effect_action(ef1));
		deptype[0][ty]++;	
	    }
	}
	
	return(levels);
    }
}


/* static list TestDependence(list n1, n2, Psysteme sc1, sc2,
 *                             statement s1, s2, effect ef1, ef2,
 *                             reference r1, r2, list llv,
 *                             list *levelsop, Ptsg *gs,*gsop)
 * input    : 
 *      list n1, n2    : enclosing loops for statements s1 and s2;
 *      Psysteme sc1, sc2: current context for each statement;
 *      statement s1, s2 : currently analyzed statements;
 *      effect ef1, ef2  : effects of each statement upon the current variable;
 *                         (maybe array regions)
 *      reference r1, r2 : current variables references;
 *      list llv        : loop variant list (variables that vary in loop nests);
 *      list *levelsop  : ? ( dependence levels from s2 to s1?);
 *      Ptsg *gs,*gsop   : depedence cone and ? (idem from s2 to s1 ?).
 * output   : dependence levels.
 * modifies : levelsop, gsop;
 * comment  : 
 * modification : l'ajout des tests de fasabilites pour le systeme initial
 * avant la projection.  
 * Yi-Qing (18/10/91)
 */
static list TestDependence(n1, sc1, s1, ef1, r1, n2, sc2, s2, ef2, r2, llv, gs,levelsop,gsop)
list n1, n2;
Psysteme sc1, sc2;
statement s1, s2;
effect ef1, ef2;
reference r1, r2;
list llv;
list *levelsop;
Ptsg *gs,*gsop;
/*boolean Finds2s1;*/
{
    Psysteme dep_syst = SC_UNDEFINED;
    Psysteme dep_syst1 = SC_UNDEFINED;
    Psysteme dep_syst2 = SC_UNDEFINED;
    Psysteme dep_syst_op = SC_UNDEFINED;
    Pbase b, tmp_base, coord;

    int l, cl;
    list levels;
    Pvecteur DiIncNonCons = NULL;


    /* Elimination of loop indices from loop variants llv */
    /* We use n2 because we take care of variables modified in 
       an iteration only for the second system. */
    MAP(STATEMENT, s, 
    {
	entity i = loop_index(statement_loop(s));
	gen_remove(&llv,i);
    },
	n2);

    ifdebug(6) 
    {
	debug(6, "", "loop variants after removing loop indices :\n");
	print_arguments(llv);
    }
    
    /* Build the dependence context system from the two initial context systems
     * sc1 and sc2. BC 
     */
    if (!build_and_test_dependence_context(r1, r2, sc1, sc2, &dep_syst, llv, n2)) 
    {
	/* the context system is not feasible : no dependence. BC */
	NbrIndepFind++;
	debug(4, "TestDependence", "context system not feasible\n");
	*levelsop = NIL;
	return(NIL);
    }
    

    /* Further construction of the dependence system; Constant and GCD tests
     * at the same time. 
     */
    if (gcd_and_constant_dependence_test(r1,r2,llv,n2,&dep_syst)) 
    {
	/* independence proved */
	/* sc_rm(dep_syst); */
	NbrIndepFind++;
	*levelsop = NIL;
	return(NIL);
    }
    
    dependence_system_add_lci_and_di(&dep_syst, n1, &DiIncNonCons);

    ifdebug(6)  
    {
	fprintf(stderr, "\ninitial system is:\n");
	syst_debug(dep_syst);
    }

    /* Consistance Test */
    if ((dep_syst = sc_normalize(dep_syst)) == NULL) 
    {
	NbrTestSimple++;	
	NbrIndepFind++;
	debug(4, "TestDependence", "initial normalized system not feasible\n");
	*levelsop = NIL;
	return(NIL);
    }

    cl = FindMaximumCommonLevel(n1, n2);

    if (TestDiCnst(dep_syst, cl, s1, ef1, s2, ef2) == TRUE ) 
    {
	/*find independences (no loop carried dependence at the same statement).*/
	NbrTestDiCnst++;
	NbrIndepFind++;
	debug(4,"TestDependence","\nTestDiCnst successed!\n");
	*levelsop = NIL;
	return(NIL);
    }
	
    is_test_exact = TRUE;
    is_test_inexact_eq = FALSE;
    is_test_inexact_fm = FALSE;
    is_dep_cnst = FALSE;

    tmp_base = base_dup(dep_syst->base);

    if (setjmp(overflow_error)) 
    {
	Pbase dep_syst_base = BASE_UNDEFINED;

	/* eliminate non di variables from the basis */
	for (coord = tmp_base; !VECTEUR_NUL_P(coord); coord = coord->succ) 
	{
	    Variable v = vecteur_var(coord);
	    l = DiVarLevel((entity) v);	    
	    if (l <= 0 || l > cl)     
		base_add_variable(dep_syst_base, v);		
	}
	dep_syst = sc_rn(dep_syst_base);

    }
    else 
    {
	if (sc_proj_optim_on_di_ofl(cl, &dep_syst) == INFAISABLE) 
	{	
	    debug(4,"TestDependence",
		  "projected system by sc_proj_optim_on_di() is not feasible\n");
	    NbrIndepFind++;
	    *levelsop = NIL;
	    return(NIL);
	}
    }

    base_rm(tmp_base);

    ifdebug(6) 
    {
	fprintf(stderr, "projected system is:\n");
	syst_debug(dep_syst);
    }
    
    if (! sc_faisabilite_optim(dep_syst)) 
    {
	debug(4,"TestDependence", "projected system not feasible\n");
	NbrIndepFind++;
	*levelsop = NIL;
	return(NIL);
    }
    
    ifdebug(6)  
    {
	fprintf(stderr, "normalised projected system is:\n");
	syst_debug(dep_syst);
	fprintf(stderr, "The list of DiIncNonCons is :\n");
	vect_debug(DiIncNonCons);
    }
    

    /* keep DiIncNonCons variable if it's zero, otherwise move it in the dep_syst*/
    if (dep_syst != NULL) 
    {
	while (DiIncNonCons != NULL){
	    Variable di;
	    int val;
	    
	    di = DiIncNonCons->var;
	    if (sc_value_of_variable(dep_syst, di, &val) ==  TRUE)
		if (val != 0){
		    sc_elim_var(dep_syst,di);
		}
	    DiIncNonCons = DiIncNonCons->succ;
	}
    }
    
    ifdebug(4) 
    {
	fprintf(stderr, 
		"normalised projected system after kill DiIncNonCons is:\n");
	syst_debug(dep_syst);
    }
    
    /* compute the levels of dep from s1 to s2 and of the opposite
       dep from s2 to s1 if it exists*/  
    
    *levelsop = NIL;
    if (Finds2s1) dep_syst_op = sc_invers(sc_dup(dep_syst));


    ifdebug(4) 
    {
	debug(4,"", "\nComputes the levels and DC for dep: (s1,e1)->(s2,e2)\n");
	fprintf(stderr, "\nThe distance system for dep is:\n");
	syst_debug(dep_syst);
    }
    
    dep_syst1 = sc_dup(dep_syst);
    b = MakeDibaseinorder(cl);
    base_rm(dep_syst1->base);
    dep_syst1->base = b; 
    dep_syst1->dimension = cl;
    
    if (dep_syst1->dimension == dep_syst1->nb_eq)
	is_dep_cnst = TRUE;

    levels = TestDiVariables(dep_syst, cl, s1, ef1, s2, ef2);
    /* if (levels == NIL) NbrTestDiVar++; */
    *gs = dependence_cone_positive(dep_syst1);
    

    ifdebug(4) 
    {	
	fprintf(stderr, "\nThe levels for dep (s1,s2) are:");
	MAP(INT, pl, 
	{
	    fprintf(stderr, " %d", pl);
	}, levels);
	
	if(!SG_UNDEFINED_P(*gs)) 
	{
	    fprintf(stderr, 
		    "\nThe lexico-positive dependence cone for"
		    " dep (s1,s2) :\n");
	    print_dependence_cone(stderr, *gs, b);
	} 
	else 
	    fprintf(stderr,"\nLexico-positive dependence cone"
		    " doesn't exist for dep (s1,s2).\n"); 
    }


    if (Finds2s1) 
    {
	
	debug(4,"","Computes the levels and DC for dep_op: (s2,e2)->(s1,e1)\n"); 
	ifdebug(4) 
	{
	    fprintf(stderr, "\nThe invers distance system for dep_op is:\n");  
	    syst_debug(dep_syst_op);
	}    	
	
	dep_syst2 = sc_dup(dep_syst_op);    
	b = MakeDibaseinorder(cl);
	base_rm(dep_syst2->base);
	dep_syst2->base = b;
	dep_syst2->dimension = cl;

	*levelsop = TestDiVariables(dep_syst_op, cl, s2, ef2, s1, ef1);
	/* if (*levelsop == NIL) NbrTestDiVar++; */
	*gsop = dependence_cone_positive(dep_syst2);
    

	ifdebug(4) 
	{	
	    fprintf(stderr, "\nThe levels for dep_op (s2,s1) is:");
	    MAP(INT, pl, 
	    {
		fprintf(stderr, " %d", pl);
	    }, *levelsop);
	    
	    if(!SG_UNDEFINED_P(*gsop)) 
	    {
		fprintf(stderr,  "\nThe lexico-positive Dependence "
			"cone for dep_op (s2,s1):\n");
		print_dependence_cone(stderr,*gsop,b);
	    } 
	    else 
		fprintf(stderr,"\nLexico-positive dependence cone "
			"does not exist for dep_op (s2,s1).\n"); 
	    
	}    
    }
    
    if (levels == NIL && *levelsop == NIL) 
    {
	NbrTestDiVar++;
	NbrIndepFind++;
	if ( s1 == s2 ) 
	{
	    /* the case of "all equals" independence at the same statement*/ 
	    NbrAllEquals++;
	}
	
	return(NIL);
    }
    

    return(levels);    
}


/* static boolean build_and_test_dependence_context(reference r1, r2,
 *                                                  Psystem sc1, sc2, *psc_dep,
 *                                                  list llv, s2_enc_loops)
 * input    : 
 *      reference r1, r2  : current array references;
 *      Psystem sc1, sc2  : context systems for r1 and r2;
 *      Psystem *psc_dep  : pointer toward the dependence context systeme;
 *      list llv          : current loop nest variant list;
 *      list s2_enc_loops : statement s2 enclosing loops;
 * output   : FALSE if one of the initial systems is unfeasible after normalization;
 *            TRUE otherwise;
 * modifies : psc_dep points toward the dependence context system built from 
 *            sc1 and sc2. Dependence distance variables (di) are introduced
 *            in sc2, along with the dsi variables to take care of variables
 *            modified in the loop nest; unrelevant constraints are removed
 *            in order to make further manipulations easier.
 * comment  :
 */
static boolean build_and_test_dependence_context(r1, r2, sc1, sc2, psc_dep, llv, 
						 s2_enc_loops)
reference r1, r2;
Psysteme sc1, sc2, *psc_dep;
list llv, s2_enc_loops;
{
    Psysteme sc_dep, sc_tmp;
    list pc;
    int l, inc;

    /* *psc_dep must be undefined */
    pips_assert("build_and_test_dependence_context", SC_UNDEFINED_P(*psc_dep));

    /* Construction of initial systems sc_dep and sc_tmp from sc1 and sc2
       if not undefined
       */
    if (SC_UNDEFINED_P(sc1) && SC_UNDEFINED_P(sc2)) 
    {
	sc_dep = sc_new();
    }
    else 
    {
	if(SC_UNDEFINED_P(sc1))
	    sc_dep = sc_new();
	else 
	{
	    /* sc_dep = sc1, but:
	     * we keep only useful constraints in the predicate 
	     * (To avoid overflow errors, and to make projections easier) 
	     */
	    Pbase variables = BASE_UNDEFINED;	    
	    
	    if (sc_normalize(sc1) == NULL) 
	    {
		debug(4, "build_and_test_dependence_context", 
		      "first initial normalized system not feasible\n");
		return(FALSE);
	    }

	    ifdebug(6) 
	    {
		debug(6, "", "Initial system sc1 before restrictions : \n");
		syst_debug(sc1);
	    }

	    MAP(EXPRESSION, ex1, 
	    {
		normalized x1 = NORMALIZE_EXPRESSION(ex1);
		
		if (normalized_linear_p(x1)) 
		{
		    Pvecteur v1;
		    Pvecteur v;		    
		    v1 = (Pvecteur) normalized_linear(x1);
		    for(v = v1; !VECTEUR_NUL_P(v); v = v->succ) 
		    { 
			if (vecteur_var(v) != TCST) 
			    variables = base_add_variable(variables, 
							  vecteur_var(v));
		    } 		    
		}
	    },
		reference_indices(r1));
	    
	    sc_dep = sc_restricted_to_variables_transitive_closure(sc1, variables);
	}

	if(SC_UNDEFINED_P(sc2))
	    sc_tmp = sc_new();
	else 
	{
	    /* sc_tmp = sc2, but:
	     * we keep only useful constraints in the predicate 
	     * (To avoid overflow errors, and to make projections easier) 
	     */
	    Pbase variables = BASE_UNDEFINED;

	    if (sc_normalize(sc2) == NULL) 
	    {
		debug(4, "build_and_test_dependence_context", 
		      "second initial normalized system not feasible\n");
		return(FALSE);
	    }
	    
	    ifdebug(6) 
	    {
		debug(6, "", "Initial system sc2 before restrictions : \n");
		syst_debug(sc2);
	    }

	    MAP(EXPRESSION, ex2,
	    {
		normalized x2 = NORMALIZE_EXPRESSION(ex2);
		
		if (normalized_linear_p(x2)) 
		{
		    Pvecteur v2;
		    Pvecteur v;
		    
		    v2 = (Pvecteur) normalized_linear(x2);
		    for(v = v2; !VECTEUR_NUL_P(v); v = v->succ) 
		    { 
			if (vecteur_var(v) != TCST) 
			    variables = base_add_variable(variables, 
							  vecteur_var(v));
		    } 
		    
		}
	    },
		reference_indices(r2));
	    
	    sc_tmp = sc_restricted_to_variables_transitive_closure(sc2, variables);
	}
	
	ifdebug(6) 
	{
	    debug(6, "", "Initial systems after restrictions : \n");
	    syst_debug(sc_dep);
	    syst_debug(sc_tmp);
	}
	
	/* introduce dependence distance variable if loop increment value
	   is known or ... */
	for (pc = s2_enc_loops, l = 1; pc != NIL ; pc = CDR(pc), l++) 
	{
	    loop lo = statement_loop(STATEMENT(CAR(pc)));
	    entity i2 = loop_index(lo);
	    inc = loop_increment_value(lo);
	    if (inc != 0)  
		sc_add_di(l, i2, sc_tmp, inc);
	    else
		sc_chg_var(sc_tmp, (Variable) i2, (Variable) GetLiVar(l));
	}
	
	/* take care of variables modified in the loop */
	for (pc = llv, l=1; pc!=NIL; pc = CDR(pc),l++) {
	    sc_add_dsi(l,ENTITY(CAR(pc)),sc_tmp);
	}

	/* (sc_tmp is destroyed by sc_fusion) */
	sc_dep = sc_fusion(sc_dep, sc_tmp);
    }

    ifdebug(6) 
    {
	debug(6, "build_and_test_dependence_context", "\ndependence context is:\n");
	syst_debug(sc_dep);
    }

    *psc_dep = sc_dep;
    return(TRUE);

}



/* static boolean gcd_and_constant_dependence_test(references r1, r2, 
 *                                                 list llv, s2_enc_loops,
 *                                                 Psysteme *psc_dep)
 * input    : 
 *      references r1, r2 : current references;
 *      list llv          : loop nest variant list;
 *      list s2_enc_loops : statement s2 enclosing loops;
 *      Psysteme *psc_dep : pointer toward the depedence system;
 * output   : TRUE if there is no dependence (GCD and constant test successful);
 *            FALSE if independence could not be proved.
 * modifies : *psc_dep;
 * comment  :
 *      *psc_dep must not be undefined on entry; it must have been initialized
 *      by build_and_test_dependence_context.
 */
static boolean gcd_and_constant_dependence_test(r1, r2, llv, s2_enc_loops, psc_dep)
reference r1, r2;
list llv, s2_enc_loops;
Psysteme *psc_dep;
{
    list pc1, pc2, pc;
    int l;

    /* Further construction of the dependence system; Constant and GCD tests
     * at the same time. 
     */
    pc1 = reference_indices(r1);
    pc2 = reference_indices(r2);
    while (pc1 != NIL && pc2 != NIL) 
    {
	expression ex1, ex2;
	normalized x1, x2;

	ex1 = EXPRESSION(CAR(pc1));
	x1 = NORMALIZE_EXPRESSION(ex1);

	ex2 = EXPRESSION(CAR(pc2));
	x2 = NORMALIZE_EXPRESSION(ex2);

	if (normalized_linear_p(x1) && normalized_linear_p(x2)) 
	{
	    Pvecteur v1, v2, v;

	    v1 = (Pvecteur) normalized_linear(x1);
	    v2 = vect_dup((Pvecteur) normalized_linear(x2));

	    for (pc = s2_enc_loops, l = 1; pc!=NIL; pc = CDR(pc), l++) 
	    {
		loop lo = statement_loop(STATEMENT(CAR(pc)));
		entity i = loop_index(lo);
		int li = loop_increment_value(lo);
		if (li!=0) 
		    vect_add_elem(&v2, (Variable) GetDiVar(l), 
				  li*vect_coeff((Variable) i, v2));
		else
		    vect_chg_var(&v2, (Variable) i, (Variable) GetLiVar(l));
	    }
    
	    for (pc = llv, l=1; pc!=NIL; pc = CDR(pc),l++) 
	    {
		vect_add_elem(&v2, (Variable) GetDsiVar(l), 
			      vect_coeff((Variable) ENTITY(CAR(pc)), v2));
	    }

	    if (! VECTEUR_UNDEFINED_P(v = vect_substract(v1, v2))) 
	    {
		Pcontrainte c = contrainte_make(v);
	    
		/* case of T(3)=... et T(4)=... */
		if (contrainte_constante_p(c) && COEFF_CST(c) != 0) 
		{
		    NbrTestCnst++;
		    debug(4,"TestDependence","TestCnst succssed!");
		    return(TRUE);
		}
		/* test of GCD */
		if (egalite_normalize(c) == FALSE)
		{
		    NbrTestGcd++;
		    debug(4,"TestDependence","TestGcd succeeded!\n");
		    return(TRUE);	
		}	
		sc_add_eg(*psc_dep, c);
	    }
	    vect_rm(v2);
	}

	pc1 = CDR(pc1); 
	pc2 = CDR(pc2);
    }


    if (pc1 != NIL || pc2 != NIL)
	/* should be an assert. BC. */
	user_warning("TestDependence", "lengths of index lists differ\n");


    return(FALSE);
}


   
/* static void dependence_system_add_lci_and_di(Psysteme *psc_dep,
 *                                              list s1_enc_loops,
 *                                              Pvecteur *p_DiIncNonCons)
 * input    : 
 *      Psysteme *psc_dep : pointer toward the dependence system;
 *      list s1_enc_loops : statement s1 enclosing loops;
 *      Pvecteur *p_DiIncNonCons : pointer toward DiIncNonCons.
 * output   : nothing;
 * modifies : the dependence systeme (addition of constraints with lci and di 
 *            variables); and di variables are added to DiIncNonCons.
 * comment  : DiIncNonCons must be undefined on entry.
 */
static void dependence_system_add_lci_and_di(psc_dep, s1_enc_loops, p_DiIncNonCons)
Psysteme *psc_dep;
list s1_enc_loops;
Pvecteur *p_DiIncNonCons;
{
    int l;
    list pc;
   
    pips_assert("dependence_system_add_lci_and_di", 
		VECTEUR_UNDEFINED_P(*p_DiIncNonCons));

    /* Addition of lck and di variables */
    for (pc = s1_enc_loops,l = 1; pc != NIL; pc = CDR(pc), l++) 
    {
	loop lo = statement_loop(STATEMENT(CAR(pc)));
	entity ind = loop_index(lo);
	int inc = loop_increment_value(lo);

	expression lb = range_lower(loop_range(lo));
	normalized nl = NORMALIZE_EXPRESSION(lb);

	/* make   nl + inc*lc# - ind = 0 */
	if (inc != 0 && inc != 1 && inc != -1 && normalized_linear_p(nl)) 
	{	   
	    Pcontrainte pc;
	    entity lc;
	    Pvecteur pv;

	    lc = MakeLoopCounter();
	    pv = vect_dup((Pvecteur) normalized_linear(nl));
	    vect_add_elem(&pv, (Variable) lc, inc);
	    vect_add_elem(&pv, (Variable) ind, -1);
	    pc = contrainte_make(pv);
	    sc_add_eg(*psc_dep, pc);
	}
	
	/* make d#i -l#i + ind = 0 ,
	   add d#i in list of DiIncNonCons*/
	if (inc == 0) 
	{
	    Pcontrainte pc;
	    Pvecteur pv = NULL;
	    entity di;

	    di = GetDiVar(l);
	    vect_add_elem(&pv, (Variable) di, 1);
	    vect_add_elem(&pv, (Variable) GetLiVar(l), -1);
	    vect_add_elem(&pv, (Variable) ind, 1);
	    pc = contrainte_make(pv);
	    sc_add_eg(*psc_dep, pc);

	    vect_add_elem(p_DiIncNonCons, (Variable) di, 1);
	}
	    
    }

    if(!BASE_NULLE_P((*psc_dep)->base)) 
    {
	vect_rm((*psc_dep)->base);
	(*psc_dep)->base = BASE_UNDEFINED;
    }
    sc_creer_base(*psc_dep);
}




/*
this function implements the last step of the CRI dependence test. ps is
a system that has been projected on di variables. ps is examined to find
out the set of possible values for di variables.

there is a loop from 1 to cl, the maximum nesting level. at each step,
the di variable corresponding to the current nesting level is examined.
dependences added to the graph depend on the sign of the values computed
for di. there is a dependence from s1 to s2 if di is positive and from
s2 to s1 if di is negative. there are no dependence if di is equal to
zero. finally, di is replaced by 0 at the end of the loop since
dependences at level l are examined with sequential enclosing loops.

ps is the projected system (the distance system for the dep: (s1->s2).

cl is the common nesting level of statement s1 and s2.

* Modification : l'ajout de variable NotPositive pour prendre en compt le cas 
* quand di<=0.
* Yi Qing (10/91)
*/

static list TestDiVariables(ps, cl, s1, ef1, s2, ef2)
Psysteme ps;
int cl;
statement s1, s2;
effect ef1, ef2;
{
    list levels = NIL;
    int l;

    for (l = 1; l <= cl; l++) 
    {
	Variable di = (Variable) GetDiVar(l);
	int min, max, val;
	int IsPositif, IsNegatif, IsNull, NotPositif;
	Psysteme pss = (l==cl) ? ps : sc_dup(ps);

	switch (dg_type)
	{
	case DG_FAST :
	{
	    if (sc_value_of_variable(pss, di, &val) == TRUE) 
	    {
		min = val;
		max = val;
	    }
	    else 
	    {
		max = MAXINT;
		min = -MAXINT;
	    }
	    break;
	}

	case DG_FULL:
	case DG_SEMANTICS :
	{
	    if (sc_minmax_of_variable_optim(pss, di, &min, &max) == FALSE) 
	    {
		return(levels);
	    }
	    break;
	}
	default:
	    pips_error("TestDiVariables", "undefined dg type\n");
	}

	IsPositif = min > 0;
	IsNegatif = max < 0;
	IsNull = (min == 0 && max == 0);
	NotPositif = (max == 0 && min <0);

	ifdebug(7)
	{
	    debug(7, "TestDiVariables", 
		  "level is %d - di variable is %s\n", l, 
		  entity_local_name((entity) di));
	    debug(7, "TestDiVariables", 
		  "min = %d   max = %d   pos = %d   neg = %d   nul = %d\n", 
		  min, max, IsPositif, IsNegatif, IsNull);
	}

	if(IsPositif) 
	{
	    levels = gen_nconc(levels, CONS(INT, l, NIL));
	    return(levels);
	}

	if(IsNegatif) 
	    return(levels);

	if (!IsNull && !NotPositif) 
	    levels = gen_nconc(levels, CONS(INT, l, NIL));
	
	if (l <= cl-1) 
	    sc_force_variable_to_zero(ps, di);
    }

    if (s1 != s2 && statement_possible_less_p(s1, s2))
	levels = gen_nconc(levels, CONS(INT, l, NIL));  

    return(levels);
}

/* Ptsg  dependence_cone_positive(Psysteme dept_syst)
 */ 
static Ptsg dependence_cone_positive(dep_sc)
Psysteme dep_sc;
{
    Psysteme sc_env= SC_UNDEFINED;
    Ptsg sg_env = sg_new();
    Pbase b;
    int n,i, j;

    if (SC_UNDEFINED_P(dep_sc))
	return(sg_env);

    sc_env = sc_empty(dep_sc->base);
    n = dep_sc->dimension;
    b = dep_sc->base;
    
    for (i=1; i<=n; i++) 
    {	
	Psysteme sub_sc = sc_dup(dep_sc);
	Pvecteur v;
	Pcontrainte pc;
	Pbase b1;
	
	for(j=1, b1=b; j<=i-1; j++, b1=b1->succ)
	{		
	    /* add the contraints  bj = 0 (1<=j<i) */
	    v = vect_new(b1->var,1);
	    pc = contrainte_make(v);
	    sc_add_eg(sub_sc,pc);
	}
	
	/* add the contraints - bi <= -1 (1<=j<i) */
	v = vect_new(b1->var,-1);
	vect_add_elem(&v, TCST , 1);
	pc = contrainte_make(v);
	sc_add_ineg(sub_sc,pc);
	
	ifdebug(7) 
	{ 
	    fprintf(stderr,
		    "\nInitial sub lexico-positive dependence system:\n");
	    syst_debug(sub_sc); 
	}
	
	if (! sc_rational_feasibility_ofl_ctrl(sub_sc, NO_OFL_CTRL, TRUE))
	{ 
	    debug(7,"dependence_cone_positive", 
		  "sub lexico-positive dependence system not feasible\n");
	    continue;
	}
	
	if ((sub_sc = sc_normalize(sub_sc))== NULL)
	{ 
	    debug(7, "Dependence_cone_positive", 
		  "normalized system not feasible\n");
	    continue;	    
	}
	
	/* We get a normalized sub lexico-positive dependence system */
	ifdebug(7) 
	{ 
	    fprintf(stderr,
		    "Normalized sub lexico-positive dependence system :\n");
	    syst_debug(sub_sc);
	}
	
	sc_env = sc_enveloppe_chernikova(sc_env,sub_sc);
	
	ifdebug(7) 
	{ 
	    fprintf(stderr, "Dependence system of the enveloppe of subs "
		    "lexico-positive dependence:\n");
	    syst_debug(sc_env);
	    
	    if (!SC_UNDEFINED_P(sc_env) && !sc_rn_p(sc_env))
	    {
		sg_env = sc_to_sg_chernikova(sc_env);
		fprintf(stderr, "Enveloppe of the subs lexico-positive "
			"dependence cones:\n");
		if(!SG_UNDEFINED_P(sg_env) && vect_size(sg_env->base) != 0) 
		{
		    print_dependence_cone(stderr,sg_env,sg_env->base);
		    sg_rm(sg_env);
		    sg_env = (Ptsg) NULL;
		}
	    }
	}
	
    } 
    
    if (!SC_UNDEFINED_P(sc_env) && sc_dimension(sc_env)!= 0)
	sg_env = sc_to_sg_chernikova(sc_env);
    
    return (sg_env);
}


/*********************************************************************************/
/* QUICK PRIVATIZATION                                                           */
/*********************************************************************************/

static void quick_privatize_graph(dep_graph)
graph dep_graph;
{

    /* we analyze arcs exiting from loop statements */
    MAP(VERTEX, v1, 
    {
	statement s1 = vertex_to_statement(v1);
	list successors = vertex_successors(v1);
	
	if (statement_loop_p(s1)) 
	{
	    loop l = statement_loop(s1);
	    list locals = loop_locals(l);
	    entity ind = loop_index(l);
	    
	    if (gen_find_eq(ind, locals) == entity_undefined) 
	    {
		if (quick_privatize_loop(s1, successors)) 
		{
		    debug(8, "quick_privatize_graph", 
			  "loop %d privatized\n", statement_number(s1));
		    
		    loop_locals(l) = CONS(ENTITY, ind, locals);
		}
		else 
		{
		    debug(1, "quick_privatize_graph", 
			  "could not privatize loop %d\n", statement_number(s1));
		}
	    }
	}
    },
	graph_vertices(dep_graph) );
}



static bool quick_privatize_loop(stat, successors)
statement stat;
list successors;
{

    debug(3, "quick_privatize_loop", "arcs from %d\n", statement_number(stat));

    MAP(SUCCESSOR, su,
    {
	dg_arc_label dal = (dg_arc_label) successor_arc_label(su);
	statement st = vertex_to_statement(successor_vertex(su));
	
	debug(3, "quick_privatize_loop", "arcs to %d\n", statement_number(st));
	
	if (! quick_privatize_statement_pair(stat, st, 
					     dg_arc_label_conflicts(dal)))
	    return(FALSE);
    },
	successors );
    
    return(TRUE);
}



static bool quick_privatize_statement_pair(s1, s2, conflicts)
statement s1, s2;
list conflicts;
{
    loop l1 = statement_loop(s1);
    entity ind1 = loop_index(l1);

    MAP(CONFLICT, c,
    {
	effect f1 = conflict_source(c);
	reference r1 = effect_reference(f1);
	entity e1 = reference_variable(r1);

	effect f2 = conflict_sink(c);
	reference r2 = effect_reference(f2);
	entity e2 = reference_variable(r2);

	debug(2, "quick_privatize_statement_pair", 
	      "conflict between %s & %s\n", entity_name(e1), entity_name(e2));

	/* equivalence or conflict not created by loop index. I give up ! */
	if (e1 != ind1)  
	    continue; 
	
	if (action_write_p(effect_action(f1)) && 
	    action_read_p(effect_action(f2))) 
	{
	    /* we must know where this read effect come from. if it
	       comes from the loop body, the arc may be ignored. */

	    list loops = load_statement_enclosing_loops(s2);

	    if (gen_find_eq(s1, loops) == entity_undefined) 
	    {
		loop l2;
		entity ind2;
		list range_effects;

		debug(3, "quick_privatize_statement_pair", 
		      "the arc goes outside the loop body.\n");

		if (! statement_loop_p(s2)) 
		{
		    debug(3, "quick_privatize_statement_pair", "s2 not a loop\n"); 
		    return(FALSE);
		}

		/* s2 is a loop. if there are no read effet in the range
		   part, ignore this conflict. */
		l2 = statement_loop(s2);
		ind2 = loop_index(l2);
		range_effects = proper_effects_of_range(loop_range(l2), 
							is_action_read);

		MAP(EFFECT, e, 
		{
		    if (reference_variable(effect_reference(e)) == ind2 &&
			action_read_p(effect_action(e))) 
		    {
			
			debug(3, "quick_privatize_statement_pair", 
			      "index read in range expressions\n"); 
			
			free_effects(make_effects(range_effects));
			return(FALSE);
		    }
		}, range_effects);
		free_effects(make_effects(range_effects));
	    }
	}
    },
	conflicts );
    
    return(TRUE);
}



    
static list loop_variant_list(stat)
statement stat;
{
    list lv = NIL;
    loop l;
    list locals;

    pips_assert("loop_variant_list", statement_loop_p(stat));

    MAP(EFFECT, ef, 
     {
	 entity en = effect_entity(ef) ;
	 if( action_write_p( effect_action( ef )) && entity_integer_scalar_p( en )) 
	     lv = gen_nconc(lv, CONS(ENTITY, en, NIL));
     }, 
	 load_statement_cumulated_effects(stat) ) ;
    
    l = statement_loop(stat);
    locals = loop_locals(l);
    MAP(ENTITY, v,
    {
	if (gen_find_eq(v,lv) == entity_undefined) 
	    lv = CONS(ENTITY, v, lv);
    }, locals);
    
    return(lv);
}

/* this function detects the no loop carried independence when Di=(0,0,...0) and s1 = s2.
*/ 
static boolean TestDiCnst(ps, cl, s1, ef1, s2, ef2)  
Psysteme ps;
int cl;
statement s1, s2;
effect ef1,ef2;
{
  int l;
    
  for (l = 1; l <=cl; l++) 
  {
      Variable di = (Variable) GetDiVar(l);
      Psysteme pss = sc_dup(ps);
      int val;

      if (sc_value_of_variable(pss, di, &val) == TRUE) 
      {
	  if (val != 0) return (FALSE);
      }
      else return (FALSE);
  }
  /* case of di zero */
  if (s1 == s2) 
  {
      NbrAllEquals++;
      return(TRUE);
  }
  else return(FALSE);
}
	    





/*********************************************************************************/
/* DG PRINTING FUNCTIONS                                                         */
/*********************************************************************************/

bool print_whole_dependence_graph(mod_name)
string mod_name;
{
    set_bool_property("PRINT_DEPENDENCE_GRAPH_WITHOUT_PRIVATIZED_DEPS",
		      FALSE);
    set_bool_property("PRINT_DEPENDENCE_GRAPH_WITHOUT_NOLOOPCARRIED_DEPS",
		      FALSE);
    return print_dependence_graph(mod_name);
}

bool print_effective_dependence_graph(mod_name)
string mod_name;
{
    set_bool_property("PRINT_DEPENDENCE_GRAPH_WITHOUT_PRIVATIZED_DEPS",
		      TRUE);
    set_bool_property("PRINT_DEPENDENCE_GRAPH_WITHOUT_NOLOOPCARRIED_DEPS",
		      FALSE);
    return print_dependence_graph(mod_name);
}

bool print_loop_carried_dependence_graph(mod_name)
string mod_name;
{
    set_bool_property("PRINT_DEPENDENCE_GRAPH_WITHOUT_PRIVATIZED_DEPS",
		      TRUE);
    set_bool_property("PRINT_DEPENDENCE_GRAPH_WITHOUT_NOLOOPCARRIED_DEPS",
		      TRUE);
    return print_dependence_graph(mod_name);
}

bool print_dependence_graph(mod_name)
string mod_name;
{
    string dg_name = NULL;
    string local_dg_name = NULL;
    graph dg;
    FILE *fp;
    statement mod_stat;

    set_current_module_entity(local_name_to_top_level_entity(mod_name));
    set_current_module_statement( (statement)
	db_get_memory_resource(DBR_CODE, mod_name, TRUE) );
    mod_stat = get_current_module_statement();

    dg = (graph) db_get_memory_resource(DBR_DG, mod_name, TRUE);

    debug_on("RICEDG_DEBUG_LEVEL");

    local_dg_name = (string) strdup(concatenate(mod_name, ".dg", NULL));
    dg_name = (string) strdup(concatenate(db_get_current_workspace_directory(), "/", 
					  local_dg_name, NULL));
    fp = safe_fopen(dg_name, "w");
    if (get_bool_property("PRINT_DEPENDENCE_GRAPH_WITHOUT_PRIVATIZED_DEPS") || 
	get_bool_property("PRINT_DEPENDENCE_GRAPH_WITHOUT_NOLOOPCARRIED_DEPS")) {
	print_graph_with_reduction(fp, mod_stat, dg);
    }
    else  
	print_graph(fp, mod_stat, dg);
    safe_fclose(fp, dg_name);

    debug_off();
    
    DB_PUT_FILE_RESOURCE(DBR_DG_FILE, strdup(mod_name), 
			 local_dg_name);

    reset_current_module_statement();
    reset_current_module_entity();
    free(dg_name);
    return TRUE;
}



static void writeresult(mod_name)
char *mod_name;
{
    FILE *fp;
    string filename;
    int i,j;


    switch (dg_type) 
    {
    case DG_FAST:
	filename = "resulttestfast"; break;
    case DG_FULL:
	filename = "resulttestfull"; break;
    case DG_SEMANTICS:
	filename = "resulttestseman"; break;
    default:
	pips_error("writeresult", "erroneous dg type.\n");
	return; /* to avoid warnings from compiler */
    }

    filename = strdup(concatenate(db_get_current_workspace_directory(), "/", 
				  mod_name, ".", filename, 0));

    fp = safe_fopen(filename, "w");

    fprintf(fp,"%s",mod_name);
    fprintf(fp, " %d %d %d %d %d %d %d %d %d %d",NbrArrayDepInit,NbrIndepFind,NbrAllEquals,
	    NbrDepCnst,NbrTestExact,NbrDepInexactEq,NbrDepInexactFM,NbrDepInexactEFM,
	    NbrScalDep,NbrIndexDep);
    for (i=0;i<=4;i++)
	for(j=0;j<=2;j++)
	    fprintf(fp," %d", deptype[i][j]);
    for (i=0;i<=4;i++)
	for(j=0;j<=2;j++)
	    fprintf(fp," %d", constdep[i][j]);
    fprintf(fp, " %d %d %d %d %d %d %d %d %d %d %d",NbrTestCnst,NbrTestGcd,
	    NbrTestSimple,NbrTestDiCnst,NbrTestProjEqDi,NbrTestProjFMDi,NbrTestProjEq,
	    NbrTestProjFM,NbrTestDiVar,NbrProjFMTotal,NbrFMSystNonAug);
    for (i=0;i<17;i++)
	    fprintf(fp," %d", FMComp[i]);
    fprintf(fp,"\n");

    safe_fclose(fp, filename);
    free(filename);
}
