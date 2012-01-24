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
/* Dependence Graph computation for Allen & Kennedy algorithm
 *
 * Remi Triolet, Yi-qing Yang
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
 *  - creation of quick_privatize.c and prettyprint.c to reduce
 *    the size of ricedg.c (FI, Oct. 1995)
 *
 * Notes:
 *  - Many values seem to be assigned to StatementToContext and never be
 *    freed;
 *
 */

#include "genC.h"
#include "local.h"

/* local variables */
/*the variables for the statistics of test of dependence and parallelization */
/* they should not be global ? FC.
 */
int NbrArrayDepInit = 0;
int NbrIndepFind = 0;
int NbrAllEquals = 0;
int NbrDepCnst = 0;
int NbrTestExact = 0;
int NbrDepInexactEq = 0;
int NbrDepInexactFM = 0;
int NbrDepInexactEFM = 0;
int NbrScalDep = 0;
int NbrIndexDep = 0;
int deptype[5][3], constdep[5][3];
int NbrTestCnst = 0;
int NbrTestGcd = 0;
int NbrTestSimple = 0; /* by sc_normalize() */
int NbrTestDiCnst = 0;
int NbrTestProjEqDi = 0;
int NbrTestProjFMDi = 0;
int NbrTestProjEq = 0;
int NbrTestProjFM = 0;
int NbrTestDiVar = 0;
int NbrProjFMTotal = 0;
int NbrFMSystNonAug = 0;
int FMComp[18]; /*for counting the number of F-M complexity less than 16.
 The complexity of one projection by F-M is multiply
 of the nbr. of inequations positive and the nbr. of
 inequations negatives who containe the variable
 eliminated.The last elem of the array (ie FMComp[17])
 is used to count cases with complexity over 16*/
bool is_test_exact = true;
bool is_test_inexact_eq = false;
bool is_test_inexact_fm = false;
bool is_dep_cnst = false;
bool is_test_Di;
bool Finds2s1;

int Nbrdo;
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

static bool PRINT_RSDG = false;

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

static int dg_type = DG_FAST;


/*********************************************************************************/
/* INTERFACE FUNCTIONS                                                           */
/*********************************************************************************/

static bool rice_dependence_graph(char */*mod_name*/);

bool rice_fast_dependence_graph(char *mod_name) {
  dg_type = DG_FAST;
  return rice_dependence_graph(mod_name);
}

bool rice_full_dependence_graph(char *mod_name) {
  dg_type = DG_FULL;
  return rice_dependence_graph(mod_name);
}

bool rice_semantics_dependence_graph(char *mod_name) {
  dg_type = DG_SEMANTICS;
  return rice_dependence_graph(mod_name);
}

bool rice_regions_dependence_graph(char *mod_name) {
  if(!same_string_p(rule_phase(find_rule_by_resource("CHAINS")),
      "REGION_CHAINS"))
    pips_user_warning("Region chains not selected - using effects instead\n");

  dg_type = DG_FAST;
  return rice_dependence_graph(mod_name);
}



/*********************************************************************************/
/* STATIC FUNCTION DECLARATIONS                                                  */
/*********************************************************************************/

static void rdg_unstructured(unstructured /*u*/);
static void rdg_statement(statement /*stat*/);
static void rdg_loop(statement /*stat*/);
static void rice_update_dependence_graph(statement /*stat*/, set /*region*/);
static list TestCoupleOfEffects(statement /*s1*/,
                                effect /*e1*/,
                                statement /*s2*/,
                                effect /*e2*/,
                                list /*llv*/,
                                Ptsg */*gs*/,
                                list */*levelsop*/,
                                Ptsg */*gsop*/);
static list TestDependence(list /*n1*/,
                           Psysteme /*sc1*/,
                           statement /*s1*/,
                           effect /*ef1*/,
                           reference /*r1*/,
                           list /*n2*/,
                           Psysteme /*sc2*/,
                           statement /*s2*/,
                           effect /*ef2*/,
                           reference /*r2*/,
                           list /*llv*/,
                           Ptsg */*gs*/,
                           list */*levelsop*/,
                           Ptsg */*gsop*/);
static bool build_and_test_dependence_context(reference /*r1*/,
                                              reference /*r2*/,
                                              Psysteme /*sc1*/,
                                              Psysteme /*sc2*/,
                                              Psysteme */*psc_dep*/,
                                              list /*llv*/,
                                              list /*s2_enc_loops*/);
static bool gcd_and_constant_dependence_test(reference /*r1*/,
                                             reference /*r2*/,
                                             list /*llv*/,
                                             list /*s2_enc_loops*/,
                                             Psysteme */*psc_dep*/);
static void dependence_system_add_lci_and_di(Psysteme */*psc_dep*/,
                                             list /*s1_enc_loops*/,
                                             Pvecteur */*p_DiIncNonCons*/);
static list TestDiVariables(Psysteme /*ps*/,
                            int /*cl*/,
                            statement /*s1*/,
                            effect /*ef1*/,
                            statement /*s2*/,
                            effect /*ef2*/);
static Ptsg dependence_cone_positive(Psysteme /*dep_sc*/);
static list loop_variant_list(statement /*stat*/);
static bool TestDiCnst(Psysteme /*ps*/,
                       int /*cl*/,
                       statement /*s1*/,
                       effect /*ef1*/,
                       statement /*s2*/,
                       effect /*ef2*/);


/**************************************** WALK THROUGH THE DEPENDENCE GRAPH */

/* The supplementary call to init_ordering_to_statement should be
 avoided if ordering.c were more clever. */
static bool rice_dependence_graph(char *mod_name) {
  FILE *fp;

  statement mod_stat;
  int i, j;
  graph chains;
  string dg_name;
  entity module = local_name_to_top_level_entity(mod_name);

  set_current_module_entity(module);

  set_current_module_statement((statement)db_get_memory_resource(DBR_CODE,
                                                                 mod_name,
                                                                 true));
  mod_stat = get_current_module_statement();

  chains = (graph)db_get_memory_resource(DBR_CHAINS, mod_name, true);

  ResetLoopCounter();

  debug_on("RICEDG_DEBUG_LEVEL");
  pips_debug(1, "Computing Rice dependence graph for %s\n", mod_name);

  ifdebug(2) {
    hash_warn_on_redefinition();
    graph_consistent_p(chains);
  }

  ifdebug(1) {
    fprintf(stderr,
            "Space for chains: %d bytes\n",
            gen_allocated_memory((gen_chunk *)chains));
    fprintf(stderr,
            "Space for obj_table: %d bytes\n",
            current_shared_obj_table_size());
  }

  hash_warn_on_redefinition();
  dg = copy_graph(chains);

  ifdebug(1) {
    fprintf(stderr,
            "Space for chains's copy: %d bytes\n",
            gen_allocated_memory((gen_chunk *)dg));
    fprintf(stderr,
            "Space for obj_table: %d bytes\n",
            current_shared_obj_table_size());
  }

  pips_debug(8,"original graph\n");
  ifdebug(8) {
    set_ordering_to_statement(mod_stat);
    prettyprint_dependence_graph(stderr, mod_stat, dg);
    reset_ordering_to_statement();
  }

  debug_off();

  if(dg_type == DG_SEMANTICS)
    set_precondition_map((statement_mapping)db_get_memory_resource(DBR_PRECONDITIONS,
                                                                   mod_name,
                                                                   true));

  set_cumulated_rw_effects((statement_effects)db_get_memory_resource(DBR_CUMULATED_EFFECTS,
                                                                     mod_name,
                                                                     true));

  debug_on("RICEDG_DEBUG_LEVEL");
  pips_debug(1, "finding enclosing loops ...\n");

  set_enclosing_loops_map(loops_mapping_of_statement(mod_stat));
  ifdebug(3) {
    fprintf(stderr, "\nThe number of DOs :\n");
    fprintf(stderr, " Nbrdo=%d", Nbrdo);
  }
  debug_on("QUICK_PRIVATIZER_DEBUG_LEVEL");

  /* we need to access the statements from their ordering for the
   dependance-graph: */
  /* Do this as late as possible as it is also used by pipsdbm... */
  set_ordering_to_statement(mod_stat);

  quick_privatize_graph(dg);
  debug_off();

  for (i = 0; i <= 4; i++) {
    for (j = 0; j <= 2; j++) {
      deptype[i][j] = 0;
      constdep[i][j] = 0;
    }
  }

  /* walk thru mod_stat to find well structured loops.
   update dependence graph for these loops. */

  rdg_statement(mod_stat);

  ifdebug(3) {
    fprintf(stderr, "\nThe results of statistique of test of dependence are:\n");
    fprintf(stderr, "NbrArrayDepInit = %d\n", NbrArrayDepInit);
    fprintf(stderr, "NbrIndepFind = %d\n", NbrIndepFind);
    fprintf(stderr, "NbrAllEquals = %d\n", NbrAllEquals);
    fprintf(stderr, "NbrDepCnst = %d\n", NbrDepCnst);
    fprintf(stderr, "NbrTestExact= %d\n", NbrTestExact);
    fprintf(stderr, "NbrDepInexactEq= %d\n", NbrDepInexactEq);
    fprintf(stderr, "NbrDepInexactFM= %d\n", NbrDepInexactFM);
    fprintf(stderr, "NbrDepInexactEFM= %d\n", NbrDepInexactEFM);
    fprintf(stderr, "NbrScalDep = %d\n", NbrScalDep);
    fprintf(stderr, "NbrIndexDep = %d\n", NbrIndexDep);
    fprintf(stderr, "deptype[][]");
    for (i = 0; i <= 4; i++)
      for (j = 0; j <= 2; j++)
        fprintf(stderr, "%d  ", deptype[i][j]);
    fprintf(stderr, "\nconstdep[][]");
    for (i = 0; i <= 4; i++)
      for (j = 0; j <= 2; j++)
        fprintf(stderr, "%d  ", constdep[i][j]);
    fprintf(stderr, "\nNbrTestCnst = %d\n", NbrTestCnst);
    fprintf(stderr, "NbrTestGcd = %d\n", NbrTestGcd);
    fprintf(stderr, "NbrTestSimple = %d\n", NbrTestSimple);
    fprintf(stderr, "NbrTestDiCnst = %d\n", NbrTestDiCnst);
    fprintf(stderr, "NbrTestProjEqDi = %d\n", NbrTestProjEqDi);
    fprintf(stderr, "NbrTestProjFMDi = %d\n", NbrTestProjFMDi);
    fprintf(stderr, "NbrTestProjEq = %d\n", NbrTestProjEq);
    fprintf(stderr, "NbrTestProjFM = %d\n", NbrTestProjFM);
    fprintf(stderr, "NbrTestDiVar = %d\n", NbrTestDiVar);
    fprintf(stderr, "NbrProjFMTotal = %d\n", NbrProjFMTotal);
    fprintf(stderr, "NbrFMSystNonAug = %d\n", NbrFMSystNonAug);
    fprintf(stderr, "FMComp[]");
    for (i = 0; i < 18; i++)
      fprintf(stderr, "%d ", FMComp[i]);
  }
  /* write the result to the file correspondant in the order of :
   module,NbrArrayDeptInit,NbrIndeptFind,NbrArrayDepInit,NbrScalDep,
   NbrIndexDep,deptype[5][3]*/
  if(get_bool_property(RICEDG_PROVIDE_STATISTICS))
    writeresult(mod_name);

  ifdebug(2) {
    fprintf(stderr, "updated graph\n");
    prettyprint_dependence_graph(stderr, mod_stat, dg);
  }

  /* FI: this is not a proper way to do it */
  if(get_bool_property("PRINT_DEPENDENCE_GRAPH") || PRINT_RSDG) {
    dg_name = strdup(concatenate(db_get_current_workspace_directory(),
                                 "/",
                                 mod_name,
                                 ".dg",
                                 NULL));
    fp = safe_fopen(dg_name, "w");
    prettyprint_dependence_graph(fp, mod_stat, dg);
    safe_fclose(fp, dg_name);
  }

  debug_off();

  DB_PUT_MEMORY_RESOURCE(DBR_DG, mod_name, (char*) dg);

  reset_ordering_to_statement();
  reset_current_module_entity();
  reset_current_module_statement();
  reset_precondition_map();
  reset_cumulated_rw_effects();
  clean_enclosing_loops();

  return true;
}

static void rdg_unstructured(unstructured u) {
  list blocs = NIL;

  CONTROL_MAP(c, {
        rdg_statement(control_statement(c));
      }, unstructured_control(u), blocs);

  gen_free_list(blocs);
}

static void rdg_statement(statement stat) {
  forloop fl = forloop_undefined;
  whileloop wl = whileloop_undefined;
  statement body = statement_undefined;
  instruction istat = statement_instruction(stat);

  switch(instruction_tag(istat)) {
    case is_instruction_block: {
      FOREACH (STATEMENT, s, instruction_block(istat))
        rdg_statement(s);
      break;
    }
    case is_instruction_test:
      rdg_statement(test_true(instruction_test(istat)));
      rdg_statement(test_false(instruction_test(istat)));
      break;
    case is_instruction_loop:
      rdg_loop(stat);
      break;
    case is_instruction_whileloop:
      wl = instruction_whileloop (istat);
      body = whileloop_body (wl);
      rdg_statement(body);
      break;
    case is_instruction_forloop:
      fl = instruction_forloop (istat);
      body = forloop_body (fl);
      rdg_statement(body);
      break;
    case is_instruction_goto:
    case is_instruction_call:
    case is_instruction_expression:
      break;
    case is_instruction_unstructured:
      rdg_unstructured(instruction_unstructured(istat));
      break;
    default:
      pips_internal_error("case default reached with tag %d",
          instruction_tag(istat));
      break;
  }
}

static void rdg_loop(statement stat) {
  set region;

  if(get_bool_property("COMPUTE_ALL_DEPENDENCES")) {
    region = region_of_loop(stat);
    ifdebug(7) {
      fprintf(stderr, "[rdg_loop] applied on region:\n");
      print_statement_set(stderr, region);
    }
    rice_update_dependence_graph(stat, region);
  } else {
    if((region = distributable_loop(stat)) == set_undefined) {
      ifdebug(1) {
        fprintf(stderr,
                "[rdg_loop] skipping loop %td (but recursing)\n",
                statement_number(stat));
      }
      rdg_statement(loop_body(statement_loop(stat)));
    } else {
      rice_update_dependence_graph(stat, region);
      set_free(region);
    }
  }
}



/* Update of dependence graph
 *
 * The update of conflict information is performed for all statements in
 * a loop. This set of statements seems to be defined by "region", while
 * the loop is defined by "stat".
 *
 * Let v denote a vertex, a an arc, s a statement attached to a vertex
 * and c a conflict, i.e. a pair of effects (e1,e2). Each arc is
 * labelled by a list of statements. Each vertex points to a list of
 * arcs. Each arc points to a list of conflicts. See dg.f.tex and
 * graph.f.tex in Documentation/Newgen for more details.
 *
 * The procedure is quite complex because:
 *
 * - each dependence arc carries a list of conflicts; when a conflict is
 * found non-existent, it has to be removed from the conflict list and
 * the arc may also have to be removed from the arc list associated to
 * the current vertex if its associated conflict list becomes empty;
 * removing an element from a list is a pain because you need to know
 * the previous element and to handle the special case of the list first
 * element removal;
 *
 * - the same conflict may appear twice, once as a def-use conflict and
 * once as a use-def conflicts; since dependence testing may be long,
 * the conflict and its symmetric conflict are processed simultaneously;
 * of course, conflict list and dependence arc updates become tricky,
 *
 * - especially if the two vertices of the dependence arc are only one:
 * you can have a s1->s1 dependence; thus you might be scanning and
 * updating the very same list of arcs and conflicts twice... but there
 * is a test on pchead and pchead1 to avoid it (pchead and pchead1
 * points to the heads of the current conflict list and of the conflict
 * list containing the symmetrical conflict); remember, you have no
 * guarantee that s1!=s2 and/or v1=!v2 and/or a1!=a2
 *
 * - note that there also is a test about symmetry: def-def and use-use
 * conflicts cannot have symmetric conflicts;
 *
 * - because of the use-def/def-use symmetry, a conflict may already
 * have been processed when it is accessed;
 *
 * - due to a strange feature in the NewGen declarations, the arcs
 * are called "successors"; variables in the procedure are named with
 * the same convention except... except when they are not; so a so-called
 * "successor" may be either an arc or a vertex (or a statement since
 * each vertex points to a statement)
 *
 * - you don't know if dg is a graph or a multigraph; apparently it's
 * a multigraph but there is no clear semantics attaches to the different
 * arcs joining a pair of vertices (Pierre Jouvelot, help!)
 *
 * - conflicts are assumed uniquely identifiable by the addresses of their
 * two effects (and by labelling the same pair of vertices - but not the
 * same arc); that calls for tricky bugs if some sharing between effects
 * exists.
 *
 * The procedure is made of many too many nested loops and tests:
 *
 * for all vertex v1 in graph dg
 *   if statement s1 associated to v1 in region
 *     for all arcs a1 outgoing from v1
 *       let v2 be the sink of a1 and s2 the statement associated to v2
 *         if s2 in region
 *           for all conflicts c12 carried by a1
 *             if c12 has not yet been refined
 *               if c12 may have a symmetric conflict
 *                 for all arcs a2 outgoing from the sink v2 of a1
 *                   if sink(a2) equals v1
 *                     for all conflicts c21
 *                       if c21 equal c12
 *                         halleluia!
 *                         compute refined dependence information for c12
 *                         and possibly c21
 *                         possibly update c21 and possibly remove a2
 *                         update c12 and possibly remove a1
 *
 * Good luck for the restructuring! I'm not sure the current procedure
 * might not end up removing as a2 the very same arc a1 it uses to
 * iterate...
 */

static void rice_update_dependence_graph(statement stat, set region) {
  list pv1, ps, pss;
  list llv, llv1;

  pips_assert("statement is a loop", statement_loop_p(stat));

  pips_debug(1, "Begin\n");

  if(dg_type == DG_FULL) {
    pips_debug(1, "computing execution contexts\n");
    set_context_map(contexts_mapping_of_nest(stat));
  }

  llv = loop_variant_list(stat);

  ifdebug(6) {
    fprintf(stderr, "The list of loop variants is :\n");
    MAP(ENTITY, e, fprintf(stderr," %s", entity_local_name(e)), llv);
    fprintf(stderr, "\n");
  }

  for (pv1 = graph_vertices(dg); pv1 != NIL; pv1 = CDR(pv1)) {
    vertex v1 = VERTEX(CAR(pv1));
    dg_vertex_label dvl1 = (dg_vertex_label)vertex_vertex_label(v1);
    statement s1 = vertex_to_statement(v1);

    if(!set_belong_p(region, (char *)s1))
      continue;

    dg_vertex_label_sccflags(dvl1) = make_sccflags(scc_undefined, 0, 0, 0);

    ps = vertex_successors(v1);
    pss = NIL;
    while(ps != NIL) {
      successor su = SUCCESSOR(CAR(ps));
      vertex v2 = successor_vertex(su);
      statement s2 = vertex_to_statement(v2);
      dg_arc_label dal = (dg_arc_label)successor_arc_label(su);
      list true_conflicts = NIL;
      list pc, pchead;

      if(!set_belong_p(region, (char *)s2)) {
        pss = ps;
        ps = CDR(ps);
        continue;
      }

      pc = dg_arc_label_conflicts(dal);
      pchead = pc;
      while(pc != NIL) {
        conflict c = CONFLICT(CAR(pc));
        effect e1 = conflict_source(c);
        effect e2 = conflict_sink(c);

        ifdebug(4) {
          fprintf(stderr, "dep %02td (", statement_number(s1));
          print_words(stderr, words_effect(e1));
          fprintf(stderr, ") --> %02td (", statement_number(s2));
          print_words(stderr, words_effect(e2));
          fprintf(stderr, ") \n");
        }

        if(conflict_cone(c) != cone_undefined) {
          /* This conflict cone has been updated. */
          ifdebug(4) {
            fprintf(stderr, " \nThis dependence has been computed.\n");
          }
          true_conflicts = gen_nconc(true_conflicts, CONS(CONFLICT, c, NIL));
        } else { /*Compute this conflit and the opposite one */
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

          Finds2s1 = false;

          /*looking for the opposite dependence from (s2,e2) to
           (s1,e1) */

          /* Make sure that you do not try to find the very same
           * conflict: eliminate symmetric conflicts like dd's,
           * and, if the KEEP-READ-READ-DEPENDENCE option is on,
           * the unusual uu's.
           */
          if(!((s1 == s2) && (action_write_p(effect_action(e1)))
              && (action_write_p(effect_action(e2)))) && !((s1 == s2)
              && (action_read_p(effect_action(e1)))
              && (action_read_p(effect_action(e2)))))
          /* && (reference_indices(effect_any_reference(e1))) != NIL) */
          {
            pips_debug (4, "looking for the opposite dependence");

            ps2su = vertex_successors(v2);
            ps2sus = NIL;
            while(ps2su != NIL && !Finds2s1) {
              s2su = SUCCESSOR(CAR(ps2su));
              v1bis = successor_vertex(s2su);
              s1bis = vertex_to_statement(v1bis);
              if(s1bis != s1) {
                ps2sus = ps2su;
                ps2su = CDR(ps2su);
                continue;
              } else {
                dals2s1 = (dg_arc_label)successor_arc_label(s2su);
                pcs2s1 = dg_arc_label_conflicts(dals2s1);
                pchead1 = pcs2s1;
                while((pcs2s1 != NIL) && !Finds2s1) {
                  cs2s1 = CONFLICT(CAR(pcs2s1));
                  e1bis = conflict_source(cs2s1);
                  e2bis = conflict_sink(cs2s1);
                  if(e1bis == e2 && e2bis == e1) {
                    Finds2s1 = true;
                    continue;
                  } else {
                    pcs2s1 = CDR(pcs2s1);
                    continue;
                  }
                }
                if(!Finds2s1) {
                  ps2sus = ps2su;
                  ps2su = CDR(ps2su);
                }
                continue;
              }
            }

            /* if (!Finds2s1)
             pips_internal_error("Expected opposite dependence are not found"); */

            if(Finds2s1) {
              ifdebug(4) {
                fprintf(stderr, "\n dep %02td (", statement_number(s2));
                print_words(stderr, words_effect(e1bis));
                fprintf(stderr, ") --> %02td (", statement_number(s1));
                print_words(stderr, words_effect(e2bis));
                fprintf(stderr, ") \n");
              }
            }

          }

          llv1 = gen_copy_seq(llv);
          /* freed in of leaked. */
          levels = TestCoupleOfEffects(s1,
                                       e1,
                                       s2,
                                       e2,
                                       llv1,
                                       &gs,
                                       &levelsop,
                                       &gsop);

          /* updating DG for the dependence (s1,e1)-->(s2,e2)*/
          if(levels == NIL) {
            debug(4, "", "\nThe dependence (s1,e1)-->(s2,e2)"
              " must be removed. \n");

            conflict_source(c) = effect_undefined;
            conflict_sink(c) = effect_undefined;
            free_conflict(c);
          } else {
            debug(4, "", "\nUpdating the dependence (s1,e1)-->(s2,e2)\n");

            if(!SG_UNDEFINED_P(gs))
              conflict_cone(c) = make_cone(levels, gs);
            else
              conflict_cone(c) = make_cone(levels, SG_UNDEFINED);
            true_conflicts = gen_nconc(true_conflicts, CONS(CONFLICT, c, NIL));
          }

          if(Finds2s1) {
            /* updating DG for the dependence (s2,e2)-->(s1,e1)*/
            if(levelsop == NIL) {
              debug(4, "", "\nThe dependence (s2,e2)-->(s1,e1)"
                " must be removed.\n");

              conflict_source(cs2s1) = effect_undefined;
              conflict_sink(cs2s1) = effect_undefined;
              /*gen_free(cs2s1);*/

              if(pchead == pchead1)
                /* They are in the same conflicts list */
                gen_remove(&pchead, cs2s1);
              else {
                gen_remove(&pchead1, cs2s1);
                if(pchead1 != NIL) {
                  dg_arc_label_conflicts(dals2s1) = pchead1;
                  successor_arc_label_(s2su) = newgen_arc_label(dals2s1);
                } else {
                  /* This successor has only one
                   conflict that has been killed.*/successor_vertex(s2su)
                      = vertex_undefined;
                  successor_arc_label_(s2su)
                      = (arc_label)dg_arc_label_undefined;
                  free_successor(s2su);
                  ps2su = CDR(ps2su);
                  if(ps2sus == NIL) {
                    vertex_successors(v2) = ps2su;
                  } else {
                    CDR(ps2sus) = ps2su;
                  }
                }
              }
            }

            else {
              debug(4, "", "\nUpdating the dependence (s2,e2)-->(s1,e1)\n");

              if(!SG_UNDEFINED_P(gsop))
                conflict_cone(cs2s1) = make_cone(levelsop, gsop);
              else
                conflict_cone(cs2s1) = make_cone(levelsop, SG_UNDEFINED);
            }
          }
        }
        pc = CDR(pc);
      }

      /* gen_free_list(dg_arc_label_conflicts(dal));*/
      if(true_conflicts != NIL) {
        dg_arc_label_conflicts(dal) = true_conflicts;
        pss = ps;
        ps = CDR(ps);
      } else {
        successor_vertex(su) = vertex_undefined;
        successor_arc_label_(su) = (arc_label)dg_arc_label_undefined;
        free_successor(su);
        ps = CDR(ps);
        if(pss == NIL)
          vertex_successors(v1) = ps;
        else
          CDR(pss) = ps;
      }
      /*ifdebug(4){ prettyprint_dependence_graph(stderr, mod_stat, dg); }*/
    }
  }

  ifdebug(8) {
    pips_debug(8,"updated graph\n");
    print_statement_set(stderr, region);
    prettyprint_dependence_graph(stderr, get_current_module_statement(), dg);
  }

  reset_context_map();

  // No leak
  gen_free_list(llv);
}


/*********************************************************************************/
/* DEPENDENCE TEST                                                               */
/*********************************************************************************/

static list TestCoupleOfEffects(statement s1,
                                effect e1,
                                statement s2,
                                effect e2,
                                list llv, /* must be freed */
                                Ptsg * gs,
                                list * levelsop,
                                Ptsg * gsop) {
  list n1 = load_statement_enclosing_loops(s1);
  Psysteme sc1 = SC_UNDEFINED;
  reference r1 = effect_any_reference( e1 );

  list n2 = load_statement_enclosing_loops(s2);
  Psysteme sc2 = SC_UNDEFINED;
  reference r2 = effect_any_reference(e2);

  switch(dg_type) {
    case DG_FAST: {
      /* use region information if some is available */
      sc1 = effect_system(e1);
      sc2 = effect_system(e2);
      break;
    }

    case DG_FULL: {
      sc1 = load_statement_context(s1);
      sc2 = load_statement_context(s2);
      break;
    }

    case DG_SEMANTICS: {
      /* This is not correct because loop bounds should be frozen on
       loop entry; we assume variables used in loop bounds are not
       too often modified by the loop body */
      transformer t1 = load_statement_precondition(s1);
      transformer t2 = load_statement_precondition(s2);

      sc1 = (Psysteme)predicate_system(transformer_relation(t1));
      sc2 = (Psysteme)predicate_system(transformer_relation(t2));
      break;
    }

    default:
      pips_error("TestCoupleOfEffects", "Unknown dependence test %d\n", dg_type);
      break;
  }

  return TestCoupleOfReferences(n1,
                                sc1,
                                s1,
                                e1,
                                r1,
                                n2,
                                sc2,
                                s2,
                                e2,
                                r2,
                                llv,
                                gs,
                                levelsop,
                                gsop);
}

/*
 * This function checks if two references have memory locations in common.
 *
 * The problem is obvious for references to the same scalar variable.
 *
 * The problem is also obvious if the references are not to the same
 * variable, except if the two variables have memory locations in common,
 * in which case we assume there are common locations and all kind of
 * dependences (although offsets in COMMON could be taken into account).
 *
 * When both references are to the same array variable, the function
 * TestDependence() is called.
 *
 * FI: When both references are relative to the same pointer variable, the
 * variable value is *assumed* to be the same in the two statements and
 * the function TestDependence() is called. Transformers on pointers
 * should be used to check that the pointer value is constant. The
 * simplest transformer would be the written memory effects for the
 * common enclosing loop.
 */

list TestCoupleOfReferences(list n1,
                            Psysteme sc1 __attribute__ ((unused)),
                            statement s1,
                            effect ef1,
                            reference r1,
                            list n2,
                            Psysteme sc2 __attribute__ ((unused)),
                            statement s2,
                            effect ef2,
                            reference r2,
                            list llv,
                            Ptsg * gs __attribute__ ((unused)),
                            list * levelsop __attribute__ ((unused)),
                            Ptsg * gsop __attribute__ ((unused))) {
  _int i, cl, dims, ty;
  list levels = NIL, levels1 = NIL;

  entity e1 = reference_variable(r1);
  entity e2 = reference_variable(r2);

  list b1 = reference_indices(r1);
  list b2 = reference_indices(r2);

  type t1 = ultimate_type(entity_type(e1));

  if(e1 != e2) {
    ifdebug(1) {
      fprintf(stderr, "dep %02td (", statement_number(s1));
      print_words(stderr, words_effect(ef1));
      fprintf(stderr, ") --> %02td (", statement_number(s2));
      print_words(stderr, words_effect(ef2));
      fprintf(stderr, ") \n");
    }
    pips_user_warning("Dependence between differents variables: "
        "%s and %s\nDependence assumed\n",
        entity_local_name(e1), entity_local_name(e2));
  }

  /* if (e1 == e2 && !entity_scalar_p(e1) && !entity_scalar_p(e2)) */
  /* FI: Why have two tests under the condition e1==e2? */

  /* FI: this test must be modified to take pointer dereferencing
   * such as p[i] into account, although p as an entity generates
   * atomic references.
   *
   * If chains.c were updated, we could also check that:
   * gen_length(b1)==gen_length(b2).
   */
  if(e1 == e2 && !entity_all_locations_p(e1) && !entity_all_locations_p(e2)
      && (!entity_atomic_reference_p(e1) || (pointer_type_p(t1)
          && gen_length(b1) > 0))) {
    if(get_bool_property("RICEDG_STATISTICS_ALL_ARRAYS")) {
      NbrArrayDepInit++;
    } else {
      if(b1 != NIL && b2 != NIL)
        NbrArrayDepInit++;
    }

    if(b1 == NIL || b2 == NIL) {
      /* A(*) reference appears in the dependence graph */
      cl = FindMaximumCommonLevel(n1, n2);

      for (i = 1; i <= cl; i++)
        levels = gen_nconc(levels, CONS(INT, i, NIL));

      if(statement_possible_less_p(s1, s2))
        levels = gen_nconc(levels, CONS(INT, cl+1, NIL));

      if(Finds2s1) {
        for (i = 1; i <= cl; i++)
          levels1 = gen_nconc(levels1, CONS(INT, i, NIL));

        if(statement_possible_less_p(s2, s1))
          levels1 = gen_nconc(levels1, CONS(INT, cl+1, NIL));

        *levelsop = levels1;
      }

      gen_free_list(llv);
    } else {
      /* llv is freed here */
      levels = TestDependence(n1,
                              sc1,
                              s1,
                              ef1,
                              r1,
                              n2,
                              sc2,
                              s2,
                              ef2,
                              r2,
                              llv,
                              gs,
                              levelsop,
                              gsop);
    }

    if(get_bool_property("RICEDG_STATISTICS_ALL_ARRAYS") || (b1 != NIL && b2
        != NIL)) {
      if(levels != NIL) {
        /* test the dependence type, constant dependence?  */
        dims = gen_length(b1);
        ty = dep_type(effect_action(ef1), effect_action(ef2));
        deptype[dims][ty]++;
        if(is_dep_cnst) {
          NbrDepCnst++;
          constdep[dims][ty]++;
        }
      }

      if(*levelsop != NIL) {
        /* test the dependence type, constant dependence?
         exact dependence? */
        dims = gen_length(b1);
        ty = dep_type(effect_action(ef2), effect_action(ef1));
        deptype[dims][ty]++;
        if(is_dep_cnst) {
          NbrDepCnst++;
          constdep[dims][ty]++;
        }
      }

      if(levels != NIL || *levelsop != NIL) {
        if(is_test_exact)
          NbrTestExact++;
        else {
          if(is_test_inexact_eq) {
            if(is_test_inexact_fm)
              NbrDepInexactEFM++;
            else
              NbrDepInexactEq++;
          } else
            NbrDepInexactFM++;
        }
      }

      ifdebug(6) {
        if(is_test_exact)
          fprintf(stderr, "\nThis test is exact! \n");
        else if(is_test_inexact_eq)
          fprintf(stderr, "\nTest not exact : "
            "non-exact elimination of equation!");
        else
          fprintf(stderr, "\nTest not exact : "
            "non-exact elimination of F-M!");
      }
    }

    return (levels);
  } else {
    /* the case of scalar variables and equivalenced arrays */

    // llv is no longer used
    gen_free_list(llv);

    cl = FindMaximumCommonLevel(n1, n2);

    for (i = 1; i <= cl; i++)
      levels = gen_nconc(levels, CONS(INT, i, NIL));

    if(statement_possible_less_p(s1, s2))
      levels = gen_nconc(levels, CONS(INT, cl+1, NIL));

    if((instruction_loop_p(statement_instruction(s1)))
        || instruction_loop_p(statement_instruction(s2)))
      NbrIndexDep++;
    else {/*scalar variable dependence */
      NbrScalDep++;
      ty = dep_type(effect_action(ef1), effect_action(ef2));
      deptype[0][ty]++;
    }

    if(Finds2s1) {
      for (i = 1; i <= cl; i++)
        levels1 = gen_nconc(levels1, CONS(INT, i, NIL));

      if(statement_possible_less_p(s2, s1))
        levels1 = gen_nconc(levels1, CONS(INT, cl+1, NIL));

      *levelsop = levels1;
      if((instruction_loop_p(statement_instruction(s1)))
          || instruction_loop_p(statement_instruction(s2)))
        NbrIndexDep++;
      else {/*case of scalar variable dependence */
        NbrScalDep++;
        ty = dep_type(effect_action(ef2), effect_action(ef1));
        deptype[0][ty]++;
      }
    }

    return (levels);
  }
}

/* static list TestDependence(list n1, n2, Psysteme sc1, sc2,
 *                             statement s1, s2, effect ef1, ef2,
 *                             reference r1, r2, list llv,
 *                             list *levelsop, Ptsg *gs,*gsop)
 * input    :
 *      list n1, n2      : enclosing loops for statements s1 and s2;
 *      Psysteme sc1, sc2: current context for each statement;
 *      statement s1, s2 : currently analyzed statements;
 *      effect ef1, ef2  : effects of each statement upon the current variable;
 *                         (maybe array regions)
 *      reference r1, r2 : current variables references;
 *      list llv         : loop variant list
 *                         (variables that vary in loop nests n1 and n2?);
 *      list *levelsop   : dependence levels from s2 to s1
 *      Ptsg *gs,*gsop   : dependence cones from s1 to s2 and from s2 to s1
 * output   : dependence levels from s1 to s2
 * modifies : levelsop, gsop, gs and returns levels
 *
 * Comments :
 *
 * This procedure has been amplified twice. The initial procedure
 * only computed the dependence levels, "levels", for conflict s1->s2.
 * The dependence cone computation was added by Yi-Qing Yang. She later
 * added the computation of the dependence levels and the dependence cone
 * for symetrical conflict, s2 -> s1, because both conflicts share the
 * same dependence system and because a set of tests can be shared.
 *
 * For her PhD, Yi-Qing Yang also added intermediate tests and instrumentation.
 *
 * This much too long procedure is made of three parts:
 *  0. The initialization of the dependence system
 *  1. A set of feasibility tests applied to the dependence system
 *  2. The computation of levels and cone for s1->s2
 *  3. The very same computation for s2->s1
 *
 * Modification :
 *
 * - ajout des tests de faisabilites pour le systeme initial,
 * avant la projection. Yi-Qing (18/10/91)
 *
 * - decoupage de la procedure par Beatrice Creusillet
 *
 * - (indirect) build_and_test_dependence_context(), in fact system, no
 *   longer has side effects on sc1 and sc2 thru sc_normalize(); FI (12/12/95)
 *
 * - sc_rm() uncommented out after call to gcd_and_constant_dependence_test()
 *
 * - base_rm(tmp_base) added after call to sc_proj_optim_on_di_ofl() just
 *   before returning with levels==NIL
 *
 * - sc_rm() added for dep_syst, dep_syst1, dep_syst_op and dep_syst2
 */
static list TestDependence(list n1,
                           Psysteme sc1,
                           statement s1,
                           effect ef1,
                           reference r1,
                           list n2,
                           Psysteme sc2,
                           statement s2,
                           effect ef2,
                           reference r2,
                           list llv,
                           Ptsg * gs,
                           list * levelsop,
                           Ptsg * gsop) {
  Psysteme dep_syst = SC_UNDEFINED;
  Psysteme dep_syst1 = SC_UNDEFINED;
  Psysteme dep_syst2 = SC_UNDEFINED;
  Psysteme dep_syst_op = SC_UNDEFINED;
  Pbase b, coord;
  /* Automatic variables read in a CATCH block need to be declared volatile as
   * specified by the documentation*/
  Pbase volatile tmp_base;

  int l, cl;
  list levels;
  Pvecteur DiIncNonCons = NULL;

  /* Elimination of loop indices from loop variants llv */
  /* We use n2 because we take care of variables modified in
   an iteration only for the second system. */
  FOREACH(STATEMENT, s, n2) {
    entity i = loop_index(statement_loop(s));
    gen_remove(&llv,i); /* llv is dead, must be freed... */
  }

  ifdebug(6) {
    pips_debug(6, "loop variants after removing loop indices :\n");
    print_arguments(llv);
  }

  /* Build the dependence context system from the two initial context systems
   * sc1 and sc2. BC
   */
  if(!build_and_test_dependence_context(r1, r2, sc1, sc2, &dep_syst, llv, n2)) {
    /* the context system is not feasible : no dependence. BC */
    /* No dep_syst() to deallocate either, FI */
    NbrIndepFind++;
    pips_debug(4, "context system not feasible\n");
    *levelsop = NIL;

    gen_free_list(llv);
    return NIL;
  }

  /* Further construction of the dependence system; Constant and GCD tests
   * at the same time.
   */

  // FI: why is dep_syst only weekly consistent here?
  assert(sc_weak_consistent_p(dep_syst));

  if(gcd_and_constant_dependence_test(r1, r2, llv, n2, &dep_syst)) {
    /* independence proved */
    /* FI: the next statement was commented out... */
    sc_rm(dep_syst);
    NbrIndepFind++;
    *levelsop = NIL;

    gen_free_list(llv);
    return NIL;
  }

  pips_assert("The dependence system is consistent", sc_consistent_p(dep_syst));

  gen_free_list(llv);

  dependence_system_add_lci_and_di(&dep_syst, n1, &DiIncNonCons);

  ifdebug(6) {
    fprintf(stderr, "\ninitial system is:\n");
    sc_syst_debug(dep_syst);
  }

  pips_assert("The dependence system is consistent", sc_consistent_p(dep_syst));

  /* Consistency Test */
  if(sc_empty_p(dep_syst = sc_normalize(dep_syst))) {
    sc_rm(dep_syst);
    NbrTestSimple++;
    NbrIndepFind++;
    pips_debug(4, "initial normalized system not feasible\n");
    *levelsop = NIL;

    return (NIL);
  }

  cl = FindMaximumCommonLevel(n1, n2);

  pips_assert("The dependence system is consistent", sc_consistent_p(dep_syst));

  if(TestDiCnst(dep_syst, cl, s1, ef1, s2, ef2) == true) {
    /* find independences (non loop carried dependence, intra-statement).*/
    /* Such dependences are counted here as independence, but other
     * parallelizer preserve them because they are useful for
     * (assembly) instructon level scheduling
     */
    NbrTestDiCnst++;
    NbrIndepFind++;
    pips_debug(4,"\nTestDiCnst succeeded!\n");
    *levelsop = NIL;

    return (NIL);
  }

  pips_assert("The dependence system is consistent", sc_consistent_p(dep_syst));

  is_test_exact = true;
  is_test_inexact_eq = false;
  is_test_inexact_fm = false;
  is_dep_cnst = false;

  tmp_base = base_dup(dep_syst->base);

  CATCH(overflow_error) {
    /* Some kind of arithmetic error, e.g. an integer overflow
     * has occured. Assume dependence to be conservative.
     */
    Pbase dep_syst_base = BASE_UNDEFINED;

    /* FI: wouldn't it be simpler to enumerate the useful di
     * variables?!?
     */
    /* eliminate non di variables from the basis */
    for (coord = tmp_base; !VECTEUR_NUL_P(coord); coord = coord->succ) {
      Variable v = vecteur_var(coord);
      l = DiVarLevel((entity)v);
      if(l <= 0 || l > cl)
        base_add_variable(dep_syst_base, v);
    }
    dep_syst = sc_rn(dep_syst_base);
  } TRY {
    if(sc_proj_optim_on_di_ofl(cl, &dep_syst) == false) {
      pips_debug(4,
          "projected system by sc_proj_optim_on_di() is not feasible\n");
      sc_rm(dep_syst);
      NbrIndepFind++;
      *levelsop = NIL;

      UNCATCH(overflow_error);
      base_rm(tmp_base);
      return (NIL);
    } else
      UNCATCH(overflow_error);
  }

  ifdebug(6) {
    fprintf(stderr, "projected system is:\n");
    sc_syst_debug(dep_syst);
  }

  pips_assert("The dependence system is consistent", sc_consistent_p(dep_syst));

  base_rm(tmp_base); // FI: delayed to help with debugging

  if(!sc_faisabilite_optim(dep_syst)) {
    /* Here, the long jump overflow buffer is handled at a lower level! */pips_debug(4, "projected system not feasible\n");
    NbrIndepFind++;
    *levelsop = NIL;

    return (NIL);
  }

  ifdebug(6) {
    fprintf(stderr, "normalised projected system is:\n");
    sc_syst_debug(dep_syst);
    fprintf(stderr, "The list of DiIncNonCons is :\n");
    vect_debug(DiIncNonCons);
  }

  /* keep DiIncNonCons variables if their value is unknown or different
   * from zero. Otherwise eliminate them from the dep_syst
   *
   * FI: I'm lost for this step. My guess DiIncNonCons==VECTEUR_NUL
   * in 99.99 % of all cases...
   */
  if(dep_syst != NULL) {
    while(DiIncNonCons != NULL) {
      Variable di;
      Value val;

      di = DiIncNonCons->var;
      if(sc_value_of_variable(dep_syst, di, &val) == true)
        if(value_notzero_p(val)) {
          sc_elim_var(dep_syst, di);
        }
      DiIncNonCons = DiIncNonCons->succ;
    }
  }

  ifdebug(4) {
    fprintf(stderr,
            "normalised projected system after DiIncNonCons elimination is:\n");
    sc_syst_debug(dep_syst);
  }

  /* Compute the levels for dependence arc s1 to s2 and of the opposite
   * dependence arc s2 to s1. Also compute the dependence cones if some
   * level exists (FI: shouldn't it be a loop-carried level?).
   *
   * For both cases, two systems are allocated, one is used to find the
   * dependence levels, the other one to compute the dependence cone.
   *
   * For s1->s2, dep_syst is used to compute the levels and dep_syst1
   * the cone.
   *
   * For s2->s1, dep_syst_op (op==oppposite) is used to find the dependence
   * levels, and dep_syst2 the dependence cone.
   */

  /* Build the dependence system for the opposite arc s2->s1
   * before dep_syst is modified
   */

  *levelsop = NIL;
  if(Finds2s1)
    dep_syst_op = sc_invers(sc_dup(dep_syst));

  /* Start processing for arc s1->s2 */

  ifdebug(4) {
    debug(4, "", "\nComputes the levels and DC for dep: (s1,e1)->(s2,e2)\n");
    fprintf(stderr, "\nThe distance system for dep is:\n");
    sc_syst_debug(dep_syst);
  }

  /* Make a proper copy of dep_syst before it is destroyed to compute
   * dependence levels. The basis must be in a specific order to check
   * lexico-positivity.
   */
  dep_syst1 = sc_dup(dep_syst);
  b = MakeDibaseinorder(cl);
  base_rm(dep_syst1->base);
  dep_syst1->base = b;
  dep_syst1->dimension = cl;

  if(dep_syst1->dimension == dep_syst1->nb_eq)
    is_dep_cnst = true;

  /* Compute dependence levels for s1->s2 */
  levels = TestDiVariables(dep_syst, cl, s1, ef1, s2, ef2);
  /* dep_syst is unfeasible or almost empty, and now useless */
  sc_rm(dep_syst);

  /* if (levels == NIL) NbrTestDiVar++;  Qui l'a enleve', pourquoi ? */

  if(levels != NIL) {

    *gs = dependence_cone_positive(dep_syst1);

    /* If the cone is not feasible, there are no loop-carried dependences.
     * (FI: the cone is strictly positive then...)
     *
     * This might not havebeen found by the previous test when computing
     * levels.
     *
     * But the dependence cone construction does not consider non-loop
     * carried dependences; so we can only remove those levels that are
     * smaller than the number of common levels
     */
    if(sg_empty(*gs)) {
      list l_tmp = levels;
      bool ok = false;

      pips_debug(5, "dependence cone not feasible\n");
      MAP(INT, ll, {if (ll == cl+1) ok = true;}, l_tmp);

      if(ok) {
        pips_debug(5, "innermost level found and kept\n");
        levels = CONS(INT, cl+1, NIL);
      } else {
        pips_debug(5, "innermost level not found, no dependences");
        levels = NIL;
      }
      gen_free_list(l_tmp);
      sg_rm(*gs);
      *gs = SG_UNDEFINED;
    }
  }
  sc_rm(dep_syst1);

  /* print results for arc s1->s2 */

  ifdebug(4) {
    fprintf(stderr, "\nThe levels for dep (s1,s2) are:");
    MAP(INT, pl,
        {
          fprintf(stderr, " %d", pl);
        }, levels);

    if(!SG_UNDEFINED_P(*gs)) {
      fprintf(stderr, "\nThe lexico-positive dependence cone for"
        " dep (s1,s2) :\n");
      print_dependence_cone(stderr, *gs, b);
    } else
      fprintf(stderr, "\nLexico-positive dependence cone"
        " doesn't exist for dep (s1,s2).\n");
  }

  /* Start the processing for arc s2->s1 */

  if(Finds2s1) {

    pips_debug(4,"Computes the levels and DC for dep_op: (s2,e2)->(s1,e1)\n");
    ifdebug(4) {
      fprintf(stderr, "\nThe invers distance system for dep_op is:\n");
      sc_syst_debug(dep_syst_op);
    }

    dep_syst2 = sc_dup(dep_syst_op);
    b = MakeDibaseinorder(cl);
    base_rm(dep_syst2->base);
    dep_syst2->base = b;
    dep_syst2->dimension = cl;

    *levelsop = TestDiVariables(dep_syst_op, cl, s2, ef2, s1, ef1);
    sc_rm(dep_syst_op);
    if(*levelsop != NIL) {
      /* if (*levelsop == NIL) NbrTestDiVar++;  Pourquoi? */
      *gsop = dependence_cone_positive(dep_syst2);
      sc_rm(dep_syst2);

      /* if the cone is not feasible, there are no loop-carried dependences;
       * this was not found by the previous test when computing levels;
       * but the dependence cone construction does not consider non-loop
       * carried dependences; so we can only remove those levels that are
       * smaller than the number of common levels
       */
      if(sg_empty(*gsop)) {
        list l_tmp = *levelsop;
        bool ok = false;

        MAP(INT, ll, {if (ll == cl+1) ok = true;}, l_tmp);

        if(ok) {
          *levelsop = CONS(INT, cl+1, NIL);
        } else {
          *levelsop = NIL;
        }
        gen_free_list(l_tmp);
        sg_rm(*gsop);
        *gsop = SG_UNDEFINED;
      }
    }

    ifdebug(4) {
      fprintf(stderr, "\nThe levels for dep_op (s2,s1) is:");
      MAP(INT, pl,
          {
            fprintf(stderr, " %d", pl);
          }, *levelsop);

      if(!SG_UNDEFINED_P(*gsop)) {
        fprintf(stderr, "\nThe lexico-positive Dependence "
          "cone for dep_op (s2,s1):\n");
        print_dependence_cone(stderr, *gsop, b);
      } else
        fprintf(stderr, "\nLexico-positive dependence cone "
          "does not exist for dep_op (s2,s1).\n");

    }
  }

  if(levels == NIL && *levelsop == NIL) {
    NbrTestDiVar++;
    NbrIndepFind++;
    if(s1 == s2) {
      /* the case of "all equals" independence at the same statement*/
      NbrAllEquals++;
    }

    return (NIL);
  }

  return (levels);
}


/* static bool build_and_test_dependence_context(reference r1, r2,
 *                                                  Psystem sc1, sc2, *psc_dep,
 *                                                  list llv, s2_enc_loops)
 * input    :
 *      reference r1, r2  : current array references;
 *      Psystem sc1, sc2  : context systems for r1 and r2;
 *                          they might be modified and even be freed (!)
 *                          by normalization procedures
 *      Psystem *psc_dep  : pointer toward the dependence context systeme;
 *      list llv          : current loop nest variant list;
 *      list s2_enc_loops : statement s2 enclosing loops;
 *
 * output  :
 *      bool           : false if one of the initial systems is unfeasible
 *                          after normalization;
 *                          true otherwise;
 *      *psc_dep          : dependence system is the function value is true;
 *                          SC_EMPTY otherwise; no need to sc_rm() in the
 *                          latter case.
 *
 * side effects :
 *      psc_dep           : points toward the dependence context system built
 *                          from sc1 and sc2, r1 and r2, and llv.
 *                          Dependence distance variables (di) are introduced
 *                          in sc2, along with the dsi variables to take care
 *                          of variables in llv
 *                          modified in the loop nest;
 *                          irrelevant (?) existencial constraints are removed.
 *                          in order to make further manipulations easier.
 *      sc1               : side_effect of sc_normalize()
 *      sc2               : side_effect of sc_normalize()
 *
 * Comment  :
 *
 * Modifications:
 *  - sc1 and sc2 cannot be sc_normalized because they may be part of
 *    statement preconditions or regions; sc_normalize() is replaced by
 *    sc_empty_p(); since regions and preconditions are normalized with
 *    stronger normalization procedure, and since the feasibility of the
 *    dependence test will be tested with an even stronger test, this
 *    should have no accuracy impact (FI, 12 December 1995)
 */
static bool build_and_test_dependence_context(reference r1,
                                              reference r2,
                                              Psysteme sc1,
                                              Psysteme sc2,
                                              Psysteme * psc_dep,
                                              list llv,
                                              list s2_enc_loops) {
  Psysteme sc_dep, sc_tmp;
  list pc;
  int l, inc;

  /* *psc_dep must be undefined */pips_assert("build_and_test_dependence_context", SC_UNDEFINED_P(*psc_dep));

  /* Construction of initial systems sc_dep and sc_tmp from sc1 and sc2
   if not undefined
   */
  if(SC_UNDEFINED_P(sc1) && SC_UNDEFINED_P(sc2)) {
    sc_dep = sc_new();
  } else {
    if(SC_UNDEFINED_P(sc1))
      sc_dep = sc_new();
    else {
      /* sc_dep = sc1, but:
       * we keep only useful constraints in the predicate
       * (To avoid overflow errors, and to make projections easier)
       */
      Pbase variables = BASE_UNDEFINED;

      if(sc_empty_p(sc1)) {
        *psc_dep = SC_EMPTY;
        pips_debug(4,
            "first initial normalized system sc1 not feasible\n");
        return (false);
      }

      ifdebug(6) {
        debug(6, "", "Initial system sc1 before restrictions : \n");
        sc_syst_debug(sc1);
      }

      FOREACH(EXPRESSION, ex1,reference_indices(r1)) {
        normalized x1 = NORMALIZE_EXPRESSION(ex1);

        if(normalized_linear_p(x1)) {
          Pvecteur v1;
          Pvecteur v;
          v1 = (Pvecteur)normalized_linear(x1);
          for (v = v1; !VECTEUR_NUL_P(v); v = v->succ) {
            if(vecteur_var(v) != TCST)
              variables = base_add_variable(variables, vecteur_var(v));
          }
        }
      }

      sc_dep = sc_restricted_to_variables_transitive_closure(sc1, variables);
    }

    if(SC_UNDEFINED_P(sc2))
      sc_tmp = sc_new();
    else {
      /* sc_tmp = sc2, but:
       * we keep only useful constraints in the predicate
       * (To avoid overflow errors, and to make projections easier)
       *
       * FI: I'm not sure this is correct. I'm afraid sc2 might
       * be unfeasible and become feasible due to a careless
       * elimination... It all depends on:
       *
       *     sc_restricted_to_variables_transitive_closure()
       *
       * There might be a better function for that purpose
       */
      Pbase variables = BASE_UNDEFINED;

      if(sc_empty_p(sc2)) {
        *psc_dep = SC_EMPTY;
        pips_debug(4,
            "second initial normalized system not feasible\n");
        return (false);
      }

      ifdebug(6) {
        debug(6, "", "Initial system sc2 before restrictions : \n");
        sc_syst_debug(sc2);
      }

      FOREACH(EXPRESSION, ex2, reference_indices(r2)) {
        normalized x2 = NORMALIZE_EXPRESSION(ex2);

        if(normalized_linear_p(x2)) {
          Pvecteur v2;
          Pvecteur v;

          v2 = (Pvecteur)normalized_linear(x2);
          for (v = v2; !VECTEUR_NUL_P(v); v = v->succ) {
            if(vecteur_var(v) != TCST)
              variables = base_add_variable(variables, vecteur_var(v));
          }

        }
      }


      sc_tmp = sc_restricted_to_variables_transitive_closure(sc2, variables);
    }

    ifdebug(6) {
      debug(6, "", "Initial systems after restrictions : \n");
      sc_syst_debug(sc_dep);
      sc_syst_debug(sc_tmp);
    }

    /* introduce dependence distance variable if loop increment value
     is known or ... */
    for (pc = s2_enc_loops, l = 1; pc != NIL; pc = CDR(pc), l++) {
      loop lo = statement_loop(STATEMENT(CAR(pc)));
      entity i2 = loop_index(lo);
      /* FI: a more powerful evaluation function for inc should be called
       * when preconditions are available. For instance, mdg could
       * be handled without having to partial_evaluate it
       *
       * It's not clear whether sc1 or sc2 should be used to
       * estimate a variable increment like N.
       *
       * It's not clear whether it's correct or not. You would like
       * N (or other variables in the increment expression) to be
       * constant within the loop nest.
       *
       * It would be better to know the precondition associated
       * to the corresponding loop statement, but the information
       * is lost in this low-level function.
       *
       * FI: this is not true, you have sc1 and sc2 at hand to call
       * sc_minmax_of_variable()
       */
      inc = loop_increment_value(lo);
      if(inc != 0)
        sc_add_di(l, i2, sc_tmp, inc);
      else
        sc_chg_var(sc_tmp, (Variable)i2, (Variable)GetLiVar(l));
    }

    /* take care of variables modified in the loop */
    for (pc = llv, l = 1; pc != NIL; pc = CDR(pc), l++) {
      sc_add_dsi(l, ENTITY(CAR(pc)), sc_tmp);
    }

    /* sc_tmp is emptied and freed by sc_fusion() */
    sc_dep = sc_fusion(sc_dep, sc_tmp);
    vect_rm(sc_dep->base);
    sc_dep->base = BASE_NULLE;
    sc_creer_base(sc_dep);

  }

  ifdebug(6) {
    pips_debug(6,
        "\ndependence context is:\n");
    sc_syst_debug(sc_dep);
  }

  *psc_dep = sc_dep;
  return (true);
}

/* static bool gcd_and_constant_dependence_test(references r1, r2,
 *                                                 list llv, s2_enc_loops,
 *                                                 Psysteme *psc_dep)
 * input    :
 *      references r1, r2 : current references;
 *      list llv          : loop nest variant list;
 *      list s2_enc_loops : enclosing loops for statement s2;
 *      Psysteme *psc_dep : pointer toward the depedence system;
 * output   : true if there is no dependence (GCD and constant test successful);
 *            false if independence could not be proved.
 * modifies : *psc_dep; at least adds dependence equations on phi variables
 * comment  :
 *  - *psc_dep must be defined on entry; it must have been initialized
 *    by build_and_test_dependence_context.
 *  - no side effects on r1, r2,...
 */
static bool gcd_and_constant_dependence_test(reference r1,
                                             reference r2,
                                             list llv,
                                             list s2_enc_loops,
                                             Psysteme * psc_dep) {
  list pc1, pc2, pc;
  int l;

  /* Further construction of the dependence system; Constant and GCD tests
   * at the same time.
   */
  pc1 = reference_indices(r1);
  pc2 = reference_indices(r2);

  while(pc1 != NIL && pc2 != NIL) {
    expression ex1, ex2;
    normalized x1, x2;

    ex1 = EXPRESSION(CAR(pc1));
    x1 = NORMALIZE_EXPRESSION(ex1);

    ex2 = EXPRESSION(CAR(pc2));
    x2 = NORMALIZE_EXPRESSION(ex2);

    if(normalized_linear_p(x1) && normalized_linear_p(x2)) {
      Pvecteur v1, v2, v;

      v1 = (Pvecteur)normalized_linear(x1);
      v2 = vect_dup((Pvecteur)normalized_linear(x2));

      for (pc = s2_enc_loops, l = 1; pc != NIL; pc = CDR(pc), l++) {
        loop lo = statement_loop(STATEMENT(CAR(pc)));
        entity i = loop_index(lo);
        int li = loop_increment_value(lo);
        Value vli = int_to_value(li);
        if(value_notzero_p(vli)) {
          Value v = vect_coeff((Variable)i, v2);
          value_product(v,vli);
          vect_add_elem(&v2, (Variable)GetDiVar(l), v);
        } else
          vect_chg_var(&v2, (Variable)i, (Variable)GetLiVar(l));
      }

      for (pc = llv, l = 1; pc != NIL; pc = CDR(pc), l++) {
        vect_add_elem(&v2,
                      (Variable)GetDsiVar(l),
                      vect_coeff((Variable)ENTITY(CAR(pc)), v2));
      }

      if(!VECTEUR_UNDEFINED_P(v = vect_substract(v1, v2))) {
        Pcontrainte c = contrainte_make(v);

        /* case of T(3)=... et T(4)=... */
        if(contrainte_constante_p(c) && value_notzero_p(COEFF_CST(c))) {
          NbrTestCnst++;
          pips_debug(4,"TestCnst succeeded!");
          return (true);
        }
        /* test of GCD */
        if(egalite_normalize(c) == false) {
          NbrTestGcd++;
          pips_debug(4,"TestGcd succeeded!\n");
          return (true);
        }
        sc_add_eg(*psc_dep, c);
      }
      vect_rm(v2);
    }

    pc1 = CDR(pc1);
    pc2 = CDR(pc2);
  }

  if(pc1 != NIL || pc2 != NIL) {
    if(fortran_module_p(get_current_module_entity())) {
      pips_internal_error("numbers of subscript expressions differ");
    } else {
      /* C assumed */
      pips_user_warning("dependence tested between two memory access paths"
			" of different lengths for variable \"%s\".\n",
			entity_user_name(reference_variable(r1)));
    }
  }

  /* Update base of *psc_dep */
  Pbase mb = sc_to_minimal_basis(*psc_dep);
  Pbase cb = sc_base(*psc_dep);
  if(!vect_in_basis_p(mb, cb)) {
    Pbase nb = base_union(cb, mb);
    sc_base(*psc_dep) = nb;
    sc_dimension(*psc_dep) = vect_size(nb); // base_dimension(nb);
    vect_rm(cb);
  }
  vect_rm(mb);

  pips_assert("The dependence system is consistent", sc_consistent_p(*psc_dep));

  return (false);
}

/* static void dependence_system_add_lci_and_di(Psysteme *psc_dep,
 *                                              list s1_enc_loops,
 *                                              Pvecteur *p_DiIncNonCons)
 * input    :
 *      Psysteme *psc_dep : pointer toward the dependence system;
 *      list s1_enc_loops : statement s1 enclosing loops;
 *      Pvecteur *p_DiIncNonCons : pointer toward DiIncNonCons.
 *
 * output   : none
 *
 * modifies :
 *
 *  *psc_dep, the dependence systeme (addition of constraints with lci and di
 *            variables, if useful; lci is a loop counter, di an iteration
 *            difference variable);
 *
 *  DiIncNonCons: di variables are added to DiIncNonCons (means Di variables
 *            for loops with non constant increment, i.e. unknown increment),
 *            if any such loop exists.
 *
 * comment  : DiIncNonCons must be undefined on entry.
 */
static void dependence_system_add_lci_and_di(psc_dep,
                                             s1_enc_loops,
                                             p_DiIncNonCons)
  Psysteme *psc_dep;list s1_enc_loops;Pvecteur *p_DiIncNonCons; {
  int l;
  list pc;

  pips_assert("dependence_system_add_lci_and_di",
      VECTEUR_UNDEFINED_P(*p_DiIncNonCons));

  /* Addition of lck, the loop counters, and di, the iteration difference,
   * variables (if useful).
   */
  for (pc = s1_enc_loops, l = 1; pc != NIL; pc = CDR(pc), l++) {
    loop lo = statement_loop(STATEMENT(CAR(pc)));
    entity ind = loop_index(lo);
    int inc = loop_increment_value(lo);

    expression lb = range_lower(loop_range(lo));
    normalized nl = NORMALIZE_EXPRESSION(lb);

    /* If the loop increment is not trivial, express the current
     * index value as the sum of the lower bound and the product
     * of the loop counter by the loop increment.
     *
     * Else, this new equation would be redundant
     */
    /* make   nl + inc*lc# - ind = 0 */
    if(inc != 0 && inc != 1 && inc != -1 && normalized_linear_p(nl)) {
      Pcontrainte pc;
      entity lc;
      Pvecteur pv;

      lc = MakeLoopCounter();
      pv = vect_dup((Pvecteur)normalized_linear(nl));
      vect_add_elem(&pv, (Variable)lc, inc);
      vect_add_elem(&pv, (Variable)ind, -1);
      pc = contrainte_make(pv);
      sc_add_eg(*psc_dep, pc);
    }

    /* If the loop increment is unknown, which is expressed by inc==0,
     * well, I do not understand what li variables are, not how they work
     */
    /* make d#i - l#i + ind = 0 ,
     add d#i in list of DiIncNonCons*/
    if(inc == 0) {
      Pcontrainte pc;
      Pvecteur pv = NULL;
      entity di;

      di = GetDiVar(l);
      vect_add_elem(&pv, (Variable)di, 1);
      vect_add_elem(&pv, (Variable)GetLiVar(l), -1);
      vect_add_elem(&pv, (Variable)ind, 1);
      pc = contrainte_make(pv);
      sc_add_eg(*psc_dep, pc);

      vect_add_elem(p_DiIncNonCons, (Variable)di, 1);
    }

  }

  /* Update basis */
  if(!BASE_NULLE_P((*psc_dep)->base)) {
    vect_rm((*psc_dep)->base);
    (*psc_dep)->base = BASE_UNDEFINED;
  }
  sc_creer_base(*psc_dep);
}



/*
 * This function implements one of the last steps of the CRI dependence test.
 *
 * System ps has been projected on di variables. ps is examined to find
 * out the set of possible values for di variables, the iteration differences
 * for which a dependence exist.
 *
 * Loops are numbered from 1 to cl, the maximum nesting level.
 * At each step, the di variable corresponding to the current nesting level
 * is examined.
 *
 * Dependences levels added to the graph depend on the sign of the values
 * computed for di. There is a dependence from s1 to s2 if di can be
 * positive and from s2 to s1 if di can be negative.
 *
 * There are no loop-carried dependence if di is equal to zero. The
 * corresponding level is cl+1.
 *
 * Finally, di is set to 0 at the end of the loop since
 * dependences at level l are examined assuming the enclosing loops are
 * at the same iteration.
 *
 * Input:
 *
 *   ps is the projected system (the distance system for the dep: (s1->s2).
 *      a copy of ps, pss is allocated to compute di's value, but ps is
 *      modified until it is empty (all di variables have been projected),
 *      or until it is unfeasible (this is detected on pss)
 *
 *   cl is the common nesting level of statement s1 and s2.
 *
 * Temporaries:
 *
 *   pss is deallocated by sc_minmax_of_variable_optim()
 *
 * Modification :
 *
 * -  variable NotPositive added to take into acounnt the case di<=0.
 *    Yi Qing (10/91)
 */

static list TestDiVariables(Psysteme ps,
                            int cl,
                            statement s1,
                            effect ef1 __attribute__ ((unused)),
                            statement s2,
                            effect ef2 __attribute__ ((unused))) {
  list levels = NIL;
  _int l;
  bool all_level_founds = false;

  pips_debug(7, "maximum common level (cl): %d\n", cl);

  for (l = 1; !all_level_founds && l <= cl; l++) {
    Variable di = (Variable)GetDiVar(l);
    Value min, max;
    int IsPositif, IsNegatif, IsNull, NotPositif;
    /* FI: Keep a consistent interface in memory allocation */
    /* Psysteme pss = (l==cl) ? ps : sc_dup(ps); */
    Psysteme pss = sc_dup(ps);

    ifdebug(7) {
      pips_debug(7, "current level: %td, variable: %s\n",
          l, entity_local_name((entity) di));
    }

    if(sc_minmax_of_variable_optim(pss, di, &min, &max) == false) {
      pips_debug(7,"sc_minmax_of_variable_optim: non feasible system\n");
      all_level_founds = true;
      break;
    }

    IsPositif = value_pos_p(min);
    IsNegatif = value_neg_p(max);
    IsNull = value_zero_p(min) && value_zero_p(max);
    NotPositif = value_zero_p(max) && value_neg_p(min);

    ifdebug(7) {
      pips_debug(7, "values: ");
      fprint_string_Value(stderr, "min = ", min);
      fprint_string_Value(stderr, "  max = ", max);
      fprintf(stderr,
              "  ==> %s\n",
              IsPositif ? "positive" : (IsNegatif ? "negative"
                                                  : (IsNull ? "null"
                                                            : "undefined")));
    }

    if(IsNegatif) {
      all_level_founds = true;
      break;
    }

    if(IsPositif) {
      pips_debug(7, "adding level %td\n", l);
      levels = gen_nconc(levels, CONS(INT, l, NIL));
      all_level_founds = true;
      break;
    }

    if(!IsNull && !NotPositif) {
      pips_debug(7, "adding level %td\n", l);
      levels = gen_nconc(levels, CONS(INT, l, NIL));
    }

    if(!all_level_founds && l <= cl - 1) {
      pips_debug(7, "forcing variable %s to 0 (l < cl)\n",
          entity_local_name((entity) di));
      /* This function does not test feasibility and does not
       * deallocate ps
       */
      sc_force_variable_to_zero(ps, di);
    }
  }

  /* If there is no dependence at a common loop level: since the system
   * is feasible, it can be a dependence at the innermost level (inside the
   * common loop nest).
   *
   * WARNING:
   *
   * If the source and target statements are identical, we do not
   * add the innermost level because the parallelization phase
   * (rice) does not appreciate.  In order to be correct, we should
   * add this level 1) because the statement may be a call to an
   * external routine, in which case we cannot be sure that all the
   * writes are performed before the reads and 2) even in the case
   * of a single assignement, the generated code must preserve the
   * order of the write and read memory operations. BC.
   */
  if(!all_level_founds && s1 != s2 && statement_possible_less_p(s1, s2)) {
    pips_debug(7, "adding innermost level %td\n", l);
    levels = gen_nconc(levels, CONS(INT, l, NIL));
  }

  return (levels);
}

/* Ptsg  dependence_cone_positive(Psysteme dept_syst)
 */
static Ptsg dependence_cone_positive(dep_sc)
  Psysteme dep_sc; {
  Psysteme volatile sc_env = SC_UNDEFINED;
  Ptsg volatile sg_env = NULL;
  Pbase b;
  int n, j;
  int volatile i;

  if(SC_UNDEFINED_P(dep_sc))
    return (sg_new());

  sc_env = sc_empty(base_dup(dep_sc->base));
  n = dep_sc->dimension;
  b = dep_sc->base;

  for (i = 1; i <= n; i++) {
    Psysteme sub_sc = sc_dup(dep_sc);
    Pvecteur v;
    Pcontrainte pc;
    Pbase b1;

    for (j = 1, b1 = b; j <= i - 1; j++, b1 = b1->succ) {
      /* add the contraints  bj = 0 (1<=j<i) */
      v = vect_new(b1->var, 1);
      pc = contrainte_make(v);
      sc_add_eg(sub_sc,pc);
    }
    /* add the contraints - bi <= -1 (1<=j<i) */
    v = vect_new(b1->var, -1);
    vect_add_elem(&v, TCST, 1);
    pc = contrainte_make(v);
    sc_add_ineg(sub_sc,pc);

    ifdebug(7) {
      fprintf(stderr, "\nInitial sub lexico-positive dependence system:\n");
      sc_syst_debug(sub_sc);
    }

    /* dans le cas d'une erreur d'overflow, on fait comme si le test
     *  avait renvoye' true. bc.
     */CATCH(overflow_error) {
      pips_debug(1, "overflow error.\n");
    } TRY
    {
      if(!sc_integer_feasibility_ofl_ctrl(sub_sc, FWD_OFL_CTRL, true)) {
        sc_rm(sub_sc);
        pips_debug(7,"sub lexico-positive dependence system not feasible\n");

        UNCATCH(overflow_error);
        continue;
      } else
        UNCATCH(overflow_error);

    }

    if(sc_empty_p(sub_sc = sc_normalize(sub_sc))) {
      sc_rm(sub_sc); /* to mimic previous behavior of sc_normalize */
      pips_debug(7, "normalized system not feasible\n");

      continue;
    }

    /* We get a normalized sub lexico-positive dependence system */ifdebug(7) {
      fprintf(stderr, "Normalized sub lexico-positive dependence system :\n");
      sc_syst_debug(sub_sc);
    }

    {
      Psysteme old_sc_env = sc_env;
      sc_env = sc_enveloppe_chernikova(sc_env, sub_sc);
      sc_rm(old_sc_env);
    }

    sc_rm(sub_sc);

    ifdebug(7) {
      fprintf(stderr, "Dependence system of the enveloppe of subs "
        "lexico-positive dependence:\n");
      sc_syst_debug(sc_env);

      if(!SC_UNDEFINED_P(sc_env) && !sc_rn_p(sc_env)) {
        sg_env = sc_to_sg_chernikova(sc_env);
        fprintf(stderr, "Enveloppe of the subs lexico-positive "
          "dependence cones:\n");
        if(!SG_UNDEFINED_P(sg_env) && vect_size(sg_env->base) != 0) {
          print_dependence_cone(stderr, sg_env, sg_env->base);
          sg_rm(sg_env);
          sg_env = (Ptsg)NULL;
        }
      }
    }

  }

  if(!SC_UNDEFINED_P(sc_env) && sc_dimension(sc_env) != 0) {

    sg_env = sc_to_sg_chernikova(sc_env);
    sc_rm(sc_env);
  } else {
    sg_env = sg_new();
  }

  return (sg_env);
}

static list loop_variant_list(stat)
  statement stat; {
  list lv = NIL;
  loop l;
  list locals;

  pips_assert("statement stat is a loop", statement_loop_p(stat));
  FOREACH (EFFECT, ef, load_cumulated_rw_effects_list(stat)) {
    entity en = effect_entity(ef);
    if(action_write_p( effect_action( ef )) && !entity_all_locations_p(en)
        && entity_integer_scalar_p(en))
      lv = gen_nconc(lv, CONS(ENTITY, en, NIL));
  }
  l = statement_loop(stat);
  locals = loop_locals(l);
  FOREACH (ENTITY, v, locals) {
    if(gen_find_eq(v, lv) == entity_undefined)
      lv = CONS(ENTITY, v, lv);
  }
  locals = statement_declarations(loop_body(l));
  FOREACH (ENTITY, v, locals) {
    if(gen_find_eq(v, lv) == entity_undefined)
      lv = CONS(ENTITY, v, lv);
  }
  return (lv);
}

/* this function detects intra-statement, non loop carried dependence
 * ( Di=(0,0,...0) and s1 = s2).
 */
static bool TestDiCnst(Psysteme ps,
                       int cl,
                       statement s1,
                       effect ef1 __attribute__ ((unused)),
                       statement s2,
                       effect ef2 __attribute__ ((unused))) {
  int l;

  for (l = 1; l <= cl; l++) {
    Variable di = (Variable)GetDiVar(l);
    Psysteme pss;
    Value val;
    bool success_p = true;

    pss = sc_dup(ps);
    success_p = sc_value_of_variable(pss, di, &val);
    sc_rm(pss);

    if(success_p) {
      if(value_notzero_p(val)) {
        return (false);
      }
    } else {
      return (false);
    }
  }

  /* case of di zero */
  if(s1 == s2) {
    NbrAllEquals++;
    return (true);
  } else
    return (false);
}



void writeresult(char *mod_name) {
  FILE *fp;
  string filename;
  int i, j;

  switch(dg_type) {
    case DG_FAST:
      filename = "resulttestfast";
      break;
    case DG_FULL:
      filename = "resulttestfull";
      break;
    case DG_SEMANTICS:
      filename = "resulttestseman";
      break;
    default:
      pips_internal_error("erroneous dg type.");
      return; /* to avoid warnings from compiler */
  }

  filename = strdup(concatenate(db_get_current_workspace_directory(),
                                "/",
                                mod_name,
                                ".",
                                filename,
                                0));

  fp = safe_fopen(filename, "w");

  fprintf(fp, "%s", mod_name);
  fprintf(fp,
          " %d %d %d %d %d %d %d %d %d %d",
          NbrArrayDepInit,
          NbrIndepFind,
          NbrAllEquals,
          NbrDepCnst,
          NbrTestExact,
          NbrDepInexactEq,
          NbrDepInexactFM,
          NbrDepInexactEFM,
          NbrScalDep,
          NbrIndexDep);
  for (i = 0; i <= 4; i++)
    for (j = 0; j <= 2; j++)
      fprintf(fp, " %d", deptype[i][j]);
  for (i = 0; i <= 4; i++)
    for (j = 0; j <= 2; j++)
      fprintf(fp, " %d", constdep[i][j]);
  fprintf(fp,
          " %d %d %d %d %d %d %d %d %d %d %d",
          NbrTestCnst,
          NbrTestGcd,
          NbrTestSimple,
          NbrTestDiCnst,
          NbrTestProjEqDi,
          NbrTestProjFMDi,
          NbrTestProjEq,
          NbrTestProjFM,
          NbrTestDiVar,
          NbrProjFMTotal,
          NbrFMSystNonAug);
  for (i = 0; i < 18; i++)
    fprintf(fp, " %d", FMComp[i]);
  fprintf(fp, "\n");

  safe_fclose(fp, filename);
  free(filename);
}


// have to be done before call :
// * set_ordering_to_statement
// * set_enclosing_loops_map
// * loading cumulated effects
graph compute_dg_on_statement_from_chains_in_place(statement s, graph chains) {
  dg = chains;
  dg_type = DG_FAST; //FIXME

  debug_on("QUICK_PRIVATIZER_DEBUG_LEVEL");
  quick_privatize_graph(dg);
  debug_off();

  memset(deptype,0,sizeof(deptype));
  memset(constdep,0,sizeof(deptype));

  rdg_statement(s);

  return dg;
}


// have to be done before call :
// * set_ordering_to_statement
// * set_enclosing_loops_map
// * loading cumulated effects
graph compute_dg_on_statement_from_chains(statement s, graph chains) {
  dg = copy_graph(chains);
  return compute_dg_on_statement_from_chains_in_place(s, dg);
}


