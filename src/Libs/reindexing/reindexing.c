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
/************************************************************************/
/* Name     : reindexing.c
 * Package  : reindexing
 * Author   : Alexis Platonoff & Antoine Cloue
 * Date     : april 1994
 * Historic :
 * - 19 jan 95 : remove my_make_rational_exp() and my_vect_var_subst(), AP
 * - 20 apr 95 : modification of get_time_ent(), AP
 * - 21 apr 95 : remove the functions dealing with "cell" in cell.c, AP
 *
 * Documents: SOON
 * Comments : This file contains the functions for the transformation of a
 * program to a single assignment form. The main function is reindexing().
 */

/* Ansi includes 	*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Newgen includes 	*/
#include "genC.h"

/* C3 includes 		*/
#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "ray_dte.h"
#include "sommet.h"
#include "sg.h"
#include "sc.h"
#include "polyedre.h"
#include "union.h"
#include "matrice.h"
#include "matrix.h"
#include "sparse_sc.h"

/* Pips includes 	*/
#include "boolean.h"
#include "ri.h"
#include "constants.h"
#include "control.h"
#include "ri-util.h"
#include "misc.h"
#include "complexity_ri.h"
#include "database.h"
#include "graph.h"
#include "dg.h"
#include "paf_ri.h"
#include "parser_private.h"
#include "property.h"
#include "reduction.h"
#include "text.h"
#include "text-util.h"
#include "tiling.h"
#include "text-util.h"
#include "pipsdbm.h"
#include "resources.h"
#include "static_controlize.h"
#include "paf-util.h"
#include "pips.h"
#include "array_dfg.h"
#include "prgm_mapping.h"
#include "conversion.h"
#include "scheduling.h"
#include "reindexing.h"

/* Macro functions  	*/

#define STRING_BDT "t"
#define STRING_PLC "p"
#define STRING_TAU "q"
#define STRING_FLAG "flag"

#define MAKE_STATEMENT(ins) (make_statement(entity_empty_label(),STATEMENT_NUMBER_UNDEFINED,STATEMENT_ORDERING_UNDEFINED,string_undefined,ins))

/* Internal variables 	*/

/* Global variables */
static int         tc;
hash_table         h_node;
hash_table         delay_table;
hash_table         ht_ab; /* Array bounds */
list               lparams;

/* Local defines */
typedef dfg_vertex_label vertex_label;
typedef dfg_arc_label arc_label;

/* We define a set of constant in order to a more generic function for the
 * insert of the declarations of the new created variables.
 */
#define INTEGER_DEC 0
#define REAL_DEC 1
#define COMPLEX_DEC 2
#define LOGICAL_DEC 3
#define DOUBLE_DEC 4
#define CHARACTER_DEC 5

#define INTEGER_DECL   "      INTEGER"
#define REAL_DECL      "      REAL"
#define COMPLEX_DECL   "      COMPLEX"
#define LOGICAL_DECL   "      LOGICAL"
#define DOUBLE_DECL    "      DOUBLE"
#define CHARACTER_DECL "      CHARACTER"
#define COLON  ":"
#define NEWLINE "\n     *"
#define ENDLINE "\n"
#define LINE_LENGHT 68
#define MARGIN_LENGHT 7 

#define IS_MIN 0
#define IS_MAX 1


#define DOUBLE_PRECISION_SIZE 8


/*===================================================================*/
/* entity get_time_ent(typ, count)
 *
 * Returns the entity that represents the global time variable "t_count"
 * for the program.  If it does not exist yet, we create it.
 * AC 94/05/30
 *
 * AP 95/04/20: entities must have in their full name the current module
 * name instead of RE_MODULE_NAME. */

entity get_time_ent(st, typ, count)
char    *typ;
int     st, count;
{
  char    name[32], *full_name;
  entity  new_coeff;
 
  if (!strncmp(typ, STRING_BDT, 1))
    sprintf(name, "%s%d%s%d", STAT_SYM, st, typ, count);
  else
    sprintf(name, "%s%d%s%d", STAT_SYM, st - BASE_NODE_NUMBER, typ, count);

  /* full_name = strdup(concatenate(RE_MODULE_NAME, MODULE_SEP_STRING,
      name, NULL));
   */
  full_name = strdup(concatenate(strdup(db_get_current_module_name()),
				 MODULE_SEP_STRING, name, NULL));

  new_coeff = gen_find_tabulated(full_name, entity_domain);
 
  /* We create it, if it does not exist yet */
  if (new_coeff == entity_undefined) {
    if (!strncmp(typ, STRING_BDT, 1))
      new_coeff = create_new_entity(st, STRING_BDT, count);
    else
      new_coeff = create_new_entity(st - BASE_NODE_NUMBER,
				    STRING_TAU, count);
  }
  
  return(new_coeff);
}


/*=========================================================================*/
/* void build_contraction_matrices(mH, mQ, mQ_inv, mC, l)
 *
 * From the matrix of Hermite mH, build the matrix mQ containing the
 * periods of the different schedule dimension.
 * 
 * AC 94/03/31 */

void build_contraction_matrices(s, ps, mH, c, mQ, mQ_inv, mC, ln, lom,
				indx, cnst)
int         s, *c;
Pmatrix     mH, *mQ, *mQ_inv, *mC;
list        *ln, *lom;
Pbase       indx, cnst;
Psysteme    ps;
{
  int         i, le = ps->nb_eq;
  Value ppc = VALUE_ONE;
  entity      ent;
  Psysteme    sc = sc_new();
  Pcontrainte cont;

  *ln = NIL;
  *lom = NIL;
  matrix_identity(*mQ, 0);
  matrix_identity(*mQ_inv, 0);
  
  if (get_debug_level() > 5)
    fprintf(stderr,"\nBuild_contraction matrice :");

  for (i = 1; i <= le; i++)
    ppc = ppcm(ppc, value_abs(MATRIX_ELEM(mH, i, i)));

  for (i = 1; i <= MATRIX_NB_LINES(*mQ); i++) {
    if (i <= le) {
	Value p = value_div(ppc,MATRIX_ELEM(mH, i, i)), x;
	MATRIX_ELEM(*mQ,i,i) = value_abs(p);
	ent = get_time_ent(s, STRING_TAU, (*c));
	(*c)++;
	ADD_ELEMENT_TO_LIST(*ln, ENTITY, ent);

	x = value_abs(MATRIX_ELEM(mH,i,i));
	ADD_ELEMENT_TO_LIST(*lom, INT, VALUE_TO_INT(x));
    }
    else 
	value_product(MATRIX_ELEM(*mQ,i,i), ppc);
  }
  
  MATRIX_DENOMINATOR(*mQ) = ppc;

  sc = matrix_to_system(*mQ, indx);

  cont = sc->egalites;

  while (ps->egalites != NULL) {
    cont->vecteur = vect_substract(cont->vecteur, (ps->egalites)->vecteur);
    cont = cont->succ;
    ps->egalites = (ps->egalites)->succ;
    ps->nb_eq--;
  }

  /* build the matrix mC */
  matrix_nulle(*mC);

  my_constraints_with_sym_cst_to_matrices(sc->egalites, indx, cnst, *mQ, *mC);

  MATRIX_DENOMINATOR(*mQ) = ppc;
  MATRIX_DENOMINATOR(*mC) = ppc;

  matrix_general_inversion(*mQ, *mQ_inv);
 
  if (get_debug_level() > 5)
    fprintf(stderr,"\nBuild_contraction_matrice fin\n");
}


/*========================================================================*/
/* build_list_of_min()
 *
 * From the list of psystem "lsys", take each inequality (that is an
 * expression of a possible lower bound), and create as many systems as
 * necessary to raise the undetermination on the domain.  This is done
 * until the counter is equal to 0 meaning that all "t" variables have
 * been treated.
 *
 * AC 94/05/05 */

Psyslist build_list_of_min(lsys, lnew, ps)
Psyslist     lsys;
list         lnew;
Psysteme     ps;
{
  Psysteme     sc_aux, ps_aux2;
  Pcontrainte  cont, cont2;
  Pvecteur     vect, vectp, vect2;
  Value        val, val2;
  Psyslist     lsys_aux = NULL, lsys_aux2 = NULL;
  Variable     var;
  
  if (get_debug_level() > 5)
    fprintf(stderr, "\nBuild list of min begin \n");

  if (lsys != NULL) {
    /* get the system corresponding to the current variable */
    sc_aux = lsys->psys;
    var = (Variable)ENTITY(CAR(lnew));
    
    if (get_debug_level() > 5) {
      fprintf(stderr, "\nSysteme en cours d'etude:");
      fprint_psysteme(stderr, sc_aux);
      fprintf(stderr, "\nVariable relative : ");
      fprint_entity_list(stderr,CONS(ENTITY, (entity)var, NIL));
      fprintf(stderr,"\nSysteme en p a ajouter: ");
      fprint_psysteme(stderr,ps);
    }

    if (sc_aux->nb_ineq == 1) {
      /* there is already an unique minimum */       
      vectp = vect_del_var(vect_dup((sc_aux->inegalites)->vecteur), var); 

      /* add inequalities on the max */
      cont = sc_aux->egalites;
      while (sc_aux->egalites != NULL) {
	cont = sc_aux->egalites;
	sc_aux->egalites = (sc_aux->egalites)->succ;
	cont->succ = NULL;
	sc_add_inegalite(sc_aux, cont);
      }
      sc_aux = sc_append(sc_aux, ps);

      /* put the value of the min in the equality */
      sc_add_egalite_at_end(sc_aux, contrainte_make(vectp));
      sc_aux = my_clean_ps(sc_aux);
      
      /* search on the next t variable */
      lsys_aux2 = build_list_of_min(lsys->succ, lnew->cdr, sc_aux);

      if (lsys_aux2 != NULL)
	lsys_aux = add_sclist_to_sclist(lsys_aux, lsys_aux2);
      else
	lsys_aux = add_sc_to_sclist(sc_aux, lsys_aux);
    }
    else {
      for (cont = sc_aux->inegalites; cont != NULL; cont = cont->succ) {
	/* for each possible minimum build new system */
	ps_aux2 = sc_dup(sc_aux);
	vect = vect_dup(cont->vecteur);
	val = vect_coeff(var, vect); 
	vectp = vect_del_var(vect_dup(cont->vecteur), var); 
	
	/* include new value of the min in the system */
	for (cont2 = ps_aux2->inegalites; cont2 != NULL; cont2 = cont2->succ) {
	  vect2 = cont2->vecteur;
	  val2 = vect_coeff(var, vect2);
	  vect2 = vect_del_var(vect2, var);
	  vect2 = vect_multiply(vect2, value_abs(val));
	  vect2 = vect_add(vect_multiply(vect_dup(vectp), val2), vect2);
	  cont2->vecteur = vect2;
	}      
	/* put the inequality on the current minimum */ 
	sc_add_inegalite(ps_aux2, contrainte_make(vect));

	/* add inequalities on the max */
	cont2 = ps_aux2->egalites;
	while (ps_aux2->egalites != NULL) {
	  cont2 = ps_aux2->egalites;
	  cont2->succ = NULL;
	  sc_add_inegalite(ps_aux2, cont2);
	  ps_aux2->egalites = (ps_aux2->egalites)->succ;
	}

	ps_aux2 = sc_append(ps_aux2, sc_dup(ps));

	/* put the minimum value that is the l in "t = omega*to + l" */
	/* in the equality part of the system. Problem if val != 1   */
	sc_add_egalite_at_end(ps_aux2, contrainte_make(vectp));

	sc_normalize(ps_aux2);

	/* get the next dimension */
	lsys_aux2 = build_list_of_min(lsys->succ, lnew->cdr, ps_aux2);

	if (lsys_aux2 != NULL)
	  lsys_aux = add_sclist_to_sclist(lsys_aux, lsys_aux2);
	else
	  lsys_aux = add_sc_to_sclist(ps_aux2, lsys_aux);
      }
    }
  }
  
  if (get_debug_level() > 5) {
    fprintf(stderr, "\n Liste de systeme build part min : ");
    sl_fprint(stderr, lsys, entity_local_name);
  }
  
  return(lsys_aux);
}


/*========================================================================*/
/* list dataflows_on_reference(cls, crhs, pred, lm)
 *
 * Selects in the list of successors cls the one that correspond to the
 * domain pred on which we are working and for the current rhs. The list
 * lm will contain the corresponding list of statement of the choosen
 * dataflow.
 *
 * AC 94/04/06 */

list dataflows_on_reference(cls, crhs, pred, lm)
list      cls, *lm;
reference crhs;
predicate pred;
{
  Psysteme  psd = sc_dup(predicate_to_system(pred)), sys_pred;
  list      ldata, ldat, lpred, ld;
  int       stat_pred;
  vertex    vert_pred;

  ldat = NIL;
  *lm = NIL;
  ldata = NIL; ld = NIL;

  if (get_debug_level() > 6) {
    fprintf(stderr, "\nDataflow on reference %s debut:",
	    entity_local_name(reference_variable(crhs)));
    fprintf(stderr, "\nDomain noeud:");
    fprint_psysteme(stderr, psd);
  }

  for (lpred = cls; lpred != NIL; POP(lpred)) {
    successor suc = SUCCESSOR(CAR(lpred));
    vert_pred = successor_vertex(suc);
    stat_pred = vertex_int_stmt(vert_pred);
    
    if (get_debug_level() > 6)
      fprintf(stderr,"\nSuccesseur en cours : %d\n",stat_pred);

    ldata = dfg_arc_label_dataflows(successor_arc_label(suc));

    for (ld = ldata; ld != NIL; POP(ld)) {
      dataflow d = DATAFLOW(CAR(ld));
      sys_pred = sc_dup(predicate_to_system(dataflow_governing_pred(d)));
      if (get_debug_level() > 6) {
	fprintf(stderr, "\nDomain arc:");
	fprint_psysteme(stderr, sys_pred);
      }
      sys_pred = sc_append(sys_pred, psd);
      sc_normalize(sys_pred);

      if (reference_equal_p(crhs, dataflow_reference(d))) {
	if (SC_RN_P(sys_pred) ||
	    sc_rational_feasibility_ofl_ctrl(sys_pred, NO_OFL_CTRL, true)) {
	  if (get_debug_level() > 6) {
	    fprintf(stderr, "\nDomain inter donc faisabilite:");
	    fprint_psysteme(stderr, sys_pred);
	  }
	  ADD_ELEMENT_TO_LIST(ldat, DATAFLOW, d);
	  ADD_ELEMENT_TO_LIST(*lm, INT, stat_pred);
	}
      }
    }
  }

  if (get_debug_level() > 6)
    fprintf(stderr, "\nDataflow on ref fin\n");

  return(ldat); 
}


/*=======================================================================*/
/* instruction build_local_time_test(t, l)
 *
 * Builds the insruction corresponding to:
 * IF t_local > ub THEN t_local = -1 ENDIF
 *
 * t = entity of the local time;
 * l = list of the ub (expression);
 * 
 * AC 94/06/14 */

instruction build_local_time_test(t, l)
entity      t;
list        l;
{
  expression  upper = EXPRESSION(CAR(l)), exp;
  instruction ins;
  call        ca;
  list        lexp;
  statement   stat;

  /* build the expression t-local = -1 */
  lexp = CONS(EXPRESSION, make_entity_expression(t, NIL), NIL);
  ADD_ELEMENT_TO_LIST(lexp, EXPRESSION, int_to_expression(-1));
  ca = make_call(entity_intrinsic(ASSIGN_OPERATOR_NAME), lexp);
  ins = make_instruction(is_instruction_call, ca);
  stat = MAKE_STATEMENT(ins);

  /* make the test expression */
  lexp = CONS(EXPRESSION, make_entity_expression(t, NIL), NIL);
  ADD_ELEMENT_TO_LIST(lexp, EXPRESSION, upper);
  ca = make_call(entity_intrinsic(GREATER_THAN_OPERATOR_NAME), lexp);
  exp = make_expression(make_syntax(is_syntax_call, ca), normalized_undefined);
  
  ins = make_instruction(is_instruction_test, 
			 make_test(exp, stat, make_empty_statement()));
  
  return(ins);
}

/*=======================================================================*/
/* expression build_global_time_test_with_exp(tg, exp)
 *
 * Builds the expression:
 *            tg == exp
 * 
 * AC 94/06/15
 */

expression build_global_time_test_with_exp(tg, exp)
entity      tg;
expression  exp;
{
  list        lexp;
  call        ca;

  lexp = CONS(EXPRESSION, make_entity_expression(tg, NIL), NIL);
  ADD_ELEMENT_TO_LIST(lexp, EXPRESSION, exp);
  ca = make_call(entity_intrinsic(EQUAL_OPERATOR_NAME), lexp);
  
  return(make_expression(make_syntax(is_syntax_call, ca),
			 normalized_undefined));
}

/*=======================================================================*/
/* instruction build_flag_assign(f, val) :
 *
 * build the instruction: flag = val.
 *
 * AC 94/06/14 */

instruction build_flag_assign(f, val)
entity   f;
bool val;
{
  call     ca;
  list     lexp;
  expression e;

  if(val)
    e = int_to_expression(1);
  else
    e = int_to_expression(0);

  lexp = CONS(EXPRESSION, make_entity_expression(f, NIL), NIL);
  ADD_ELEMENT_TO_LIST(lexp, EXPRESSION, e);
  ca = make_call(entity_intrinsic(ASSIGN_OPERATOR_NAME), lexp);
  
  return(make_instruction(is_instruction_call, ca));
}


/*=======================================================================*/
/* instruction build_flag_test(f, t)
 *
 * Builds the instruction: 
 *          IF flag == 1 THEN  t = t + 1 ENDIF
 *
 * AC 94/06/14
 */

instruction build_flag_test(f, t)
 entity       f, t;
{
 call         ca;
 list         lexp;
 instruction  ins;
 statement    stat;
 expression   exp;

 ins = make_increment_instruction(t, 1);
 stat = MAKE_STATEMENT(ins);

 lexp = CONS(EXPRESSION, make_entity_expression(f, NIL), NIL);
 ADD_ELEMENT_TO_LIST(lexp, EXPRESSION, int_to_expression(1));
 ca = make_call(entity_intrinsic(EQUAL_OPERATOR_NAME), lexp);
 exp = make_expression(make_syntax(is_syntax_call, ca), normalized_undefined);
 
 return(make_instruction(is_instruction_test, 
			 make_test(exp, stat, make_empty_statement())));
}

/*=======================================================================*/
/* instruction make_init_time(en, ex) : build the instruction: en = ex
 *
 * AC 94/06/15
 */

instruction make_init_time(en, ex)
 entity      en;
 expression  ex;
{
 call        ca;
 list        lexp = NIL;

 lexp = CONS(EXPRESSION, make_entity_expression(en, NIL), NIL);
 ADD_ELEMENT_TO_LIST(lexp, EXPRESSION, ex);
 ca = make_call(entity_intrinsic(ASSIGN_OPERATOR_NAME), lexp);

 return(make_instruction(is_instruction_call, ca));
}


/*======================================================================*/
/* instruction make_increment_instruction(t, i): returns the instruction:
 *    t = t + i
 *
 * AC 94/06/09
 */

instruction make_increment_instruction(t, i)
 entity      t;
 int         i;
{
 call        ca;
 expression  exp1;
 list        lexp;

 lexp = CONS(EXPRESSION, make_entity_expression(t, NIL), NIL);
 ADD_ELEMENT_TO_LIST(lexp, EXPRESSION, int_to_expression(i));
 ca = make_call(entity_intrinsic(PLUS_OPERATOR_NAME), lexp);
 exp1 = make_expression(make_syntax(is_syntax_call, ca), 
			normalized_undefined);

 lexp = CONS(EXPRESSION, make_entity_expression(t, NIL), NIL);
 ADD_ELEMENT_TO_LIST(lexp, EXPRESSION, exp1);
 ca = make_call(entity_intrinsic(ASSIGN_OPERATOR_NAME), lexp);

 return(make_instruction(is_instruction_call, ca));
}


/*=======================================================================*/
/* int get_number_of_ins(e): gives back the number of the node.
 * 
 * AC 94/07/28
 */

int get_number_of_ins(e)
entity    e;
{
  int       n = 0;
  char      *c;

  c = (char*) malloc(32);

  c = strcpy(c, (entity_local_name(e)+4));
  n = atoi(c) + BASE_NODE_NUMBER;

  if (get_debug_level() > 5)
    fprintf(stderr,"\nNumero de l'instruction de %s = %d",
	    entity_local_name(e), n);
  
  free(c);

  return(n);
}


/*=======================================================================*/
/* void reindexing((char*) mod_name):
 * 
 */

bool reindexing(mod_name)
char*           mod_name;
{
  extern int tc;

  graph           the_dfg;
  bdt             the_bdt;
  plc             the_plc;
  entity          ent;
  static_control  stco;
  statement       mod_stat, new_mod_stat;
  int nb_nodes;
  list l;
  statement_mapping STS;

  /* Initialize debugging functions */
  debug_on("REINDEXING_DEBUG_LEVEL");
  if (get_debug_level() > 0)
    user_log("\n\n *** COMPUTE REINDEXING for %s\n", mod_name);

  /* We get the required data: module entity, code, static_control, */
  /* dataflow graph.  */
  ent = local_name_to_top_level_entity( mod_name );

  if (ent != get_current_module_entity())
  {
    reset_current_module_entity();
    set_current_module_entity(ent);
  }

  /* mod_stat = copy_statement((statement) db_get_memory_resource(DBR_CODE, mod_name,
     true));*/
  mod_stat = (statement) db_get_memory_resource(DBR_CODE, mod_name, false);
  STS = (statement_mapping)db_get_memory_resource(DBR_STATIC_CONTROL,
					   mod_name, false);
  set_current_stco_map(STS);
  stco = get_stco_from_current_map(mod_stat);

  lparams = static_control_params(stco);

  if (stco == static_control_undefined) 
    pips_internal_error("This is an undefined static control !");
  
  if (!static_control_yes(stco)) 
    pips_internal_error("This is not a static control program !");
  

  /* The DFG, the BDT and the PLC */
  the_dfg = (graph)db_get_memory_resource(DBR_ADFG, mod_name, true);
  the_bdt = (bdt)db_get_memory_resource(DBR_BDT, mod_name, true);
  the_plc = (plc)db_get_memory_resource(DBR_PLC, mod_name, true);

  if (get_debug_level() > 0)  
  {
    fprint_dfg(stderr, the_dfg);
    fprint_bdt(stderr, the_bdt);
    fprint_plc(stderr, the_plc);
  }

  /* First we count the number of nodes to initialize the hash tables */
  nb_nodes = 0;
  for(l = graph_vertices(the_dfg); !ENDP(l); POP(l))
    nb_nodes++;
  h_node = hash_table_make(hash_int, nb_nodes+1);
  delay_table = hash_table_make(hash_int, nb_nodes+1);
  ht_ab = hash_table_make(hash_int, nb_nodes+1);

  /* The temporary variables counter */
  tc = 0;

  new_mod_stat = re_do_it(the_dfg, the_bdt, the_plc);

  /* Remove the old code: */
  free_instruction(statement_instruction(mod_stat));
  /* And replace it by the new one: */
  statement_instruction(mod_stat) = make_instruction_block(CONS(STATEMENT,
								new_mod_stat,
								NIL));
  DB_PUT_MEMORY_RESOURCE(DBR_REINDEXED_CODE, strdup(mod_name),
			 (char*) mod_stat);
  reset_current_module_statement(); 
  set_current_module_statement((statement) 
			       db_get_memory_resource(DBR_CODE, 
						      mod_name, true) ); 
  mod_stat = get_current_module_statement(); 

  DB_PUT_MEMORY_RESOURCE(DBR_CODE, strdup(mod_name), (char*) mod_stat); 
  reset_current_module_statement(); 

  /* print the values of the delay */
  if (get_debug_level() > 0)
    fprint_delay(stderr, the_dfg, delay_table);

  if(get_debug_level() > 0)
    user_log("\n\n *** REINDEXING done\n");

  hash_table_free(h_node);
  hash_table_free(delay_table);

  reset_current_stco_map();
  reset_current_module_entity();

  debug_off();

  return(true);
}
