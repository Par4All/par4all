/*

  $Id$

  Copyright 1989-2014 MINES ParisTech

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
/* Name     : cell.c
 * Package  : reindexing
 * Author   : Alexis Platonoff & Antoine Cloue
 * Date     : april 1995
 * Historic :
 *
 * Documents: SOON
 * Comments : This file contains the functions manipulating the cells.
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
#include "matrice.h"
#include "union.h"
#include "matrix.h"
#include "sparse_sc.h"

/* Pips includes 	*/
#include "boolean.h"
#include "ri.h"
#include "constants.h"
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
#include "pip.h"
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

#define MAKE_STATEMENT(ins) \
          (make_statement(entity_empty_label(), \
			  STATEMENT_NUMBER_UNDEFINED, \
			  STATEMENT_ORDERING_UNDEFINED, \
			  string_undefined, ins))

/* Internal variables 	*/

/* Local defines */
typedef dfg_vertex_label vertex_label;
typedef dfg_arc_label arc_label;

/* type cell that contains all information for the reindexation of */
/* an instruction. Filled during function prepare_reindexing().    */
typedef struct scell {
        int          statement; /* number of the node         */
        predicate    domain;    /* complete domain of the bdt */
        predicate    edge_dom;  /* predicate of the schedule  */

	Pbase        var_base;  /* (to) = Rmat(i) + Smat(n)   */
        Pbase        con_base;
	Pbase        Rbase_out;
	Pbase        Tbase_out;

	Pmatrix      Tmat;      /* (t) = Tmat(i) + Bmat(n)         */
	Pmatrix      Tmat_inv;  /* (i) = Tmat_inv(t) + Bmat_inv(n) */
	Pmatrix      Bmat;
	Pmatrix      Bmat_inv;

	Pmatrix      Rmat;      /* (to) = Rmat (i) + Smat           */
	Pmatrix      Rmat_inv;  /* (i) = Rmat_inv(to) + Smat_inv(n) */
	Pmatrix      Smat;
	Pmatrix      Smat_inv;

	list         lomega;    /* list of the period(s) of bdt */
	list         lp;        /* list of variable p created   */
	list         lt;        /* list of variable t created   */
	list         ltau;      /* list of variable tau created */

        Psysteme     Nbounds;    /* new systeme of bounds in (t,p,..)    */
        Pbase        Nindices;   /* list of the new indices              */
	Psyslist     t_bounds;   /* bounds of global time for the ins    */
	Psyslist     p_topology; /* topology of the plc in function of t */

        struct scell  *succ;
                    } scell, *Pscell;

/* type of a test, the same as the "normal" test except that the */
/* condition field is not an expression but a Psysteme.          */
typedef struct mytest {
        Psysteme       test;
        instruction    true;
        instruction    false;

        struct mytest  *succ;
                      } mytest, *Pmytest;

/* type of new instruction which contains information about the loop */
/* bounds we will construct around it - used in build_first_comb().  */
typedef struct newinst {
        instruction    ins;

	/* information about the local time */
	list           lomega;
	list           lb;
	list           ub;

	struct newinst *succ;
                       } newinst, *Pnewinst;

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


/*======================================================================*/
/*
 *
 * AC 94/08/05
 */

static Pbase include_time_in_base(pc, e)
Pscell    pc;
entity   e;
{
  list     l;

  l = gen_copy_seq(base_to_list(pc->Tbase_out));
  if(l == NIL) {
    l = CONS(ENTITY, e, l);
  }
  else {
    l = CDR(l);
    l = CONS(ENTITY, e, l);
  }
  
  return(list_to_base(l));
}


/*========================================================================*/
/* Pmytest create_mytest(ps, ins1, ins2): create a Pmytest.
 *
 * AC 94/06/08
 */

static Pmytest create_mytest(ps, ins1, ins2)

 Psysteme     ps;
 instruction  ins1, ins2;
{
 Pmytest      te = (Pmytest)malloc(sizeof(mytest));

 te->test = ps;
 te->true = ins1;
 te->false = ins2;
 te->succ = NULL;

 return(te);
}


/*======================================================================*/
/* Pmytest add_elt_to_test_list(te, lte): add a Pmytest element at the
 * end of a Pmytest list lte.
 *
 * AC 94/06/08
 */

static Pmytest add_elt_to_test_list(te, lte)

 Pmytest  te, lte;
{
 Pmytest  aux, aux2;

 if (lte == NULL) return(te);
 else
   {
    aux = lte;
    while (aux != NULL)
      {
       aux2 = aux;
       aux = aux->succ;
      }
    aux2->succ = te;

    return(lte);
   }
}

/*========================================================================*/
/* void fprint_mytest(fp, t): print a list of Pmytest.
 *
 * AC 94/06/08
 */

static void fprint_mytest(fp, t)

 FILE     *fp;
 Pmytest  t;
{
 Pmytest  l;
 int      i = 1;

 for (l = t; l!= NULL; l = l->succ)
   {
    fprintf(fp,"\nTest n. %d:", i);
    fprint_psysteme(fp, l->test);
    sa_print_ins(fp, l->true);
    i++;
   }
}

/*========================================================================*/
/* Pmytest add_ltest_to_ltest(l1, l2): add the Pmytest list l2 at the end
 * of the Pmytest list l1.
 *
 * AC 94/06/08
 */

static Pmytest add_ltest_to_ltest(l1, l2)

 Pmytest  l1, l2;
{
 Pmytest  l3, l4;

 if (l1 == NULL) return(l2);
 if (l2 == NULL) return(l1);

 l3 = l1;

 while (l3 != NULL)
   {
    l4 = l3;
    l3 = l3->succ;
   }
 l4->succ = l2;

 return(l1);
}


/*=========================================================================*/
/* void calculate_delay(exp, pcs, pcd)
 *
 * Calculates the delay if possible for the instruction of Pscell pcs that
 * is the instruction source.
 *
 * AC 94/06/29 */

static void calculate_delay(exp, pcs, pcd, tau)
expression  exp;
Pscell       pcd, pcs;
entity      tau;
{
  Psysteme    ps;
  Pcontrainte con;
  int         os = 0, od = 0,di;
  Value del, d, lbs = VALUE_ZERO, lbd = VALUE_ZERO;
  Pvecteur    vect, vect2;
  bool with_t;

  if (get_debug_level() > 5)
    fprintf(stderr,"\nDebut calculate delai pour %d", pcs->statement);

  with_t = (tau != entity_undefined);

  di = (int)hash_get(delay_table, (char *)pcs->statement);
  d = int_to_value(di);

  if (di != INFINITY) {
    if (get_debug_level() > 5) {
      fprintf(stderr,"\nd != infinity pour noeud %d", pcs->statement);
    }

    if (pcs->lomega == NIL)	/* A cst Schedule for the source */
      del = INFINITY;
    else {
      os = INT(CAR(pcs->lomega));
      if (pcd->lomega == NIL)	/* A cst schedule for the destination */
	del = INFINITY;
      else {
	od = INT(CAR(pcd->lomega));

	if (get_debug_level() > 5) {
	  fprintf(stderr,"\n os = %d", os);
	  fprintf(stderr,"\n od = %d", od);
	  fprintf(stderr,"\nExpression :");
	  fprint_list_of_exp(stderr, CONS(EXPRESSION, exp, NIL));
	}

	if (os != od)	 /* Diff period for source and dest */
	  del = INFINITY;
	else {
	  /* first see if the expression has the */ 
	  /* proper form that is : tau - d       */
	  Variable Tau;

	  if(with_t)
	    Tau = (Variable) tau;
	  else
	    Tau = TCST;

	  vect = normalized_linear(expression_normalized(exp));
	  if (get_debug_level() > 5) {
	    fprintf(stderr,"\nVecteur :");
	    pu_vect_fprint(stderr, vect);
	    fprintf(stderr,"\nEntite Tau : %s", pu_variable_name(Tau));
	    fprint_string_Value(stderr, "\nCoeff = ", vect_coeff(Tau, vect));
	    fprintf(stderr, "\n");
	  }

	  if (value_notone_p(vect_coeff(Tau, vect)) && with_t)
	    del = INFINITY;
	  else {
	    /* vect2 is equal to +d in the formula */
	    if(with_t)
	      vect2 = vect_substract(vect_new(Tau, 1), vect);
	    else {
	      vect2 = vect_dup(vect);
	      vect_chg_sgn(vect2);
	    }

	    if (get_debug_level() > 5) {
	      fprintf(stderr,"\nVecteur 2 :");
	      pu_vect_fprint(stderr, vect2);
	      fprintf(stderr,"\nIndices of Dest :");
	      pu_vect_fprint(stderr, pcd->Nindices);
	    }

	    if (!cst_vector_p(vect2, pcd->Nindices))
	      del = INFINITY;
	    else {
	      /* the delay is calculated by the formula: */
	      /*    delay = abs((lbd-lbs)/omega + d  )   */

	      Pvecteur pd;
	      Value vp;

	      ps = (pcs->t_bounds)->psys;
	      con = ps->egalites;
	      if (!CONTRAINTE_UNDEFINED_P(con)) 
		lbs = value_uminus(vect_coeff(TCST, con->vecteur));
	      pd = vect_dup(con->vecteur);
	      vect_chg_sgn(pd);

	      if (get_debug_level() > 5) {
		fprintf(stderr, "\nLBS :");
		pu_vect_fprint(stderr, con->vecteur);
		fprint_string_Value(stderr,"\nlbs = ", lbs);
	      }
	  
	      ps = (pcd->t_bounds)->psys;
	      con = ps->egalites;
	      if (!CONTRAINTE_UNDEFINED_P(con)) 
		lbd = value_uminus(vect_coeff(TCST, con->vecteur));
	      pd = vect_cl_ofl_ctrl(pd, (Value)  1,
				    vect_dup(con->vecteur), NO_OFL_CTRL);

	      if (get_debug_level() > 5) {
		fprintf(stderr, "\nLBD :");
		pu_vect_fprint(stderr, con->vecteur);
		fprint_string_Value(stderr,"\nlbd = ", lbd);
		fprintf(stderr, "\nValue of PD :");
		pu_vect_fprint(stderr, pd);
	      }

	      if(is_vect_constant_p(pd))
		vp = VALUE_ZERO;
	      else
		  vp = value_mod(vect_pgcd_all(pd),int_to_value(os));
	      if(value_zero_p(vp)) {
		pd = vect_div(pd, int_to_value(os));
		pd = vect_cl_ofl_ctrl(pd, VALUE_ONE,
				      vect2 , NO_OFL_CTRL);
		
		if (get_debug_level() > 5) {
		  fprintf(stderr, "\nValue of the delay :");
		  pu_vect_fprint(stderr, pd);
		}

		if(is_vect_constant_p(pd)) {
		    del = value_abs(vect_coeff(TCST, pd));
		}
		else
		  del = INFINITY;
	      }
	      else
		del = INFINITY;
	    }
	  }
	}
      }
    } 
    if(value_gt(del,d)) {
	(void) hash_del(delay_table, (char *)pcs->statement);
	hash_put(delay_table, (char *)pcs->statement,
		 (char *)VALUE_TO_INT(del));

      if (get_debug_level() > 5)
	  fprint_string_Value(stderr,"\nDelai = ", del);
    }
  }
  if (get_debug_level() > 5)
    fprintf(stderr,"\nFin calculate delai");
}


/*====================================================================*/
/* static void prepare_array_bounds(Pscell pc):
 *
 * Compute for the instruction corresponding to the scell "pc" the size
 * of all the dimensions of the array defined by this instruction.
 *
 * These sizes are memorized in the hash table "ht_ab".
 *
 * Parameters :
 *             pc: scell of the current instruction
 *
 * Note: On the time dimensions, these sizes are not the true ones
 * because they correspond to the local time bounds, not the minor
 * time bounds. The computation of the minor bounds is done afterwards
 * (see make_array_bounds())
 *
 * AP 22/08/94 */
static void prepare_array_bounds(pc)
Pscell pc;
{
  extern hash_table ht_ab;

  Pscell pc_aux;
  int sn;
  list llp, llt;
  expression lower, upper;
  Psyslist sl_tbou, sl_ptopo, llp1, llp2, llp3;
  list lcr_range = NIL, ab_val, lcr_ab;
  range cr;

  /* statement number ("ht_ab" key) */
  sn = pc->statement;

  /* We get the current value of the bounds: should not exit yet */
  if((ab_val = (list) hash_get(ht_ab, (char *) sn))
     == (list) HASH_UNDEFINED_VALUE)
    lcr_range = NIL;
  else
    user_error("prepare_array_bounds", "\n Bounds already exits\n");

  for(pc_aux = pc; pc_aux != NULL; pc_aux = pc_aux->succ) {
    lcr_ab = lcr_range;

    if(pc_aux->statement != sn)
      user_error("prepare_array_bounds", "\n Not the same stmt\n");

    /* the P topology is a list of system giving the bounds of the */
    /* space loop counters ; the order is from the inner loop to the */
    /* outer loop. We reverse the order. */
    llp1 = pc_aux->p_topology;
    llp2 = NULL;
    while (llp1 != NULL){   
      llp3 = sl_new();
      llp3->psys = llp1->psys;
      llp3->succ = llp2;
      llp2 = llp3;
      llp1 = llp1->succ;
    }
    sl_ptopo = llp2;

    /* Idem for the T bounds */
    llp1 = pc_aux->t_bounds;
    llp2 = NULL;
    while (llp1 != NULL){   
      llp3 = sl_new();
      llp3->psys = llp1->psys;
      llp3->succ = llp2;
      llp2 = llp3;
      llp1 = llp1->succ;
    }
    sl_tbou = llp2;

    /* Two cases depending on whether there is a non null scheduling or
     * not. */
    if(sl_tbou->psys->nb_ineq == 0) {
      for(llp = pc_aux->lp; !ENDP(llp); POP(llp)) {
	entity cind = ENTITY(CAR(llp));
	Psysteme ps = sl_ptopo->psys;

	cr = make_bounds(ps, cind, IS_ARRAY_BOUNDS,
			 gen_copy_seq(pc_aux->lp), lcr_range);
	lower = range_lower(cr);
	upper = range_upper(cr); 

	if (get_debug_level() > 6) {
	  fprintf(stderr,
		  "\nNew lb and ub expressions (only Space) :\n\tLB: %s\n\tUB: %s\n",
		  words_to_string(words_expression(lower)),
		  words_to_string(words_expression(upper)));
	}

	if(lcr_ab == NIL) {
	  cr = make_range(lower, upper,
			  int_to_expression(1));
	  lcr_ab = CONS(RANGE, cr, NIL);
	  lcr_range = gen_nconc(lcr_range, lcr_ab);
	}
	else {
	  cr = RANGE(CAR(lcr_ab));
	  range_lower(cr) = merge_expressions(range_lower(cr),
					      lower, IS_MIN);
	  range_upper(cr) = merge_expressions(range_upper(cr),
					      upper, IS_MAX);
	}
	lcr_ab = CDR(lcr_ab);

	sl_ptopo = sl_ptopo->succ;
      }
    }
    else {
      /* First the time bounds */
      for(llt = pc_aux->lt; !ENDP(llt); POP(llt)) {
	entity cind = ENTITY(CAR(llt));
	Psysteme ps = sl_tbou->psys;
      
	cr = make_bounds(ps, cind, IS_ARRAY_BOUNDS,
			 gen_copy_seq(pc_aux->lt), lcr_range); 
	lower = range_lower(cr);
	upper = range_upper(cr); 

	if (get_debug_level() > 6) {
	  fprintf(stderr,
		  "\nNew lb and ub expressions (time) :\n\tLB: %s\n\tUB: %s\n",
		  words_to_string(words_expression(lower)),
		  words_to_string(words_expression(upper)));
	}

	if(lcr_ab == NIL) {
	  cr = make_range(lower, upper,
			   int_to_expression(1));
	  lcr_ab = CONS(RANGE, cr, NIL);
	  lcr_range = gen_nconc(lcr_range, lcr_ab);
	}
	else {
	  cr = RANGE(CAR(lcr_ab));
	  range_lower(cr) = merge_expressions(range_lower(cr),
					      lower, IS_MIN);
	  range_upper(cr) = merge_expressions(range_upper(cr),
					      upper, IS_MAX);
	}
	lcr_ab = CDR(lcr_ab);

	sl_tbou = sl_tbou->succ;
      }
      /* Then the space bounds */
      for(llp = pc_aux->lp; !ENDP(llp); POP(llp)) {
	entity cind = ENTITY(CAR(llp));
	Psysteme ps = sl_ptopo->psys;
      
	cr = make_bounds(ps, cind,
			 IS_ARRAY_BOUNDS,
			 gen_nconc(gen_copy_seq(pc_aux->lt),
				   gen_copy_seq(pc_aux->lp)) , lcr_range);
	lower = range_lower(cr);
	upper = range_upper(cr); 

	if (get_debug_level() > 6) {
	  fprintf(stderr,
		  "\nNew lb and ub expressions (space) :\n\tLB: %s\n\tUB: %s\n",
		  words_to_string(words_expression(lower)),
		  words_to_string(words_expression(upper)));
	}

	if(lcr_ab == NIL) {
	  cr = make_range(lower, upper,
			   int_to_expression(1));
	  lcr_ab = CONS(RANGE, cr, NIL);
	  lcr_range = gen_nconc(lcr_range, lcr_ab);
	}
	else {
	  cr = RANGE(CAR(lcr_ab));
	  range_lower(cr) = merge_expressions(range_lower(cr),
					      lower, IS_MIN);
	  range_upper(cr) = merge_expressions(range_upper(cr),
					      upper, IS_MAX);
	}
	lcr_ab = CDR(lcr_ab);

	sl_ptopo = sl_ptopo->succ;
      }
    }
  }

  hash_put(ht_ab, (char *) sn, (char *) lcr_range);

  if (get_debug_level() > 6) {
    fprintf(stderr, "\n====================\n====================\n");
  }
}

#define DOUBLE_PRECISION_SIZE 8

/*====================================================================*/
/* void make_array_bounds(vertex cv):
 *
 * Do the declaration of the array defined by the instruction
 * corresponding to "cv". The bounds of this array are given by a hash
 * table ("ht_ab"), see prepare_array_bounds().
 *
 * The bounds of the first time dimension must be computed using the
 * delay.  If there is no delay (d < 0 or d = infinity), then these bounds
 * are expressed from the minor time point of view, so the values given by
 * the hash table must be translate from the local time to the minor time.
 *
 * With the following notation: local lower bound = llb, local upper bound
 * = lub, local period = lp, minor lower bound = mlb, minor upper bound =
 * mub, minor period = mp
 *
 * We have: mlb = 0
 *          mub = (lub - llb)/lp
 *          mp  = 1
 *
 * AP 22/08/94 */
void make_array_bounds(cv)
     vertex cv;
{
  extern hash_table ht_ab, h_node, delay_table;

  range cr;
  int d, cn;
  entity mod_entity, var_ent;
  string name;
  list lcr, lt, la, lom;
  basic ba;
  int type_decl;
  string num;
  Pscell pc;
  bool is_first;

  cn = dfg_vertex_label_statement(vertex_vertex_label(cv));
  if (cn == ENTRY_ORDER)
    return;
  if (cn == EXIT_ORDER)
    return;

  pc = (Pscell)hash_get(h_node, (char *)cn);
  lcr = (list) hash_get(ht_ab, (char *) cn);
  d = (int)hash_get(delay_table, (char *) cn);

  mod_entity = get_current_module_entity();

  /* Find the array entity */
  num = (string) malloc(32);
  (void) sprintf(num, "%d", cn-BASE_NODE_NUMBER);
  name = strdup(concatenate(SA_MODULE_NAME, MODULE_SEP_STRING,
			    SAI, num, (char *) NULL));
  var_ent = gen_find_tabulated(name, entity_domain);
  if(var_ent == entity_undefined)
    user_error("make_array_bounds",
	       "\nOne ins (%d) has no array entity : %s\n",
	       cn-BASE_NODE_NUMBER, name);

  /* Find the type of this array */
  ba = variable_basic(type_variable(entity_type(var_ent)));
  switch(basic_tag(ba))
    {
    case is_basic_int: { type_decl = INTEGER_DEC; break; }
    case is_basic_complex: { type_decl = COMPLEX_DEC; break; }
    case is_basic_logical: { type_decl = LOGICAL_DEC; break; }
    case is_basic_string: { type_decl = CHARACTER_DEC; break; }
    case is_basic_float: {
      if(basic_float(ba) == DOUBLE_PRECISION_SIZE)
	type_decl = REAL_DEC;
      else
	type_decl = DOUBLE_DEC;
      break;
    }
    default: user_error("make_array_bounds", "\nBad array type\n");
    }
  
  /* No bounds => scalar variable. */
  if(lcr != NIL) {
    /* Modifies the bounds of the time dimensions (see above). */
    is_first = true;
    for(lt = pc->ltau, la = lcr, lom = pc->lomega; !ENDP(lt);
	POP(lt), POP(la), POP(lom)) {
      cr = RANGE(CAR(la));

      if(is_first && (d >= 0) && (d != INFINITY)) {
	range_upper(cr) = int_to_expression(d);

	is_first = false;
      }
      else {
	normalized nor_ub, nor_lb;
	Pvecteur pv_aux;
	Value cr_om;

	cr_om = int_to_value(INT(CAR(lom)));
	nor_ub = NORMALIZE_EXPRESSION(range_upper(cr));
	nor_lb = NORMALIZE_EXPRESSION(range_lower(cr));

	if((normalized_tag(nor_ub) != is_normalized_linear) ||
	   (normalized_tag(nor_ub) != is_normalized_linear) )
	    user_error("make_array_bounds", 
		       "\nArray bounds should be linear\n");
	
	pv_aux = vect_substract((Pvecteur) normalized_linear(nor_ub),
				(Pvecteur) normalized_linear(nor_lb));

	if (value_zero_p(value_mod(vect_pgcd_all(pv_aux), value_abs(cr_om))))
	  range_upper(cr) = make_vecteur_expression(vect_div(pv_aux, cr_om));
	else
	  range_upper(cr) = make_rational_exp(pv_aux, cr_om);
      }
      range_lower(cr) = int_to_expression(0);
    }
  }

  set_array_declaration(var_ent, lcr);
}


/*========================================================================*/
/* static Psysteme include_trans_on_LC_in_sc(ps, pc): transform the system
 * ps with the new variable we can find in the scell pc.
 *
 * AC 94/05/10
 */

static Psysteme include_trans_on_LC_in_sc(ps, pc)

 Psysteme     ps;
 Pscell        pc;
{
 Pcontrainte  cont;
 Psysteme     int_ps;

 my_matrices_to_constraints_with_sym_cst(&cont, pc->var_base,\
                                        pc->Tbase_out, pc->con_base,\
                                        pc->Tmat_inv, pc->Bmat_inv);
 int_ps = sc_make(cont, NULL);

 return(change_base_in_sc(ps, base_to_list(pc->var_base), int_ps));
}

/*========================================================================*/
/* reference include_trans_on_LC_in_ref(re, pc): include in the reference
 * "re" the new variables introduced by pc.
 *
 * AC 94/05/10
 */

static reference include_trans_on_LC_in_ref(re, pc)

 reference    re;
 Pscell        pc;
{
 Pcontrainte  cont;
 Psysteme     int_ps;
 list         l, li = NIL, la, lnew = NIL;
 expression   exp;
 Pvecteur     v, vc;
 int          d = 1;
 Value co, va, ppc;
 entity       e;

 li = base_to_list(pc->var_base);
 if (li != NIL)
   {
    if (get_debug_level() > 5)
      {
       fprintf(stderr, "\nReference en entree: \n");
       fprintf(stderr, "\n\tVariable : ");
       fprint_entity_list(stderr, CONS(ENTITY, reference_variable(re),NIL));
       fprintf(stderr, "\n\tIndices : ");
       fprint_list_of_exp(stderr, reference_indices(re));
      }

    my_matrices_to_constraints_with_sym_cst(&cont, pc->var_base,\
                                        pc->Nindices, pc->con_base,\
                                        pc->Rmat_inv, pc->Smat_inv);
    int_ps = sc_make(cont, NULL);
    sc_normalize(int_ps);
 
    if (get_debug_level() > 5)
    {
      fprintf(stderr,"\nSysteme qui remplace :");
      fprint_psysteme(stderr,int_ps);
    }

    for (l = reference_indices(re); l != NIL; l = CDR(l)) {
      exp = EXPRESSION(CAR(l));
      analyze_expression(&exp, &d);
      if (d != 1) pips_internal_error("d != 1");
      NORMALIZE_EXPRESSION(exp);
      v = normalized_linear(expression_normalized(exp));
      la = li;
      for (cont = int_ps->egalites; cont != NULL; cont = cont->succ)
      {
	e = ENTITY(CAR(la));
	co = vect_coeff((Variable) e, cont->vecteur);
	if (base_contains_variable_p(v, (Variable) e))
	{
	    Value x;
	  vc = vect_dup(cont->vecteur);
	  va = vect_coeff((Variable)e, v);
	  ppc = ppcm(co, va);
	  x = value_div(ppc,co);
	  vc = vect_multiply(vc, value_abs(x));
	  x = value_div(ppc,va);
	  v = vect_multiply(v, value_abs(x));
	  vect_erase_var(&v, (Variable)e);
	  vect_erase_var(&vc, (Variable)e);
	  if (value_posz_p(value_mult(co,va))) v = vect_substract(v, vc);
	  else v = vect_add(v, vc);
	  vect_normalize(v);
	}
	la = CDR(la);
      }
      exp = Pvecteur_to_expression(v);
      ADD_ELEMENT_TO_LIST(lnew, EXPRESSION, exp);
    }
    
    reference_indices(re) = lnew;
  }

 if (get_debug_level() > 5)
   {
    fprintf(stderr, "\nReference en sortie: \n");
    fprintf(stderr, "\n\tVariable : ");
    fprint_entity_list(stderr, CONS(ENTITY, reference_variable(re),NIL));
    fprintf(stderr, "\n\tIndices : ");
    fprint_list_of_exp(stderr, reference_indices(re));
   }

 return(re);
}


/*========================================================================*/
/* static Psysteme include_trans_on_LC_in_sc2(ps, pc): transform the system
 * ps with the new variable we can find in the scell pc.
 *
 * AC 94/05/10
 */

static Psysteme include_trans_on_LC_in_sc2(ps, pc, b)

 Psysteme     ps;
 Pscell        pc;
 Pbase        b;
{
 Pcontrainte  cont;
 Psysteme     int_ps;

 my_matrices_to_constraints_with_sym_cst(&cont, pc->var_base, b, pc->con_base,
					 pc->Tmat_inv, pc->Bmat_inv);
 int_ps = sc_make(cont, NULL);

 return(change_base_in_sc(ps, base_to_list(pc->var_base), int_ps));
}


/*========================================================================*/
/* void prepare_reindexing(v, b, p)
 *
 * It associates to the vertex v a structure scell. First it changes the
 * base of the instruction from base (i,j,...) of loop counters to base
 * (t, p,...) given by the time function and the placement function. All
 * the results are put in matrix. Then we introduce a matrix called Q that
 * mappes time x space coordinates to minor time x space coordinates using
 * the time periodicty. Returns the list of global time variables created
 *
 * AC 94/03/21 */

list prepare_reindexing(v, b, p)
vertex       v;
bdt          b;
plc          p;
{
  Pscell        pc = NULL, pc_aux = NULL;
  placement    pv;
  Psysteme ps_b, ps_p, ps_aux, old_ps, new_ps, int_ps, sys_p, pps, pcond,
  time_ps = SC_UNDEFINED;
  int          cn = dfg_vertex_label_statement(vertex_vertex_label(v));
  static_control stco;
  list lexp, bv, lnewp, lnewb, lpar, ciel, lnew, lnewp2, lnewp3, ltau =
    NIL, lnewq, lnewt, lom, lt = NIL;
  int pcount, bcount = 0, diff, d, qcount = 0, count_loop, i, time_count;
  Value den = VALUE_ONE, detP, detU;
  expression   exp;
  Pvecteur     vec, Tindx, Tcnst, Qindx, Qcnst, new_indx;
  entity       ent;
  Pmatrix mT, mB, mT_inv, mQ, mC, mQ_inv, mT_invB, mH, mP, mR, mQ_invC,
  mS, mR_inv, mS_inv, mI, mIc, mU, mId;
  Pcontrainte  cont;

  bool      with_t = true;
  Psyslist     lsys, lsys_aux, ltime, lsys_time, ltopo;

  lnewp = NIL;
  lnewq = NIL;

  if (cn != ENTRY_ORDER) {
    stco = get_stco_from_current_map(adg_number_to_statement(cn));
 
    if (get_debug_level()>5)
      fprintf(stderr, "\n***debut index pour %d***\n", cn);

    /* List of englobing loop counters */
    ciel = static_control_to_indices(stco);

    /* List of structure parameters. */
    lpar = lparams;

    Tindx = list_to_base(ciel);
    Tcnst = list_to_base(lpar);

    /* Two cases : loops around, or not */
    if (ciel != NIL) {
      if (get_debug_level()>1) {
	fprintf(stderr,"\nCIEL: ");
	fprint_entity_list(stderr,ciel);
	fprintf(stderr,"\nLPAR: ");
	fprint_entity_list(stderr,lpar);
      }

      /* first we extract the bdt corresponding to this node in the global */
      /* bdt b. idem for the plc.  */
      bv = extract_bdt(b, cn);
      pv = extract_plc(p, cn);

      /* we write the equations p = f(i,j,...) and put them in system
         ps_p. Note : we only keep those with loop indices. */
      ps_p = sc_new();
      pcount = 0;
      for (lexp = placement_dims(pv); !ENDP(lexp); POP(lexp)) {
	normalized nor;

	exp = EXPRESSION(CAR(lexp));
	nor = NORMALIZE_EXPRESSION(exp);

	if(normalized_tag(nor) != is_normalized_linear)
	  user_error("prepare_reindexing", "PLC not linear\n");

	vec = normalized_linear(nor);

	if(vars_in_vect_p(vec,ciel)) {
	  ps_p = sc_add_egalite_at_end(ps_p, contrainte_make(vec)); 
	  ent = create_new_entity(cn-BASE_NODE_NUMBER, STRING_PLC, pcount);
	  ADD_ELEMENT_TO_LIST(lnewp, ENTITY, ent);
	  pcount++;
	}
      }

      /* we write the equations t = f(i,j,...) and put them in system
       ps_b we have a double loop on the schedule because of possible
       predicate and posible multi-dimensionnal expression. We attach
       a scell to each domain of the schedule.  */
      for (; !ENDP(bv); POP(bv)) {
	schedule sched = SCHEDULE(CAR(bv));
	predicate pred = schedule_predicate(sched);
	lnewb = NIL;
	lnewq = NIL;
	lnew = NIL;
	ps_b = sc_new();
	den = 1;
	ltime = NULL;
	
	old_ps = sc_dup(predicate_to_system(dfg_vertex_label_exec_domain((dfg_vertex_label)vertex_vertex_label(v))));
	old_ps = sc_append(old_ps, predicate_to_system(pred));

	if (get_debug_level() > 1)
	  fprintf(stderr,"\n**Computing Cell**\n\n");
	time_count = gen_length(schedule_dims(sched));

	if(time_count > 1)
	  user_error("prepare_reindexing",
		     "Multi dimensional case not treated yet\n");

	ent = create_new_entity(cn-BASE_NODE_NUMBER, STRING_BDT, bcount);
	ADD_ELEMENT_TO_LIST(lnewb, ENTITY, ent);
	exp = EXPRESSION(CAR(schedule_dims(sched)));
	analyze_expression(&exp, &d);
	bcount++;

	/* we take care of a possible denominator. the denominator */
	/* of system ps_b has the value "den" */
	if (d == 1) {
	  normalized nor;
	  
	  nor = NORMALIZE_EXPRESSION(exp);
	  if(normalized_tag(nor) != is_normalized_linear)
	    user_error("prepare_reindexing", "Bdt is not linear\n");
	  
	  vec = vect_multiply(normalized_linear(nor), den);
	  if (vars_in_vect_p(vec, ciel)) {
	    sc_add_egalite(ps_b, contrainte_make(vec));
	  }
	  else {
	    ps_b = SC_UNDEFINED;
	    with_t = false; 
	    time_ps = sc_make(contrainte_make(vec), NULL);
	  }
	}
	else
	  user_error("prepare reindexing",
		     "Rational case not treated yet !\n");
	
	if (get_debug_level() > 5) {
	  fprintf(stderr, "\nNouvelles variables b:\n");
	  fprint_entity_list(stderr, lnewb);
	  fprintf(stderr, "\nNouvelles variables p:\n");
	  fprint_entity_list(stderr, lnewp);
	  fprintf(stderr,"\nContraintes sur B:\n");
	  fprint_psysteme(stderr, ps_b);
	  fprintf(stderr,"\nContraintes sur P:\n");
	  fprint_psysteme(stderr, ps_p);
        }

	/* count the number of equalities (diff) we need to add to
	 * build an inversible system */
	if (SC_UNDEFINED_P(ps_b)) {
	  diff = -1 * gen_length(ciel);
	  lnew = NIL;
        }
	else {
	  diff = gen_length(lnewb) - gen_length(ciel);
	  lnew = gen_copy_seq(lnewb);
        }

	if (get_debug_level() > 5)
	  fprintf(stderr,"\nDiff = %d\n",diff);

	lnewp2 = gen_copy_seq(lnewp);
	lnewp3 = NIL;
	ps_aux = sc_dup(ps_p);
	ps_aux = sc_reverse_constraints(ps_aux);

	/* Now we build the system on the system coming from the local */
	/* time by adding equalities coming from the placement system  */
	/* who are independant with the first ones.  */
	if (diff < 0) {
	  Pcontrainte ct = ps_aux->egalites;

	  /* we first append to ps_b the vectors coming from ps_p who */
	  /* are not linked with those of ps_b.  */
	  while ((diff != 0) && !CONTRAINTE_UNDEFINED_P(ps_aux->egalites)) {
	    ps_aux->egalites = ct->succ;
	    ct->succ = NULL;

	    if (value_notone_p(den))
	      ct->vecteur = vect_multiply(ct->vecteur, den);

	    if (!SC_UNDEFINED_P(ps_b))
	      {sc_add_egalite(ps_b, ct);}
	    else
	      ps_b = sc_make(ct, NULL);

	    if (vecteurs_libres_p(ps_b, Tindx, Tcnst)) {
	      /* the new vector is acceptable, i.e. free */
	      diff++;
	      /* update the list of new considered entities */
	      ent = ENTITY(CAR(lnewp2));
	      ADD_ELEMENT_TO_LIST(lnew, ENTITY, ent);
	      ADD_ELEMENT_TO_LIST(lnewp3, ENTITY, ent);
	    }
	    else {
	      /* the vector is unacceptable, continue to search */
	      ps_b->egalites = (ps_b->egalites)->succ;
	      ps_b->nb_eq--;
	    }
	    lnewp2 = CDR(lnewp2);

	    ct = ps_aux->egalites;
	    if (get_debug_level() > 5)
	      fprint_psysteme(stderr,ps_b);
	  }

	  /* we have to complete the system with others free vectors */
	  if (diff != 0) {
	    list llnew = NIL;

	    if (get_debug_level() > 5) {
	      fprintf(stderr,"\nBase incomplete !!!\n");
	      fprintf(stderr,"\nCiel :");
	      fprint_entity_list(stderr, ciel);
	      fprintf(stderr,"\nLpar :");
	      fprint_entity_list(stderr, lpar);
	      fprintf(stderr,"\nLnew :");
	      fprint_entity_list(stderr, lnew);
	    }

	    if(SC_UNDEFINED_P(ps_b))
	      ps_b = sc_new();

	    ps_b = base_complete(ps_b, ciel, lpar, &llnew); 
	    ps_b = sc_reverse_constraints(ps_b);
	    if (get_debug_level() > 5)
	      fprint_psysteme(stderr,ps_b);

	    for(; !ENDP(llnew); POP(llnew)) {
	      ent = create_new_entity(cn-BASE_NODE_NUMBER, STRING_PLC, pcount);
	      ADD_ELEMENT_TO_LIST(lnew, ENTITY, ent);
	      ADD_ELEMENT_TO_LIST(lnewp3, ENTITY, ent);
	      ADD_ELEMENT_TO_LIST(lnewp, ENTITY, ent);
	      vec = vect_new((Variable) ent, 1);
	      ps_p = sc_add_egalite_at_end(ps_p, contrainte_make(vec)); 
	      pcount++;
	    }
	  }
        }

	ps_b = sc_reverse_constraints(ps_b);

	if(ps_b->nb_eq != gen_length(ciel))
	  pips_internal_error("We are building a non-squared matrix");

	/* Now build the matrix of base changement :
         *  t: new indices, i: old indices, n:structure parameters
         *  t = mT.i + mB.n
         *  i = mT_inv.t - (mT_inv*mB).n
         */
	mT = matrix_new(ps_b->nb_eq, gen_length(ciel));
	mB = matrix_new(ps_b->nb_eq, gen_length(lpar)+1);
	mR = matrix_new(MATRIX_NB_LINES(mT),MATRIX_NB_COLUMNS(mT));
	mS = matrix_new(MATRIX_NB_LINES(mT),MATRIX_NB_COLUMNS(mB));
	mR_inv = matrix_new(MATRIX_NB_LINES(mT),MATRIX_NB_COLUMNS(mT));
	mS_inv = matrix_new(MATRIX_NB_LINES(mT),MATRIX_NB_COLUMNS(mB));
	my_constraints_with_sym_cst_to_matrices(ps_b->egalites, Tindx,
						Tcnst, mT, mB);
	MATRIX_DENOMINATOR(mT) = den;
	MATRIX_DENOMINATOR(mB) = den;
	matrix_normalize(mT);
	matrix_normalize(mB);

	/* calculate the inverse matrix of mT called mT_inv */
	/* we are sure that mT is reversible because it has */
	/* been built in a way where all vectors were free */
	mT_inv = matrix_new(ps_b->nb_eq, gen_length(ciel));
	matrix_general_inversion(mT, mT_inv);

	mT_invB = matrix_new(MATRIX_NB_LINES(mT),MATRIX_NB_COLUMNS(mB));
	matrix_multiply(mT_inv, mB, mT_invB );
	matrix_coef_mult(mT_invB, VALUE_MONE);
	matrix_normalize(mT_inv);
	matrix_normalize(mT_invB);

	/* we get the execution domain of the node and we will change   */
	/* the old variables by the new ones (t,p...). First we find    */
	/* the expression of the old variables in function of the new   */
	/* ones and put the result in int_ps, then make the replacement */
	/* in old_ps. old_ps will be the "domain" of the scell.  */

	old_ps = sc_dup(predicate_to_system(dfg_vertex_label_exec_domain((dfg_vertex_label)vertex_vertex_label(v)))); 
	old_ps = sc_append(old_ps, predicate_to_system(pred));
	pred = make_predicate(sc_dup(old_ps));

	if (get_debug_level() > 5) {
	  fprintf(stderr,"\nOld indices:");
	  fprint_entity_list(stderr, ciel);
	  fprintf(stderr,"\nNew indices:");
	  fprint_entity_list(stderr, lnew);
	  fprintf(stderr, "\nDomain of the old indices:");
	  fprint_psysteme(stderr, old_ps);
	  fprintf(stderr, "\nSystem of basis change 1:");
	  fprint_psysteme(stderr, ps_b);
	  fprintf(stderr, "\nT = ");
	  matrix_fprint(stderr,mT);
	  fprintf(stderr, "\nB = ");
	  matrix_fprint(stderr,mB);
	  fprintf(stderr,"\nMatrice mT_inv :");
	  matrix_fprint(stderr, mT_inv);
	  fprintf(stderr,"\nMatrice mT_invB :");
	  matrix_fprint(stderr, mT_invB);
        }

	new_indx = list_to_base(lnew);
	my_matrices_to_constraints_with_sym_cst(&cont, Tindx, new_indx,\
						Tcnst, mT_inv, mT_invB);
	int_ps = sc_make(cont, NULL);
	sc_normalize(int_ps);
     
	/* we include the new parameters in the system of constraints */
	new_ps = change_base_in_sc(old_ps, ciel, int_ps); 

	if (get_debug_level() > 5) {
	  fprintf(stderr, "\nSystem of basis change 2:");
	  fprint_psysteme(stderr, int_ps);
	  fprintf(stderr, "\nDomain after the basis change:");
	  fprint_psysteme(stderr, new_ps);
        }

	pcond = sc_rn(list_to_base(lnew));
	pps = sc_dup(new_ps);
	sc_transform_eg_in_ineg(pps);

	if(!sc_consistent_p(pps)) {
	  pps->base = (Pbase) NULL;
	  sc_creer_base(pps);
	  pps->nb_eq = nb_elems_list(sc_egalites(pps));
	  pps->nb_ineq = nb_elems_list(sc_inegalites(pps));
	}

	algorithm_row_echelon(pps, list_to_base(lnew), &pcond, &int_ps);

	new_ps = sc_append(new_ps, pcond);

	lnewt = gen_copy_seq(lnew);
    
	if (get_debug_level() > 5) {
	  fprintf(stderr, "\nDomain after row_echelon:");
	  fprint_psysteme(stderr, int_ps);
        }

	
	if (with_t) {
	  /* Now we have to build matrix mQ that maps normal to minor
	   * time x space coordinates, using the time period of each
	   * new variable given by omega and the its value at the
	   * origin */
	  mH = matrix_new(ps_b->nb_eq, gen_length(ciel));
	  mP = matrix_new(ps_b->nb_eq, gen_length(ciel));
	  mU = matrix_new(ps_b->nb_eq, gen_length(ciel));

	  matrix_hermite(mT, mP, mH, mU, &detP, &detU);

	  if (get_debug_level() > 5) {
	    fprintf(stderr, "\nHermite of mT = ");
	    matrix_fprint(stderr,mH);
	  }
  
	  matrix_free(mU);
	  matrix_free(mP);

	  if (get_debug_level() > 1) {
            fprintf(stderr,"\nSeparate begin");
	  }
	  count_loop = gen_length(lnewb);
	  lsys = separate_variables(sc_dup(int_ps), gen_copy_seq(lnew),\
				    &sys_p, count_loop);
	  if (get_debug_level() > 1) {
            fprintf(stderr,"\nSeparate end");
	  }

	  if (get_debug_level() > 1) {
            fprintf(stderr, "\n Separate systems: ");
            sl_fprint(stderr, lsys, entity_local_name);
            fprintf(stderr, "\nSystems for PLC:");
            fprint_psysteme(stderr, sys_p);
	  }

	  lsys_time = lsys;
	  ltime = NULL;
	  ltopo = NULL;

	  /* we separate the list of psystems in p from the one in t */
	  i = 1;
	  while (lsys_time != NULL) {
	    if (i <= time_count) {
	      /* this is a system in t */
	      ltime = add_sc_to_sclist(sc_dup(lsys_time->psys), ltime);
	      lsys_time = lsys_time->succ;
	      i++;
	    }
	    else {
	      /* this is a system in p */
	       ltopo = add_sc_to_sclist(sc_dup(lsys_time->psys), ltopo);
	       lsys_time = lsys_time->succ;
	     }
	  }

	  ltime = build_list_of_min(ltime, lnew, SC_UNDEFINED);

	  if (get_debug_level() > 5) {
            fprintf(stderr, "\n Systems for BDT after build_min: ");
            sl_fprint(stderr, ltime, entity_local_name);
            fprintf(stderr, "\n Systems for PLC after build_min: ");
            sl_fprint(stderr, ltopo, entity_local_name);
	  }

	  /* for each system of ltime build a Pscell */
	  for (lsys_aux = ltime; lsys_aux != NULL; lsys_aux =
	       lsys_aux->succ) {
	    qcount = 0;
            int_ps = sc_dup(lsys_aux->psys);
            int_ps = sc_append(int_ps, sc_dup(sys_p));

            if (get_debug_level() > 5) {
	      fprintf(stderr,"\nWe work on system: ");
	      fprint_psysteme(stderr, int_ps);
	    }

            /* tau: new indices with minor time
             * tau = mQ.t + mC.n
             * t = mQ_inv.tau - (mQ_inv*mC).n
             */
            mQ = matrix_new(MATRIX_NB_LINES(mT),MATRIX_NB_COLUMNS(mT));
            mQ_inv = matrix_new(MATRIX_NB_LINES(mT),MATRIX_NB_COLUMNS(mT));
	    mC = matrix_new(MATRIX_NB_LINES(mT), MATRIX_NB_COLUMNS(mB));
            mQ_invC = matrix_new(MATRIX_NB_LINES(mT), MATRIX_NB_COLUMNS(mB));

            build_contraction_matrices(cn, int_ps, mH, &qcount, &mQ, &mQ_inv,\
				       &mC, &lnewq, &lom,
				       list_to_base(lnew), Tcnst);

            matrix_multiply(mQ_inv, mC, mQ_invC);
	    matrix_coef_mult(mQ_invC, -1);
            matrix_normalize(mQ_inv);
            matrix_normalize(mQ_invC);

	    ltau = gen_copy_seq(lnewq);
	    lnewq = find_new_variables(lnew, lnewq);

            new_indx = list_to_base(lnewq);
            Qindx = list_to_base(lnew);
	    Qcnst = Tcnst;

            if (get_debug_level() >5) {
	      fprintf(stderr, "\nmQ :");
	      matrix_fprint(stderr, mQ);
	      fprintf(stderr, "\nmC:");
	      matrix_fprint(stderr, mC);
	      fprintf(stderr, "\nmQ_inv :");
	      matrix_fprint(stderr, mQ_inv);
	      fprintf(stderr, "\nmQ_invC:");
	      matrix_fprint(stderr, mQ_invC);
	    }

	    my_matrices_to_constraints_with_sym_cst(&cont, Qindx, new_indx,\
						    Qcnst, mQ_inv,
						    mQ_invC);

            mIc = matrix_new(MATRIX_NB_LINES(mT), MATRIX_NB_COLUMNS(mB));
            mI = matrix_new(MATRIX_NB_LINES(mT),MATRIX_NB_COLUMNS(mB));
            mId = matrix_new(MATRIX_NB_LINES(mT), MATRIX_NB_COLUMNS(mB));

	    matrix_multiply(mQ, mT, mR);
            matrix_normalize(mR);

	    matrix_multiply(mT_inv, mQ_inv, mR_inv);
            matrix_normalize(mR_inv);

	    matrix_multiply(mQ, mB, mIc);
            matrix_add(mS, mIc, mC);
            matrix_normalize(mS);

            matrix_multiply(mR_inv, mS, mS_inv);
            matrix_coef_mult(mS_inv, -1);
            matrix_normalize(mS_inv);
	  }
	}
	else {
	  /* case where the schedule is constant */
	  /* should be matrix_undefined better than NULL */
	  mQ = NULL; /* should be: Id */
	  mQ_inv = NULL; /* idem */
	  mC = NULL; /* should be: 0 */
	  lnewq = lnew;
	  mR = mT;
	  mS = mB;
	  mR_inv = mT_inv;
	  mS_inv = mT_invB;

	  lom = CONS(INT, 1, NIL);
	  ltau = NIL;
	  ltime  = add_sc_to_sclist(time_ps, ltime);

	  count_loop = gen_length(lnewb);
	  ltopo = separate_variables_2(sc_dup(int_ps), gen_copy_seq(lnew),\
				       &sys_p, count_loop);
	}

	/* we fill the scell at last ! */
	pc_aux = (Pscell)malloc(sizeof(scell));
	pc_aux->statement = cn;
	pc_aux->domain = pred;
	pc_aux->edge_dom = make_predicate(sc_dup(predicate_to_system(schedule_predicate(sched))));
	pc_aux->var_base = Tindx;
	pc_aux->con_base = Tcnst;
	pc_aux->Rbase_out = list_to_base(lnewq);
	pc_aux->Tbase_out = list_to_base(lnewt);
	pc_aux->Tmat = mT;
	pc_aux->Tmat_inv = mT_inv;
	pc_aux->Bmat = mB;
	pc_aux->Bmat_inv = mT_invB;
	pc_aux->Rmat = mR;
	pc_aux->Rmat_inv = mR_inv;
	pc_aux->Smat = mS;
	pc_aux->Smat_inv = mS_inv;
	pc_aux->lomega = lom;
	pc_aux->lp = lnewp3;
	pc_aux->lt = lnewb;
	pc_aux->ltau = ltau;
	pc_aux->Nbounds = new_ps;
	pc_aux->Nindices = list_to_base(lnewq);
	pc_aux->t_bounds = ltime;
	pc_aux->p_topology = ltopo;
	pc_aux->succ = pc;
	pc = pc_aux;

	if (get_debug_level() > 1) {
	  fprintf(stderr,"\nNoeud %d :\n", cn);
	  fprintf(stderr, "\nNouvelles variables:\n");
	  fprint_entity_list(stderr, lnewq);
	  fprintf(stderr, "\nRmat = ");
	  matrix_fprint(stderr,mR);
	  fprintf(stderr, "\nBasen de Smat = ");
	  pu_vect_fprint(stderr, Tcnst);
	  fprintf(stderr, "\nSmat = ");
	  matrix_fprint(stderr,mS);
	  fprintf(stderr, "\nSmat inverse = ");
	  matrix_fprint(stderr,mS_inv);
	  fprintf(stderr, "\nR_inv = ");
	  matrix_fprint(stderr,mR_inv);
	  fprintf(stderr, "\nNew Psystem ");
	  fprint_psysteme(stderr, new_ps);
	  fprintf(stderr,"\nOMEGU = %d",INT(CAR(lom)));
	  fprintf(stderr,"Liste des extremums sur temps :");
	  sl_fprint(stderr, pc_aux->t_bounds, entity_local_name);
	  fprintf(stderr,"Liste sur topology de p :");
	  sl_fprint(stderr, pc_aux->p_topology, entity_local_name);
	}
	
	/* Y a des matrices a effacer ICI !!*/
	
	if (gen_length(lnewb) > gen_length(lt)) lt = gen_copy_seq(lnewb);
	/* end of loop on schedule */ 
      }
    } 
    else { 
      /* No loop around the instruction */
      if (get_debug_level()>5) 
 	fprintf(stderr, "\nNo Loop Counters\n"); 

      /* first we extract the bdt, which should be constant. The plc */
      /* is null. */
      bv = extract_bdt(b, cn);

      /* The bdt is constant, but it can have different domains */
      for (; !ENDP(bv); POP(bv)) {
	schedule sched = SCHEDULE(CAR(bv));
	predicate pred = schedule_predicate(sched);
	lnewb = NIL;
	ltime = NULL;
	
	old_ps = sc_dup(predicate_to_system(dfg_vertex_label_exec_domain((dfg_vertex_label)vertex_vertex_label(v))));
	old_ps = sc_append(old_ps, predicate_to_system(pred));

	if (get_debug_level() > 1)
	  fprintf(stderr,"\n**Computing Cell**\n\n");

	lexp = schedule_dims(sched);
	if(gen_length(lexp) > 1)
	  user_error("prepare_reindexing",
		     "\nA constant schedule can not be multi-dimensional\n");

	ent = create_new_entity(cn-BASE_NODE_NUMBER, STRING_BDT, bcount);
	ADD_ELEMENT_TO_LIST(lnewb, ENTITY, ent);
	exp = EXPRESSION(CAR(lexp));
	analyze_expression(&exp, &d);

	/* we take care of a possible denominator. */
	if (d == 1) {
	  NORMALIZE_EXPRESSION(exp);
	  vec = normalized_linear(expression_normalized(exp));
	  time_ps = sc_make(contrainte_make(vec), NULL);
	  lom = CONS(INT, 1, NIL);
	}
	else
	  user_error("prepare reindexing",
		     "Rational case not treated yet !");
	
	pc_aux = (Pscell)malloc(sizeof(scell));
	pc_aux->statement = cn;
	pc_aux->domain = pred;
	pc_aux->edge_dom = make_predicate(sc_dup(predicate_to_system(schedule_predicate(sched))));
	pc_aux->var_base = NULL;
	pc_aux->con_base = Tcnst;
	pc_aux->Rbase_out = NULL;
	pc_aux->Tbase_out = NULL;
	pc_aux->Tmat = matrix_new(0,0);
	pc_aux->Tmat_inv = matrix_new(0,0);
	pc_aux->Bmat = matrix_new(0,0);
	pc_aux->Bmat_inv = matrix_new(0,0);
	pc_aux->Rmat = matrix_new(0,0);
	pc_aux->Rmat_inv = matrix_new(0,0);
	pc_aux->Smat = matrix_new(0,0);
	pc_aux->Smat_inv = matrix_new(0,0);
	pc_aux->lomega = lom;
	pc_aux->lp = NIL;
	pc_aux->lt = lnewb;
	pc_aux->ltau = NIL;
	pc_aux->Nbounds = NULL;
	pc_aux->Nindices = NULL;
	pc_aux->t_bounds = add_sc_to_sclist(time_ps, NULL);
	pc_aux->p_topology = NULL;
	pc_aux->succ = pc;
	pc = pc_aux;
      }
    } 

    hash_put(h_node, (char *)cn, (char *)pc);

    /* Prepare the array bounds */
    prepare_array_bounds(pc);
 
    if (get_debug_level()>5)
      fprintf(stderr, "\n***Fin index***\n");
  }
  
  return(lt);
}


/*=========================================================================*/
/* bool compatible_pc_p(pcd, pcs, s, d)
 *
 * Tests if the domain on which we work, that is pcd->domain with the
 * conditions on the edge is included in the domain of the selectionned
 * scell pcs (we transform)
 *
 * AC 94/04/07 */

static bool compatible_pc_p(pcd, pcs, s, d)
Pscell     pcd, pcs;
int       s;
dataflow  d;
{
  bool   bool = false;
  list      ltrans = dataflow_transformation(d);
  Psysteme  pss, psd;
  Pdisjunct dis = DJ_UNDEFINED;
  Ppath     pa = pa_new();

  if (get_debug_level() > 6) {
    fprintf(stderr,"\ncompatible pc debut:\n");
    fprintf(stderr,"\narc %d -> %d", pcs->statement, pcd->statement);
    fprintf(stderr,"\nDataflow:");
    fprint_dataflow(stderr, s, d);
  }

  if ((pcd == NULL) || (pcs == NULL))
    bool = true;
  else {
    psd = sc_dup(predicate_to_system(dataflow_governing_pred(d)));
    psd = sc_append(psd, predicate_to_system(pcd->domain));
    if (get_debug_level() > 6) {
      fprintf(stderr,"\nSysteme destination avec arc:\n");
      fprint_psysteme(stderr,psd);
    }

    pss = sc_dup(predicate_to_system(pcs->domain));
    if (get_debug_level() > 6) {
      fprintf(stderr,"\nTransformation:");
      fprint_list_of_exp(stderr, ltrans);
      fprintf(stderr,"\nSysteme source:\n");
      fprint_psysteme(stderr, pss);
    }
    if(pss != SC_UNDEFINED) {
      pss = include_trans_in_sc(s, pss, ltrans);
      if (get_debug_level() > 6) {
	fprintf(stderr,"\nSysteme source:\n");
	fprint_psysteme(stderr, pss);
      }

      dis = dj_append_system(dis, pss);
      pa->psys = psd;
      pa->pcomp = dis;
      if (!pa_faisabilite(pa))
	bool = true;
    }
    else
      bool = true;
   }
  
  if (get_debug_level() > 6) {
    if (bool)
      fprintf(stderr, "\ncellules compatibles ."); 
    else
      fprintf(stderr, "\ncellules non compatibles ."); 
    fprintf(stderr,"\ncompatible pc fin:\n");
  }
  
  return(bool);
}


/*=========================================================================*/
/* reference make_reindex(crt_df, crt_m, assign_i, pc):
 *
 * AC 94/04/22
 */

static reference make_reindex(crt_df, crt_m, assign_i, pc)
dataflow    crt_df;
int         crt_m;
instruction assign_i;
Pscell       pc;
{
  extern hash_table h_node;
  
  Pscell       source_pc;
  bool     not_found = true;
  reference   old_ref, new_ref;
  list        ltrans = dataflow_transformation(crt_df), lsub = NIL;
  Pmatrix     mL, mC, mA, mAC, mB, mBC, term1, term2, term3, term4, term5;
  Pmatrix     term6, term7;
  Psysteme    ps = sc_new();
  Pvecteur    Abase, ACbase, Bbase, Lbase;
  Pcontrainte cont;
  expression  exp;
  Pvecteur    vec;
  
  if (get_debug_level() > 6)
    fprintf(stderr,"\nMake reindex debut pour le noeud %d:\n", crt_m);
  
  source_pc = (Pscell)hash_get(h_node, (char *)crt_m);
  old_ref = dataflow_reference(crt_df);
  
  if (get_debug_level() > 6) { 
    fprintf(stderr, "\nOld dataflow reference : ");
    print_reference(old_ref);
  }

  if (source_pc != NULL)   {
    /* first, find the good scell we should work on */
    for ( ; (source_pc != NULL) && not_found; ) {
      if (compatible_pc_p(pc, source_pc, crt_m, crt_df))
	not_found = false;
      else
	source_pc = source_pc->succ;
    }

    if (not_found)
      user_error("make reindex"," pas trouve de domaine commun !");

    if(source_pc->var_base != NULL) {
      if (pc->var_base != NULL) {
	/* gotcha the good scell, we build now the list of subscripts */
	/* that is gamma(rhs) = f(gamma(lhs))                        */
	mA = pc->Rmat_inv;
	mAC = pc->Smat_inv;
	mB = source_pc->Rmat;
	mBC = source_pc->Smat;
	Abase = pc->Rbase_out;
	ACbase = pc->con_base;
	Lbase = pc->var_base;
	Bbase = source_pc->Rbase_out;

	/* first we transform the L application into matrices */
	/* (j) = (L)(i) <=> (j) = mL * (i) + mC * (n)         */

	if (get_debug_level() > 6)
	  fprintf(stderr,"\nApplication transformation :\n");

	for (; ltrans != NIL; POP(ltrans)) {
	  exp = EXPRESSION(CAR(ltrans));
	  if (get_debug_level() > 6) {
	    fprint_list_of_exp(stderr, CONS(EXPRESSION, exp,NIL));
	    fprintf(stderr,"\n");
	  }
	  NORMALIZE_EXPRESSION(exp);
	  vec = normalized_linear(expression_normalized(exp));
	  ps = sc_add_egalite_at_end(ps, contrainte_make(vec));
	}
	
	mL = matrix_new(MATRIX_NB_LINES(mB),MATRIX_NB_COLUMNS(mA));
	mC = matrix_new(MATRIX_NB_LINES(mB),MATRIX_NB_COLUMNS(mAC));
	
	my_constraints_with_sym_cst_to_matrices(ps->egalites, Lbase,
						ACbase, mL, mC);
	
	sc_rm(ps);
	ps = sc_new();
	
	/* the formula we have to apply now is :                */
	/*    gamma_b = mB * (mL(mA(gamma_a)+mAC) + mC) + mBC   */
	/*    gamma_b = mB*mL*mA(gamma_a)+mB*mL*mAC+mB*mC+mBC   */
	
	term1 = matrix_new(MATRIX_NB_LINES(mL),MATRIX_NB_COLUMNS(mA));
	term2 = matrix_new(MATRIX_NB_LINES(mB),MATRIX_NB_COLUMNS(mL));
	term3 = matrix_new(MATRIX_NB_LINES(mL),MATRIX_NB_COLUMNS(mAC));
	term4 = matrix_new(MATRIX_NB_LINES(mB),MATRIX_NB_COLUMNS(mC));
	term5 = matrix_new(MATRIX_NB_LINES(mB),MATRIX_NB_COLUMNS(mC));
	term6 = matrix_new(MATRIX_NB_LINES(mB),MATRIX_NB_COLUMNS(mC));
	term7 = matrix_new(MATRIX_NB_LINES(mB),MATRIX_NB_COLUMNS(mC));
	
	if (get_debug_level() > 6) {
	  fprintf(stderr,"\nMatrice A:\n");
	  matrix_fprint(stderr, mA);
	  fprintf(stderr,"\nMatrice mL:\n");
	  matrix_fprint(stderr, mL);
	  fprintf(stderr,"\nMatrice mB:\n");
	  matrix_fprint(stderr, mB);
	  fprintf(stderr,"\nMatrice mC:\n");
	  matrix_fprint(stderr, mC);
	}
	
	matrix_multiply(mL, mA, term1);
	matrix_multiply(mB, term1, term2);
	
	/* constant term */
	matrix_multiply(mL, mAC, term3);
	matrix_multiply(mB, term3, term4);
	matrix_multiply(mB, mC, term5);
	
	matrix_add(term6, term5, term4);
	matrix_add(term7, mBC, term6);
	
	matrix_normalize(term2);
	matrix_normalize(term7);
	
	if (get_debug_level() > 4) {
	  fprintf(stderr,"\nBase de depart:\n");
	  pu_vect_fprint(stderr, Abase);
	  fprintf(stderr,"\nBase d arrivee:\n");
	  pu_vect_fprint(stderr, Bbase);
	  fprintf(stderr,"\nBase matrice constante:\n");
	  pu_vect_fprint(stderr, ACbase);
	  fprintf(stderr,"\nMatrice produit:\n");
	  matrix_fprint(stderr, term2);
	  fprintf(stderr,"\nMatrice constante:\n");
	  matrix_fprint(stderr, term7);
	}
	
	my_matrices_to_constraints_with_sym_cst_2(&cont, Bbase, Abase,\
						  ACbase, term2, term7);
	
	matrix_free(mL);
	matrix_free(mC);
	matrix_free(term1);
	matrix_free(term2);
	matrix_free(term3);
	matrix_free(term4);
	matrix_free(term5);
	matrix_free(term6);
	matrix_free(term7);
	
	for (; cont != NULL; cont = cont->succ) {
	  exp = Pvecteur_to_expression(vect_dup(cont->vecteur));
	  if (expression_undefined_p(exp))
	    exp = int_to_expression(0);
	  expression_normalized(exp) = make_normalized(is_normalized_linear,
						       cont->vecteur);
	  ADD_ELEMENT_TO_LIST(lsub, EXPRESSION, exp);
	}
      }
      else {
	/* gotcha the good scell, we build now the list of subscripts */
	/* that is gamma(rhs) = f(gamma(lhs))                        */
	mB = source_pc->Rmat;
	mBC = source_pc->Smat;
	Bbase = source_pc->Rbase_out;
	ACbase = pc->con_base;
	
	/* first we transform the L application into matrices */
	/* (j) = (L)(i) <=> (j) = mL * (i) + mC * (n)         */
	
	if (get_debug_level() > 6)
	  fprintf(stderr,"\nApplication transformation : destination empty\n");
	
	for (; ltrans != NIL; POP(ltrans)) {
	  exp = EXPRESSION(CAR(ltrans));
	  if (get_debug_level() > 6) {
	    fprint_list_of_exp(stderr, CONS(EXPRESSION, exp,NIL));
	    fprintf(stderr,"\n");
	  }
	  NORMALIZE_EXPRESSION(exp);
	  vec = normalized_linear(expression_normalized(exp));
	  ps = sc_add_egalite_at_end(ps, contrainte_make(vec));
	}

	if(get_debug_level() > 6) {
	  fprintf(stderr,"mB :\n");
	  matrix_fprint(stderr, mB);
	  fprintf(stderr,"mBC :\n");
	  matrix_fprint(stderr, mBC);
	  fprintf(stderr,"Bbase :\n");
	  pu_vect_fprint(stderr, Bbase);
	  fprintf(stderr,"ACbase :\n");
	  pu_vect_fprint(stderr, ACbase);
	  fprintf(stderr,"Trans:\n");
	  fprint_psysteme(stderr, ps);
	}

	mC = matrix_new(MATRIX_NB_LINES(mB), vect_size(ACbase)+1);
	constraints_with_sym_cst_to_matrices(ps->egalites, NULL, ACbase,
					     matrix_new(0,0), mC);

	if(get_debug_level() >6) {
	  fprintf(stderr,"mC :\n");
	  matrix_fprint(stderr, mC);
	}
 
	sc_rm(ps);
	ps = sc_new();
	
	/* the formula we have to apply now is :                */
	/*    gamma_b = mB * mC + mBC   */
	
	term1 = matrix_new(MATRIX_NB_LINES(mB),0);
	term5 = matrix_new(MATRIX_NB_LINES(mB),MATRIX_NB_COLUMNS(mC));
	term6 = matrix_new(MATRIX_NB_LINES(mB),MATRIX_NB_COLUMNS(mC));
	
	if (get_debug_level() > 6) {
	  fprintf(stderr,"\nMatrice mB:\n");
	  matrix_fprint(stderr, mB);
	  fprintf(stderr,"\nMatrice mC:\n");
	  matrix_fprint(stderr, mC);
	}
	
	/* constant term */
	matrix_multiply(mB, mC, term5);
	matrix_add(term6, term5, mBC);
	matrix_normalize(term6);
	
	if (get_debug_level() > 4) {
	  fprintf(stderr,"\nBase d arrivee:\n");
	  pu_vect_fprint(stderr, Bbase);
	  fprintf(stderr,"\nBase matrice constante:\n");
	  pu_vect_fprint(stderr, ACbase);
	  fprintf(stderr,"\nMatrice constante:\n");
	  matrix_fprint(stderr, term6);
	}
	
	my_matrices_to_constraints_with_sym_cst_2(&cont, Bbase, NULL,
						  ACbase, term1, term6);
	
	matrix_free(mC);
	matrix_free(term1);
	matrix_free(term5);
	matrix_free(term6);
	
	for (; cont != NULL; cont = cont->succ) {
	  exp = Pvecteur_to_expression(vect_dup(cont->vecteur));
	  if (expression_undefined_p(exp))
	    exp = int_to_expression(0);
	  expression_normalized(exp) = make_normalized(is_normalized_linear,
						       cont->vecteur);
	  ADD_ELEMENT_TO_LIST(lsub, EXPRESSION, exp);

	  if (get_debug_level() > 4) {
	    fprintf(stderr,"\nNouvelles contraintes:\n");
	    vecteur_fprint(stderr, cont, pu_variable_name);
	    fprintf(stderr, "\nNouvelles expressions: %s\n",
		    words_to_string(words_expression(exp)));
	  }
	}
      }
    }
    else
      if(ltrans != NIL)
	user_error("make_reindex", "\nTransformation should be empty\n");
  }
  
  if (get_debug_level() > 6) {
    fprintf(stderr,"\nListe d expression :\n");
    fprint_list_of_exp(stderr, lsub);
  }

  /* calculate the delay if necessary */
  if (lsub != NULL) {
    entity tau = entity_undefined;
   
    if(pc->ltau != NIL)
      tau = ENTITY(CAR(pc->ltau));
    calculate_delay(EXPRESSION(CAR(lsub)), source_pc, pc, tau);
  }

  new_ref = my_build_new_ref(IS_NEW_ARRAY, crt_m, lsub, old_ref);
 
  if (get_debug_level() > 6) {
    fprintf(stderr,"\nNouvelle reference :\t ");
    print_reference(new_ref);
    fprintf(stderr,"\nMake reindex fin\n");
  }

  return(new_ref);
}

/*======================================================================*/
/* void substitute_expressions(expression exp, list l_ind, list l_exp)
 *
 * Substitute in expression "exp" the loop indices contains in "l_ind" by
 * the respective expressions contains in "l_exp". This substitution is
 * done recursively
 *
 * Parameters:
 *  exp: expression in which we do the substitution
 *  l_ind: list of the loop indices that have to be subtituted
 *  l_exp: list of the expressions used in the substitution
 *
 * AP 95/09/20 */
void substitute_expressions(exp, l_ind, l_exp)
expression exp;
list l_ind, l_exp;
{
  syntax sy;
  sy = expression_syntax(exp);

  if(syntax_tag(sy) == is_syntax_reference) {
    reference crt_ref = syntax_reference(sy);

    if(reference_indices(crt_ref) == NIL) {
      entity crt_e = reference_variable(crt_ref);
      bool is_in_list = false;
      list li, le = l_exp;

      for(li = l_ind; (li != NIL) && (! is_in_list); li = CDR(li)) {
	if(same_entity_p(crt_e, ENTITY(CAR(li))))
	  is_in_list = true;
	else
	  le = CDR(le);
      }
      if(is_in_list)
	expression_syntax(exp) = expression_syntax(EXPRESSION(CAR(le)));
    }
  }
  else if(syntax_tag(sy) == is_syntax_call) {
    list ael = call_arguments(syntax_call(sy));
    for(; !ENDP(ael); POP(ael)) {
      expression ae = EXPRESSION(CAR(ael));
      substitute_expressions(ae, l_ind, l_exp);
    }
  }
  else
    pips_internal_error("Bad syntax tag");
}

/*======================================================================*/
/* void substitute_loop_indices(instruction ins, list l_ind, list l_exp)
 *
 * Substitute in instruction "ins" the loop indices contains in "l_ind" by
 * the respective expressions contains in "l_exp".
 *
 * Parameters:
 *  ins: instruction in which we do the substitution
 *  l_ind: list of the loop indices that have to be subtituted
 *  l_exp: list of the expressions used in the substitution
 *
 * AP 95/09/20 */
void substitute_loop_indices(ins, l_ind, l_exp)
instruction ins;
list l_ind, l_exp;
{
  if(instruction_tag(ins) == is_instruction_call) {
    call c = instruction_call(ins);
    if(ENTITY_ASSIGN_P(call_function(c))) {
      expression rhs_exp;
      list args = call_arguments(c);

      if(gen_length(args) != 2)
	pips_internal_error("Assign call without 2 args");

      /* There are two args: lhs = rhs, we want the references of the rhs */
      rhs_exp = EXPRESSION(CAR(CDR(args)));

      substitute_expressions(rhs_exp, l_ind, l_exp);
    }
  }
  else
    pips_internal_error("Instruction is not an assign call");
}

/*======================================================================*/
/* list build_third_comb(old_ref, cls, assign_i, pc, te, sc, c, linit,
 * lmax): make the reindexation on a reference which we know it is the
 * last one of the instruction, so we make the reindexation on the
 * current reference, then encapsulate the new instruction in its test
 * C(t,p).
 *
 * Parameters:
 *             old_ref: current reference
 *             cls: current list of predecessors
 *             assign_i: current instruction
 *             pc: current Pscell
 *             te:
 *             sc: current domain
 *             c: counter of the local time variables
 *             linit:
 *             lmax:
 *
 * Result:
 *
 * AC 94/05/15 */

static list 
build_third_comb(old_ref, cls, assign_i, pc, te, sc, c, linit, lmax)
list         cls, *linit, *lmax;
instruction  assign_i;
Pscell        pc;
reference    old_ref;
Psysteme     sc, te;
int          *c;
{
  instruction  ins, new_i;
  Psysteme     new_ps, pps, pps2, pps3 = SC_UNDEFINED, te_dup, pcond;
  predicate    pred;
  dataflow     df;
  reference    new_ref;
  int          m, i;
  list         ldf, lm, lp, lom, lb = NIL, ub = NIL;
  statement    stat;
  range        ran;
  loop         loo;
  Psyslist     llp, llp1, llp2, llp3;
  list         li = NIL, lstat;
  expression   exp;
  entity       ent = entity_undefined, tim;
  Pbase        nbase;
  Pcontrainte  cont;
  Psysteme     int_ps;
  list         la, lnew = NIL;
  Pvecteur     vc;
  Value         co;
  entity       e;
    
  if (get_debug_level() > 4)
    fprintf(stderr, "\nBuild_third_comb begin");
    
  ldf = NIL; lm = NIL;

  /* special treatment for the old loop counters appearing as scalar
     variable in the instruction */
  if(pc != NULL) {
    li = base_to_list(pc->var_base);
    if (li != NIL) {
      my_matrices_to_constraints_with_sym_cst(&cont, pc->var_base,\
					      pc->Nindices, pc->con_base,\
					      pc->Rmat_inv, pc->Smat_inv);
      int_ps = sc_make(cont, NULL);
      sc_normalize(int_ps);
 
      if (get_debug_level() > 5) {
	fprintf(stderr, "\nIns BEFORE substitution\n");
	sa_print_ins(stderr, assign_i);
      }
      if (get_debug_level() > 5) {
	fprintf(stderr,"\nSubstitution with system:");
	fprint_psysteme(stderr,int_ps);
      }

      la = li;
      lnew = NIL;
      for (cont = int_ps->egalites; cont != NULL; cont = cont->succ) {
	e = ENTITY(CAR(la));
	co = vect_coeff((Variable) e, cont->vecteur);
	vc = vect_dup(cont->vecteur);
	vect_erase_var(&vc, (Variable)e);
	if (value_posz_p(co))
	  vc = vect_multiply(vc, VALUE_MONE);
	vect_normalize(vc);
	exp = Pvecteur_to_expression(vc);
	if(value_gt(co,VALUE_ONE) || value_lt(co,VALUE_MONE)) {
	  exp = make_op_exp(DIVIDE_OPERATOR_NAME, exp,
			    int_to_expression(
				VALUE_TO_INT(value_abs(co))));
	}
	lnew = gen_nconc(lnew, CONS(EXPRESSION, exp, NIL));
	la = CDR(la);

	if (get_debug_level() > 5) {
	  fprintf(stderr, "\nSubstitute %s by %s\n",
		  entity_local_name(e),
		  words_to_string(words_expression(exp)));
	}
      }
      substitute_loop_indices(assign_i, li, lnew);
      if (get_debug_level() > 5) { 
	fprintf(stderr, "\nIns AFTER substitution\n");
	sa_print_ins(stderr, assign_i);
      }
    }
  }

  li = NIL;

  /* get the list of dataflows */
  if (pc != NULL)
    ldf = dataflows_on_reference(cls, old_ref, pc->domain, &lm);
    
  if (ldf == NIL) {
    entity var;
    Pvecteur vec;
    expression cst_sched;
	
    /* special case where we replace just the old variables by the */
    /* new ones without changing the name of the reference         */
    if (get_debug_level() > 5)
      fprintf(stderr,"\nOOPPS!! pas de reference\n");
	
    new_ref = include_trans_on_LC_in_ref(copy_reference(old_ref), pc);
    ins = copy_instruction(assign_i);
    rhs_subs_in_ins(ins, old_ref, new_ref);
	
    /* Duplication because "te" is reused when there are two or
     * more dataflows, i.e. "ldf" has two or more elements. */
    new_ps = sc_dup(te);
    if (get_debug_level() > 6) {
      fprint_psysteme(stderr, new_ps);
    }
	
    /* modify te to take t_local into account */
    pps = sc_dup(pc->Nbounds);
    if (get_debug_level() > 6) {
      fprint_psysteme(stderr, pps);
    }
	
    /* include the information on the test */
    pps = sc_append(pps, new_ps);
	
    nbase = pc->Tbase_out;
    pcond = sc_rn(nbase);
    if(nbase != NULL) {
      /* recalculate the bounds for each variable of nbase */
      /*       pps2 = new_loop_bound(pps, nbase); */
      sc_transform_eg_in_ineg(pps);

      if(!sc_consistent_p(pps)) {
	pps->base = (Pbase) NULL;
	sc_creer_base(pps);
	pps->nb_eq = nb_elems_list(sc_egalites(pps));
	pps->nb_ineq = nb_elems_list(sc_inegalites(pps));
      }

      algorithm_row_echelon(pps, nbase, &pcond, &pps2);
    }
    else
      pps2 = pps;
	
    if (get_debug_level() > 6) {
      fprint_psysteme(stderr, pps2);
      fprint_psysteme(stderr, pcond);
    }
	
    if(!sc_empty_p(pcond)) {
      normalized ncs;

      /* deux derniers parametres a revoir */
      llp = separate_variables(pps2, base_to_list(nbase), &pps3, 0);
	    
      /* build the "forall p" loop around the instruction */
      lp = gen_nreverse(gen_copy_seq(pc->lp));
      if (get_debug_level() > 4) {
	fprintf(stderr,"\nListe des p : ");
	fprint_entity_list(stderr, lp);
      }
	    
      /* the list is in the wrong order: reorder it */
      llp1 = llp;
      llp2 = NULL;
      if(nbase != NULL) {
	while (llp1 != NULL) {   
	  llp3 = sl_new();
	  llp3->psys = llp1->psys;
	  if (get_debug_level() > 6) {
	    fprint_psysteme(stderr, llp1->psys);
	  }
	  llp3->succ = llp2;
	  llp2 = llp3;
	  llp1 = llp1->succ;
	}
      }
	    
      /* Bounds of the local time */
      cst_sched =
	make_vecteur_expression(pc->t_bounds->psys->egalites->vecteur);
      ncs = NORMALIZE_EXPRESSION(cst_sched);
	    
      if(get_debug_level() > 6) {
	fprintf(stderr, "\nConstant Schedule : %s\n",
		words_to_string(words_expression(cst_sched)));
      }
	    
      /* get a possible max for global time */
      ADD_ELEMENT_TO_LIST((*lmax), EXPRESSION, cst_sched);
	    
      /* We substitute the time variable by its constante value */
      var = ENTITY(CAR(pc->lt));
      vec = (Pvecteur) normalized_linear(ncs);
	    
      /* build the loop on p's, around our ins */
      stat = MAKE_STATEMENT(ins);
      for (; lp != NIL; lp = CDR(lp)) {
	pps = llp2->psys;
		
	if (get_debug_level() > 6) {
	  fprintf(stderr,
		  "\nBEFORE subs: var = %s, val = 1, vec = ",
		  entity_local_name(var));
	  pu_vect_fprint(stderr, vec);
	  fprint_psysteme(stderr, pps);
	}
		
	/* The substitution */
	substitute_var_with_vec(pps, var, 1, vec);
		
	if (get_debug_level() > 6) {
	  fprintf(stderr, "\nAFTER subs:\n");
	  fprint_psysteme(stderr, pps);
	}
		
	ran = make_bounds(pps, ENTITY(CAR(lp)), IS_LOOP_BOUNDS, NIL, NIL);
	loo = make_loop(ENTITY(CAR(lp)), ran, stat,
			entity_empty_label(),
			make_execution(is_execution_parallel,UU),
			NIL);
	new_i = make_instruction(is_instruction_loop, loo);
	stat = MAKE_STATEMENT(new_i);
	llp2 = llp2->succ;
      }  

      lstat = NIL;
      ADD_ELEMENT_TO_LIST(lstat, STATEMENT, stat);
	    
      if (get_debug_level() > 6) {
	fprintf(stderr,
		"\nInstruction en milieu de build_third_comb() :\n");
	sa_print_ins(stderr, ins);
      }
	    
      /* encapsulate everything in a block */
      ins = make_instruction_block(lstat);

      /* Add the condition introduces by algorithm_row_echelon() */
      stat = generate_optional_if(pcond, MAKE_STATEMENT(ins));
	    
      /* Put the test on global time around it */
      tim = get_time_ent(0, STRING_BDT, 0);
      exp = build_global_time_test_with_exp(tim, cst_sched);
	    
      ins = make_instruction(is_instruction_test, 
			     make_test(exp, stat,
				       make_empty_statement()));
	    
      if (get_debug_level() > 6) {
	fprintf(stderr,
		"\nInstruction en sortie de build_third_comb() :\n");
	sa_print_ins(stderr, ins);
      }
	    
      ADD_ELEMENT_TO_LIST(li, INSTRUCTION, ins);
    }
  }
  else {
    if (get_debug_level() > 4)
      fprintf(stderr,"\nNombre de dataflow: %d", gen_length(ldf));

    while (ldf != NIL) {
      bool with_t;

      df = DATAFLOW(CAR(ldf));
      m = INT(CAR(lm));
      lb = NIL;
      ub = NIL;
      lstat = NIL;
	    
      /* Duplication because "te" is reused when there are two or
       * more dataflows, i.e. "ldf" has two or more elements. */
      te_dup = sc_dup(te);
	    
      /* Says if the schedule is constant or not. */
      with_t = (pc->ltau != NIL);
	    
      if(with_t) {
	/* The local time of the scell is split into as many local
	 * time as there are dataflows. Each has its own bounds,
	 * which are computed below. However, there relation with
	 * the minor time remains the same, i.e.: t = omega*tau +
	 * l, where "l" is the lower bound calculated before
	 * (cf. prepare_reindexing) and saved into the field
	 * "t_bounds" of the scell.*/
	ent = get_time_ent((pc->statement)-BASE_NODE_NUMBER,
			   STRING_BDT, (*c));
	(*c)++;
		
	/* take the new local time into account in the base */
	nbase = include_time_in_base(pc, ent);
		
	if (get_debug_level() > 6) {
	  fprintf(stderr,"\nBase : ");
	  pu_vect_fprint(stderr, nbase);
	}
		
	/* modify te to take t_local into account */
	sc_chg_var(te_dup, (Variable)ENTITY(CAR(pc->lt)),
		   (Variable)ent);
      }
      else {
	nbase = pc->Tbase_out;
      }
	    
      if (m == ENTRY_ORDER)
	new_ref = include_trans_on_LC_in_ref(copy_reference(old_ref),
					     pc);
      else
	new_ref = make_reindex(df, m, assign_i, pc);
      new_i = copy_instruction(assign_i);
      rhs_subs_in_ins(new_i, old_ref, new_ref);
	    
      pred = dataflow_governing_pred(df);
      new_ps = sc_dup(predicate_to_system(pred));
	    
      if (get_debug_level() > 4) {
	fprintf(stderr,"\nCurrent Dataflow:");
	fprint_dataflow(stderr, m, df);
	fprintf(stderr,"\nSystem of the edge:");
	fprint_psysteme(stderr, new_ps);
      }
	    
      /* take the test into account */
      if (!SC_UNDEFINED_P(new_ps)) {
	/* we put around the ins the test C(t, p) */
	new_ps = include_trans_on_LC_in_sc2(new_ps, pc, nbase);
	new_ps = sc_append(new_ps, te_dup);
      }
      else
	new_ps = te_dup;
	    
      if (get_debug_level() > 6) {
	fprintf(stderr,"\nnew system:");
	fprint_psysteme(stderr, new_ps);
      }
	
      /* We need some consistency */
      if(!sc_consistent_p(new_ps) && !SC_UNDEFINED_P(new_ps)) {
	new_ps->base = NULL;
	sc_creer_base(new_ps);
	new_ps->nb_eq = nb_elems_list(sc_egalites(new_ps));
	new_ps->nb_ineq = nb_elems_list(sc_inegalites(new_ps));
	sc_dimension(new_ps) = base_dimension(sc_base(new_ps));
      }
    
      if (SC_RN_P(new_ps) ||
	  sc_rational_feasibility_ofl_ctrl(new_ps, NO_OFL_CTRL, true)) {
	entity tau, var;
	int omega, val;
	Pvecteur vec;
		
	/* modify te to take t_local into account */
	pps = sc_dup(pc->Nbounds);
		
	if(with_t)
	  sc_chg_var(pps, (Variable)ENTITY(CAR(pc->lt)),
		     (Variable)ent);
		
	if (get_debug_level() > 6) {
	  fprint_psysteme(stderr, pps);
	}
		
	/* include the information on the test */
	pps = sc_append(pps, new_ps);
		
	if (get_debug_level() > 6) {
	  fprint_psysteme(stderr, pps);
	}

	pcond = sc_rn(nbase);
	if(nbase != NULL) {
	  /* recalculate the bounds for each variable of nbase */
	  /* pps2 = new_loop_bound(pps, nbase); */
	  sc_transform_eg_in_ineg(pps);

	  if(!sc_consistent_p(pps)) {
	    pps->base = (Pbase) NULL;
	    sc_creer_base(pps);
	    pps->nb_eq = nb_elems_list(sc_egalites(pps));
	    pps->nb_ineq = nb_elems_list(sc_inegalites(pps));
	  }

	  algorithm_row_echelon(pps, nbase, &pcond, &pps2);
	}
	else
	  pps2 = pps;
	
	if (get_debug_level() > 6) {
	  fprint_psysteme(stderr, pps2);
	  fprint_psysteme(stderr, pcond);
	}
	
	if(!sc_empty_p(pcond)) { 
	  /* deux derniers parametres a revoir */
	  llp = separate_variables(pps2, base_to_list(nbase),
				   &pps3, 0);
		    
	  /* build the "forall p" loop around the instruction */
	  lp = gen_nreverse(gen_copy_seq(pc->lp));
	  if (get_debug_level() > 4) {
	    fprintf(stderr,"\nListe des p : ");
	    fprint_entity_list(stderr, lp);
	  }
		    
	  /* the list is in the wrong order: reorder it */
	  llp1 = llp;
	  llp2 = NULL;
		    
	  if(nbase != NULL) {
	    /* extract the list concerning the p's and reorder it */
	    for (i = 1; i <= gen_length(pc->ltau); i++)
	      llp1 = llp1->succ;
	    
	    while (llp1 != NULL) {   
	      llp3 = sl_new();
	      llp3->psys = llp1->psys;
	      if (get_debug_level() > 6) {
		fprint_psysteme(stderr, llp1->psys);
	      }
	      llp3->succ = llp2;
	      llp2 = llp3;
	      llp1 = llp1->succ;
	    }
	  }
		    
	  stat = MAKE_STATEMENT(new_i);
		    
	  /* Bounds of the local time */
	  if(with_t) {
	    lom = pc->lomega;
	    get_bounds_expression(llp, CONS(ENTITY, ent, NIL),
				  &lb, &ub);
	    
	    /* in lb we 've got the list of the lower bound of local time
	     * (of the current dataflow).  we are in the case of an
	     * instruction dependant of the time */
	  }
	  else {
	    expression cst_sched;

	    cst_sched =
	      make_vecteur_expression
		(pc->t_bounds->psys->egalites->vecteur);
	    NORMALIZE_EXPRESSION(cst_sched);
	    ub = CONS(EXPRESSION, cst_sched, NIL);

	    if(get_debug_level() > 6) {
	      fprintf(stderr, "\nConstant Schedule : %s\n",
		      words_to_string
		      (words_expression(cst_sched)));
	    }
	  }
	  
	  /* get a possible max for global time */
	  *lmax = gen_nconc((*lmax), remove_minmax(CONS(EXPRESSION,
							EXPRESSION(CAR(ub)),
							NIL)));
		    
	  if(with_t) {
	    Pvecteur vec_l;
	    statement sa;

	    /* initialize the local time, put the test
	       introduces by algorithm_row_echelon and put it in
	       the initialization list of stat called linit.  */
	    if (sc_rn_p(pcond))
	      sa = MAKE_STATEMENT(make_init_time(ent, EXPRESSION(CAR(lb))));
	    else if (sc_empty_p(pcond))
	      sa = MAKE_STATEMENT(make_init_time(ent,
						 int_to_expression(-1)));
	    else
	      sa = st_make_nice_test(Psysteme_to_expression(pcond),
				     CONS(STATEMENT,
					  MAKE_STATEMENT(make_init_time(ent, EXPRESSION(CAR(lb)))),
					  NIL),
				     CONS(STATEMENT,
					  MAKE_STATEMENT(make_init_time(ent, int_to_expression(-1))),
					  NIL));

	    ADD_ELEMENT_TO_LIST((*linit), STATEMENT, sa);
	    
	    /* HERE WE HAVE TO SUBSTITUTE minor time (tau) TO
	     * local time (var) IN THE Ps LOOP BOUNDS : var =
	     * omega*tau + l. "l" is the lower bound of the
	     * local time of the current scell, NOT the lower
	     * bound of the local time of the current dataflow
	     * (cf. above).  */
	    vec_l =
	      vect_dup(pc->t_bounds->psys->egalites->vecteur);
	    tau = ENTITY(CAR(pc->ltau));
	    omega = INT(CAR(pc->lomega));
	    var = ent;
	    val = 1;
	    vec = vect_add(vec_l, vect_new((Variable) tau, omega));
	  }
	  else {
	    /* We substitute the time variable by its constante
	       value */
	    var = ENTITY(CAR(pc->lt));
	    val = 1;
	    vec = (Pvecteur) normalized_linear
	      (expression_normalized(EXPRESSION(CAR(ub))));
	  }
		    
	  /* build the loop on p's */
	  for (; lp != NIL; lp = CDR(lp)) {
	    pps = llp2->psys;
	      
	    if (get_debug_level() > 6) {
	      fprintf(stderr,
		      "\nBEFORE subs: var = %s, val = 1, vec = ",
		      entity_local_name(var));
	      pu_vect_fprint(stderr, vec);
	      fprint_psysteme(stderr, pps);
	    }
	    
	    /* The substitution */
	    substitute_var_with_vec(pps, var, val, vec);
		      
	    if (get_debug_level() > 6) {
	      fprintf(stderr, "\nAFTER subs:\n");
	      fprint_psysteme(stderr, pps);
	    }
		      
	    ran = make_bounds(pps, ENTITY(CAR(lp)), IS_LOOP_BOUNDS, NIL, NIL);
	    loo = make_loop(ENTITY(CAR(lp)), ran, stat,
			    entity_empty_label(),
			    make_execution(is_execution_parallel,
					   UU), NIL);
	    new_i = make_instruction(is_instruction_loop, loo);
	    stat = MAKE_STATEMENT(new_i);
	    llp2 = llp2->succ;
	  }  
		    
	  lstat = NIL;
	  ADD_ELEMENT_TO_LIST(lstat, STATEMENT, stat);
		    
	  if (get_debug_level() > 6) {
	    fprintf(stderr,
		    "\nInstruction en milieu de build_third_comb() :\n");
	    sa_print_ins(stderr, new_i);
	  }
		    
	  tim = get_time_ent(0, STRING_BDT, 0);
	  if(with_t) {
	    /* increment the local time of omega */
	    ins = make_increment_instruction(ent,
					     INT(CAR(pc->lomega)));
	    ADD_ELEMENT_TO_LIST(lstat, STATEMENT, MAKE_STATEMENT(ins));
	  
	    /* build test on local time: if t>ub then t=-1 */
	    ins = build_local_time_test(ent, ub);
	    ADD_ELEMENT_TO_LIST(lstat, STATEMENT, MAKE_STATEMENT(ins));
	  
	    /* build the test on global time: IF t == t_local */
	    exp = build_global_time_test_with_exp(tim,
						  make_entity_expression(ent,
									 NIL));
	  }
	  else {
	    /* build the test on global time IF t == cst */
	    exp = build_global_time_test_with_exp(tim, EXPRESSION(CAR(ub)));
	  }
		    
	  /* encapsulate everything in a block */
	  ins = make_instruction_block(lstat);
	  stat = MAKE_STATEMENT(ins);
		    
	  /* Put the test on global time around it */
	  ins = make_instruction(is_instruction_test, 
				 make_test(exp, stat,
					   make_empty_statement()));
		    
	  if (get_debug_level() > 6) {
	    fprintf(stderr,
		    "\nInstruction en sortie de build_third_comb() :\n");
	    sa_print_ins(stderr, ins);
	  }
		    
	  ADD_ELEMENT_TO_LIST(li, INSTRUCTION, ins);
	}
	else
	  /* We can reuse the local time variable */
	  (*c)--;
      }
      else
	/* We can reuse the local time variable */
	(*c)--;
	    
      ldf = CDR(ldf);
      lm = CDR(lm);
	    
      if (get_debug_level() > 4)
	fprintf(stderr,"\nFin d'un dataflow");
    }
	
    /* end of while */
  }
    
  if (get_debug_level() > 4)
    fprintf(stderr, "\nBuild_third_comb end");
    
  return(li);
}


/*======================================================================*/
/* Pmytest build_third_subcomb(old_ref, cls, assign_i, pc, sc):
 * make reindexation on a reference we know it is not the last one,
 * which means that we can't build the new complete instruction. We
 * work on the copy of the instruction, and we return a list of
 * "mytest", the test is the possible new test introduced by the
 * dataflows, and the reindexed instruction is in the field "true".
 *
 * Parameters:
 *             old_ref: current reference
 *             cls: current list of predecessors
 *             assign_i: current instruction
 *             pc: current Pscell
 *             sc: current domain
 *
 * Result:
 *
 * AC 94/05/15 */

static Pmytest build_third_subcomb(old_ref, cls, assign_i, pc, sc)
list         cls;
instruction  assign_i;
Pscell        pc;
reference    old_ref;
Psysteme     sc;
{
  instruction  ins, new_i;
  Psysteme     new_ps;
  predicate    pred;
  dataflow     df;
  reference    new_ref;
  int          m;
  list         ldf = NIL, lm;
  Pmytest      lins = NULL, tes = NULL;

  if (get_debug_level() > 4)
    fprintf(stderr, "\nBuild_third_subcomb begin");

  /* get the list of dataflows */
  if (pc != NULL)
    ldf = dataflows_on_reference(cls, old_ref, pc->domain, &lm);

  if (ldf == NIL) {
    /* special case where we replace juste the old variables by the */
    /* new ones without changing the name of the reference          */
    if (get_debug_level() > 5) fprintf(stderr,"\nOOPPS!! pas de reference\n");
    new_ref = include_trans_on_LC_in_ref(copy_reference(old_ref), pc);
    ins = copy_instruction(assign_i);
    rhs_subs_in_ins(ins, old_ref, new_ref);
    lins = create_mytest(SC_UNDEFINED, ins, instruction_undefined);
  }
  else {
    if (get_debug_level() > 4)
      fprintf(stderr,"\nNombre de dataflow: %d", gen_length(ldf));

    while (ldf != NIL) {
      df = DATAFLOW(CAR(ldf));
      m = INT(CAR(lm));
    
      /* make the reindexation really if the source ("m") is not the
       * original sin, oopps!, node.  */
      if (m == ENTRY_ORDER)
	new_ref = include_trans_on_LC_in_ref(copy_reference(old_ref), pc);
      else
	new_ref = make_reindex(df, m, assign_i, pc);
      new_i = copy_instruction(assign_i);
      rhs_subs_in_ins(new_i, old_ref, new_ref);

      /* include the right conditions on the instruction we build */
      pred = dataflow_governing_pred(df);
      new_ps = sc_dup(predicate_to_system(pred));

      if (!SC_UNDEFINED_P(new_ps)) {
	/* we put around the ins the test C(t, p) */
	new_ps = include_trans_on_LC_in_sc(new_ps, pc);
	new_ps = sc_append(new_ps, sc);
	tes = create_mytest(new_ps, new_i, instruction_undefined);
      }
      else
	tes = create_mytest(sc, new_i, instruction_undefined);

      /* put the new element in the list we return */
      lins = add_elt_to_test_list(tes, lins);

      if (get_debug_level() > 4) {
	fprint_mytest(stderr, lins);
	fprintf(stderr,"\nFin d'un dataflow");
      }

      ldf = CDR(ldf);
      lm = CDR(lm);
    }
  }

  if (get_debug_level() > 4)
    fprintf(stderr, "\nBuild_third_subcomb end");

  return(lins);
}

/*======================================================================*/
/* list build_second_comb(pc, clrhs, assign_i, cls, sc, linit, lmax):
 * we treat here each reference of the right hand side of the
 * instruction (rhs) and we distinguish two type of reference, the
 * last one and the others. We first treat "the others" where we apply
 * build_third_subcomb(), and then we apply on the last one
 * build_third_comb().
 *
 * Parameters:
 *             pc: current Pscell
 *             clrhs: current list of rhs
 *             assign_i: current instruction
 *             cls: current list of predecessors
 *             sc: current bdt domain
 *             linit: list of initialization instructions
 *             lmax:
 *
 * Result:
 *
 * AC 94/04/28 */

static list build_second_comb(pc, clrhs, assign_i, cls, sc, linit, lmax)
Pscell        pc;
list         clrhs, cls, *linit, *lmax;
instruction  assign_i;
Psysteme     sc;
{
  instruction  ins;
  list         li = NIL, li2 = NIL, linit_aux = NIL, lmax_aux = NIL;
  reference    crhs = REFERENCE(CAR(clrhs));
  Pmytest      lins = NULL, lins1, lins2, lins3;
  Psysteme     tes;
  int          count = 0;

  /* the case clrhs = NIL should have treated in build_first_comb() */
  if (get_debug_level() > 4)
    fprintf(stderr, "\nBuild_second_comb begin");

  /* First we treat the case where we have several reference: we build a
   * list of "mytest" containing the copy of the instruction where the
   * current refernce has been replaced by its value and reindexed by the
   * function build_third_subcomb().  */
  while (clrhs->cdr != NIL) {
    lins2 = NULL;

    if (lins == NULL)
      lins = build_third_subcomb(crhs, cls, assign_i, pc, sc);
    else {
      for (lins1 = lins; lins1 != NULL; lins1 = lins1->succ) {
	ins = lins1->true;
	tes = lins1->test;

	lins3 = build_third_subcomb(crhs, cls, ins, pc, tes);
	lins2 = add_ltest_to_ltest(lins2, lins3);
      }
      lins = lins2;
    }
    clrhs = clrhs->cdr;
    crhs = REFERENCE(CAR(clrhs));
  }

  /* Now we are in the case where clrhs->cdr == NIL.  We call here the
   * function build_third_comb().  What we build here is not a list of
   * mytest but a list of instruction. */
  if (lins != NULL) {
    lins2 = NULL;

    for (lins1 = lins; lins1 != NULL; lins1 = lins1->succ) {
      ins = lins1->true;
      tes = lins1->test;
      linit_aux = NIL;
      lmax_aux = NIL;

      li2 = build_third_comb(crhs, cls, ins, pc, tes, sc, &count, 
			     &linit_aux, &lmax_aux);

      *linit = ADD_LIST_TO_LIST((*linit), linit_aux);
      *lmax = ADD_LIST_TO_LIST((*lmax), remove_minmax(lmax_aux));
      li = ADD_LIST_TO_LIST(li2, li);
    }
  }
  else {
    linit_aux = NIL;
    lmax_aux = NIL;

    li = build_third_comb(crhs, cls, assign_i, pc, SC_RN, sc, &count, 
			  &linit_aux, &lmax_aux);

    *linit = ADD_LIST_TO_LIST((*linit), linit_aux);
    *lmax = ADD_LIST_TO_LIST((*lmax), remove_minmax(lmax_aux));
  }

  return(li);
}

/*======================================================================*/
/* instruction build_first_comb(pc, ci, cls, cn, linit, lmax):
 *
 * Parameters:
 *             pc: Pcurrent Pscell
 *             ci: current instruction
 *             cls: list of the predecessors of the instruction
 *             cn: number of the instruction
 *             linit: list of the initialization instructions
 *             lmax: list
 *
 * Result:
 *
 * AC 94/04/22 
 */

static list build_first_comb(pc, ci, cls, cn, linit, lmax)
Pscell        pc;
instruction  ci;
list         cls, *linit, *lmax;
int          cn;
{
  instruction  assign_i = copy_instruction(ci), new_i;
  list         n_indices = NIL, clrhs, lp;
  Psysteme     new_sc, pps;
  list         lins = NIL, linit_a = NIL, lmax_a = NIL;
  statement    stat;
  Psyslist     llp;
  loop         loo;
  range        ran;
  expression   exp, exp2;
  Pvecteur     vec;
  entity       tim = get_time_ent(0, STRING_BDT, 0);
  instruction  ins;

  if (pc != NULL)
    n_indices = base_to_list(pc->Nindices);
  
  if (get_debug_level() > 1) {
    fprintf(stderr,"\n** instruction build_firstomb_comb debut pour %d**\n",cn);
    fprintf(stderr,"\nInstruction en entree : ");
    sa_print_ins(stderr, assign_i);
  }

  /* Change the lhs of ci to a new array indexed by the englobing loop
   * indices: SAIn(i1,...ip), (i1,...,ip) are the new indices of the
   * englobing loops of ci.  */
  my_lhs_subs_in_ins(assign_i, SAI, cn, n_indices);

  if (get_debug_level() > 1) {
    fprintf(stderr,"\nInstruction apres lhs_subs");
    sa_print_ins(stderr, assign_i);
  }

  /* construct the list of rhs */
  clrhs = get_rhs_of_instruction(assign_i);

  if (get_debug_level() > 2)
    fprintf(stderr,"\nNombre de reference a droite : %d\n",
	    gen_length(clrhs)); 

  if (pc != NULL) {
    if (pc->succ != NULL) {
      /* the domain of the bdt is in multiple parts, we build a comb by
       * calling recursively build_first_comb() */
      new_sc = sc_dup(predicate_to_system(pc->edge_dom));

      if (gen_length(clrhs) != 0) { 
	linit_a = NIL;
	lmax_a = NIL;

	lins = build_second_comb(pc, clrhs, assign_i, cls, new_sc, 
				 &linit_a, &lmax_a); 
	*linit = ADD_LIST_TO_LIST((*linit), linit_a);
	*lmax = ADD_LIST_TO_LIST((*lmax), remove_minmax(lmax_a));
      }
      else {
	ADD_ELEMENT_TO_LIST(lins, INSTRUCTION, assign_i);
	hash_put(delay_table, (char *)cn, (char *)INFINITY);
      }

      linit_a = NIL;
      lmax_a = NIL;
      lins = ADD_LIST_TO_LIST(lins, build_first_comb(pc->succ, ci, cls, 
						     cn, &linit_a,
						     &lmax_a));
      *linit = ADD_LIST_TO_LIST((*linit), linit_a);
      *lmax = ADD_LIST_TO_LIST((*lmax), remove_minmax(lmax_a));
    }
    else {
      /* the domain of the bdt is in one part */
      if (gen_length(clrhs) != 0) {
	linit_a = NIL;
	lmax_a = NIL;
	lins = build_second_comb(pc, clrhs, assign_i, cls, SC_RN, 
				 &linit_a, &lmax_a); 
	*linit = ADD_LIST_TO_LIST((*linit), linit_a);
	*lmax = ADD_LIST_TO_LIST((*lmax), remove_minmax(lmax_a));
      }
      else {
	if (get_debug_level() > 4) {
	  fprintf(stderr,"\nNo right hand side reference\n");
	}

	/* case of no right hand side reference */
	stat = MAKE_STATEMENT(assign_i);

	/* build the "forall p" loop around the instruction */
	lp = gen_nreverse(gen_copy_seq(pc->lp));
	if (get_debug_level() > 4) {
	  fprintf(stderr,"\nListe des p : ");
	  fprint_entity_list(stderr, lp);
	}
	llp = (pc->p_topology);

	for (; lp != NIL; lp = CDR(lp)) {
	  pps = llp->psys;
	  
	  ran = make_bounds(pps, ENTITY(CAR(lp)), IS_LOOP_BOUNDS, NIL, NIL);
	  loo = make_loop(ENTITY(CAR(lp)), ran, stat, entity_empty_label(),
			  make_execution(is_execution_parallel,UU), NIL);
	  new_i = make_instruction(is_instruction_loop, loo);
	  stat = MAKE_STATEMENT(new_i);
	  llp = llp->succ;
	}  
	hash_del(delay_table, (char *)cn);
	hash_put(delay_table, (char *)cn, (char *) INFINITY);

	/* the instruction has a constant schedule, in the system of
	 * pc->t_bounds we have the value of the bdt */
	vec = (((pc->t_bounds)->psys)->egalites)->vecteur;
	if (!VECTEUR_NUL_P(vec))
	  exp2 = Pvecteur_to_expression(vec);
	else
	  exp2 = int_to_expression(0);

	/* put the test on global time IF t == bdt_value */
	exp = build_global_time_test_with_exp(tim, exp2);

	if (get_debug_level() > 6) {
	  fprint_list_of_exp(stderr, CONS(EXPRESSION, exp, NIL));
	}

	(*lmax) = gen_nconc((*lmax), remove_minmax(CONS(EXPRESSION, exp2,
							NIL)));

	ins = make_instruction(is_instruction_test, 
			       make_test(exp, stat, make_empty_statement()));

	ADD_ELEMENT_TO_LIST(lins, INSTRUCTION, ins);
      }
    }
    /* If it exists, initialize the minor time and put it in the
     * initialization list of stat called linit.  */
    if(pc->ltau != NIL) {
      ins = make_init_time(get_time_ent(cn, STRING_TAU, 0), 
			   int_to_expression(0));
      ADD_ELEMENT_TO_LIST((*linit), STATEMENT, MAKE_STATEMENT(ins));
    }
  }

  return(lins);
}


/*=======================================================================*/
/* void re_do_it(graph the_dfg): function that redo the code by calling
 * the different functions necessary to do the job.
 * 
 * AC 94/07/25
 */

statement re_do_it(the_dfg, the_bdt, the_plc)
     graph       the_dfg;
     bdt         the_bdt;
     plc         the_plc;
{
  extern hash_table h_node;

  list        vl, sl, lstatg, ltim = NIL, laux, lins = NIL, lent;
  statement   new_mod_stat;
  range       ran;
  loop        loo;
  statement   stat;
  instruction ins, ins2;
  list        linit, lstat, lmax = NIL;
  entity      tim, fla;
  expression   upper, lower, incr;
  call         ca;
  Psysteme     ps;
  Variable        var;


  /* let's have a reverse DFG, i.e. the "successors" of an instruction i */
  /* are the instructions that may write the values used by i; */
  the_dfg = my_dfg_reverse_graph(the_dfg);

  if (get_debug_level() > 2) {
    fprintf(stderr,"\nGraphe renverse comme la creme:");
    fprint_dfg(stderr, the_dfg);
  }

  /* we prepare here the different elements of the reindexing. The default
   * delay is -1. */
  for (vl = graph_vertices(the_dfg); !ENDP(vl); POP(vl)) {
    vertex cv = VERTEX(CAR(vl));
    hash_put(delay_table, 
	     (char *) dfg_vertex_label_statement(vertex_vertex_label(cv)),
	     (char *) -1);
    laux = prepare_reindexing(cv, the_bdt, the_plc);
    if ( gen_length(laux) > gen_length(ltim) )
      ltim = laux;
  }

  if(gen_length(ltim) > 1)
    user_error("re_do_it", "\nMulti dimensional case not treated yet\n");

  tim = ENTITY(CAR(ltim));

  /* Initialize the list of instructions of the body of the global
   * time loop. */
  sl = NIL;

  /* Initialize the list of instructions of initialization */
  linit = NIL;
  
  /* now we make the reindexation in the code */
  for (vl = graph_vertices(the_dfg); !ENDP(vl); POP(vl)) {
    vertex      cv = VERTEX(CAR(vl));
    int         cn = dfg_vertex_label_statement(vertex_vertex_label(cv));
    statement   cs;
    instruction ci;
    list        cls;
    Pscell       pc;

    /* loop on vertices */
    if (cn != ENTRY_ORDER) { 
      cs = adg_number_to_statement(cn);
      ci = statement_instruction(cs);

      if (get_debug_level() > 2)
	print_detailed_ins(ci);

      if (!assignment_statement_p(cs))
	user_error("reindexing", "Pas une assignation\n");

      /* cls is the list of predecessors of cv */
      cls = vertex_successors(cv);

      pc = (Pscell)hash_get(h_node, (char *)cn);

      /* Do the reindexation */
      lins = build_first_comb(pc, ci, cls, cn, &linit, &lmax);

      /* two cases here: either the list has one element and we do not
       * have to introduce a flag but we have to increment the minor time
       * (if it exists). Or, the list has more than one element and we
       * have to introduce a flag and put the incrementation of the minor
       * time in a test on the flag (if it exists). In both case, if there
       * is no minor time then we do nothing.  */

      if(pc->ltau != NIL) {
	/* This list is to be the new value of "lins". */
	lstatg = NIL;

	if (lins->cdr != NIL) {
	  /* introducing the flag */
	  fla = create_new_entity(cn-BASE_NODE_NUMBER, STRING_FLAG, 0);

	  for (; lins != NIL; lins = CDR(lins)) {
	    instruction aux_ins;

	    /* Each instruction is composed of tests perfectly nested with
             * a non empty instruction in the true part. */
	    ins = INSTRUCTION(CAR(lins));

	    if(instruction_tag(ins) != is_instruction_test)
		user_error("re_do_it", "Not a test\n");

	    aux_ins = statement_instruction(test_true(instruction_test(ins)));
	    while(instruction_tag(aux_ins) != is_instruction_block) {
		if(instruction_tag(aux_ins) != is_instruction_test)
		    user_error("re_do_it", "Not a test\n");

		aux_ins = statement_instruction
		    (test_true(instruction_test(aux_ins)));
	    }
	    lstat = instruction_block(aux_ins);

	    /* create the statement: flag = true, and add it at the
	     * end of the block */
	    ins2 = build_flag_assign(fla, true);
	    ADD_ELEMENT_TO_LIST(lstat, STATEMENT, MAKE_STATEMENT(ins2));
	    instruction_block(statement_instruction
			      (test_true(instruction_test(ins)))) = lstat;

	    /* Put this statement in "lstatg", the new value of
	     * "lins". */
	    ADD_ELEMENT_TO_LIST(lstatg, STATEMENT, MAKE_STATEMENT(ins));
	  }
	  
	  /* add the test on the flag */
	  ins = build_flag_test(fla, ENTITY(CAR(pc->ltau)));
	  ADD_ELEMENT_TO_LIST(lstatg, STATEMENT, MAKE_STATEMENT(ins));

	  /* set the flag to false, i.e. reinitialization */
	  ins = build_flag_assign(fla, false);
	  ADD_ELEMENT_TO_LIST(lstatg, STATEMENT, MAKE_STATEMENT(ins));
	
	  /* initialize the flag to false in "linit" */
	  ins = build_flag_assign(fla, false);
	  ADD_ELEMENT_TO_LIST(linit, STATEMENT, MAKE_STATEMENT(ins));
	}
	else {
	  ins = INSTRUCTION(CAR(lins));
	  lstat = instruction_block(statement_instruction(test_true(instruction_test(ins))));
	  
	  /* increment the minor time */
	  ins2 = make_increment_instruction(ENTITY(CAR(pc->ltau)),1);
	  ADD_ELEMENT_TO_LIST(lstat, STATEMENT, MAKE_STATEMENT(ins2));

	  instruction_block(statement_instruction(test_true(instruction_test(ins)))) = lstat;

	  /* no flag to introduce */
	  ADD_ELEMENT_TO_LIST(lstatg, STATEMENT, MAKE_STATEMENT(ins));
	}
      }
      else {
	lstatg = NIL;
	for (; lins != NIL; lins = CDR(lins)) {
	  ins = INSTRUCTION(CAR(lins));
	  ADD_ELEMENT_TO_LIST(lstatg, STATEMENT, MAKE_STATEMENT(ins));
	}
      }
      /* put all the pieces of lstatg in one statement */
      ins = make_instruction_block(lstatg);

      /* put the new statement in the list representing the program */
      sl = ADD_ELEMENT_TO_LIST(sl, STATEMENT, MAKE_STATEMENT(ins));
    }
  /* end of loop on vertices */
  }

  if (get_debug_level() > 1) {
    fprintf(stderr,
	    "\n\n****************************************************");
    fprintf(stderr,"\nCODE BEFORE DELAY");
    fprintf(stderr,
	    "\n****************************************************\n\n");
    sa_print_ins(stderr, make_instruction_block(sl));
    fprintf(stderr,
	    "\n\n****************************************************\n");
  }

  /* now, we have to take into account the value of the delay calculated, */
  /* i.e. we go through the code and replace each first time dimension of */
  /* each instruction by the same expression modulo the delay.  */
  sl = add_delay_information(delay_table, sl);

  /* ICI il faudrait pouvoir construire les boucles sur les differentes */
  /* dimension du temps global et non 1 seule dimension !! */
  /* voir aussi le probleme des bornes de ces variables */

  /* Computation of the array bounds, one per instruction */
  for (vl = graph_vertices(the_dfg); !ENDP(vl); POP(vl)) {
    vertex cv = VERTEX(CAR(vl));
    make_array_bounds(cv);
  }

  /* Build the loop on the global time, the body is "sl" */
  stat = MAKE_STATEMENT(make_instruction_block(sl));
  tim = get_time_ent(0, STRING_BDT, 0);

  /* We have to simplify the list of the possible upper bounds to put in
   * our MAX expression. This is done by using the function
   * simplify_minmax() with a context specifying that each structure
   * parameter is positive. The list "lparams" is a global variables
   * giving these parameters. */
  ps = sc_new();
  for (lent = lparams; lent != NIL; lent = CDR(lent))    {
    var = (Variable)ENTITY(CAR(lent));
    sc_add_inegalite(ps, contrainte_make(vect_new(var, (Value)-1)));
  }
  sc_creer_base(ps);
  ca = make_call(entity_intrinsic("MAX"), simplify_minmax(lmax, ps,
							  IS_MAX));
  upper = make_expression(make_syntax(is_syntax_call, ca),
			  normalized_undefined);
  lower = int_to_expression(0);
  incr = int_to_expression(1);
  ran = make_range(lower, upper, incr);

  loo = make_loop(tim, ran, stat, entity_empty_label(),
		  make_execution(is_execution_sequential,UU), NIL);
  ins = make_instruction(is_instruction_loop, loo);

  /* add the statement to all statements of initialization */
  ADD_ELEMENT_TO_LIST(linit, STATEMENT, MAKE_STATEMENT(ins));

  /* "sl" becomes the lists of all the instructions of the parallel
   * program. */
  sl = linit;

  new_mod_stat = MAKE_STATEMENT(make_instruction_block(sl));

  if (get_debug_level() > 1) {
    fprintf(stderr,
	    "\n\n****************************************************");
    fprintf(stderr,"\nCODE FINAL");
    fprintf(stderr,
	    "\n****************************************************\n\n");
    sa_print_ins(stderr, statement_instruction(new_mod_stat));
    fprintf(stderr,
	    "\n\n****************************************************\n");}
  
  return(new_mod_stat);
}
