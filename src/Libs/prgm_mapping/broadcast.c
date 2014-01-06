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
/*
 * -- broadcast.c
 * 
 * package prgm_mapping : Alexis Platonoff, april 1993 --
 * 
 * This file contains the functions used for the broadcast detection.
 * 
 */

#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>

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

#include "ri.h"
#include "ri-util.h"
#include "graph.h"
#include "dg.h"
#include "paf_ri.h"
#include "constants.h"
#include "misc.h"
#include "static_controlize.h"
#include "paf-util.h"
#include "array_dfg.h"
#include "prgm_mapping.h"

/* Local defines */
typedef dfg_vertex_label vertex_label;
typedef dfg_arc_label arc_label;

/* global variables */
extern list prgm_parameter_l;

/* ======================================================================== */
/*
 * void broadcast(graph g) : This function walks through all the node list of
 * the graph "g" (it is a DFG) and for each dataflow it detects if it is a
 * broadcast or not. If so the communication field of the dataflow is updated
 * with the broadcast vector.
 * 
 * The detection on each dataflow is done with broadcast_of_dataflow() that
 * needs the dataflow itself and the corresponding sink statement.
 */
void
broadcast(g)
  graph           g;
{
  list            nodes_l,	/* List of the nodes of the DFG */
                  su_l,		/* List of successors of one node */
                  df_l;		/* List of dataflows of one successor */
  int             sink_stmt;	/* Statement number associated with one
				 * successor */

  /* We look for broadcasts on all nodes */
  for (nodes_l = graph_vertices(g); nodes_l != NIL; nodes_l = CDR(nodes_l)) {
    su_l = vertex_successors(VERTEX(CAR(nodes_l)));

    /* We look for broadcasts on all successors of one node */
    for (; su_l != NIL; su_l = CDR(su_l)) {
      successor       su = SUCCESSOR(CAR(su_l));
      vertex          sv = successor_vertex(su);

      sink_stmt = vertex_int_stmt(sv);
      df_l = dfg_arc_label_dataflows((dfg_arc_label) successor_arc_label(su));

      /* We look for broadcasts on all dataflows of one successor */
      for (; df_l != NIL; df_l = CDR(df_l))
	broadcast_of_dataflow(DATAFLOW(CAR(df_l)), sink_stmt, VERTEX_DOMAIN(sv));
    }
  }
}


/* ======================================================================== */
/*
 * entity base_find_var_with_rank(Pbase b, int r): returns the entity
 * corresponding to the variable of the Pbase at the rth position.
 */
entity
base_find_var_with_rank(b, r)
  Pbase           b;
  int             r;
{
  int             i;
  Pvecteur        v = (Pvecteur) b;

  if (r > base_dimension(b))
    pips_internal_error("rank %d too high", r);

  for (i = 1; i < r; v = v->succ, i++);

  return ((entity) v->var);
}


/* ======================================================================== */
/*
 * void broadcast_of_dataflow(dataflow df, int stmt, predicate exec_domain):
 *
 * detects if the dataflow "df" can produce a broadcast communication. "stmt"
 * gives the sink statement corresponding to this dataflow, it is used to get
 * the indices of the englobing loops. "exec_domain" gives the execution
 * domain of the sink instruction.
 * 
 * A broadcast corresponds to the fact that the same memory cell is read by
 * different instances (or operations) of an instruction. An operation is
 * represented by the name of the instruction and a vector of iteration (that
 * gives the value of the indices of the englobing loops).
 * 
 * One field of the dataflow is "transformations" which gives a list of
 * expressions each representing the value of a source index function of the
 * sink indices. Let s be the source indices and d the sink indices,
 * "transformations" represents a system like: A.d = s
 * 
 * A dataflow is a broadcast if and only if, the source indices fixed, their
 * exist several possible values for the sink indices. This means to find the
 * solution of a system of linear integer equations: d = d0 + KerA, KerA is
 * the kernel of the linear application associated to matrix A, so, a basis
 * of KerA gives the direction of the broadcast.
 * 
 * We solve this system with the hermite reduction ([Min83]): P.A.Q = H,
 * where P and Q are unimodular and H is the hermite form. Let r be the rank
 * of H and Q_r a sub-matrix of Q (the m-r last columns). [Min83] gives the
 * solution as: d = d0 + Q_r.y where y is a vector of arbitrary integers.
 * Then Q_r gives a basis of KerA.
 */
void
broadcast_of_dataflow(df, stmt, exec_domain)
  dataflow        df;
  int             stmt;
  predicate       exec_domain;
{
  matrice         A,		/* Matrix of transformation equations (n x m) */
                  B, P,		/* Left matrix for the hermite decomposition
				 * (n x n) */
                  Q,		/* Right matrix for the hermite decomposition
				 * (m x m) */
                  H,		/* Matrix of Hermite (n x m) */
                  Q_r;		/* Sub-matrix of Q: the m-r last columns */
  int             n,		/* Number of transformation equations */
                  m,		/* Number of englobing loops */
                  r,		/* Rank of H */
                  i, j;
  Value           det_p, det_q;
  predicate       new_pred,	/* Equations of broadcast vectors */
                  gov_pred;	/* Governing predicate */
  communication   comm;		/* Communication of the dataflow */
  Pbase           b_loop_ind;	/* Base of index of the englobing loops */
  Psysteme        sc_trans,	/* Equations of transformations */
                  diff_sc;	/* Equations of broadcast vectors */
  list            trans_l,	/* List of the transformations */
                  li, lp;
  static_control  stct;

  trans_l = dataflow_transformation(df);
  gov_pred = dataflow_governing_pred(df);
  comm = dataflow_communication(df);
  diff_sc = sc_new();

  /* Transformation system of equations */
  sc_trans = make_expression_equalities(trans_l);

  /* Englobing loops indices (li) and structure parameters (lp) */
  stct = get_stco_from_current_map(adg_number_to_statement(stmt));
  li = static_control_to_indices(stct);
  lp = gen_append(prgm_parameter_l, NIL);

  /* Base of the index of the englobing loops */
  b_loop_ind = list_to_base(li);

  /* We compute the size the matrix A */
  n = sc_trans->nb_eq;
  m = base_dimension(b_loop_ind);

  if (n == 0) {
    /*
     * The number of equations is zero (no equations for the transformation
     * means that the source instruction is not inside a loop body) then each
     * index of the base (indices of the loop nest containing the sink
     * instruction) loop is a broadcast vector.
     */
    for (j = 1; j <= m; j++) {
      entity          ent = base_find_var_with_rank(b_loop_ind, j);
      sc_add_egalite(diff_sc, contrainte_make(vect_new((char *) ent, 1)));
    }
  } else if (m == 0) {
    /*
     * There is no englobing index for the sink statement, this means that
     * this statement is not inside a loop body, so, it is not a broadcast.
     */
  } else {
    /*
     * n > 0: we compute the matrix H and the associated matrices P and Q.
     * The kernal of A is not of dim 0 only if its rank (r) is less than its
     * number of columns (m).
     */
    A = matrice_new(n, m);
    B = matrice_new(n, 1);
    pu_contraintes_to_matrices(sc_trans->egalites, b_loop_ind, A, B, n, m);

    P = matrice_new(n, n);
    Q = matrice_new(m, m);
    H = matrice_new(n, m);
    matrice_hermite(A, n, m, P, H, Q, &det_p, &det_q);

    r = matrice_hermite_rank(H, n, m);
    if (r > n)
      r = n;
    if (r > m)
      r = m;

    if (m - r > 0) {
      Q_r = matrice_new(m, m - r);
      for (i = 1; i <= m; i++)
	for (j = 1; j <= m - r; j++)
	  ACCESS(Q_r, m, i, j) = ACCESS(Q, m, i, j + r);
      Q_r[0] = DENOMINATOR(Q);

      for (j = 1; j <= m - r; j++) {
	Pvecteur        vect = NULL;
	for (i = 1; i <= m; i++) {
	  entity          ent = base_find_var_with_rank(b_loop_ind, i);
	  vect = vect_add(vect_new((char *) ent, ACCESS(Q_r, m, i, j)), vect);
	}
	sc_add_egalite(diff_sc, contrainte_make(vect));
      }
    }
  }

  /*
   * We update the dataflow communication of "df" only if the systeme has at
   * least one equation, i.e. one broadcast vector.
   */
  if (diff_sc->nb_eq != 0) {
    Psysteme        sc_ed, sc_gp, df_domain, impl_sc;
    Pcontrainte     pd, pdprec;

    /*
     * We make sure that each broadcast vector is a real one, i.e. the set of
     * values it can take has strictly more than one value. For this, we get
     * the implicit equation of the dataflow domain, the execution domain of
     * the sink intersection the governing predicate.
     */
    if (exec_domain == predicate_undefined)
      sc_ed = sc_new();
    else
      sc_ed = (Psysteme) predicate_system(exec_domain);
    if (gov_pred == predicate_undefined)
      sc_gp = sc_new();
    else
      sc_gp = (Psysteme) predicate_system(gov_pred);

    df_domain = sc_new();
    df_domain = sc_intersection(df_domain, sc_ed, sc_gp);

    impl_sc = find_implicit_equation(df_domain);

    if (impl_sc != NULL) {
      for (pd = diff_sc->egalites, pdprec = NULL; pd != NULL; pd = pd->succ) {
	Psysteme        aux_ps = sc_dup(impl_sc);

	sc_add_egalite(aux_ps, contrainte_make(vect_dup(pd->vecteur)));
	aux_ps->base = NULL;
	sc_creer_base(aux_ps);
	aux_ps = sc_normalize(aux_ps);

	/*
	 * We remove the current diffusion vector if it corresponds to one of
	 * the implicit equations.
	 */
	if ((aux_ps == NULL) || (aux_ps->nb_eq == impl_sc->nb_eq)) {
	  diff_sc->nb_eq = diff_sc->nb_eq - 1;
	  if (pdprec == NULL)
	    diff_sc->egalites = pd->succ;
	  else {
	    pdprec->succ = pd->succ;
	  }
	} else
	  pdprec = pd;
      }
    }
    if (diff_sc->nb_eq != 0) {

      /* Create the basis of the new sc */
      sc_creer_base(diff_sc);

      new_pred = make_predicate((char *) diff_sc);
      if (comm == communication_undefined)
	comm = make_communication(new_pred, predicate_undefined,
				  predicate_undefined);
      else
	communication_broadcast(comm) = new_pred;

      dataflow_communication(df) = comm;
    } else
      sc_rm(diff_sc);
  } else
    sc_rm(diff_sc);
}


/* ======================================================================== */
list
contraintes_to_list(pc)
  Pcontrainte     pc;
{
  Pcontrainte     cpc, spc;
  list            l_pc = NIL;

  for (cpc = pc; cpc != NULL;) {
    spc = cpc;
    cpc = cpc->succ;
    spc->succ = NULL;
    l_pc = gen_nconc(l_pc, CONS(CHUNK, (chunk *) spc, NIL));
  }
  return (l_pc);
}


/* ======================================================================== */
Pcontrainte
list_to_contraintes(l_pc)
  list            l_pc;
{
  Pcontrainte pc = CONTRAINTE_UNDEFINED, cpc = CONTRAINTE_UNDEFINED;
  list        l;

  for (l = l_pc; !ENDP(l); POP(l)) {
    Pcontrainte     spc = (Pcontrainte) CHUNK(CAR(l));
    if (CONTRAINTE_UNDEFINED_P(pc)) {
      pc = spc;
      cpc = pc;
    } else {
      cpc->succ = spc;
      cpc = spc;
    }
  }
  return (pc);
}


/* ======================================================================== */
/*
 * bool compare_eq_occ(eq1, eq2):
 *
 * "eq1" and "eq2" are two Pcontrainte(s). This function returns true if the
 * field "eq_sat" of "eq1" is greater or equal to the one of "eq2". This
 * field represents the number of occurences of the equations in a list of
 * systems.
 */
boolean
compare_eq_occ(eq1, eq2)
  chunk          *eq1, *eq2;
{
  Pcontrainte     c1, c2;
  c1 = (Pcontrainte) eq1;
  c2 = (Pcontrainte) eq2;

  return (*(c1->eq_sat) >= *(c2->eq_sat));
}


/* ======================================================================== */
void
fprint_l_psysteme(fp, l_ps)
  FILE           *fp;
  list            l_ps;
{
  list            l;
  int             i = 1;
  for (l = l_ps; !ENDP(l); POP(l)) {
    fprintf(fp, "Syst %d:\n", i++);
    fprint_psysteme(fp, (Psysteme) CHUNK(CAR(l)));
  }
}


/* ======================================================================== */
/*
 * void count_eq_occ(list l_ps):
 *
 * We have a list of systems "l_ps", each
 * system is a list of equations. This function counts for each equations the
 * number of systems in which it appears. As each equation is represented
 * with the Pcontrainte data structure, the result is put into the "eq_sat"
 * field.
 */
void
count_eq_occ(l_ps)
  list            l_ps;
{
  list            l, ll;
  int             c = 0, cc;

  /* Initialization of eq_sat to 1, each equation appears at least once. */
  for (l = l_ps; !ENDP(l); POP(l)) {
    Psysteme        ps = (Psysteme) CHUNK(CAR(l));
    Pcontrainte     pc = ps->egalites;
    for (; pc != NULL; pc = pc->succ) {
      pc->eq_sat = (int *) malloc(sizeof(int));
      *(pc->eq_sat) = 1;
    }
  }

  /* We count the occurence of each equation in all the others systems */
  for (l = l_ps; !ENDP(l); POP(l)) {
    Psysteme        ps = (Psysteme) CHUNK(CAR(l));
    Pcontrainte     pc = ps->egalites;
    c++;
    for (; pc != NULL; pc = pc->succ) {
      Pvecteur        pv = pc->vecteur;
      cc = 0;
      for (ll = l_ps; !ENDP(ll); POP(ll)) {
	cc++;
	if (cc > c) {
	  Psysteme        pps = (Psysteme) CHUNK(CAR(ll));
	  Pcontrainte     ppc = pps->egalites;
	  for (; ppc != NULL; ppc = ppc->succ) {
	    Pvecteur        ppv = ppc->vecteur;
	    if ((vect_substract(pv, ppv) == VECTEUR_NUL) ||
		(vect_add(pv, ppv) == VECTEUR_NUL)) {
	      (*(ppc->eq_sat))++;
	      (*(pc->eq_sat))++;
	    }
	  }
	}
      }
    }
  }
}


/* ======================================================================== */
/*
 * void   sort_eq_in_systems(list l_ps):
 *
 * We have a list of systems, each
 * system is a list of equations. We sort each list of equations so as to
 * have first the equations that appear the most often in all the systems.
 * 
 * For example, with the two following list of systems L = (S1, S2), S1 = {x1 =
 * 0, x2 = 0} and S2 = {x3 = 0, x2 = 0}, the lists of equations of S1 and S2
 * are reordered as follows: S1 = {x2 = 0, x1 = 0} and S2 = {x2 = 0, x3 = 0}.
 * 
 * Note: We do not sort anything if there is only one system in "l_ps".
 */
void
sort_eq_in_systems(l_ps)
  list            l_ps;
{
  list            l, l_eq, sl_eq;
  Psysteme        crt_ps;

  if (gen_length(l_ps) <= 1)
    return;

  /* We count the occurence of each equation */
  count_eq_occ(l_ps);

  /* We sort the list of equations of each systems using these numbers */
  for (l = l_ps; !ENDP(l); POP(l)) {
    crt_ps = (Psysteme) CHUNK(CAR(l));

    /*
     * Our sorting function (general_merge_sort()) works upon NewGen
     * lists. So we have to transform our chained Pcontrainte(s) into a
     * list of Pcontrainte(s), each Pcontrainte having no successor. After
     * sorting, we transform it back into chained Pcontrainte(s).  */
    l_eq = contraintes_to_list(crt_ps->egalites);
    sl_eq = general_merge_sort(l_eq, compare_eq_occ);
    crt_ps->egalites = list_to_contraintes(sl_eq);
  }
}


/* ======================================================================== */
/*
 * void mapping_on_broadcast(int stmt, Psysteme K)
 *
 * This function constructs some dimension of the placement function, of
 * statement "stmt", by mapping on them some of the broadcast directions given
 * in "K".
 *
 * First, it gets the dimensions already mapped, they constitutes a space A.
 * Second, it looks for the greatest subspace K' of K (the broadcast space)
 * such as A and K' are disjoint.
 * Third, it maps some of the remaining dimensions of the placement function
 * with some of the dimension of K'.
 */
void mapping_on_broadcast(stmt, K)
int stmt;
Psysteme K;
{
  extern plc pfunc;
  extern hash_table StmtToPdim, StmtToMu, StmtToLamb;

  list plcs, ind_l, mu_list, la_list;
  Psysteme A, Kp;
  placement stmt_plc = placement_undefined;
  static_control stct;
  Pbase ind_base;
  int p_dim;

if (get_debug_level() > 5) {
fprintf(stderr, "[mapping_on_broadcast] BEGIN **********\n");
fprintf(stderr, "[mapping_on_broadcast] with stmt %d and K: \n", stmt);
fprint_psysteme(stderr, K);
}

  /* We get the placement function of statement "stmt". */
  for(plcs = plc_placements(pfunc); stmt_plc == placement_undefined; POP(plcs)) {
    placement crt_plc = PLACEMENT(CAR(plcs));
    if(stmt == placement_statement(crt_plc))
      stmt_plc = crt_plc;
  }

  /* Don't do anything if all the placement function dimensions' are already
   * mapped.
   */
  p_dim = (int) hash_get(StmtToPdim, (char *) stmt);

if (get_debug_level() > 5) {
fprintf(stderr, "[mapping_on_broadcast] dim(PLC) = %d, already mapped:\n", p_dim);
fprint_pla_pp_dims(stderr, stmt_plc);
}

  if(gen_length(placement_dims(stmt_plc)) == p_dim)
    return;

  stct = get_stco_from_current_map(adg_number_to_statement(stmt));
  ind_l = static_control_to_indices(stct);
  ind_base = list_to_base(ind_l);

  mu_list = (list) hash_get(StmtToMu, (char *) stmt);

if (get_debug_level() > 5) {
fprintf(stderr, "[mapping_on_broadcast] Mu :");
fprint_entity_list(stderr, mu_list);
fprintf(stderr, "\n");
}

/* First, it gets the dimensions already mapped, they constitutes a space A. */
  A = broadcast_dimensions(stmt_plc, mu_list);

if (get_debug_level() > 5) {
fprintf(stderr, "[mapping_on_broadcast] Space A:\n");
fprint_psysteme(stderr, A);
}

/* Second, it looks for the greatest subspace K' of K (the broadcast space)
 * such as A and K' are disjoint.
 */
{
  Pcontrainte K_dims;
  Psysteme Ps;

  if( (A != SC_EMPTY) && (A->nb_eq != 0) ) {
    Kp = sc_new();

    for(K_dims = K->egalites; K_dims != NULL; K_dims = K_dims->succ) {
      Ps = sc_dup(A);

      sc_add_egalite(Ps, contrainte_dup(K_dims));

      if(vecteurs_libres_p(Ps, ind_base, BASE_NULLE))
	sc_add_egalite(Kp, contrainte_dup(K_dims));

      sc_rm(Ps);
    }
  }
  else
    Kp = K;
}

  sc_rm(A);

if (get_debug_level() > 5) {
fprintf(stderr, "[mapping_on_broadcast] Space K':\n");
fprint_psysteme(stderr, Kp);
}

/* Third, it maps some of the remaining dimensions of the placement function
 * with some of the dimension of K'.
 */
  if(Kp->nb_eq != 0) {
    list new_dims = NIL, mu_l, par_l;
    Pcontrainte Kp_dims;
    int count_dim, i;


    /* To each dimension of the placement function, we have associated a
     * variable mu.
     */
    mu_l = mu_list;
    la_list = (list) hash_get(StmtToLamb, (char *) stmt);

if (get_debug_level() > 5) {
fprintf(stderr, "[mapping_on_broadcast] Mu :");
fprint_entity_list(stderr, mu_l);
fprintf(stderr, "\n");
fprintf(stderr, "[mapping_on_broadcast] Lambda :");
fprint_entity_list(stderr, la_list);
fprintf(stderr, "\n");
}


    Kp_dims = Kp->egalites;
    count_dim = gen_length(placement_dims(stmt_plc));
    for(i = 0; i<count_dim; i++) { POP(mu_l); }
    for(; (Kp_dims != NULL) && (count_dim < p_dim); Kp_dims = Kp_dims->succ) {
      Ppolynome new_pp;
      list lam_l = la_list;

      par_l = gen_append(prgm_parameter_l, NIL);

if (get_debug_level() > 5) {
fprintf(stderr, "[mapping_on_broadcast] Params :");
fprint_entity_list(stderr, par_l);
fprintf(stderr, "\n");
}

      new_pp = make_polynome(1.0, (Variable) ENTITY(CAR(lam_l)), (Value) 1);

      for(i = 0; i<=gen_length(ind_l); i++) { POP(lam_l); }
      for(; !ENDP(lam_l); POP(lam_l), POP(par_l)) {
        polynome_add(&new_pp,
		     polynome_mult(make_polynome(1.0,
					         (Variable) ENTITY(CAR(par_l)),
					         (Value) 1),
				   make_polynome(1.0,
					         (Variable) ENTITY(CAR(lam_l)),
					         (Value) 1)));
      }
      polynome_add(&new_pp,
		   polynome_mult(make_polynome(1.0,
					       (Variable) ENTITY(CAR(mu_l)),
					       (Value) 1),
			         vecteur_to_polynome(Kp_dims->vecteur)));

      new_dims = gen_nconc(new_dims, CONS(CHUNK, (chunk *) new_pp, NIL));

if (get_debug_level() > 5) {
fprintf(stderr, "[mapping_on_broadcast] K' current dim : ");
pu_vect_fprint(stderr, Kp_dims->vecteur);
fprintf(stderr, "\n");
fprintf(stderr, "[mapping_on_broadcast] PLC added dim %d : ", count_dim);
polynome_fprint(stderr, new_pp, pu_variable_name, pu_is_inferior_var);
fprintf(stderr, "\n");
}

      POP(mu_l);
      count_dim++;
    }

    placement_dims(stmt_plc) = gen_nconc(placement_dims(stmt_plc), new_dims);
  }

if (get_debug_level() > 5) {
fprintf(stderr, "[mapping_on_broadcast] Dims now mapped:\n");
fprint_pla_pp_dims(stderr, stmt_plc);

fprintf(stderr, "[mapping_on_broadcast] END **********\n\n");
}

}


/* ======================================================================== */
/*
 * list broadcast_conditions(list lambda, list df_l, list *sigma):
 *
 * processes a list of dataflows "df_l" with the broadcast conditions. The
 * dataflows not processed are returned.
 * 
 * The goal is to accept conditions upon variables (in "lambda"). These
 * conditions are the broadcast conditiions. There is one condition for each
 * edge with a broadcast. Each condition is a system of equations. We'll say
 * that a condition is accepted iff at least one equation is accepted. An
 * equation is accepted iff it does not make trivial any of the prototypes.
 * For triviality, see is_not_trivial_p().
 * 
 * An accepted equation can be represented as a substitution that expresses
 * one variable function of the others.
 */
list
broadcast_conditions(lambda, df_l, sigma)
  list            lambda, df_l, *sigma;
{
  extern hash_table StmtToLamb, DtfToSink, StmtToPdim;

  list            l, rem_df_l = NIL, acc_df_l = NIL, l_M_local = NIL, ml, dl;

  for (l = df_l; !ENDP(l); POP(l)) {
    dataflow        df = DATAFLOW(CAR(l));
    communication   com = dataflow_communication(df);
    predicate       bp = predicate_undefined;

    if (get_debug_level() > 4) {
      fprintf(stderr, "[broadcast_conditions] \t\t\tDF: ");
      fprint_dataflow(stderr, 0, df);
      fprintf(stderr, "\n");
    }
    if (is_broadcast_p(df)) {
      bp = communication_broadcast(com);
    }
    if (get_debug_level() > 4) {
      fprintf(stderr, "[broadcast_conditions] \t\t\tComm pred: ");
      if (bp == predicate_undefined)
	fprintf(stderr, "predicate_undefined");
      else
	fprint_pred(stderr, bp);
      fprintf(stderr, "\n");
    }
    if (bp == predicate_undefined)
      rem_df_l = gen_nconc(rem_df_l, CONS(DATAFLOW, df, NIL));
    else {
      list            ind_l, par_l, proto_lambda, l_bdir, bl;
      Psysteme        ps_pdir = (Psysteme) predicate_system(bp), ps_bdir;
      Pcontrainte     pc, prec_pc, new_pc;
      int             stmt = (int) hash_get(DtfToSink, (char *) df),
      n, m1, m2, i,
                      j, k, p_dim;
      static_control  stct = get_stco_from_current_map(adg_number_to_statement(stmt));
      matrice         A, B, inv_A, R, Rt, Bz;

      k = ps_pdir->nb_eq;
      ind_l = static_control_to_indices(stct);
      par_l = gen_append(prgm_parameter_l, NIL);
      m1 = gen_length(ind_l);
      m2 = gen_length(par_l) + 1;	/* add 1, because of TCST */

      p_dim = (int) hash_get(StmtToPdim, (char *) stmt);
      l_bdir = stmt_bdt_directions(stmt, ind_l, par_l);

      /* We remove broadcast vectors contained in the time space */
      for (bl = l_bdir; !ENDP(bl); POP(bl)) {
	ps_bdir = (Psysteme) CHUNK(CAR(bl));
	if (ps_bdir->nb_eq > 0) {
	  prec_pc = NULL;
	  for (pc = ps_pdir->egalites; pc != NULL; pc = pc->succ) {
	    Psysteme        ps_aux = sc_dup(ps_bdir);
	    sc_add_egalite(ps_aux, contrainte_make(pc->vecteur));

	    if (vecteurs_libres_p(ps_aux, list_to_base(ind_l),
				  list_to_base(par_l))) {
	      prec_pc = pc;
	    } else {
	      (ps_pdir->nb_eq)--;
	      if (prec_pc == NULL) {
		ps_pdir->egalites = pc->succ;
	      } else {
		prec_pc->succ = pc->succ;
	      }
	    }
	  }

	  if (get_debug_level() > 4) {
	    fprintf(stderr, "[broadcast_conditions] \t\t\tk before elim bdt = %d\n", k);
	  }
	  k = ps_pdir->nb_eq;
	}
      }

      if (get_debug_level() > 4) {
	fprintf(stderr, "[broadcast_conditions] \t\t\tk = %d\n", k);
      }
      if (k == m1) {
	/* No condition: broadcast on all the processors */
      } else if (k == 0) {
	/* No condition yet, we'll try to zero out its distance */
	rem_df_l = gen_nconc(rem_df_l, CONS(DATAFLOW, df, NIL));
      } else {
	Psysteme        M_local, ps_eps;

	acc_df_l = gen_nconc(acc_df_l, CONS(DATAFLOW, df, NIL));

	ps_eps = completer_base(ps_pdir, ind_l, par_l);

	if (get_debug_level() > 4) {
	  fprintf(stderr, "[broadcast_conditions] \t\t\tFull sys\n");
	  fprint_psysteme(stderr, ps_eps);
	  fprintf(stderr, "\n");
	}
	n = ps_eps->nb_eq;
	if (m1 != n)
	  user_error("broadcast_conditions", "m1 should be equal to n\n");

	A = matrice_new(n, n);
	B = matrice_new(n, m2);
	contraintes_with_sym_cst_to_matrices(ps_eps->egalites,
					     list_to_base(ind_l),
				       list_to_base(par_l), A, B, n, n, m2);

	inv_A = matrice_new(n, n);
	matrice_general_inversion(A, inv_A, n);

	/* R is a sub-matrix of inv_A containing the (n-k) last columns. */
	R = matrice_new(n, n - k);
	for (i = 1; i <= n; i++)
	  for (j = 1; j <= (n - k); j++)
	    ACCESS(R, n, i, j) = ACCESS(inv_A, n, i, j + k);
	R[0] = 1;

	Rt = matrice_new((n - k), n);
	matrice_transpose(R, Rt, n, (n - k));

	proto_lambda = get_stmt_index_coeff(stmt, StmtToLamb);

	Bz = matrice_new((n - k), 1);
	matrice_nulle(Bz, (n - k), 1);
	pu_matrices_to_contraintes(&new_pc, list_to_base(proto_lambda),
				   Rt, Bz, (n - k), n);

	matrice_free(A);
	matrice_free(B);
	matrice_free(inv_A);
	matrice_free(R);
	matrice_free(Rt);
	matrice_free(Bz);

	M_local = sc_make(new_pc, NULL);

	if (get_debug_level() > 4) {
	  fprintf(stderr, "[broadcast_conditions] \t\t\tM_local:\n");
	  fprint_psysteme(stderr, M_local);
	  fprintf(stderr, "\n");
	}
	l_M_local = gen_nconc(l_M_local, CONS(CHUNK, (chunk *) M_local, NIL));
      }
    }
  }
  /*
   * We have a list of systems, each system is a list of equations. We sort
   * each list of equations so as to have first the equations that appear the
   * most often in all the systems.
   * 
   * Thus, these equations are treated first.
   */
  sort_eq_in_systems(l_M_local);

  for (dl = acc_df_l, ml = l_M_local; !ENDP(dl); POP(dl), POP(ml)) {
    dataflow        df = DATAFLOW(CAR(dl));
    Psysteme        M_local = (Psysteme) CHUNK(CAR(ml));

    if (get_debug_level() > 4) {
      fprintf(stderr, "[broadcast_conditions] \t\t\tSIGMA before:\n");
      fprint_vvs(stderr, *sigma);
      fprintf(stderr, "\n");
    }
    /* If one condition is not are satisfied, we try to cut this edge AND we
     * try to ensure that a subspace of the distribution space will coincide
     * with a subspace of the broadcast space.
     */
    if (!solve_system_by_succ_elim(M_local, sigma)) {
      int stmt = (int) hash_get(DtfToSink, (char *) df);

      rem_df_l = gen_nconc(rem_df_l, CONS(DATAFLOW, df, NIL));
      mapping_on_broadcast(stmt,
			   (Psysteme) predicate_system
			   (communication_broadcast
			    (dataflow_communication(df))));
    }

    if (get_debug_level() > 4) {
      fprintf(stderr, "[broadcast_conditions] \t\t\tSIGMA after:\n");
      fprint_vvs(stderr, *sigma);
      fprintf(stderr, "\n");
    }
  }

  return (rem_df_l);
}


/* ========================================================================= */
list
stmt_bdt_directions(stmt, ind_l, par_l)
  int             stmt;
  list            ind_l, par_l;
{
  extern bdt      the_bdt;
  list            l_dirs = NIL;

  if (the_bdt != bdt_undefined) {
    list            bl;
    bool         sc_found = false;
    for (bl = bdt_schedules(the_bdt); (bl != NIL) && (!sc_found); bl = CDR(bl)) {
      schedule        sc = SCHEDULE(CAR(bl));
      if (schedule_statement(sc) == stmt) {
	list            dl;
	Psysteme        ps = sc_new();
	for (dl = schedule_dims(sc); dl != NIL; dl = CDR(dl)) {
	  expression      e = EXPRESSION(CAR(dl));
	  if (!expression_constant_p(e)) {
	    list            l;
	    bool         ind_not_null;
	    Pvecteur        pv;

	    NORMALIZE_EXPRESSION(e);
	    pv = (Pvecteur) normalized_linear(expression_normalized(e));

	    ind_not_null = false;
	    for (l = ind_l; (!ENDP(l)) && (!ind_not_null); POP(l)) {
	      entity          var = ENTITY(CAR(l));
	      if (vect_coeff((Variable) var, pv) != 0)
		ind_not_null = true;
	    }
	    if (ind_not_null) {
	      Psysteme        aux_ps = sc_dup(ps);
	      sc_add_egalite(aux_ps, contrainte_make(pv));

	      if (vecteurs_libres_p(aux_ps, list_to_base(ind_l), list_to_base(par_l))) {
		ps = aux_ps;
	      } else {
	      }
	    }
	  }
	}
	sc_creer_base(ps);
	l_dirs = gen_nconc(l_dirs, CONS(CHUNK, (chunk *) ps, NIL));
      }
    }
  }
  return (l_dirs);
}
