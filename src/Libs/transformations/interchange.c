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
/* functions to perform loop interchange */
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif

#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>


#include "boolean.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"
#include "matrice.h"
#include "matrix.h"
#include "sparse_sc.h"

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "misc.h"
#include "text.h"

#include "ri-util.h"
#include "effects-util.h"
#include "conversion.h"
#include "transformations.h"

/* statement gener_DOSEQ(cons *lls,Pvecteur pvg[], Pbase base_oldindex,
 * Pbase base_newindex)
 * generation of loops interchange code for the nested loops  (cons *lls).
 * the new nested loops will be ;
 *  DOSEQ Ip = ...
 *    DOSEQ Jp = ...
 *      DOSEQ Kp = ...
 *        ...
 *  ENDDO
 */
static statement
gener_DOSEQ(
    list lls,
    Pvecteur *pvg,
    Pbase base_oldindex,
    Pbase base_newindex,
    Psysteme sc_newbase)
{
    statement state_lhyp = statement_undefined;
    instruction instr_lhyp = instruction_undefined;
    loop l_old = loop_undefined;
    loop l_hyp = loop_undefined;
    range rl = range_undefined;
    expression lower = expression_undefined;
    expression upper = expression_undefined;
    statement bl = statement_undefined;
    statement s_loop = statement_undefined;
    Pbase pb = BASE_NULLE;

    bl=loop_body(instruction_loop(statement_instruction(STATEMENT(CAR(lls)))));
    statement_newbase(bl,pvg,base_oldindex);
    /* make the parallel loops from inner loop to upper loop*/

    for(pb=base_reversal(base_newindex);lls!=NIL; lls=CDR(lls)) {
	/* traitement of current loop */
	s_loop = STATEMENT(CAR(lls));
	l_old = instruction_loop(statement_instruction(s_loop));

	/*new bounds de new index correspondant a old index de cet loop*/
	make_bound_expression(pb->var,base_newindex,sc_newbase,&lower,&upper);
	rl = make_range(lower,upper,int_to_expression(1));


	if (CDR(lls)!=NULL) {
	    /* make  the inner sequential loops
	       they will be the inner parallel loops after
	       integration the phase  of parallelization*/
	    l_hyp = make_loop(pb->var,
			      rl,
			      bl,
			      loop_label(l_old),
			      make_execution(is_execution_sequential,UU),
			      loop_locals(l_old));
	    bl = makeloopbody(l_hyp,s_loop, true);
	    pb=pb->succ;
	}
    }

    /* Make the last outer loop which is sequential and can be labelled */

    l_hyp = make_loop(pb->var,rl,bl,loop_label(l_old),
		      make_execution(is_execution_sequential,UU),
		      loop_locals(l_old));
    instr_lhyp = make_instruction(is_instruction_loop,l_hyp);
    state_lhyp = make_statement(statement_label(s_loop),
				statement_number(s_loop),
				statement_ordering(s_loop),
				statement_comments(s_loop),
				instr_lhyp,NIL,NULL,
				statement_extensions(s_loop), make_synchronization_none());
    return(state_lhyp);
}

/* Implementation of the loops interchange method. The outer most loop
 * is exchanged with the inner most one. The legality of the
 * transformation is not checked against the data dependencies nor
 * against the control flow nor against the local declarations.
 *
 * "lls" is the list of nested loops. The user loop selection is used to
 * build lls, a list of loop statements. The innermost loop is first
 * in lls and the outermost loop is last in lls.
 *
 * If the innermost and outermost loops have the same loop label, the
 * code generation is OK.
 *
 * If neither the innermost nor the outermost loop has a loop label,
 * the code generation is OK.
 *
 * If both loops have different non-empty labels, the innermost label
 * must be preserved in case it is the target of a goto. And the
 * outermost loop label, if any, has to be moved inside the nested
 * loop body. So the initial innermost loop must be made labelless.
 *
 * Note: the initial outermost loop label cannot be the target of a
 * goto because the loop nest would not be a loop nest after
 * controlization.
 *
 * FI: should be replaced by interchange_two_loops(lls, 1, n)
 * FI: used to be called "interchange()"
 */

statement interchange_inner_outermost_loops(list lls,
                                __attribute__((unused)) bool (*unused)(statement))
{
  int n = gen_length(lls);
  /* might be 0 and n-1*/
  statement s = interchange_two_loops(lls, 1, n);
  return s;
}

#if 0
static statement old_interchange_inner_outermost_loops(list lls,
                                __attribute__((unused)) bool (*unused)(statement))
{
  Psysteme sci;			/* sc initial */
  Psysteme scn;			/* sc nouveau */
  Psysteme sc_row_echelon;
  Psysteme sc_newbase;
  Pbase base_oldindex = NULL;
  Pbase base_newindex = NULL;
  matrice A;
  matrice G;
  matrice AG;
  int n ;				/* number of indices */
  int m ;				/* number of constraints */
  statement s_lhyp;
  Pvecteur *pvg;
  Pbase pb;
  expression lower, upper;
  Pvecteur pv1, pv2;
  loop l;
  /* Initially, the outermost loop. Stays the outermost loop.  */
  loop oml =
    instruction_loop(statement_instruction(STATEMENT(CAR(gen_last(lls)))));
  /* The initial outermost loop label */
  entity omll = loop_label(oml);
  /* Initially, the innermost loop. Stays the innermost loop. */
  loop iml =
    instruction_loop(statement_instruction(STATEMENT(CAR(lls))));
  /* The initial innermost loop label */
  entity imll = loop_label(iml);

  debug_on("LOOP_INTERCHANGE_DEBUG_LEVEL");
  pips_debug(8,"begin:\n");

  /* make the  system "sc" of constraints of iteration space */
  sci = loop_iteration_domaine_to_sc(lls, &base_oldindex);

  /* create the  matrix A of coefficients of  index in (Psysteme)sci */
  n = base_dimension(base_oldindex);
  m = sci->nb_ineq;
  A = matrice_new(m,n);
  sys_matrice_index(sci, base_oldindex, A, n, m);

  /* computation of the matrix of basis change  for loops interchange */
  G = matrice_new(n,n);
  matrice_identite(G,n,0);
  matrice_swap_columns(G,n,n,1,n);

  /* the new matrice of constraints AG = A * G */
  AG = matrice_new(m,n);
  matrice_multiply(A,G,AG,m,n,n);

  /* create the new system of constraintes (Psysteme scn) with
     AG and sci */
  scn = sc_dup(sci);
  matrice_index_sys(scn,base_oldindex,AG,n,m);

  /* computation of the new iteration space in the new basis G */
  sc_row_echelon = new_loop_bound(scn,base_oldindex);

  /* change of basis for index */
  change_of_base_index(base_oldindex,&base_newindex);
  sc_newbase=sc_change_baseindex(sc_dup(sc_row_echelon),
				 base_oldindex,base_newindex);

  /* generation of interchange  code */
  /*  generation of bounds */
  for (pb=base_newindex; pb!=NULL; pb=pb->succ) {
    make_bound_expression(pb->var,base_newindex,sc_newbase,&lower,&upper);
  }

  /* loop body generation */
  pvg = (Pvecteur *)malloc((unsigned)n*sizeof(Svecteur));
  scanning_base_to_vect(G,n,base_newindex,pvg);
  pv1 = sc_row_echelon->inegalites->succ->vecteur;
  pv2 = vect_change_base(pv1,base_oldindex,pvg);

  l = instruction_loop(statement_instruction(STATEMENT(CAR(lls))));
  lower = range_upper(loop_range(l));
  upper= expression_to_expression_newbase(lower, pvg, base_oldindex);


  s_lhyp = gener_DOSEQ(lls,pvg,base_oldindex,base_newindex,sc_newbase);

  /* Fix Fortran loop labels. Should this be made part of gener_DOSEQ? */
  if(!c_language_module_p(get_current_module_entity())
     && (!entity_empty_label_p(omll) || gen_length(lls)>2)
     && omll!=imll) {
    /* A corresponding continue should be added to the loop nest
       body, the body of the initial innermost loop , iml */
    statement nlb = loop_body(iml);
    /* The initial continue statements are assumed lost when lls is
       built and transformed. */
    list lll = CONS(ENTITY, imll, NIL); // loop label list
    FOREACH(STATEMENT, ls, CDR(lls)) {
      entity ll = loop_label(statement_loop(ls));
      if(!entity_empty_label_p(ll) && !gen_in_list_p(ll, lll)) {
	statement cs = make_continue_statement(ll);
	insert_statement(nlb, cs, false);
	lll = CONS(ENTITY, ll, lll);
      }
    }
    /* get rid of the innermost loop label? */
    //loop_label(iml) = entity_empty_label();
  }

  pips_debug(8,"end\n");
  debug_off();

  return s_lhyp;
}
#endif

/* See comments for interchange_inner_outermost_loops(). Continue
   statements for loop labels are not fixed. */
statement interchange_two_loops(list lls, int n1, int n2)
{
  Psysteme sci;			/* sc initial */
  Psysteme scn;			/* sc nouveau */
  Psysteme sc_row_echelon;
  Psysteme sc_newbase;
  Pbase base_oldindex = NULL;
  Pbase base_newindex = NULL;
  matrice A;
  matrice G;
  matrice AG;
  int n = gen_length(lls); /* number of loops and loop indices */
  int m ;				/* number of constraints */
  statement s_lhyp;
  Pvecteur *pvg;
  Pbase pb;
  expression lower, upper;
  Pvecteur pv1 ;
  loop l;
  statement s1 = STATEMENT(gen_nth(n1-1,lls));
  statement s2 = STATEMENT(gen_nth(n2-1,lls));
  loop l1 = statement_loop(s1);
  loop l2 = statement_loop(s2);
  entity ll1 = loop_label(l1);
  entity ll2 = loop_label(l2);
  //execution e1 = copy_execution(loop_execution(l1));
  //execution e2 = copy_execution(loop_execution(l2));
  execution e[n];

  debug_on("LOOP_INTERCHANGE_DEBUG_LEVEL");
  pips_debug(8,"\n begin: n1=%d, n2=%d\n", n1, n2);

  /* Preserve the parallelism information */
  int ln = 0;
  FOREACH(STATEMENT, ls, lls) {
    loop l = statement_loop(ls);
    e[ln] = copy_execution(loop_execution(l));
    ln++;
  }

  /* Build the  system "sci" with the constraints of the iteration
     space defined by lls */
  sci = loop_iteration_domaine_to_sc(lls, &base_oldindex);

  /* create the matrix A of coefficients for the loop indices in
     (Psysteme)sci */
  n = base_dimension(base_oldindex);
  m = sci->nb_ineq;
  A = matrice_new(m,n);
  sys_matrice_index(sci, base_oldindex, A, n, m);

  /* Computate of the unimodular matrix for loop interchange */
  G = matrice_new(n,n);
  matrice_identite(G,n,0);
  matrice_swap_columns(G,n,n, n1, n2);

  /* the new matrice of constraints AG = A * G */
  AG = matrice_new(m,n);
  matrice_multiply(A,G,AG,m,n,n);

  /* create the new system of constraintes (Psysteme scn) with AG
     and sci */
  scn = sc_dup(sci);
  matrice_index_sys(scn,base_oldindex,AG,n,m);

  /* computation of the new iteration space in the new basis G */
  sc_row_echelon = new_loop_bound(scn,base_oldindex);

  /* changeof basis for index */
  change_of_base_index(base_oldindex,&base_newindex);
  sc_newbase =
    sc_change_baseindex(sc_dup(sc_row_echelon),
			base_oldindex,base_newindex);

  /* generation of interchange  code */
  /*  generation of bounds */
  for (pb=base_newindex; pb!=NULL; pb=pb->succ) {
    make_bound_expression(pb->var,base_newindex,sc_newbase,&lower,&upper);
  }

  /* loop body generation */
  pvg = (Pvecteur *)malloc((unsigned)n*sizeof(Svecteur));
  scanning_base_to_vect(G,n,base_newindex,pvg);
  pv1 = sc_row_echelon->inegalites->succ->vecteur;
  (void)vect_change_base(pv1,base_oldindex,pvg);

  l = instruction_loop(statement_instruction(STATEMENT(CAR(lls))));
  lower = range_upper(loop_range(l));
  upper= expression_to_expression_newbase(lower, pvg, base_oldindex);


  s_lhyp = gener_DOSEQ(lls,pvg,base_oldindex,base_newindex,sc_newbase);

  /* Fix Fortran loop labels. Should this be made part of gener_DOSEQ? */
  if(!c_language_module_p(get_current_module_entity())
     && (!entity_empty_label_p(ll2) || gen_length(lls)>2)
     && ll1!=ll2) {
    /* A corresponding continue should be added to the loop nest
       body, the body of the initial innermost loop , iml */
    list nlsl = statement_to_loop_statement_list(s_lhyp);
    loop iml = statement_loop(STATEMENT(CAR(gen_last(nlsl))));
    statement nlb = loop_body(iml);
    /* The initial continue statements are assumed lost when lls is
       built and transformed, except for the innermost loop. */
    entity imll = loop_label(iml);
    list lll =  CONS(ENTITY, imll, NIL); // loop label list
    nlsl = gen_nreverse(nlsl);
    FOREACH(STATEMENT, ls, nlsl) {
      entity ll = loop_label(statement_loop(ls));
      if(!entity_empty_label_p(ll) && !gen_in_list_p(ll, lll)) {
	statement cs = make_continue_statement(ll);
	insert_statement(nlb, cs, false);
	lll = CONS(ENTITY, ll, lll);
      }
    }
    gen_free_list(nlsl);
    gen_free_list(lll);
    /* get rid of the innermost loop label? */
    //loop_label(iml) = entity_empty_label();
  }

  /* Add the parallelism information */
  ln = 0;
  list nlsl = statement_to_loop_statement_list(s_lhyp);
  pips_assert("The loop list retrieved has the expected lenght",
	      ((int) gen_length(nlsl))==n);
  FOREACH(STATEMENT, ls, nlsl) {
    loop l = statement_loop(ls);
    free_execution(loop_execution(l));
    if(ln==n1-1)
      loop_execution(l) = e[n2-1];
    else if(ln==n2-1)
      loop_execution(l) = e[n1-1];
    else
      loop_execution(l) = e[ln];
    ln++;
  }

  pips_assert("Statement s_lhyp is consistent",
	      statement_consistent_p(s_lhyp));

  pips_debug(8, "end\n");
  debug_off();

  return s_lhyp;
}
