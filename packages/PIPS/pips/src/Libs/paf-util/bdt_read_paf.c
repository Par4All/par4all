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
/* Name     : bdt_read_paf.c
 * Package  : paf-util
 * Author   : Alexis Platonoff
 * Date     : april 1993
 * Historic :
 * - 16 july 93, changes in paf_ri, AP
 * - 2 august 93, moved from (package) scheduling to paf-util, AP
 * - 10 nov 93: add of reorganize_bdt() and same_predicate_p(), AP
 *
 * Documents:
 * Comments :
 * These functions are used to store a timing function (BDT) in a Newgen
 * structure from the reading of a file generate by the PAF parallelizer:
 * ".bdt" file, which contains the BDT functions.
 *
 * Each BDT function is associated with an instruction and a predicate. The
 * Newgen structure is called "bdt" which is a list of BDT function of type
 * "schedule". These two Newgen structures are defined in the file paf_ri.f.tex
 * (see comments on them in this file). We also use the library of PIPS and
 * its RI Newgen structure.
 *
 * The ".bdt" file is read with Yacc and Lex programs (parse.y and scan.l).
 * The Newgen structure "bdt" is update during the parsing of the file.
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

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
#include "matrix.h"

#include "genC.h"

#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "constants.h"
#include "misc.h"
#include "paf_ri.h"
typedef dfg_arc_label arc_label;
typedef dfg_vertex_label vertex_label;
#include "graph.h"
#include "paf-util.h"

#define POSITIVE 1
#define NEGATIVE 0
#define INS_NAME_LENGTH 4
#define DOT "."
#define BDT_STRING "bdt"


/* Static global variables */
static int	crt_ins;	/* Current stmt (an integer) */
static list	pred_l,		/* Current list of predicates */
		lin_exp_l;	/* Current list of linear expressions */
static expression crt_exp;	/* Current expression */


/* Global variables */

/* This global variable is the current BDT being computed. Its type is defined
 * in paf_ri.h
 */
bdt	base;

/*============================================================================*/
/* bdt bdt_read_paf(char *s) : computes the BDT of the PAF program name given
 * in argument and returns it.
 *
 * bdtyyparse() do the parsing over the ".bdt" file and put the timing function
 * into the global variable "base" which is returned.
 *
 */
bdt bdt_read_paf(s)
char *s;
{
 extern bdt base;

 FILE *bdt_file;
 char *bdt_file_name;

 bdt_file_name = strdup(concatenate(s, DOT, BDT_STRING, (char *) NULL));

 if( (bdt_file = fopen(bdt_file_name, "r")) == NULL)
   {
     fprintf(stderr, "Cannot open file %s\n", bdt_file_name);
     exit(1);
   }

#if defined(HAS_BDTYY)

 bdtyyin = bdt_file;
 (void) bdtyyparse();

#else

 pips_internal_error("not bdtyy{in,parse} compiled in (HAS_BDTYY undef)");

#endif

 fclose(bdt_file);

 reorganize_bdt(base);

 return(base);
}

/*============================================================================*/
bool same_predicate_p(p1, p2)
predicate p1, p2;
{

  if( (p1 == predicate_undefined) && (p2 == predicate_undefined) )
    return(true);
  else if(p1 == predicate_undefined)
    return(false);
  else if(p2 == predicate_undefined)
    return(false);

  if( (p1 == NULL) && (p2 == NULL) )
    return(true);
  else if(p1 == NULL)
    return(false);
  else if(p2 == NULL)
    return(false);

  /* If the predicate are defined, we consider them as always different */
  return(false);
}


/*============================================================================*/
/* void reorganize_bdt(bdt base):
 */
void reorganize_bdt(base)
bdt base;
{
  list sch_l, new_sch_l = NIL, l;

  for(sch_l = bdt_schedules(base); !ENDP(sch_l); POP(sch_l)) {
    schedule sch = SCHEDULE(CAR(sch_l));
    int stmt = schedule_statement(sch);
    predicate pred = schedule_predicate(sch);
    bool to_add = true;

    for(l = new_sch_l; (!ENDP(l)) && to_add; POP(l)) {
      schedule nsch =  SCHEDULE(CAR(l));
      int nstmt = schedule_statement(nsch);
      predicate npred = schedule_predicate(nsch);

      if(stmt == nstmt) {
	if(same_predicate_p(pred, npred)) {
	  expression nexp = EXPRESSION(CAR(schedule_dims(sch)));
	  schedule_dims(nsch) = gen_nconc(schedule_dims(nsch),
					  CONS(EXPRESSION, nexp, NIL));
	  to_add = false;
	}
      }
    }
    if(to_add)
      new_sch_l = gen_nconc(new_sch_l, CONS(SCHEDULE, sch, NIL));
  }
  bdt_schedules(base) = new_sch_l;
}


/*============================================================================*/
/* void bdt_init_new_base() : initializes the computation of the BDT, i.e. the
 * creation of the BDT in the "base" variable.
 */
void bdt_init_new_base()
{
 extern bdt base;

 base = make_bdt(NIL);
}

/*============================================================================*/
/* void bdt_init_new_ins(char *s_ins): initializes the computation of the
 * current statement for which we are now parsing its BDT. This statement is
 * represented by its ordering contained in its name "s_ins".
 * Also, we initialize "lin_exp_l" used for the parsing of the lisp
 * expressions and "pred_l" used for the parsing of the predicates.
 */
void bdt_init_new_ins(s_ins)
char * s_ins;
{
 extern int crt_ins;
 extern list pred_l, lin_exp_l;

 /* In PAF, a statement name is a string "ins_#", where "#" is the number
  * associated with the statement. We get this number.
  */
 crt_ins = atoi(strdup(s_ins + INS_NAME_LENGTH));

 pred_l = NIL;
 lin_exp_l = NIL;
}

/*============================================================================*/
/* void bdt_new_shedule(char *s_func): the parser has found all the predicates
 * of a schedule. We create this new schedule and put it in "base". This
 * predicate is formed with the list of expressions of "pred_l". The function
 * expressions_to_predicate() translates this list of expressions into a
 * predicate. The schedule statement number is the current one (crt_ins"). The
 * schedule has one dimension, the corresponding expression is found in
 * "crt_exp".
 */
void bdt_new_shedule(string s_func __attribute__ ((unused)))
{
    extern int crt_ins;
    extern expression crt_exp;
    extern list pred_l, lin_exp_l;

    bdt_schedules(base) = CONS(SCHEDULE,
			       make_schedule(crt_ins,
					     expressions_to_predicate(pred_l),
					     CONS(EXPRESSION, crt_exp, NIL)),
			       bdt_schedules(base));
    
    lin_exp_l = NIL;
    crt_exp = expression_undefined;
}

/*============================================================================*/
/* void bdt_save_pred(int option): computes one expression of the predicate.
 * Each expression is used twice ; indeed, an expression may be greater or
 * equal than zero (>=) and smaller than zero (<). "option" says in which case
 * we are: POSITIVE indicates that the predicate is >=, with NEGATIVE it is <.
 * However, the C3 library always represents its inequalities with <=. So, the
 * inequality "A >= 0" becomes "-A <= 0" and "A < 0" becomes "A + 1 <= 0".
 *
 * This function updates the global list "pred_l" that contains the current
 * list of predicates. When a new predicate expression is parsed, the POSITIVE
 * is always considered first (that is why only in that case we use "crt_exp").
 * When the NEGATICE case is considered, the corresponding expression (used in
 * the POSITIVE case) is the first expression of the list "pred_l". So, we only
 * have to replace this expression by it equivalent for the NEGATIVE case (that
 * is why the expression is multiplied by -1).
 */
void bdt_save_pred(option)
int option;
{
 extern list pred_l, lin_exp_l;
 extern expression crt_exp;

 expression aux_pred;

 if(option == POSITIVE)
   {
    if(crt_exp == expression_undefined)
       pips_internal_error("current expression is undefined");

    pred_l = CONS(EXPRESSION, negate_expression(crt_exp), pred_l);
    crt_exp = expression_undefined;
   }
 else
 /* option == NEGATIVE */
   {
    aux_pred = make_op_exp(PLUS_OPERATOR_NAME,
                           negate_expression(EXPRESSION(CAR(pred_l))),
                           int_to_expression(1));

    pred_l = CONS(EXPRESSION, aux_pred, CDR(pred_l));
   }

/* Initialization of global variables */
 lin_exp_l = NIL;
}

/*============================================================================*/
/* void bdt_elim_last_pred(): When POSITIVE and NEGATIVE cases of one predicate
 * have been completed, we eliminate the corresponding expression which is the
 * first one of the list "pred_l".
 */
void bdt_elim_last_pred()
{
 extern list pred_l;

 pred_l = CDR(pred_l);
}

/*============================================================================*/
/* void bdt_save_int(int i): The parser has found an integer as a part of a
 * lisp expression. We save it in our global variable "lin_exp_l".
 *
 * If "lin_exp_l" is empty, then this integer becomes the current expression.
 * If not, it becomes an argument of the first lisp expression of "lin_exp_l".
 */
void bdt_save_int(i)
int i;
{
 extern list lin_exp_l;
 extern expression crt_exp;
 expression aux_exp;

 aux_exp = int_to_expression(i);

 if(lin_exp_l == NIL)
    crt_exp = aux_exp;
 else
   {
    lisp_expression crt_le = LISP_EXPRESSION(CAR(lin_exp_l));
    lisp_expression_args(crt_le) = gen_nconc(lisp_expression_args(crt_le),
                                             CONS(EXPRESSION, aux_exp, NIL));
   }
}

/*============================================================================*/
/* void bdt_save_id(string s): The parser has found a variable as a part of a
 * lisp expression. We save it in our global variable "lin_exp_l".
 *
 * If "lin_exp_l" is empty, then this variable becomes the current expression.
 * If not, it becomes an argument of the first lisp expression of "lin_exp_l".
 */
void bdt_save_id(s)
string s;
{
 extern list lin_exp_l;
 extern expression crt_exp;
 expression aux_exp;

 aux_exp = make_id_expression(s);

 if(lin_exp_l == NIL)
    crt_exp = aux_exp;
 else
   {
    lisp_expression crt_le = LISP_EXPRESSION(CAR(lin_exp_l));
    lisp_expression_args(crt_le) = gen_nconc(lisp_expression_args(crt_le),
                                             CONS(EXPRESSION, aux_exp, NIL));
   }
}

/*============================================================================*/
/* void bdt_init_op_exp(string op_name): initializes a new lisp expression with
 * the operation "op_name". This expression is put at the beginning of
 * "lin_exp_l", it is the expression the parser is currently reading.
 *
 * If "op_name" is the string "0" then the operator used is "crt_op_name", else
 * the operator name is contained in "op_name".
 */
void bdt_init_op_exp(op_name)
string op_name;
{
 extern list lin_exp_l;

 lin_exp_l = CONS(LISP_EXPRESSION, make_lisp_expression(op_name, NIL), lin_exp_l);
}

/*============================================================================*/
/* void bdt_save_exp(): the parser has completed the reading of one lisp
 * expression, this is the first lisp expression of "lin_exp_l". We extract it
 * from this list and translate it into a Pips expression. If there is no other
 * lisp expression in "lin_exp_l", then this expression becomes the current
 * expression, else it becomes an argument of the next lisp expression which is
 * now the first object of "lin_exp_l".
 */
void bdt_save_exp()
{
 extern expression crt_exp;

 expression aux_exp;
 lisp_expression aux_le;

 aux_le = LISP_EXPRESSION(CAR(lin_exp_l));
 aux_exp = lisp_exp_to_ri_exp(aux_le);

 lin_exp_l = CDR(lin_exp_l);

 if(lin_exp_l == NIL)
   crt_exp = aux_exp;
 else
   {
    lisp_expression crt_le = LISP_EXPRESSION(CAR(lin_exp_l));
    lisp_expression_args(crt_le) = gen_nconc(lisp_expression_args(crt_le),
                                             CONS(EXPRESSION, aux_exp, NIL));
   }
}


