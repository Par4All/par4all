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
#include <stdio.h>
#include <setjmp.h>
#include <string.h>
#include <stdlib.h>

#include "genC.h"
#include "ri.h"
#include "constants.h"
#include "ri-util.h"
#include "misc.h"

/* C3 includes          */
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
#include "matrix.h"


/* Pips includes        */

#include "complexity_ri.h"
#include "database.h"
#include "dg.h"
#include "parser_private.h"
#include "property.h"
#include "reduction.h"
#include "tiling.h"


#include "text.h"
#include "text-util.h"
#include "graph.h"
#include "paf_ri.h"
#include "paf-util.h"
#include "pipsdbm.h"
#include "resources.h"
#include "array_dfg.h"
#include "pip.h"
#include "static_controlize.h"
#include "scheduling.h"

/* Local defines */
typedef dfg_arc_label arc_label;
typedef dfg_vertex_label vertex_label;


/*======================================================================*/
/* entity create_named_entity(name): create a scalar integer entity named
 * "name".
 *
 * AC 93/11/05 */
entity create_named_entity(name)
char    *name;
{
    entity  ent;
    basic   b;
    string  full_name;

    full_name = concatenate(BDT_MODULE_NAME, MODULE_SEP_STRING, name,
			    NULL);

    if((ent = gen_find_tabulated(full_name, entity_domain))
       == entity_undefined) {
	ent = make_entity(strdup(full_name),
			  type_undefined,
			  make_storage(is_storage_rom, UU),
			  make_value(is_value_unknown, UU));
	b = make_basic(is_basic_int, 4);
	entity_type(ent) = (type) MakeTypeVariable(b, NIL);
    }

    return(ent);
}

/*============================================================================*/
/* void fprint_bdt_with_stat(fp, obj): prints the bdt "obj" with the 
 * corresponding statement.
 *
 * AC 94/02/12
 */

void fprint_bdt_with_stat(fp, obj)

 FILE   *fp;
 bdt    obj;
{
 list   sched_l;

 fprintf(fp,"\n Base de Temps :\n");
 fprintf(fp,"===============\n");

 if (obj == bdt_undefined) 
    {
     fprintf(fp, "\tNon calculee\n\n");
     return;
    }

 sched_l = bdt_schedules(obj);

 for (; sched_l != NIL; sched_l = CDR(sched_l)) 
    {
     schedule  crt_sched;
     int       crt_stmt;
     list      dim_l;
     predicate crt_pred;

     crt_sched = SCHEDULE(CAR(sched_l));
     crt_stmt = schedule_statement(crt_sched);
     dim_l = schedule_dims(crt_sched);
     crt_pred = schedule_predicate(crt_sched);

     /* PRINT */
     fprintf(fp,"ins_%d: =>", crt_stmt);
     print_statement(adg_number_to_statement(crt_stmt));

     if (crt_pred != predicate_undefined) 
        {
         Psysteme ps = (Psysteme)predicate_system(crt_pred);
         Pcontrainte peq;

         if (ps != NULL) 
            {
             fprintf(fp,"\t pred: ");

             for (peq = ps->inegalites; peq!=NULL;\
                 pu_inegalite_fprint(fp,peq,entity_local_name),peq=peq->succ);

             for (peq = ps->egalites; peq!=NULL;\
                 pu_egalite_fprint(fp,peq,entity_local_name),peq=peq->succ);

             fprintf(fp,"\n");
            }
         else
         fprintf(fp, "\t pred: TRUE\n");
        }
     else
         fprintf(fp, "\t pred: TRUE\n");

     fprintf(fp, "\t dims: ");
     for (; dim_l != NIL; dim_l = CDR(dim_l)) 
        {
         expression exp = EXPRESSION(CAR(dim_l));
         fprintf(fp,"%s", words_to_string(words_expression(exp)));
         if (CDR(dim_l) != NIL)
            fprintf(fp," , ");
        }
    fprintf(fp,"\n");
  }
}

/*==========================================================================*/
/* bool my_contrainte_normalize(Pcontrainte c, bool is_egalite):
 * this is the same function as contrainte_normalize, except that it divides
 * inequalities by the pgcd of all the terms, including the TCST one.
 *
 * AC 93/12/23
 */

bool my_contrainte_normalize(c, is_egalite)

 Pcontrainte   c;
 bool       is_egalite;
{
 bool       is_c_norm = true;
 Value        a, nb0 = VALUE_ZERO;

 if ((c != NULL) && (c->vecteur != NULL)
     && value_notzero_p(a = vect_pgcd_except(c->vecteur, TCST)))
    {
     nb0 = value_mod(value_abs(vect_coeff(TCST, c->vecteur)),a);
     if (is_egalite) 
	{
         if (value_zero_p(nb0))
	    {
             (void) vect_div(c->vecteur, value_abs(a));
             c->vecteur = vect_clean(c->vecteur);
            }
         else is_c_norm= false;
        }

     else 
	{
         if (value_zero_p(nb0) )
	    {
             vect_chg_coeff(&(c->vecteur), TCST,
			    value_uminus(vect_coeff(TCST, c->vecteur)));
             (void) vect_div(c->vecteur, value_abs(a));
             c->vecteur = vect_clean(c->vecteur);
             vect_chg_coeff(&(c->vecteur), TCST,
			    value_uminus(vect_coeff(TCST, c->vecteur)));
            }
        }
    }

 return (is_c_norm);
}

/*===============================================================*/
/* bool my_inegalite_normalize(Pcontrainte ineg): same function
 * as inegalite_normalize except that it uses my_contrainte_nor-
 * malize insteadof contrainte_normalize.
 *
 * AC 93/12/23
 */

bool my_inegalite_normalize(ineg)

 Pcontrainte ineg;
{
 return(my_contrainte_normalize(ineg, false));
}

/*===============================================================*/
/* Psysteme my_sc_normalize(ps): that function suppresses redondant
 * equations and reduce its equations to the pgcd. There is no test
 * of faisability.
 *
 * AC 93/12/23
 */

Psysteme my_sc_normalize(ps)

 Psysteme     ps;
{
 Pcontrainte  eq;

 ps = sc_elim_db_constraints(ps);

 if (ps)
    {
     for (eq = ps->egalites; eq != NULL; eq=eq->succ)
        {
         /* normalisation de chaque equation */
         if (eq->vecteur)
           {
            vect_normalize(eq->vecteur);
           }
        }

     for (eq = ps->inegalites; eq!=NULL; eq=eq->succ)
        {
         if (eq->vecteur)
            {
             vect_normalize(eq->vecteur);
            }
        }
     ps = sc_elim_db_constraints(ps);
     sc_elim_empty_constraints(ps, true);
     sc_elim_empty_constraints(ps, false);

     ps->base = (Pvecteur)NULL;
     sc_creer_base(ps);
    }

 return(ps);
}

/*==================================================================*/
/* Psysteme predicate_to_system((predicate) p) : transform a
 * predicate in a system.
 *
 * AC 93/11/02
 */

Psysteme predicate_to_system(p)

 predicate   p;
{
 Psysteme    s;

 if (p == predicate_undefined) s = SC_RN;
 else s = predicate_system(p);

 return(s);
}

/*==================================================================*/
/* schedule true_copy_schedule( s ): really copies a schedule s.
 * The function copy_schedule is not sufficient here because it
 * doesn't copy the predicate properly for example.
 *
 * AC 93/12/15
 */

schedule true_copy_schedule(s)

 schedule     s;
{
 int          stat;
 predicate    pred;
 expression   exp, exp_copy;
 list         lexp, lexp_copy = NIL;
 Psysteme     sys, sys_copy;
 schedule     scopy;

 stat = schedule_statement(s);

 sys = predicate_to_system(schedule_predicate(s));
 sys_copy = sc_dup(sys);
 pred = make_predicate(sys_copy);

 lexp = schedule_dims(s);
 for ( ; lexp != NIL; lexp = CDR(lexp))
    {
     exp = EXPRESSION(CAR(lexp));
     exp_copy = copy_expression(exp);
     ADD_ELEMENT_TO_LIST(lexp_copy, EXPRESSION, exp_copy);
    }

 scopy = make_schedule(stat, pred, lexp_copy);

 return(scopy);
}

/*======================================================================*/
/* bdt true_copy_bdt(b): really copies a bdt b, cf. true_copy_schedule().
 *
 * AC 93/12/21
 */

bdt true_copy_bdt(b)

 bdt       b;
{
 bdt       b_copy;
 list      lbdt, lbdt_copy = NIL;
 schedule  sched, sched_copy;

 lbdt = bdt_schedules(b);

 for (; lbdt != NIL; lbdt = CDR(lbdt))
    {
     sched = SCHEDULE(CAR(lbdt));
     sched_copy = true_copy_schedule(sched);
     ADD_ELEMENT_TO_LIST(lbdt_copy, SCHEDULE, sched_copy);
    }
 b_copy = make_bdt(lbdt_copy);

 return(b_copy);
}

/*======================================================================*/
/* bool system_contains_var(ps, var): test if a system contains the
 * variable"var".
 *
 * AC 94/01/04
 */

bool system_contains_var(ps, var)

 Psysteme   ps;
 Variable   var;
{
 ps->base = NULL;
 sc_creer_base(ps);
 return(base_contains_variable_p(ps->base, var));
}

/*==================================================================*/
/* void poly_chg_var(pp, v_old, v_new) : change the variable "v_old"
 * in the variable "v_new" in the polynom pp.
 *
 * AC 93/11/04
 */

void poly_chg_var(pp, v_old, v_new)

 Ppolynome  pp;
 Variable   v_old, v_new;
{
 while (!POLYNOME_NUL_P(pp))
    {
     Pmonome pmcur = polynome_monome(pp);
     if (pmcur != (Pmonome)NIL)
	{
         Pvecteur pvcur = monome_term(pmcur);
         vect_chg_var(&pvcur, v_old, v_new);
        }
     pp = polynome_succ(pp);
    }
}

/*=======================================================================*/
/* Psysteme suppress_sc_in_sc((Psysteme) in_ps1, (Psysteme) in_ps2)
 * Input  : 2 Psystemes.
 * Output : Psysteme : Scan in_ps1 and remove from it Pcontraintes
 *              in in_ps2. No sharing, No remove input object.
 * cf. function "adg_suppress_2nd_in_1st_ps" from array_dfg.c
 *
 * AC 93/12/14
 */

Psysteme suppress_sc_in_sc(in_ps1, in_ps2)

 Psysteme     in_ps1, in_ps2;
{
 Psysteme     ret_ps = SC_RN;
 Pcontrainte  eq1, eq2, ineq1, ineq2, cont;

 if (in_ps1 == SC_RN) return(ret_ps);
 if (in_ps2 == SC_RN) return(in_ps1);

 for (eq1 = in_ps1->egalites; eq1 != NULL; eq1 = eq1->succ)
    {
     bool ok = true;
     for (eq2 = in_ps2->egalites; eq2 != NULL; eq2 = eq2->succ)
        {
         ok = ok && (!vect_equal(eq1->vecteur, eq2->vecteur));
         if (!ok) break;
        }
     if (ok)
        {
         cont = contrainte_make(eq1->vecteur);
         ret_ps = sc_append(ret_ps,sc_make(cont,CONTRAINTE_UNDEFINED));
        }
    }
 for (ineq1 = in_ps1->inegalites; ineq1 != NULL; ineq1 = ineq1->succ)
    {
     bool ok = true;
     for (ineq2=in_ps2->inegalites; ineq2 != NULL; ineq2=ineq2->succ)
        {
         ok = ok && !(vect_equal(ineq1->vecteur, ineq2->vecteur));
         if (!ok) break;
        }
     if (ok)
        {
         cont = contrainte_make(ineq1->vecteur);
         ret_ps = sc_append(ret_ps,sc_make(CONTRAINTE_UNDEFINED,cont));
        }
    }

 return(ret_ps);
}


/*======================================================================*/
/* void analyze_expression(e, d): this function analyzes an expression
 * that is, it reduces the different components of the expression to
 * the common denominator and make as many simplification as possible.
 * ex:
 *      u + v - (u + 2*v)/3    => (2*u + v)/3
 *      4*(u + v)/2 - v +w     => 2*u + v + w
 *
 * AC 93/12/14
 */

void analyze_expression(e, d)

 expression  *e;
 int         *d;
{
 expression  exp1, exp2, exp_a, exp_b;
 list        lexp1, lexp2;
 syntax      syn;
 tag         t;
 int         d1, d2, val;
 entity      ent, func;
 call        ca;
 Pvecteur    vect1, vect2;

 unnormalize_expression(*e);
 NORMALIZE_EXPRESSION(*e);

 if (normalized_tag(expression_normalized(*e))==is_normalized_complex) {

 /* We have to analyze the complex expression */
 unnormalize_expression(*e);
 syn = expression_syntax(*e);
 t = syntax_tag(syn);

 switch(t) {
    case is_syntax_call:
       ent = call_function(syntax_call(syn));
       lexp1 = call_arguments(syntax_call(syn));
       exp1 = EXPRESSION(CAR(lexp1));
       lexp2 = CDR(lexp1);
       exp2 = EXPRESSION(CAR(lexp2));

       if (ENTITY_DIVIDE_P(ent))
          {
           NORMALIZE_EXPRESSION(exp1);
           d2 = expression_to_int(exp2);
           if (normalized_tag(expression_normalized(exp1))==\
					is_normalized_complex)
              {
               analyze_expression(&exp1, &d1);
               d2 = d1 * d2;
               exp2 = int_to_expression(d2);
	       lexp2=call_arguments(syntax_call(expression_syntax(exp1)));
	       lexp2->cdr = NULL;
	       ADD_ELEMENT_TO_LIST(lexp2, EXPRESSION, exp2);
               *d = d2;
	       *e = exp1;
              }
           else
              {
               vect1 = normalized_linear(expression_normalized(exp1));
               val = VALUE_TO_INT(vect_pgcd_all(vect1));
               if ((val%d2) == 0)
                  {
                   vect1 = vect_div(vect1, d2);
                   *e = Pvecteur_to_expression(vect1);
                   unnormalize_expression(*e);
                   *d = 1;
                  }
               else
                  {
                   unnormalize_expression(exp1);
                   *d = expression_to_int(exp2);
                  }
              }
          }

       else if (ENTITY_MULTIPLY_P(ent))
          {
           if (expression_constant_p(exp1))
              {
               analyze_expression(&exp2, &d1);
               val = expression_to_int(exp1);
               *d = d1;
               if (d1 != 1)
                  {
                   func=call_function(syntax_call(expression_syntax(exp2)));
                   lexp2=call_arguments(syntax_call(expression_syntax(exp2)));
                   exp_a = EXPRESSION(CAR(lexp2));
                   NORMALIZE_EXPRESSION(exp_a);
                   lexp2 = CDR(lexp2);
                   vect1 = normalized_linear(expression_normalized(exp_a));
                   vect1 = vect_multiply(vect1, val);
                   val = VALUE_TO_INT(vect_pgcd_all(vect1));
                   d2 = expression_to_int(exp1);
                   if ((val%d1) == 0)
                      {
		       vect1 = vect_div(vect1, d1);
                       *e = Pvecteur_to_expression(vect1);
                       unnormalize_expression(*e);
                       *d = 1;
                      }
                   else
                      {
		       if ((d1%val) == 0)
			  {
                           vect1 = vect_div(vect1, val);
			   exp_a = Pvecteur_to_expression(vect1);
			   unnormalize_expression(exp_a);
			   exp_b = int_to_expression(d1/val);
			   lexp2 = CONS(EXPRESSION, exp_b, NIL);
			   lexp1 = CONS(EXPRESSION, exp_a, lexp2);
			   ca = make_call(func, lexp1);
			   syn = make_syntax(is_syntax_call,ca);
			   *e = make_expression(syn, normalized_undefined);
			   *d = d1/val;
			  }
                       else
			  {
                           exp_a = Pvecteur_to_expression(vect1);
                           unnormalize_expression(exp_a);
                           lexp1 = CONS(EXPRESSION, exp_a, lexp2);
                           ca = make_call(func, lexp1);
                           syn = make_syntax(is_syntax_call,ca);
                           *e = make_expression(syn, normalized_undefined);
			  }
                      }
                  }
               else
                  {
                   NORMALIZE_EXPRESSION(exp2);
                   vect1 = normalized_linear(expression_normalized(exp2));
                   vect1 = vect_multiply(vect1, val);
                   *e = Pvecteur_to_expression(vect1);
                   unnormalize_expression(*e);
                  }
              }
           else
              {
               analyze_expression(&exp1, &d1);
               val = expression_to_int(exp2);
               *d = d1;
               if (d1 != 1)
                  {
                   func=call_function(syntax_call(expression_syntax(exp1)));
                   lexp1=call_arguments(syntax_call(expression_syntax(exp1)));
                   exp_a = EXPRESSION(CAR(lexp1));
                   NORMALIZE_EXPRESSION(exp_a);
                   lexp1 = CDR(lexp1);
                   vect1 = normalized_linear(expression_normalized(exp_a));
                   vect1 = vect_multiply(vect1, val);

                   val = VALUE_TO_INT(vect_pgcd_all(vect1));
                   d2 = expression_to_int(exp2);

                   if ((val%d1) == 0)
                      {
                       vect1 = vect_div(vect1, d1);
                       *e = Pvecteur_to_expression(vect1);
                       unnormalize_expression(*e);
                       *d = 1;
                      }
                   else
                      {
                       if ((d1%val) == 0)
                          {
                           vect1 = vect_div(vect1, val);
                           exp_a = Pvecteur_to_expression(vect1);
                           unnormalize_expression(exp_a);
                           exp_b = int_to_expression(d1/val);
                           lexp2 = CONS(EXPRESSION, exp_b, NIL);
                           lexp1 = CONS(EXPRESSION, exp_a, lexp2);
                           ca = make_call(func, lexp1);
                           syn = make_syntax(is_syntax_call,ca);
                           *e = make_expression(syn, normalized_undefined);
                           *d = d1/val;
                          }
                       else
                          {
                           exp_a = Pvecteur_to_expression(vect1);
                           unnormalize_expression(exp_a);
                           lexp1 = CONS(EXPRESSION, exp_a, lexp1);
                           ca = make_call(func, lexp1);
                           syn = make_syntax(is_syntax_call,ca);
                           *e = make_expression(syn, normalized_undefined);
                          }
                      }
                  }
               else
                  {
                   NORMALIZE_EXPRESSION(exp1);
                   vect1 = normalized_linear(expression_normalized(exp2));
                   vect1 = vect_multiply(vect1,val);
                   *e = Pvecteur_to_expression(vect1);
                   unnormalize_expression(*e);
                  }
              }
          }
       else
          {
	   /* case of + or - */
           analyze_expression(&exp1, &d1);
           analyze_expression(&exp2, &d2);
           *d = sol_ppcm(d1, d2);

           if ((d1 == 1) && (d2 != 1))
              {
               func=call_function(syntax_call(expression_syntax(exp2)));
               lexp2=call_arguments(syntax_call(expression_syntax(exp2)));
               exp_a = EXPRESSION(CAR(lexp2));
               NORMALIZE_EXPRESSION(exp_a);
               NORMALIZE_EXPRESSION(exp1);
               vect1 = normalized_linear(expression_normalized(exp1));
               vect2 = normalized_linear(expression_normalized(exp_a));
               lexp2 = CDR(lexp2);

               vect1 = vect_multiply(vect1, d2);
               if (ENTITY_PLUS_P(ent))
                    vect1 = vect_add(vect1, vect2);
               else vect1 = vect_substract(vect1, vect2);

               val = VALUE_TO_INT(vect_pgcd_all(vect1));
               if ((val%(*d)) == 0)
                  {
                   vect1 = vect_div(vect1, d2);
                   *e = Pvecteur_to_expression(vect1);
                   unnormalize_expression(*e);
                   *d = 1;
                  }
               else
                  {
                   exp_a = Pvecteur_to_expression(vect1);
                   unnormalize_expression(exp_a);
                   lexp1 = CONS(EXPRESSION, exp_a, lexp2);
                   ca = make_call(func, lexp1);
                   syn = make_syntax(is_syntax_call, ca);
                   *e = make_expression(syn, normalized_undefined);
                  }
              }
           else if ((d1 != 1) && (d2 == 1))
              {
               func=call_function(syntax_call(expression_syntax(exp1)));
               lexp1=call_arguments(syntax_call(expression_syntax(exp1)));
               exp_a = EXPRESSION(CAR(lexp1));
               NORMALIZE_EXPRESSION(exp_a);
               NORMALIZE_EXPRESSION(exp2);
               vect1 = normalized_linear(expression_normalized(exp_a));
               vect2 = normalized_linear(expression_normalized(exp2));
               lexp1 = CDR(lexp1);

               vect2 = vect_multiply(vect2, d1);
               if (ENTITY_PLUS_P(ent))
                    vect1 = vect_add(vect1, vect2);
               else vect1 = vect_substract(vect1, vect2);

               val = VALUE_TO_INT(vect_pgcd_all(vect1));
               if ((val%(*d)) == 0)
                  {
                   vect1 = vect_div(vect1, d2);
                   *e = Pvecteur_to_expression(vect1);
                   unnormalize_expression(*e);
                   *d = 1;
                  }
               else
                  {
                   exp_a = Pvecteur_to_expression(vect1);
                   unnormalize_expression(exp_a);
                   lexp1 = CONS(EXPRESSION, exp_a, lexp1);
                   ca = make_call(func, lexp1);
                   syn = make_syntax(is_syntax_call, ca);
                   *e = make_expression(syn, normalized_undefined);
                  }
              }
           else if ((d1 != 1) && (d2 != 1))
              {
               func=call_function(syntax_call(expression_syntax(exp1)));
               exp_a = EXPRESSION(CAR(call_arguments(syntax_call\
                                          (expression_syntax(exp1)))));
               exp_b = EXPRESSION(CAR(call_arguments(syntax_call\
                                          (expression_syntax(exp2)))));
               NORMALIZE_EXPRESSION(exp_a);
               NORMALIZE_EXPRESSION(exp_b);
               vect1 = normalized_linear(expression_normalized(exp_a));
               vect2 = normalized_linear(expression_normalized(exp_b));

               vect1 = vect_multiply(vect1, (*d/d1));
               vect2 = vect_multiply(vect2, (*d/d2));
               if (ENTITY_PLUS_P(ent))
                     vect1 = vect_add(vect1, vect2);
               else  vect1 = vect_substract(vect1, vect2);

               val = VALUE_TO_INT(vect_pgcd_all(vect1));
               if ((val%(*d)) == 0)
                  {
                   vect1 = vect_div(vect1, (*d));
                   *e = Pvecteur_to_expression(vect1);
                   unnormalize_expression(*e);
                   *d = 1;
                  }
               else
                  {
                   exp_a = Pvecteur_to_expression(vect1);
                   unnormalize_expression(exp_a);
                   exp_b = int_to_expression(*d);
                   lexp1 = CONS(EXPRESSION, exp_b, NIL);
                   lexp1 = CONS(EXPRESSION, exp_a, lexp1);
                   ca = make_call(func, lexp1);
                   syn = make_syntax(is_syntax_call, ca);
                   *e = make_expression(syn, normalized_undefined);
                  }
              }
           else
              {
               NORMALIZE_EXPRESSION(exp1);
               NORMALIZE_EXPRESSION(exp2);
               vect1 = normalized_linear(expression_normalized(exp1));
               vect2 = normalized_linear(expression_normalized(exp2));

               if (ENTITY_PLUS_P(ent))
                     vect1 = vect_add(vect1, vect2);
               else  vect1 = vect_substract(vect1, vect2);

               *e = Pvecteur_to_expression(vect1);
               unnormalize_expression(*e);
              }
          }
    break;

    case is_syntax_reference:
       *d = 1;
    break;
    }
 }

 else
   {
    /* case of an already normalized expression, do nothing */
    unnormalize_expression(*e);
    *d =1;
   }
}

/*=======================================================================*/
/* bool exp_equals_p(e1,e2): tests if two expressions are equal. This
 * function is better than expression_equal_p() because it takes into
 * account the case where the expressions are not linear.
 *
 * AC 94/01/03
 */

bool exp_equals_p(e1, e2)

 expression  e1, e2;
{
 expression  e12, e22;
 int         d1, d2;

 analyze_expression(&e1, &d1);
 analyze_expression(&e2, &d2);

 if (d1 == d2)
    {
     if (d1 == 1)
	{
	 if ((e1 != expression_undefined) && \
	     (e2 != expression_undefined))
	    {
             NORMALIZE_EXPRESSION(e1);
	     NORMALIZE_EXPRESSION(e2);
	     return (vect_equal(normalized_linear(e1),\
				normalized_linear(e2)));
            }
         else return(false);
	}
     else
	{
         e12 = EXPRESSION(CAR(call_arguments(syntax_call\
					     (expression_syntax(e1)))));
	 e22 = EXPRESSION(CAR(call_arguments(syntax_call\
	                        	     (expression_syntax(e2)))));
         NORMALIZE_EXPRESSION(e12);
	 NORMALIZE_EXPRESSION(e22);
	 return (vect_equal(normalized_linear(e12),\
			    normalized_linear(e22)));
	}
    }
 else return(false);
}

/*=======================================================================*/
/* bool list_of_exp_equals_p(l1,l2): tests if two lists of expressions
 * have all their terms equals. Beware, the equality of the length of the
 * two lists is not checked !
 *
 * AC 94/01/03
 */

bool list_of_exp_equals_p(l1, l2)

 list        l1, l2;
{
 bool     is_equal = true;
 expression  exp1, exp2;

 while ((is_equal) && (l1 != NIL))
    {
     exp1 = EXPRESSION(CAR(l1));
     exp2 = EXPRESSION(CAR(l2));
     is_equal = exp_equals_p(exp1, exp2);
     l1 = CDR(l1);
     l2 = CDR(l2);
    }

 return (is_equal);
}

/*************************************************************************/

