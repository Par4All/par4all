/* 
 * $Id$
 * 
 * This file provides functions to test the feasibility of a system 
 * of constraints. 
 *
 * Arguments of these functions :
 * 
 * - s or sc is the system of constraints.
 * - ofl_ctrl is the way overflow errors are handled
 *     ofl_ctrl == NO_OFL_CTRL
 *               -> overflow errors are not handled
 *     ofl_ctrl == OFL_CTRL
 *               -> overflow errors are handled in the called function
 *     ofl_ctrl == FWD_OFL_CTRL
 *               -> overflow errors must be handled by the calling function
 * - ofl_res is the result of the feasibility test when ofl_ctrl == OFL_CTRL
 *   and there is an overflow error.
 * - integer_p (low_level function only) is a boolean :
 *     integer_p == TRUE to test if there exists at least one integer point 
 *               in the convex polyhedron defined by the system of constraints.
 *     integer_p == FALSE to test if there exists at least one rational point 
 *               in the convex polyhedron defined by the system of constraints.
 *     (This has an impact only upon Fourier-Motzkin feasibility test).
 *
 * Last modified by Beatrice Creusillet, 13/12/94.
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <malloc.h>

#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"
#include "sc-private.h"


/*****duong - set timeout with signal and alarm*****/
#include <signal.h>
#define FM_TIMEOUT timeout_for_FM 
//get timeout from environment by extern variable timeout_for_FM. default 3 minutes

void 
catch_alarm_FM (int sig)
{  
  fprintf(stderr,"CATCH ALARM sc_fourier_motzkin_feasibility_ofl_ctrl. Timeout_for_FM = %d\n", FM_TIMEOUT);
  alarm(0); //clear the alarm
  
  THROW(timeout_error);

}
/*****duong*****/
#define MAX_NB_VARS_CAN_ADD 10
#define MAX_COEFF_CAN_HAVE 131
#define CHOSEN_NUMBER 64
int number_of_variables_added = 0; // number of variable added
static int S_counter = 0;
static int FM_counter = 0;


EXCEPTION timeout_error; // needed for sc_fourier_motzkin_feasibility_ofl_ctrl
//EXCEPTION any_exception_error; //needed for internal_sc_feasibility
//EXCEPTION user_exception_error; //needed for internal_sc_feasibility

static boolean FM_overflow_or_timeout = FALSE; // check if FM timeout or not
static boolean S_overflow_or_timeout = FALSE; // check if Simplex overflow or timeout or not

/* 
 * INTERFACES
 */

boolean 
sc_rational_feasibility_ofl_ctrl(sc, ofl_ctrl, ofl_res)
Psysteme sc;
int ofl_ctrl;
boolean ofl_res;
{
    return sc_feasibility_ofl_ctrl(sc, FALSE, ofl_ctrl, ofl_res);
}

boolean 
sc_integer_feasibility_ofl_ctrl(sc,ofl_ctrl, ofl_res)
Psysteme sc;
int ofl_ctrl;
boolean ofl_res;
{
    return sc_feasibility_ofl_ctrl(sc, TRUE, ofl_ctrl, ofl_res);
}

/*
 * LOW LEVEL FUNCTIONS 
 */

/*  just a test to improve the Simplex/FM decision.
 * c is a list of constraints, equalities or inequalities
 * pc is the number of constraints in the list
 * pv is the number of non-zero coefficients in the system
 *
 * pc and pv MUST be initialized. They are multiplied by weight.
 */
static void 
decision_data(c, pc, pv, weight)
Pcontrainte c;
int *pc, *pv, weight;
{
    Pvecteur v;
    
    for(; c!=NULL; c=c->succ)
    {
	v=c->vecteur;
	if (v!=NULL) 
	{
	    (*pc)+=weight;
	    for(; v!=NULL; v=v->succ) 
		if (var_of(v)!=TCST) (*pv)+=weight;
	}
    }
}

/* chose the next variable in base b for projection in system s.
 * tries to avoid Fourier potential explosions when combining inequalities.
 * - if there are equalities, chose the var with the min |coeff| (not null)
 * - if there are only inequalities, chose the var that will generate the
 *   minimum number of constraints with pairwise combinations.
 * - if ineq is TRUE, consider variables even if no equalities.
 *
 * (c) FC 21 July 1995
 */
static Variable
chose_variable_to_project_for_feasability(Psysteme s, Pbase b, boolean ineq)
{
  Pcontrainte c = sc_egalites(s);
  Pvecteur v;
  Variable var = NULL;
  Value val;
  int size = vect_size(b);

  ifscdebug(8)
    {
      fprintf(stderr, "[chose_variable_to_project_for_feasability] b/s:\n");
      vect_fprint(stderr, b, default_variable_to_string);
      sc_fprint(stderr, s, default_variable_to_string);
    }

  if (size==1) return var_of(b);
  assert(size>1);
    
  if (c)
  {
    /* find the lowest coeff 
     */
    Variable minvar = TCST;
    Value minval = VALUE_ZERO;

    for (; c; c=c->succ)
    {
      for (v = contrainte_vecteur(c); v; v=v->succ)
      {
	var = var_of(v);

	if (var!=TCST)
	{
	  val = value_abs(val_of(v));
	  if ((value_notzero_p(minval) && value_lt(val,minval))
	      || value_zero_p(minval)) 
	    minval = val, minvar = var;
		     
	  if (value_one_p(minval)) return minvar;
	}
      }
    }

    /* shouldn't find empty equalities ?? */
    /* assert(minvar!=TCST); */
    var = minvar;
  }
  
  if (!var && ineq)
  {
    /* only inequalities, reduce the explosion
     */
    int i;
    two_int_infop t = (two_int_infop) malloc(2*size*sizeof(int));
    Pbase tmp;
    int min_new;

    c = sc_inegalites(s);

    /* initialize t
     */
    for (i=0; i<size; i++) t[i][0]=0, t[i][1]=0;

    /* t[x][0 (resp. 1)] = number of negative (resp. positive) coeff
     */
    for (; c; c=c->succ)
    {
      for (v = contrainte_vecteur(c); v; v=v->succ)
      {
	var = var_of(v); 
	if (var!=TCST)
	{
	  ifscdebug(9) 
	    fprintf(stderr, "%s\n", default_variable_to_string(var));

	  for (i=0, tmp=b; tmp && var_of(tmp)!=var; 
	       i++, tmp=tmp->succ);
	  assert(tmp);
		    
	  t[i][value_posz_p(val_of(v))]++;
	}
      }
    }

    /* t[x][0] = number of combinations, i.e. new created constraints.
     */
    for (i=0; i<size; i++) t[i][0] *= t[i][1];

    for (tmp=b->succ, var=var_of(b), min_new=t[0][0], i=1;
	 min_new && i<size; 
	 i++, tmp=tmp->succ) {
      if (t[i][0]<min_new)
	min_new = t[i][0], var=var_of(tmp); 
    }

    free(t);
  }

  ifscdebug(8)
    fprintf(stderr, "[chose_variable_to_project_for_feasability] "
	    "suggesting %s\n", default_variable_to_string(var));

  return var;
}

/* project in s1 (which is modified) 
 * using equalities or both equalities and inequalities.
*/
static boolean sc_fm_project_variables
(Psysteme * ps1, boolean integer_p, boolean use_eq_only, int ofl_ctrl)
{
  Pbase b = base_copy(sc_base(*ps1));
  Variable var;
  boolean faisable = TRUE;
  
   while (b && faisable)
  {
    var = chose_variable_to_project_for_feasability(*ps1, b, !use_eq_only);

    /* if use_eq_only */
    if (!var) break;

    vect_erase_var(&b, var);

    ifscdebug(8)
      {
	fprintf(stderr, 
		"[sc_fm_project_variables] system before %s projection:\n", 
		var);
	sc_fprint(stderr, *ps1, default_variable_to_string);
      }
	    
    sc_projection_along_variable_ofl_ctrl(ps1, var, ofl_ctrl);

    ifscdebug(8)
    {
      fprintf(stderr, 
	      "[sc_fm_project_variables] system after projection:\n");
      sc_fprint(stderr, *ps1, default_variable_to_string);
    }
	    
    if (sc_empty_p(*ps1))
    {
      faisable = FALSE;
      break;
    }

    if (integer_p) {
      *ps1 = sc_normalize(*ps1);
      if (SC_EMPTY_P(*ps1)) 
      { 
	faisable = FALSE; 
	break;
      }
    }    
  }//of while    
  
  base_rm(b);
  return faisable;
}

static boolean 
switch_method_when_error(Psysteme d, boolean int_p, int ofl_ctrl)
{
  boolean ok = TRUE;
  char * label;
  char * filename;
  
  //if there's once more exception, donot rethrow directly to sc_feasibility_ofl_ctrl
  //but produce an user_exception_error and catch in internal_sc_feasibility
    CATCH(any_exception_error) 
    {
      //if both S and FM fail, then print a message here, and throw overflow_error

	ifscprintexact(6) {
	  fprintf(stderr,"\n*****Both tried Simplex and FM, but failed!!!\n");
	  label = "LABEL - Both tried Simplex and FM, but failed !!!";
	  filename = "S_and_FM_fail_sc_dump.out";
	  sc_default_dump_to_file(d,label,0,filename);
	}
	ifscprintexact(5) {
	  fprintf(stderr,"\n*****Both tried Simplex and FM, but failed!!!\n");
	  label = "LABEL - Both tried Simplex and FM, but failed !!!";
	  filename = "S_and_FM_fail_sc_dump.out";
	  sc_default_dump_to_file(d,label,0,filename); 
	}

	THROW(user_exception_error);	
    }
    TRY
    {
	if (!(FM_overflow_or_timeout)) {

	  ifscprintexact(2) { //Test temporary - to be removed
	    fprintf(stderr,"\n *** * *** Simplex die %d th\n",S_counter);
	  }
	 
	  //print or not all the systems troublesome with Simplex here 	  

	  ifscprintexact(8) {
	    fprintf(stderr,"\nSimplex overflow or timeout. Let's try Fourier-Motzkin ...");
	    //sc_default_dump(d);
	    //fprintf(stderr,"Exception with Simplex %dth\n",S_counter);
	    //sc_dump(d);
	    //when using sc_pre_process_for_simplex, cannot use default_variable_to_string
	    label = "LABEL - System of constraints given to internal_sc_feasibility - Simplex : ";
	    filename = "S_fail_sc_dump.out";
	    sc_default_dump_to_file(d,label,S_counter,filename);
	  }
	  ifscprintexact(5) {	    
	    fprintf(stderr,"\nSimplex overflow or timeout. Let's try Fourier-Motzkin ...");
	    label = "LABEL - System of constraints given to internal_sc_feasibility - Simplex : ";
	    filename = "S_fail_sc_dump.out";
	    sc_default_dump_to_file(d,label,S_counter,filename);
	  }
	  FM_counter ++;
	  ok = sc_fourier_motzkin_feasibility_ofl_ctrl(d, int_p, ofl_ctrl);
	  //if timeout, then go to CATCH(overflow_error) within S_overflow_or_timeout, else return ok.
	  ifscprintexact(5) {	     fprintf(stdout," Pass !!!\n");}
	  FM_overflow_or_timeout = FALSE;
	  UNCATCH(any_exception_error);
	  return ok;
	}//of if (!(FM_overflow_or_timeout))

	if (!(S_overflow_or_timeout)) {
	  
	  //I don't think this case would happen (FM die while haven't tested S), but who'll know ...
	  
	  //print or not all the systems troublesome with FM here
	  
	  ifscprintexact(2) { //Test temporary - to be removed
	    fprintf(stderr,"\n *** * *** FM die %d th\n",FM_counter);
	  }

	  ifscprintexact(7) {
	     fprintf(stderr,"\nFourier-Motzkin timeout or overflow. Let's try Simplex ...");
	    //sc_default_dump();	  	  
	    label = "LABEL - System of constraints given to internal_sc_feasibility - FM : ";
	    filename = "FM_fail_sc_dump.out";
	    sc_default_dump_to_file(d,label,FM_counter,filename);
	  }
	  ifscprintexact(5) {
	    fprintf(stderr,"\nFourier-Motzkin timeout or overflow. Let's try Simplex ...");
	    //sc_default_dump();	  	  
	    label = "LABEL - System of constraints given to internal_sc_feasibility - FM : ";
	    filename = "FM_fail_sc_dump.out";
	    sc_default_dump_to_file(d,label,FM_counter,filename);
	  }
	  S_counter ++;
	  ok = sc_simplexe_feasibility_ofl_ctrl(d, ofl_ctrl);
	  //if overflow, then go to CATCH(overflow_error) within FM_overflow_or_timeout, else return ok. 
	  ifscprintexact(5) {fprintf(stdout," Pass !!!\n");}
	  S_overflow_or_timeout = FALSE;
	  UNCATCH(any_exception_error);
	  return ok;
	}//of (!(S_overflow_or_timeout))
	fprintf(stderr,"Error: shouldn't reach here, DN.\n");
	return ok;// 
    }//of TRY
    //UNCATCH() already put in TRY{}
}

#define SIMPLEX_METHOD		1
#define FM_METHOD		0
#define PROJECT_EQ_METHOD	2
#define NO_PROJ_METHOD		0

static boolean internal_sc_feasibility
  (Psysteme sc, int method, boolean int_p, int ofl_ctrl)
{
  Psysteme w = NULL;
  boolean ok = TRUE; 
   
  if ((method & PROJECT_EQ_METHOD))//if method = 01 then 01&10 = 0, if method = 11 then 11&10= 1
  {
    w = sc_dup(sc);//there's a change in sc_dup
    //w = sc_copy(sc);//copy the system of constraints
    ok = sc_fm_project_variables(&w, int_p, TRUE, ofl_ctrl);
    }
  
  //  if nb_ineg >= 10 and nb_eg < 6 then use Simplex (replace n eg by 2n ineg 
  //  if nb_ineg >= 10 and nb_eg >= 6 then project eg and use Simplex
  //  if nb_ineg < 10 then use FM

  /* maybe the S/FM should be chosen again as #ref has changed... */
  
  if (ok) 
  {

    /*duong : if cannot solve the problem by one method, we'll switch to the other
    Attention: there are 2 exceptions in Simplex : simplex_arithmetic_error and timeout_error
    and also 2 exception in FM : overflow_error and timeout_error
    We catch all exception here. 

    If the problem cannot be solved by both methods, we'll throw the exception overflow_error 
    (in fact maybe timeout_error, overflow_error, or simplex_arithmetic_error)
    */

    CATCH(any_exception_error) // CATCH if first call of methods fails
    {
	CATCH(user_exception_error) //CATCH if second call of methods fails
	  {
	    if (w) sc_rm(w);  // free resource
	    if (ofl_ctrl==FWD_OFL_CTRL)
	      THROW(overflow_error);// throw overflow_error like before
	    return TRUE; //if not ofl_ctrl then default is TRUE
	  }
	TRY
	  {
	    ok = switch_method_when_error(w? w:sc,int_p, ofl_ctrl);	   
	  }
	// if no more exception then free resource, return ok to sc_feasibility_ofl_ctrl
	UNCATCH(user_exception_error); 
	if (w) sc_rm(w);  
	return ok;
	
    
    }//of CATCH()
        
    if (method & SIMPLEX_METHOD)
    {
      S_overflow_or_timeout = TRUE;
      S_counter ++;
      ok = sc_simplexe_feasibility_ofl_ctrl(w? w: sc, ofl_ctrl);      
      S_overflow_or_timeout = FALSE; //if no overflow then go to this line 
    }
    else
    { 
      FM_overflow_or_timeout = TRUE;
      FM_counter ++;
      ok = sc_fourier_motzkin_feasibility_ofl_ctrl(w? w: sc, int_p, ofl_ctrl);
      FM_overflow_or_timeout = FALSE;//if no timeout then go to this line
    }
  
    UNCATCH(any_exception_error);
  
  }//of if(ok)
  
  if (w) sc_rm(w);  

  return ok;
}

boolean 
sc_feasibility_ofl_ctrl(sc, integer_p, ofl_ctrl, ofl_res)
Psysteme sc;
boolean integer_p;
int ofl_ctrl;
boolean ofl_res;
{ 
  int
    method = 0,
    n_var,
    n_cont_eq = 0, n_ref_eq = 0,
    n_cont_in = 0, n_ref_in = 0;
  boolean 
    ok = FALSE,
    catch_performed = FALSE;

  if (sc_rn_p(sc)) /* shortcut */
    return TRUE;
  
  n_var = sc->dimension,
  decision_data(sc_egalites(sc), &n_cont_eq, &n_ref_eq, 1);
  decision_data(sc_inegalites(sc), &n_cont_in, &n_ref_in, 1);

  /* else
   */
  switch (ofl_ctrl) 
  {
  case OFL_CTRL :
    ofl_ctrl = FWD_OFL_CTRL;
    catch_performed = TRUE;
    //CATCH(any_exception_error)
    CATCH(overflow_error) 
      {
	ok = ofl_res;
	catch_performed = FALSE;
	/* 
	 *   PLEASE do not remove this warning.
	 *
	 *   FC 30/01/95
	 */
		

	fprintf(stderr, "[sc_feasibility_ofl_ctrl] "
		"arithmetic error (%s[%d,%deq/%dref,%din/%dref]) -> %s\n",
		method&SIMPLEX_METHOD ? "Simplex" : "Fourier-Motzkin", 
		n_var, n_cont_eq, n_ref_eq, n_cont_in, n_ref_in,
		ofl_res ? "TRUE" : "FALSE");
	break;
      }		
  default: /* anyway, try */
    {
      /* a little discussion about what to decide (FC, 05/07/2000)
       *
       * - FM is good at handling equalities which are simply projected, 
       *   but may explode with many inequalities when they are combined.
       *   it is quite fast with few inequalities anyway.
       *
       * - SIMPLEX switches every eq to 2 inequalities, adding hyperplanes.
       *   thus it is not that good with equalities. Maybe equalities could
       *   be handled as such but I don't think that it is the case.
       *   it is quite slow with small systems because of the dense matrix
       *   to build and manipulate.
       *
       * suggestion implemented here: 
       *  1/ project equalities as much as possible,
       *     if there are many of them... ???
       *  2/ chose between FM and SIMPLEX **after** that?
       */
      /* use_simplex = (n_cont_in >= NB_CONSTRAINTS_MAX_FOR_FM || 
		     (n_cont_in>=10 && n_ref_in>2*n_cont_in));
		     (use_simplex && n_cont_eq >= 20) => proj */
      
      if (n_cont_in >= 10) 
      {
	method = SIMPLEX_METHOD;
	if (n_cont_eq >= 6)
	  method |= PROJECT_EQ_METHOD;// then method might be 3 = 11 in binary.
      }
      else
      {
	method = FM_METHOD;
      }

      /* fprintf(stderr, "in=%d eq=%d method=%d\n", n_cont_in, n_cont_eq, method);*/

      /* STATS
	 extern void init_log_timers(void);
	 extern void get_string_timers(char **, char **);
	 for (method=0; method<4; method++) {
	 char * t1, * t2;
	 init_log_timers(); */     
     
      ok = internal_sc_feasibility(sc, method, integer_p, ofl_ctrl);
      
	/* STATS 
	   get_string_timers(&t1, &t2);
	   fprintf(stderr, "FEAS %d %d %d %d %d %d %s",
	   n_cont_eq, n_ref_eq, n_cont_in, n_ref_in, method, ok, t1); } */

    }
  }  

  if (catch_performed)
    // UNCATCH(any_exception_error);
    UNCATCH(overflow_error);
  
  return ok;
}

/* boolean sc_fourier_motzkin_faisabilite_ofl(Psysteme s):
 * test de faisabilite d'un systeme de contraintes lineaires, par projections
 * successives du systeme selon les differentes variables du systeme
 *
 *  resultat retourne par la fonction :
 *
 *  boolean	: TRUE si le systeme est faisable
 *		  FALSE sinon
 *
 *  Les parametres de la fonction :
 *
 *  Psysteme s    : systeme lineaire 
 *
 * Le controle de l'overflow est effectue et traite par le retour 
 * du contexte correspondant au dernier CATCH(overflow_error) effectue.
 * 
 */
boolean 
sc_fourier_motzkin_feasibility_ofl_ctrl(s, integer_p, ofl_ctrl)
Psysteme s;
boolean integer_p;
int ofl_ctrl;
{
  Psysteme s1;// a copy of the system to calculate on
  boolean faisable = TRUE;
   
  CATCH(any_exception_error) {
    // maybe timeout_error or overflow_error
           
    alarm(0); // clear the alarm 
    
    if (ofl_ctrl == FWD_OFL_CTRL) {
	ifscprintexact(2) {
	  fprintf(stderr,"\nThis is an exception rethrown from [sc_fourier_motzkin_feasibility_ofl_ctrl]\n ");
	}	
	RETHROW(); //rethrow whatever the exception is
    }     
    return TRUE;// default is feasible
  }
  
  //start the alarm
  signal(SIGALRM, catch_alarm_FM);   
  alarm(FM_TIMEOUT);  
  
  if (s == NULL) return TRUE;
  s1 = sc_copy(s);
  
  ifscdebug(8)
    {
      fprintf(stderr, "[sc_fourier_motzkin_feasibility_ofl_ctrl] system:\n");
      sc_fprint(stderr, s1, default_variable_to_string);
    }
  
  s1 = sc_elim_double_constraints(s1);

  if (s1 != NULL)
  {
    /* a small basis if possible... (FC).
     */
    base_rm(sc_base(s1));
    sc_creer_base(s1);

    faisable = sc_fm_project_variables(&s1, integer_p, FALSE, ofl_ctrl);
	
    sc_rm(s1);
  }
  else 
    /* sc_kill_db_eg a de'sallouer s1 a` la detection de 
       sa non-faisabilite */
    faisable = FALSE;
  
  alarm(0);// clear the alarm.

  UNCATCH(any_exception_error);

  return faisable;
}

/*Afterward : duong test****************************************************/

//to be removed. duong
boolean
my_test_sc_simplex(Psysteme sc, int integer_p, int ofl_ctrl) 
{
  Psysteme w = NULL;
  boolean ok = TRUE;

  w = sc_copy(sc);

  fprintf(stderr,"********System after duplication by sc_copy********\n");
  sc_default_dump(w);//sc_default_dump_to_file(w);

  ok = sc_fm_project_variables(&w, integer_p, TRUE, ofl_ctrl);
  //integer_p = integer or rational, TRUE = project equation only
   
  if (ok) {

    //sc_pre_process_for_simplex(w);//duong.
    //If projection of some variables makes too big coefficients, 
    //then undo the projection on these variables by pre_process

    fprintf(stderr,"********System after projection********\n");
    sc_default_dump(sc);//sc_default_dump_to_file(w);

    w = sc_normalize(w);//recompute the base here.

    fprintf(stderr,"********System after normalization********\n");
    sc_default_dump(sc);//sc_default_dump_to_file(w);
    if (SC_EMPTY_P(w)) 
      { 
	ok = FALSE; 
	return ok;
      }
     
    //fprintf(stderr,"********System after sort********\n");
    //sc_lexicographic_sort(w, my_is_inferior_pvarval);
    fprintf(stderr,"********System after change by pre_process_for_Simplex********\n");
    sc_pre_process_for_simplex(w);
    sc_default_dump(sc);//sc_default_dump_to_file(w);
  
    //Now let's try Simplex
    ok = sc_simplexe_feasibility_ofl_ctrl(w,ofl_ctrl);
  }

  if (w) sc_rm(w);
  return ok;
}

//There's a problem with variables created inside sc, if using default_variable_to_string
//I don't know how to use sc_variable_name_push inside sc. 
//So have to use variable_dump_name here
boolean
sc_pre_process_for_simplex(Psysteme sc)
{
boolean flag,changed;
Pvecteur v = VECTEUR_NUL;
Pcontrainte c = NULL;

 number_of_variables_added = 0;
 changed = TRUE;

// doing within inequations
// fprintf(stderr,"\nDN sc_pre_process_for_simplex: only with inequation, for the moment ! \n");
 for(c = sc->inegalites; c != NULL; c = c->succ) {
    flag = FALSE;
    for(v = c->vecteur; v != VECTEUR_NUL; v = v->succ) {

      if (value_pos_p(val_of(v))){ //of if : test positive or negative

          if (value_gt(val_of(v),(Value)MAX_COEFF_CAN_HAVE)) {
            //if (flag) {
	      changed = try_to_change_system(sc,c,v,TRUE);//positive sign
	      if (!changed) {
		fprintf(stderr,"\n Cannot add more variable ! \n");
		if (number_of_variables_added) return TRUE;
		else return FALSE;
	      }
	      //}else{
	      // flag = TRUE;
	      //}//flag = decrease n-1 big coeff, no flag = decrease all big coeff
	  }

      }else{// of if : test positive or negative
	
          if (value_lt(val_of(v),value_uminus((Value)MAX_COEFF_CAN_HAVE))) {
	    //if (flag) {
	      changed = try_to_change_system(sc,c,v,FALSE);//negative sign
	      if (!changed) {
		fprintf(stderr,"\n Cannot add more variable ! \n");
		if (number_of_variables_added) return TRUE;
		else return FALSE;
	      }
	      // }else{
	      //flag = TRUE;
	      //}
	  }
      }//of if : test positive or negative

    }//of inside for 
 }//of outside for

// doing within equations

 if (number_of_variables_added) return TRUE;
 return FALSE;

}//of sc_pre_process_for_simplex

/*use only with sc_pre_process_for_simplex*/

//We only act on c->vecteur, not on c; v->val, not on v :)
boolean 
try_to_change_system(Psysteme sc, Pcontrainte c, Pvecteur v, boolean positive_sign){
Variable tmp = NULL;
Pcontrainte c_tmp = NULL;
Pvecteur v_tmp = VECTEUR_NUL;
char variable_added[20];

    if (number_of_variables_added == MAX_NB_VARS_CAN_ADD) return FALSE;    
  
    // create new variable
    sprintf(variable_added,"variable_added_%d",number_of_variables_added);
    tmp = variable_make(variable_added);

    //add the new variable into the equation or inequation with coeff = (Bigcoeff - CHOSEN_NUMBER)
    //add new <variable tmp, coeff> in the head of list of vecteur
    if (positive_sign) {
       value_substract(v->val,(Value)(CHOSEN_NUMBER));
       c->vecteur = vect_chain(c->vecteur,tmp,v->val);
    }
    else{
       value_addto(v->val,(Value)(CHOSEN_NUMBER));
       c->vecteur = vect_chain(c->vecteur,tmp,v->val); 
    }

    // Change the value of coeff with (CHOSEN_NUMBER) in the current variable.
    if (positive_sign) {
       value_assign(v->val,(Value)(CHOSEN_NUMBER));
       //can use vect_chg_coeff here
    }else {
       value_assign(v->val,value_uminus((Value)CHOSEN_NUMBER));
    }    
    
    //create new equation new_coeff.old_variable - coeff.variable_added == 0
    //create new vecteur
    v_tmp = vect_new(var_of(v),VALUE_ONE);
    v_tmp = vect_chain(v_tmp,tmp,VALUE_MONE);

    //become comtrainte
    c_tmp = contrainte_make(v_tmp);
    
    //add the equation into the system
    //as append the new contrainte directly into the system
    sc->egalites = contrainte_append(sc->egalites,c_tmp);
    sc->nb_eq ++;
    
  
    //recompute base ??? inside sc_simplex_feasibilty_ofl_ctrl, we already do it
    //should call inside simplex

    number_of_variables_added ++;
    //fprintf(stderr,"\nIn [try_to_change_system]: number_of_variables_added = %d\n",number_of_variables_added);
    
    return TRUE;
}//of try_to_change_system

