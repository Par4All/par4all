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
 * Pls see dn_implementation.ps for recent changes.
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

#ifdef FILTERING

#include <signal.h>

#define EXCEPTION_PRINT_LINEAR_SIMPLEX TRUE
#define EXCEPTION_PRINT_FM TRUE
#define EXCEPTION_PRINT_JANUS TRUE

#define FILTERING_TIMEOUT_FM filtering_timeout_FM
#define FILTERING_TIMEOUT_LINEAR_SIMPLEX filtering_timeout_S
#define FILTERING_TIMEOUT_JANUS filtering_timeout_J

#define FILTERING_DIMENSION_FEASIBILITY filtering_dimension_feasibility
#define FILTERING_NUMBER_CONSTRAINTS_FEASIBILITY filtering_number_constraints_feasibility
#define FILTERING_DENSITY_FEASIBILITY filtering_density_feasibility
#define FILTERING_MAGNITUDE_FEASIBILITY (Value) filtering_magnitude_feasibility

#endif

#define SWITCH_HEURISTIC_FLAG sc_switch_heuristic_flag

static int feasibility_sc_counter = 0;

boolean FM_timeout = FALSE;
boolean J_timeout = FALSE;
boolean S_timeout = FALSE;

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

/*****duong - set timeout with signal and alarm*****/
#ifdef FILTERING

static void 
filtering_catch_alarm_FM (int sig)
{  
  alarm(0); //clear the alarm
  FM_timeout = TRUE;
}
static void 
filtering_catch_alarm_J (int sig)
{  
  alarm(0); //clear the alarm
  J_timeout = TRUE;
}
static void
filtering_catch_alarm_S (int sig)
{
  alarm(0);
  S_timeout = TRUE;
}
#endif
/*  just a test to improve the Simplex/FM decision.
 * c is a list of constraints, equalities or inequalities
 * pc is the number of constraints in the list
 * pv is the number of non-zero coefficients in the system
 * magnitude is the biggest coefficent in the system
 * pc, pv and magnitude MUST be initialized. They are multiplied by weight.
 *
 * Modif: (this function were not in use, but may be needed in the futur)
 *        can get the magnitude of the sc (the biggest coefficent)
 *        should be called from anywhere, for the filters = remove static DN:25/02/03
 */
void
decision_data(c, pc, pv, magnitude, weight)
Pcontrainte c;
int *pc, *pv;
Value *magnitude;
int weight;
{
  Pvecteur v;    
  for(; c!=NULL; c=c->succ) {
    v=c->vecteur;
    if (v!=NULL) {
      (*pc)+=weight;
      for(; v!=NULL; v=v->succ) 
	if (var_of(v)!=TCST) {
	  (*pv)+=weight;
	  if value_gt(value_abs(val_of(v)),(*magnitude)) value_assign(*magnitude,value_abs(val_of(v)));
	}
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

#define LINEAR_SIMPLEX_PROJECT_EQ_METHOD 11
#define LINEAR_SIMPLEX_NO_PROJECT_EQ_METHOD 12
#define FM_METHOD 13
#define JANUS_METHOD 14
#define ALL_METHOD 15
// We can add other heuristic if we need in an easy way.
//Note: FM is not good for big sc, but good enough for small sc.
//Simplex build a tableau, so it's not good for small sc. 
#define HEURISTIC1 1 //Keep the old heuristic of Fabien.
#define HEURISTIC2 2 //Replace Simplex by Janus in heuristic 1
#define HEURISTIC3 3 //Only for experiment. Test successtively 3 methods to see the coherence
#define HEURISTIC4 4 //The best?: (Linear Simplex vs Janus) try to use the method that succeeded recently. If failed than turn to another. Rely on the fact that the sc are similar. [optional?] : If the 2 methods fail, then call FM. This can solve many sc, in fact. 
static int method_used = 0;//means LINEAR_SIMPLEX :-)

static boolean internal_sc_feasibility
(Psysteme sc, int heuristic, boolean int_p, int ofl_ctrl)
{   
  boolean ok = TRUE;
  int method, n_var, n_cont_eq = 0, n_ref_eq = 0, n_cont_in = 0, n_ref_in = 0;
  Value magnitude;

 feasibility_sc_counter ++;
  
  //We can put the size filters here! filtering timeout is integrated in the methods themself
  //size filtering: dimension,number_constraints, density, magnitude

#ifdef FILTERING
   //Begin size filters
  
  if (TRUE) {
    int dimens; int nb_cont_eq = 0; int nb_ref_eq = 0; int nb_cont_in = 0; int nb_ref_in = 0;

    dimens = sc->dimension; value_assign(magnitude,VALUE_ZERO);
    decision_data(sc_egalites(sc), &nb_cont_eq, &nb_ref_eq, &magnitude, 1);
    decision_data(sc_inegalites(sc), &nb_cont_in, &nb_ref_in, &magnitude, 1);
  
    if ((FILTERING_DIMENSION_FEASIBILITY)&&(dimens>=FILTERING_DIMENSION_FEASIBILITY)) {
      char *directory_name = "feasibility_dimension_filtering_SC_OUT";
      sc_default_dump_to_files(sc,feasibility_sc_counter,directory_name);
    }
    if ((FILTERING_NUMBER_CONSTRAINTS_FEASIBILITY)&&((nb_cont_eq + nb_cont_in) >= FILTERING_NUMBER_CONSTRAINTS_FEASIBILITY)) {
      char *directory_name = "feasibility_number_constraints_filtering_SC_OUT";
      sc_default_dump_to_files(sc,feasibility_sc_counter,directory_name);  
    } 
    if ((FILTERING_DENSITY_FEASIBILITY)&&((nb_ref_eq + nb_ref_in) >= FILTERING_DENSITY_FEASIBILITY)) {
      char *directory_name = "feasibility_density_filtering_SC_OUT";
      sc_default_dump_to_files(sc,feasibility_sc_counter,directory_name);  
    }
    if ((value_notzero_p(FILTERING_MAGNITUDE_FEASIBILITY))&&(value_gt(magnitude,FILTERING_MAGNITUDE_FEASIBILITY))) {
      char *directory_name = "feasibility_magnitude_filtering_SC_OUT";
      sc_default_dump_to_files(sc,feasibility_sc_counter,directory_name);
    }
  }
  
  //End size filters
#endif

  switch(heuristic) {
  case (HEURISTIC1):
    {
      method=0, n_var =0, n_cont_eq = 0, n_ref_eq = 0, n_cont_in = 0, n_ref_in = 0;
      n_var = sc->dimension; value_assign(magnitude,VALUE_ZERO);
      decision_data(sc_egalites(sc), &n_cont_eq, &n_ref_eq, &magnitude, 1);
      decision_data(sc_inegalites(sc), &n_cont_in, &n_ref_in, &magnitude, 1);
      // HEURISTIC1
      //  if nb_ineg >= 10 and nb_eg < 6 then use LSimplex (replace n eg by 2n ineg) 
      //  if nb_ineg >= 10 and nb_eg >= 6 then project eg and use LSimplex
      //  if nb_ineg < 10 then use FM

      if (n_cont_in >= 10) {	
	if (n_cont_eq >= 6) {method=LINEAR_SIMPLEX_PROJECT_EQ_METHOD;}
	else {method=LINEAR_SIMPLEX_NO_PROJECT_EQ_METHOD;}
      } else {
	method = FM_METHOD;
      }
      break;
    }
  case (HEURISTIC2):
    {
      method=0, n_var =0, n_cont_eq = 0, n_ref_eq = 0, n_cont_in = 0, n_ref_in = 0;
      n_var = sc->dimension; value_assign(magnitude,VALUE_ZERO);
      decision_data(sc_egalites(sc), &n_cont_eq, &n_ref_eq, &magnitude, 1);
      decision_data(sc_inegalites(sc), &n_cont_in, &n_ref_in, &magnitude, 1);

      if (n_cont_in >= 10) {	
	method = JANUS_METHOD;	
      } else { method = FM_METHOD;}
      break;
    }
  case (HEURISTIC3):
    {
      method=0, n_var =0, n_cont_eq = 0, n_ref_eq = 0, n_cont_in = 0, n_ref_in = 0;
      n_var = sc->dimension; value_assign(magnitude,VALUE_ZERO);
      decision_data(sc_egalites(sc), &n_cont_eq, &n_ref_eq, &magnitude, 1);
      decision_data(sc_inegalites(sc), &n_cont_in, &n_ref_in, &magnitude, 1);
      
      method = ALL_METHOD;
      break; 
    }
  case (HEURISTIC4) : //test by FM one more time here? That helps in some cases!
    {
      method=0, n_var =0, n_cont_eq = 0, n_ref_eq = 0, n_cont_in = 0, n_ref_in = 0;
      n_var = sc->dimension; value_assign(magnitude,VALUE_ZERO);
      decision_data(sc_egalites(sc), &n_cont_eq, &n_ref_eq, &magnitude, 1);
      decision_data(sc_inegalites(sc), &n_cont_in, &n_ref_in, &magnitude, 1);
 
      CATCH(overflow_error) {

	//	ifscdebug(5) {fprintf(stderr,"nb_exceptions af %d\n",linear_number_of_exception_thrown);}

	if (n_cont_in >= 10) {
	  if (method_used==JANUS_METHOD) {
	    method_used = 0;//LINEAR_SIMPLEX
	    ifscdebug(5) {fprintf(stderr,"J failes so change to LS ...");}
	    if (n_cont_eq>=6) {	      
	      ok = sc_simplexe_feasibility_ofl_ctrl_timeout_ctrl(sc,TRUE,int_p,ofl_ctrl);
	    }else{
	      ok = sc_simplexe_feasibility_ofl_ctrl_timeout_ctrl(sc,FALSE,int_p,ofl_ctrl);
	    }
	    ifscdebug(5) {fprintf(stderr," ...Passed\n");}
	    linear_number_of_exception_thrown --;
	  } else {
	    method_used = JANUS_METHOD;
	    ifscdebug(5) {fprintf(stderr,"LS failed so change to J ...");}
	    ok = sc_janus_feasibility_ofl_ctrl_timeout_ctrl(sc,ofl_ctrl);	
	    ifscdebug(5) {fprintf(stderr," ...Passed\n");}
	    linear_number_of_exception_thrown--; 
	  }
	  //	  ok = sc_fourier_motzkin_feasibility_ofl_ctrl_timeout_ctrl(sc,int_p,ofl_ctrl);
	} else {
	  ifscdebug(5) {fprintf(stderr,"\nFM fail with small sc => bug??? ...");}
	  if (method_used == JANUS_METHOD) {
	    ok = sc_janus_feasibility_ofl_ctrl_timeout_ctrl(sc,ofl_ctrl);
	    //Janus
	  }else {	    
	    if (n_cont_eq>=6) {	      
	      ok = sc_simplexe_feasibility_ofl_ctrl_timeout_ctrl(sc,TRUE,int_p,ofl_ctrl);
	    }else{
	      ok = sc_simplexe_feasibility_ofl_ctrl_timeout_ctrl(sc,FALSE,int_p,ofl_ctrl);
	    }
	  }
	  ifscdebug(5) {fprintf(stderr," ...Passed\n");}
	  linear_number_of_exception_thrown--;
	}	
      }
      TRY {
	if (n_cont_in >= 10) {
	  if (method_used == JANUS_METHOD) {
	    ok = sc_janus_feasibility_ofl_ctrl_timeout_ctrl(sc,ofl_ctrl);
	  }else {	    
	    if (n_cont_eq>=6) {	      
	      ok = sc_simplexe_feasibility_ofl_ctrl_timeout_ctrl(sc,TRUE,int_p,ofl_ctrl);
	    }else{
	      ok = sc_simplexe_feasibility_ofl_ctrl_timeout_ctrl(sc,FALSE,int_p,ofl_ctrl);
	    }
	  }
	}else{
	  //FM
	  ok = sc_fourier_motzkin_feasibility_ofl_ctrl_timeout_ctrl(sc,int_p,ofl_ctrl);
	}
	UNCATCH(overflow_error);    
      }//of TRY
      
      break;
    }

  default: //use heuristic1
    {
      method=0, n_var =0, n_cont_eq = 0, n_ref_eq = 0, n_cont_in = 0, n_ref_in = 0;
      n_var = sc->dimension; value_assign(magnitude,VALUE_ZERO);
      decision_data(sc_egalites(sc), &n_cont_eq, &n_ref_eq, &magnitude, 1);
      decision_data(sc_inegalites(sc), &n_cont_in, &n_ref_in, &magnitude, 1);
      // HEURISTIC1
      //  if nb_ineg >= 10 and nb_eg < 6 then use Simplex (replace n eg by 2n ineg) 
      //  if nb_ineg >= 10 and nb_eg >= 6 then project eg and use Simplex
      //  if nb_ineg < 10 then use FM

      if (n_cont_in >= 10) {	
	if (n_cont_eq >= 6) {method=LINEAR_SIMPLEX_PROJECT_EQ_METHOD;}
	else {method=LINEAR_SIMPLEX_NO_PROJECT_EQ_METHOD;}
      } else {
	method = FM_METHOD;
      }
      break;
    }
  
  }//of switch heuristic

  // fprintf(stderr, "in=%d eq=%d method=%d magnitude=", n_cont_in, n_cont_eq, method);print_Value(magnitude);

  switch(method){
  
  case (LINEAR_SIMPLEX_PROJECT_EQ_METHOD): 
    {
      ok = sc_simplexe_feasibility_ofl_ctrl_timeout_ctrl(sc,TRUE,int_p,ofl_ctrl);
      break;
    }
  case (LINEAR_SIMPLEX_NO_PROJECT_EQ_METHOD): 
    {
      ok = sc_simplexe_feasibility_ofl_ctrl_timeout_ctrl(sc,FALSE,int_p,ofl_ctrl);
      break;
    }
  case (FM_METHOD): 
    {
      ok = sc_fourier_motzkin_feasibility_ofl_ctrl_timeout_ctrl(sc,int_p,ofl_ctrl);
      break;
    }
  case (JANUS_METHOD): 
    {
      ok = sc_janus_feasibility_ofl_ctrl_timeout_ctrl(sc,ofl_ctrl);
      break;
    }//end of case JANUS 
  case (ALL_METHOD):
    {

      boolean okS = TRUE,okJ = TRUE,okFM = TRUE;
      CATCH(overflow_error) {
	ifscdebug(5) {
	  fprintf(stderr,"WARNING: Janus or Simplex failed. Let's go with FM\n");
	}
	okFM = sc_fourier_motzkin_feasibility_ofl_ctrl_timeout_ctrl(sc,int_p,ofl_ctrl);
	ok = okFM;// the most reliable???
      }
      TRY {
	if (n_cont_eq >= 10) {
	  okS = sc_simplexe_feasibility_ofl_ctrl_timeout_ctrl(sc,TRUE,int_p,ofl_ctrl);
	} else { 
	  okS = sc_simplexe_feasibility_ofl_ctrl_timeout_ctrl(sc,FALSE,int_p,ofl_ctrl);
	}
	okJ = sc_janus_feasibility_ofl_ctrl_timeout_ctrl(sc,ofl_ctrl);
    
	okFM = sc_fourier_motzkin_feasibility_ofl_ctrl_timeout_ctrl(sc,int_p,ofl_ctrl);
      
	ifscdebug(5) {
	  if (okS != okFM) {
	    fprintf(stderr,"WARNING: okS %d != okFM %d\n",okS,okFM);
	    sc_default_dump(sc);
	  }
	  if (okJ != okFM) {
	    fprintf(stderr,"WARNING: okJ %d != okFM %d\n",okJ,okFM);
	    sc_default_dump(sc);
	  }
	  assert(okS == okFM);
	  assert(okJ == okFM);
	}
	ok = okFM; 
	UNCATCH(overflow_error);
      }     
      break;
    }  
  default:
    {
      //rien faire      
      //ifscdebug(5){fprintf(stderr,"methoddefault");assert(FALSE);}
       break;
    }

  }//of switch method

  return ok;
}


boolean 
sc_feasibility_ofl_ctrl(sc, integer_p, ofl_ctrl, ofl_res)
Psysteme sc;
boolean integer_p;
int ofl_ctrl;
boolean ofl_res;
{ 
  boolean 
    ok = FALSE,
    catch_performed = FALSE;
  int heuristic = 0;
  
  ifscdebug(5) {    
    if (sc->dimension < 0) {
      sc_default_dump(sc);
      sc_fix(sc);
      assert(FALSE);
    }
  }

  if (sc_rn_p(sc)) {
    ifscdebug(5) {
      fprintf(stderr,"\n sc_rn is given to sc_feasibility_ofl_ctrl : return TRUE");
    }// this should be treated somewhere else -> faster
    return TRUE;
  }
  if (sc_empty_p(sc)) {
    ifscdebug(5) {
      fprintf(stderr,"\n sc_empty is given to sc_feasibility_ofl_ctrl : return FALSE");
    }// this should be treated somewhere else -> faster
    return FALSE;
  }

  switch (ofl_ctrl) {

  case OFL_CTRL :
    ofl_ctrl = FWD_OFL_CTRL;
    catch_performed = TRUE;    
    CATCH(overflow_error) {
	ok = ofl_res;
	catch_performed = FALSE;
	/* 
	 *   PLEASE do not remove this warning.
	 *
	 *   FC 30/01/95
	 */
	linear_number_of_exception_thrown--;
	fprintf(stderr, "\n[sc_feasibility_ofl_ctrl] "
		"arithmetic error (%d) -> %s\n",
		heuristic, ofl_res ? "TRUE" : "FALSE");
 
      break;
      }		
  default: {
      //What we need to do here: choose a method or a heuristic predifined by a variable of environment
      //and catch an overflow exception if it happens DN240203
      heuristic = SWITCH_HEURISTIC_FLAG;//default, heuristic flag is 0         
      ok = internal_sc_feasibility(sc, heuristic, integer_p, ofl_ctrl);
    }
  }  
  
  if (catch_performed)
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
 * Back to the version modifying the sc. Calls of this function should store the sc first
 * or pls call sc_fourier_motzkin_ofl_ctrl_timeout_ctrl DN210203 
 */
boolean 
sc_fourier_motzkin_feasibility_ofl_ctrl(s, integer_p, ofl_ctrl)
Psysteme s;
boolean integer_p;
int ofl_ctrl;
{
  boolean faisable = TRUE;

  CATCH(any_exception_error) {
    // maybe timeout_error or overflow_error
    if (ofl_ctrl == FWD_OFL_CTRL) {
	RETHROW(); //rethrow whatever the exception is
    } else {
      fprintf(stderr,"\nWARNING [sc_fourier_motzkin_feasibility_ofl_ctrl] without OFL_CTRL => RETURN TRUE\n");
      return TRUE;// default is feasible
    }
  }
 
  if (s == NULL) return TRUE;
  
  ifscdebug(8)
    {
      fprintf(stderr, "[sc_fourier_motzkin_feasibility_ofl_ctrl] system:\n");
      sc_fprint(stderr, s, default_variable_to_string);
    }

  
  s = sc_elim_double_constraints(s);

  if (s != NULL)
  {
    /* a small basis if possible... (FC).
     */
    base_rm(sc_base(s));
    sc_creer_base(s);

    faisable = sc_fm_project_variables(&s, integer_p, FALSE, ofl_ctrl);

      sc_rm(s); //should remove the copy of the sc.
      s = NULL;

  }
  else 
    /* sc_kill_db_eg a de'sallouer s a` la detection de 
       sa non-faisabilite */
    faisable = FALSE;

  UNCATCH(any_exception_error);

  return faisable;
}

boolean
sc_fourier_motzkin_feasibility_ofl_ctrl_timeout_ctrl(sc,int_p,ofl_ctrl)
Psysteme sc;
boolean int_p;
int ofl_ctrl;
{
  Psysteme w = NULL;
  boolean ok = TRUE;

  if (sc->dimension == 0) return TRUE;

  CATCH(any_exception_error) {

    ifscdebug(5) {
      fprintf(stderr,"sc_fourier_motzkin_feasibility_ofl_ctrl_timeout_ctrl fails");
      //      sc_default_dump(sc);
    }

#ifdef FILTERING
    if (FILTERING_TIMEOUT_FM) alarm(0);
    if (EXCEPTION_PRINT_FM) {
      char *directory_name = "feasibility_FM_fail_SC_OUT";	
      sc_default_dump_to_files(sc,feasibility_sc_counter,directory_name);       
    }
#endif

    //if (w) sc_rm(w);
    if (ofl_ctrl==FWD_OFL_CTRL) {
      linear_number_of_exception_thrown -=2;//there r 2 exceptions here!
      THROW(overflow_error);//need to handle numberofexception somewhere
    } else {
      fprintf(stderr,"\nWARNING [sc_fourier_motzkin_feasibility_ofl_ctrl_timeout_ctrl] without OFL_CTRL => RETURN TRUE\n");
      return TRUE;
    }    
  }
  TRY {

    w = sc_copy(sc);

#ifdef FILTERING
    //start the alarm
    if (FILTERING_TIMEOUT_FM) {
      signal(SIGALRM, filtering_catch_alarm_FM);
      alarm(FILTERING_TIMEOUT_FM);
      FM_timeout = FALSE;
    }
#endif

    if (w) {
      ok= sc_fourier_motzkin_feasibility_ofl_ctrl(w,int_p,ofl_ctrl);   
    }
    else ok = TRUE;
    
#ifdef FILTERING
    if (FILTERING_TIMEOUT_FM) {
      alarm(0);
      if (FM_timeout) {
	char *directory_name = "feasibility_FM_timeout_filtering_SC_OUT";
	sc_default_dump_to_files(sc,feasibility_sc_counter,directory_name);
      }
    }
#endif

  }//of TRY

  //  if (w) sc_rm(w);  //sc_fourier_motzkin_feasibility_ctrl_ofl has freed the memory. base
    
  UNCATCH(any_exception_error);

  return ok;
}

boolean 
sc_simplexe_feasibility_ofl_ctrl_timeout_ctrl(sc, project_eq_p, int_p, ofl_ctrl)
Psysteme sc;
boolean project_eq_p;
boolean int_p;
int ofl_ctrl;
{ 
  Psysteme w = NULL;
  boolean ok = TRUE;

  if (sc->dimension == 0) return TRUE;
  
  CATCH(any_exception_error) {

    ifscdebug(5) {
	fprintf(stderr,"sc_simplexe_feasibility_ofl_ctrl_timeout_ctrl fails");
	//      sc_default_dump(sc);
    }
  
#ifdef FILTERING
    if (FILTERING_TIMEOUT_LINEAR_SIMPLEX) alarm(0);

    if (EXCEPTION_PRINT_LINEAR_SIMPLEX) {
      char *directory_name = "feasibility_linear_simplex_fail_SC_OUT";	
      sc_default_dump_to_files(sc,feasibility_sc_counter,directory_name);
    }
#endif

    if (w) sc_rm(w);
    if (ofl_ctrl == FWD_OFL_CTRL) {
      linear_number_of_exception_thrown -=2;//there r 2 exceptions here!
      THROW(overflow_error);//need to handle numberofexception somewhere
    } else {
      fprintf(stderr,"\nWARNING [sc_simplex_feasibility_ofl_ctrl_timeout_ctrl] without OFL_CTRL => RETURN TRUE\n");
      return TRUE;// default is feasible
    }
  }
  TRY {      

#ifdef FILTERING
    if (FILTERING_TIMEOUT_LINEAR_SIMPLEX) {
      signal(SIGALRM, filtering_catch_alarm_S);
      alarm(FILTERING_TIMEOUT_LINEAR_SIMPLEX); 
      S_timeout = FALSE;
    }
#endif

    if (project_eq_p) {
      w = sc_copy(sc);
      ok = sc_fm_project_variables(&w, int_p, TRUE, ofl_ctrl);
      ok = sc_simplexe_feasibility_ofl_ctrl(w,ofl_ctrl);
    }else {    
      ok = sc_simplexe_feasibility_ofl_ctrl(sc,ofl_ctrl);
    } 

#ifdef FILTERING
    if (FILTERING_TIMEOUT_LINEAR_SIMPLEX) {
      alarm(0);
      if (S_timeout) {
	char *directory_name = "feasibility_linear_simplex_timeout_filtering_SC_OUT";
	sc_default_dump_to_files(sc,feasibility_sc_counter,directory_name);
      }
    }
#endif
     
  }//of TRY  

  if (w) sc_rm(w);    
  UNCATCH(any_exception_error);

  return ok;
}

boolean
sc_janus_feasibility_ofl_ctrl_timeout_ctrl(sc,ofl_ctrl)
Psysteme sc;
boolean ofl_ctrl;
{
  Psysteme w = NULL;
  int ok;

  //DN: We should be sure that the sc is not null in the sc_feasibility_ofl_ctrl, but for direct call of Janus ... 
  if (sc) {
    if (sc->dimension == 0) return TRUE;
  }
  else return TRUE;

  //sc_empty_p is filtered, (sc_fix is called), so there's no other reason for janus to fail ...
  //maybe a sc_not_easy_to_see_empty
  // TODO aware of vectors of one element: 0 <= 1.

  CATCH(any_exception_error) {

    ifscdebug(5) {
      fprintf(stderr,"sc_janus_feasibility_ofl_ctrl_timeout_ctrl fails");
      //      sc_default_dump(sc);
    }

#ifdef FILTERING
    if (FILTERING_TIMEOUT_JANUS) alarm(0);

    if (EXCEPTION_PRINT_JANUS) {
      char *directory_name = "feasibility_janus_fail_SC_OUT";	
      sc_default_dump_to_files(w,feasibility_sc_counter,directory_name);
    }
#endif

    if (w) sc_rm(w);
    linear_number_of_exception_thrown -=1;//there r 2 exceptions here!
    THROW(overflow_error);//need to handle number of exceptions somewhere
  }
  TRY { 
    if (sc) {w = sc_copy(sc);}
    else return TRUE;
    if (w) {sc_fix(w);}

    if (w) {

#ifdef FILTERING
      //start the alarm
      if (FILTERING_TIMEOUT_JANUS) {
	signal(SIGALRM, filtering_catch_alarm_J);
	alarm(FILTERING_TIMEOUT_JANUS);
	J_timeout = FALSE;
      }
#endif
 
      ok = sc_janus_feasibility(w); //sc_janus_feasibility returns type int, not boolean

#ifdef FILTERING
      if (FILTERING_TIMEOUT_JANUS) {
	alarm(0);
	if (J_timeout){
	  char *directory_name = "feasibility_janus_timeout_filtering_SC_OUT";
	  sc_default_dump_to_files(sc,feasibility_sc_counter,directory_name);
	}
      }
#endif

    } else return TRUE;
    
  }//of TRY

  UNCATCH(any_exception_error);

  if (ok<3) {    
    // result found
      if (w) sc_rm(w);
      if (ok > 0) return TRUE;
      else return FALSE;
    } else {
      // result not found
      ifscdebug(5) {
	if (ok ==7) fprintf(stderr,"TRIED JANUS BUT OVERFLOW !!\n");
	if (ok ==6) fprintf(stderr,"TRIED JANUS BUT BUG OF PROGRAMMATION IN JANUS !!\n");
	if (ok ==5) fprintf(stderr,"TRIED JANUS BUT WRONG PARAMETER !!\n");
	if (ok ==4) fprintf(stderr,"TRIED JANUS BUT ARRAY OUT OF BOUNDARY !!\n");
	if (ok ==3) fprintf(stderr,"TRIED JANUS BUT NUMBER OF PIVOTAGE TOO BIG, MIGHT BOUCLE !!\n");
	if (ok ==8) fprintf(stderr,"TRIED JANUS BUT pivot anormally small !!\n");
	if (ok ==9) fprintf(stderr,"Janus is not ready for this system of constraints !!\n");//DN20112002
      }
      if (w) sc_rm(w);
      if (ofl_ctrl == FWD_OFL_CTRL) {
	THROW(overflow_error);
      } else {
	fprintf(stderr,"\nWARNING [sc_janus_feasibility_ofl_ctrl_timeout_ctrl] without OFL_CTRL => RETURN TRUE\n");
	return TRUE;// default is feasible
      }
    }  
}
