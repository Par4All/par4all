/******************************************************************
 *
 *		     BOTTOM UP ARRAY RESIZING
 *
 *
*******************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "database.h"
#include "pipsdbm.h"
#include "resources.h"
#include "makefile.h"
#include "misc.h"
#include "control.h"
#include "properties.h"
#include "semantics.h"
#include "transformer.h"
#include "pipsmake.h"
#include "abc_private.h"
#include "effects-generic.h"
#include "effects-convex.h"
#include "effects-simple.h"
#include "conversion.h"
#include "text-util.h" /* for words_to_string*/
#include "instrumentation.h"
#include "transformations.h"

static int number_of_right_array_declarations = 0;
static string current_mod ="";

#define PREFIX_DEC  "$DEC"

static bool
parameter_p(entity e)
{

  return storage_rom_p(entity_storage(e)) && 
    !(value_undefined_p(entity_initial(e))) &&
    value_symbolic_p(entity_initial(e)) &&
    type_functional_p(entity_type(e));
}

/*La valeur retournée est TRUE si la variable v est un parameter 
  ou une variable common. Sinon, elle rend la valeur FALSE.*/

static bool
variable_is_param_common_p(entity e)
{
  if ( (parameter_p(e)) || (variable_in_common_p(e)) )  return (TRUE);
  else 
    return (FALSE);
}

/*Rendre TRUE si la variable v n'est pas un parameter, ni une variable common
  et v, v_phi ne sont pas identiques.
  En vice-versa, retourner la valeur FALSE.*/

static bool
variable_to_project_p(Variable v_phi, Variable v)
{
  if (v!=v_phi) {
    entity e = (entity) v;
    if (variable_is_param_common_p(e) || storage_formal_p(entity_storage(e))) return FALSE;
    return TRUE;
  }
  return FALSE;
}

static boolean extract_constraint_on_var(Pvecteur p_var, Variable var, int val,  Pvecteur *ptmp)
{
  boolean divisible=TRUE; 
  Pvecteur p_tmp = VECTEUR_NUL,pv;
  for (pv = p_var; !VECTEUR_NUL_P(pv) && divisible; pv=pv->succ) {
    Variable v1=vecteur_var(pv); 
    if (v1 ==TCST) 
      divisible &= value_zero_p(value_mod(vect_coeff(TCST,p_var), val));
    else if (v1!= var)
      divisible &=(basic_int_p(variable_basic(type_variable(entity_type((entity) v1 )))) 
		   &&  value_zero_p(value_mod(pv->val, val)));
  }
  if (divisible) {
    p_tmp = vect_dup(p_var);
    vect_erase_var(&p_tmp,var);
    for (pv = p_tmp; !VECTEUR_NUL_P(pv); pv=pv->succ) {
      Variable v1=vecteur_var(pv);
      vect_chg_coeff(&pv,v1, value_uminus(value_div(pv->val, val)));
    }
    *ptmp = p_tmp;
    return(TRUE);
  }
  else {
    *ptmp = VECTEUR_NUL;
    return (FALSE);
  }
}

/* Traiter les égalités d'un système de contraintes ps.
  . Si la variable var apparaît dans ces égalités, *pe va contenir le vecteur
    définissant var. Ex. :
     Equation de la forme: k*var + q1*C1 + ... + p1*N1 + ... == 0
     *pe contient le vecteur : (-q1/k)*C1 + ... + (-p1/k)*N1
  . Sinon, pe est nulle. */

static bool
extract_constraint_from_equalitites(Psysteme ps, Variable var, Pvecteur *pe)
{
  Pcontrainte pc;
  Value v_phi = VALUE_ZERO;
  Pvecteur p_var = VECTEUR_NUL, ptmp= VECTEUR_NUL; 
  boolean result=FALSE;   
  if (!SC_UNDEFINED_P(ps) && !CONTRAINTE_UNDEFINED_P(ps->egalites) 
      && CONTRAINTE_NULLE_P(ps->egalites))  {
    *pe = VECTEUR_NUL;
    return(FALSE);
  }
  for (pc = ps->egalites; pc != NULL; pc = pc->succ) {    
    /* équation de la forme: k*var + q1*C1 + ... + p1*N1 + ... == 0 */
    p_var = contrainte_vecteur(pc);
    v_phi = vect_coeff(var,p_var);
    if (v_phi) {
      result =  extract_constraint_on_var(p_var,var,v_phi,&ptmp);
      *pe=ptmp;
      return(result);}
  }
  *pe = VECTEUR_NUL;
  return(FALSE);
}

/*Simplifier le Pvecteur pv et extraire les parametres qui apparaissent dans pv.
  Le Pvecteur résultat ne contient que les entiers et les variables commons.*/
static Pvecteur
vect_partial_eval(Pvecteur pv)
{
  Pvecteur v = VECTEUR_NUL, v_tmp = VECTEUR_NUL, v_p_tmp = VECTEUR_NUL;
  Value v_tcst = VALUE_ZERO;
  while (!VECTEUR_NUL_P(pv)) {
    Variable var = pv->var;
    if (var == TCST)
      v_tcst = value_plus(v_tcst, val_of(pv));
    else if (parameter_p((entity) var)) {
      Value val = int_to_value(expression_to_int(symbolic_expression(value_symbolic(entity_initial((entity) var)))));
      pips_debug(8, "The variable %s is a parameter and its value is %lld\n",
		 entity_local_name((entity) var), val);
      v_tcst = value_plus(v_tcst, value_direct_multiply(val, val_of(pv)));
    }
    else {
      v_tmp = vect_new(pv->var, pv->val);
      if (v == VECTEUR_NUL) {
	v_p_tmp = v_tmp;
	v = v_tmp;
      }
      else {
	v_p_tmp->succ = v_tmp;
	v_p_tmp = v_tmp;
      }
    }
    pv = pv->succ;
  }

  if (v_tcst != VALUE_ZERO) {
    v_tmp = vect_new(TCST, v_tcst);
    if (v == VECTEUR_NUL) {
      v_p_tmp = v_tmp;
      v = v_tmp;
    }
    else v_p_tmp->succ = v_tmp;
  }
  return (v);
}

/* Faire la comparaison entre deux Pvecteurs basés sur le système des préconditions ps_prec.
  Cette fonction fait une combinaison sur 8 cas de la faisabilité d'un système des contraintes
  pour déterminer le Pvecteur supérieur et le Pvecteur inférieur.
  Ces 8 cas proviennent des 3 systèmes de contraintes :
    i)   sc_egal = { ps_prec , pv1 - pv2 = 0 }
   ii)   sc_inf  = { ps_prec , pv1 - pv2 < 0 }  =  { ps_prec , pv1 - pv2 + 1 <= 0 }
  iii)   sc_sup  = { ps_prec , pv2 - pv1 < 0 }  =  { ps_prec , pv2 - pv1 + 1 <= 0 }

  et le tableau de résultats est :
  
  sc_egal    || .T. |     .T.    |     .T.    |    .T.    | .F. |    .F.    |    .F.    | .F.
  --------------------------------------------------------------------------------------------
  sc_inf     || .T. |     .T.    |     .F.    |    .F.    | .T. |    .T.    |    .F.    | .F.
  --------------------------------------------------------------------------------------------
  sc_sup     || .T. |     .F.    |     .T.    |    .F.    | .T. |    .F.    |    .T.    | .F.
  ============================================================================================
  Conclusion ||  *  | pv1 <= pv2 | pv1 >= pv2 | pv1 = pv2 |  *  | pv1 < pv2 | pv1 > pv2 |  *
  
  ( " * "  correspondant au cas non-determiné )
 
  Dans le cas non-determiné, le Pvecteur retourné est VECTEUR_UNDEFINED et
  on doit donc traiter le cas de VECTEUR_UNDEFINED avec les autres :
  
    a-/  VECTEUR_UNDEFINED  et  VECTEUR_UNDEFINED
    b-/  VECTEUR_UNDEFINED  et  VECTEUR_ONE (1)
    c-/  VECTEUR_UNDEFINED  et  un vecteur quelconque sauf VECTEUR_ONE */
static Pvecteur
sc_minmax_of_pvector(Psysteme ps_prec, Pvecteur pv1, Pvecteur pv2, boolean is_min)
{
  Psysteme ps_egal = SC_UNDEFINED, 
    ps_inf = SC_UNDEFINED, 
    ps_sup = SC_UNDEFINED,
    pt=  SC_UNDEFINED;
  Pcontrainte pc_egal = CONTRAINTE_UNDEFINED, 
    pc_inf = CONTRAINTE_UNDEFINED, 
    pc_sup = CONTRAINTE_UNDEFINED;
  Pvecteur p1, p2, p_egal, p_inf, p_sup,pvt,pv_1;
  boolean  egal = FALSE, inf = FALSE, sup = FALSE;
  Pvecteur p_one = vect_new(TCST, VALUE_ONE);

  if (VECTEUR_UNDEFINED_P(pv1) && VECTEUR_UNDEFINED_P(pv2)) {
    vect_rm(p_one);
    return VECTEUR_UNDEFINED;
  }
  else if (VECTEUR_UNDEFINED_P(pv1) && !VECTEUR_UNDEFINED_P(pv2)) {
    if (!vect_equal(pv2, p_one)) {
      vect_rm(p_one);
      return (pv2);
    }
    else {
      if (is_min)  
	return(p_one);
      else 
	return VECTEUR_UNDEFINED;
    }
  }
  else if ( VECTEUR_UNDEFINED_P(pv2) && !VECTEUR_UNDEFINED_P(pv1) ) { 
    if (!vect_equal(pv1, p_one)) {
      vect_rm(p_one);
       return (pv1);
       }
    else {
      if (is_min)  
	return(p_one);
      else 
	return VECTEUR_UNDEFINED;
    }
  }
  p1 = vect_partial_eval(pv1);
  p2 = vect_partial_eval(pv2);
  p_egal = vect_substract(p1, p2);
  if (VECTEUR_NUL_P(p_egal)) return(p1);

  /* Creation des trois systemes */
  pvt=vect_dup(p_egal);
  pc_egal = contrainte_make(pvt);
  pt=sc_dup(ps_prec);
  ps_egal = sc_equation_add(pt, pc_egal);
  base_rm(ps_egal->base);
  ps_egal->base = BASE_NULLE;
  sc_creer_base(ps_egal);
  ps_egal = sc_elim_redund(ps_egal);
  
  pv_1= vect_new(TCST, VALUE_ONE);
  p_inf = vect_add(p_egal,pv_1);
  vect_rm(pv_1);
  pc_inf = contrainte_make(p_inf); 
  pt=sc_dup(ps_prec);
  ps_inf = sc_inequality_add(pt, pc_inf);
  base_rm(ps_inf->base);
  ps_inf->base = BASE_NULLE;
  sc_creer_base(ps_inf);
  ps_inf = sc_elim_redund(ps_inf);

  pv_1= vect_new(TCST, VALUE_ONE);
  p_sup = vect_substract(pv_1,p_egal);
  vect_rm(pv_1);
  pc_sup = contrainte_make(p_sup); 
  pt=sc_dup(ps_prec);
  ps_sup = sc_inequality_add(pt, pc_sup);
  base_rm(ps_sup->base);
  ps_sup->base = BASE_NULLE;
  sc_creer_base(ps_sup);
  ps_sup = sc_elim_redund(ps_sup);
  
  CATCH (overflow_error) {
    pips_debug(8, "Overflow detected !\n");  
    sc_free(ps_egal);
    sc_free(ps_inf);
    sc_free(ps_sup);
    return VECTEUR_UNDEFINED;     
  }
  TRY {
    egal = !SC_UNDEFINED_P(ps_egal) && 
      sc_rational_feasibility_ofl_ctrl(ps_egal, OFL_CTRL, TRUE);
    inf =  !SC_UNDEFINED_P(ps_inf) && 
      sc_rational_feasibility_ofl_ctrl(ps_inf, OFL_CTRL, TRUE);
    sup =  !SC_UNDEFINED_P(ps_sup) && 
      sc_rational_feasibility_ofl_ctrl(ps_sup, OFL_CTRL, TRUE);
    sc_free(ps_egal);
    sc_free(ps_inf);
    sc_free(ps_sup);
    UNCATCH (overflow_error);
    if (is_min) { /* Recherche du  minimum  */
     if ( (egal && inf && !sup) || (egal && !inf && !sup) || (!egal && inf && !sup) ) {
	pips_debug(8, "p1 is minimum\n");
	return (p1);
      }
      if ( (egal && !inf && sup) || (egal && !inf && !sup) || (!egal && !inf && sup) ) {
	pips_debug(8, "p2 is minimum\n");
	return (p2);     
      }
      else {
	pips_debug(8, "Non-determined\n");
	return VECTEUR_UNDEFINED;
      }
    }
    else { /* Recherche du  maximum  */
      if ( (egal && inf && !sup) || (egal && !inf && !sup) || (!egal && inf && !sup) ) {
	pips_debug(8, "p2 is maximum\n");
	return (p2);
      }
      if ( (egal && !inf && sup) || (egal && !inf && !sup) || (!egal && !inf && sup) ) {
	pips_debug(8, "p1 is maximum\n");
	return (p1);     
      }
      else {
	pips_debug(8, "Non-determined\n");
	return VECTEUR_UNDEFINED;     
      }
    }    
  }
}

/*Traiter les inégalités d'un système de contraintes.
  Si la variable var apparaît dans les inégalités, cette fonction va retourner la borne inférieure et la borne supérieure
  de la variable var sous la forme de Pvecteur pmin et pmax. Sinon, la valeur retournée est FALSE et les Pvecteurs sont nuls.
  Dans cette fonction, il y a des appels à la fonction sc_min_max_of_pvector() pour comparer deux vecteurs. */

static bool
extract_constraint_from_inequalities(Psysteme ps, Variable var, Psysteme ps_prec, Pvecteur *pe, Pvecteur *pmin, Pvecteur *pmax)
{
  Pcontrainte pc;
  Value v_phi = VALUE_ZERO;
  Pvecteur p_var = VECTEUR_NUL, ptmp = VECTEUR_NUL, p_max = VECTEUR_NUL, p_min = VECTEUR_NUL ; 
  boolean result;
  p_max = *pe;
  if (VECTEUR_NUL_P(*pe)) 
    p_min = vect_new(TCST, VALUE_ONE);
  else  p_min = *pe;
  
  if (!SC_UNDEFINED_P(ps) && !CONTRAINTE_UNDEFINED_P(ps->inegalites) 
      && CONTRAINTE_NULLE_P(ps->inegalites))  {
    *pmax = p_max;
    *pmin = p_min;
    return(FALSE);
  }  
  for (pc = ps->inegalites; pc != NULL; pc = pc->succ) {
    p_var = contrainte_vecteur(pc);
    v_phi = vect_coeff(var,p_var);
    if (v_phi) {
      result =  extract_constraint_on_var(p_var,var,v_phi,&ptmp);
      
      if (value_pos_p(v_phi)) 
	p_max = sc_minmax_of_pvector(ps_prec, p_max, ptmp, FALSE);
      else if (value_neg_p(v_phi)) {
	p_min = sc_minmax_of_pvector(ps_prec, p_min, ptmp, TRUE);
      }
    }
  }
  *pmax = p_max;
  *pmin = p_min;
  return (TRUE);
}
  
/*Cette fonction a été écrite pour déterminer les valeurs minimum et maximum d'une variable dans
  un système de contraintes, elle est donc la fonction principale du programme. 
  . La valeur retournée est FALSE si le système de contraintes est infaisable ou les valeurs min, max sont
    indéterminables. Et vice-versa, la valeur retournée est TRUE.
  . Les pointeurs pmin et pmax contiennent les valeurs des bornes supérieure et inférieure
    de la variable var dans le système de contraintes ps. Ces valeurs sont des Pvecteurs.
  Cette fonction contient les appels aux fonctions sc_egalites_of_variable() et sc_inegalites_of_variable()
  pour traiter les égalités et les inégalités du système ps.  */
static bool
sc_min_max_of_variable(Psysteme ps, Variable var, Psysteme ps_prec, Pvecteur *min, Pvecteur *max)
{
  Pbase b;
  Pvecteur pe = VECTEUR_NUL;
  Psysteme ps_e, ps_i;
  boolean ok1,ok2;
  assert(var!=TCST);  
  *min = vect_new(TCST, VALUE_MIN);
  *max = vect_new(TCST, VALUE_MAX);
  
  /* faire la projection sur toutes les variables sauf var, parametres formels et commons */
  for (b = ps->base; !VECTEUR_NUL_P(b); b = b->succ) {
    Variable v = var_of(b);
    if (variable_to_project_p(var, v)) {
      if (SC_EMPTY_P(ps = sc_projection_pure(ps, v))) 
	return FALSE;
    }
  }
  if (SC_EMPTY_P(ps = sc_normalize(ps)))
    return FALSE;
  
  if (SC_UNDEFINED_P(ps) || ( sc_nbre_inegalites(ps)==0  && sc_nbre_egalites(ps)==0))
    return(FALSE);  
  ps_e = sc_dup(ps);
  ps_i = sc_dup(ps); 
  ok1 = extract_constraint_from_equalitites(ps_e, var, &pe);  
  ok2 = extract_constraint_from_inequalities(ps_i, var, ps_prec, &pe, min, max);
  if (ok2) {
    pips_debug(8, "The upper bound has been found\n");
    return (TRUE);
  }
  vect_rm(pe);
  sc_rm(ps_e); 
  sc_rm(ps_i);
  pips_debug(8, "The upper bound has not been found\n");
  return (FALSE);
}

static void bottom_up_adn_array_region(region reg, entity e, Psysteme pre)
{  
  variable v = type_variable(entity_type(e));   
  list l_dims = variable_dimensions(v);
  int length = gen_length(l_dims);
  //  dimension last_dim =  find_ith_dimension(l_dims,length);
  entity phi = make_phi_entity(length);
  expression upper = expression_undefined;
  Pvecteur min,max;
  if (!region_undefined_p(reg))
    {
      /* there are cases when there is no region for one array */
      Psysteme ps = sc_dup(region_system(reg));
      if (sc_min_max_of_variable(ps, (Variable) phi, pre, &min, &max))
	upper = Pvecteur_to_expression(max);
      sc_rm(ps);
    }      
  if (expression_undefined_p(upper))
    upper = make_unbounded_expression();   
  if (!unbounded_expression_p(upper))
    number_of_right_array_declarations++;
  
  user_log("%s\t%s\t%s\t%s\t%d\t%s\t", PREFIX_DEC, 
	   db_get_memory_resource(DBR_USER_FILE,current_mod,TRUE), 
	   current_mod,entity_local_name(e),length,words_to_string(words_expression(upper)));  
  print_array_declaration(e);
  user_log("\n");
  user_log("---------------------------------------------------------------------------------------\n");  
}


/*This function finds in the list of regions the read and write regions of e.
  If there are 2 regions, it returns the union region */
static region find_union_regions(list l_regions,entity e)
{
  region reg = region_undefined;
  while (!ENDP(l_regions))
    {
      region re = REGION(CAR(l_regions));
      reference ref = region_reference(re);
      entity array = reference_variable(ref); 
      if (same_entity_lname_p(array,e))
	{
	  if (region_undefined_p(reg))
	    reg = region_dup(re);
	  else
	    reg =  regions_must_convex_hull(reg,re);
	}
      l_regions = CDR(l_regions);
    }
  return reg;
}

bool array_resizing_bottom_up(char* mod_name)
{
  entity mod_ent = local_name_to_top_level_entity(mod_name);
  list l_decl = code_declarations(entity_code(mod_ent)), l_regions = NIL; 
  statement  mod_stmt = (statement) db_get_memory_resource(DBR_CODE, mod_name, TRUE);
  transformer mod_pre;
  Psysteme pre;
  /* If we use summary_precondition and the preconditions are calculated 
     intraprocedurally (no activate PRECONDITION_INTER_FULL/FAST) 
     => pips_error in db_get_memory_resource because 
     summary_precondition of the module is not available. 
     
     NN : I don't see the reason to use summary_precondition because 
     bottom_up_adn is intraprocedural, if summary_precondition is used, 
     we do not have more useful information but may infeasible precondition 
     for no called subroutine => do nothing for that subroutine that is in fact
     a favour point of this approach in comparison with top_down_adn

     So I replaced summary_precondition by precondition*/

  /* transformer mod_pre = (transformer) db_get_memory_resource(DBR_SUMMARY_PRECONDITION, mod_name, TRUE);*/
 
  current_mod = mod_name;  
  set_precondition_map((statement_mapping)
		       db_get_memory_resource(DBR_PRECONDITIONS,mod_name,TRUE));
  set_rw_effects((statement_effects) 
      db_get_memory_resource(DBR_REGIONS, mod_name, TRUE));
  regions_init(); 
  debug_on("ARRAY_RESIZING_BOTTOM_UP_DEBUG_LEVEL");
  debug(1," Begin bottom up array resizing for %s\n", mod_name);
    l_regions = load_rw_effects_list(mod_stmt);  

  /* version la plus rapide mais perte des declarations locales
     notament des pointeurs que l'on peut vouloir conserver dans le cas de
     COCCINELLE
     l_regions = effects_effects((effects) 
      db_get_memory_resource(DBR_SUMMARY_REGIONS, mod_name, TRUE));
  */
  mod_pre = load_statement_precondition(mod_stmt);
  pre = predicate_system(transformer_relation(mod_pre));
  user_log("\n-------------------------------------------------------------------------------------\n");
  user_log("Prefix \tFile \tModule \tArray \tNdim \tNew declaration\tOld declaration\n");
  user_log("---------------------------------------------------------------------------------------\n");
  
  /* This version computes new upper bound for all kind of unnormalized array declarations
     Modification NN: for formal argument arrays only*/
  while (!ENDP(l_decl))
    {
      entity e = ENTITY(CAR(l_decl));
      if (unnormalized_array_p(e))
	{
	  /*  storage s = entity_storage(e);
	      if (storage_formal_p(s))
	      { */
	  region reg = find_union_regions(l_regions,e);
	  bottom_up_adn_array_region(reg,e,pre);
	  /*  }*/
	}
      l_decl = CDR(l_decl);
    }
  user_log(" \n The total number of right array declarations : %d \n"
	  ,number_of_right_array_declarations );
  
  debug(1,"End bottom up array resizing for %s\n", mod_name);
  debug_off();  
  regions_end();
  reset_precondition_map();
  reset_rw_effects();
  current_mod = "";
  DB_PUT_MEMORY_RESOURCE(DBR_CODE, mod_name, mod_stmt);
  return TRUE;
}






















