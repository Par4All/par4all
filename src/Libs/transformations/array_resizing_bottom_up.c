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
#include "effects.h"
#include "ri-util.h"
#include "effects-util.h"
#include "database.h"
#include "pipsdbm.h"
#include "resources.h"
#include "misc.h"
#include "properties.h"
#include "semantics.h"
#include "abc_private.h"
#include "effects-generic.h"
#include "effects-convex.h"
#include "effects-simple.h"
#include "text-util.h" /* for words_to_string*/
#include "alias_private.h"
#include "transformations.h"

static int number_of_right_array_declarations = 0;
static string current_mod ="";
static int opt = 0; /* 0 <= opt <= 7*/
static char *file_name = NULL;
static FILE * instrument_file; /*To store new array declarations*/

#define PREFIX "$ARRAY_DECLARATION"

static bool
parameter_p(entity e)
{

  return storage_rom_p(entity_storage(e)) && 
    !(value_undefined_p(entity_initial(e))) &&
    value_symbolic_p(entity_initial(e)) &&
    type_functional_p(entity_type(e));
}

/*La valeur retournée est true si la variable v est un parameter 
  ou une variable common. Sinon, elle rend la valeur FALSE.*/

static bool
variable_is_param_common_p(entity e)
{
  if ( (parameter_p(e)) || (variable_in_common_p(e)) )  return (true);
  else 
    return (false);
}

/*Rendre true si la variable v n'est pas un parameter, ni une variable common
  et v, v_phi ne sont pas identiques.
  En vice-versa, retourner la valeur FALSE.*/

static bool
variable_to_project_p(Variable v_phi, Variable v)
{
  if (v!=v_phi) {
    entity e = (entity) v;
    if (variable_is_param_common_p(e) || storage_formal_p(entity_storage(e))) return false;
    return true;
  }
  return false;
}

static bool extract_constraint_on_var(Pvecteur p_var, Variable var, int val,  Pvecteur *ptmp)
{
  bool divisible=true; 
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
    return(true);
  }
  else {
    *ptmp = VECTEUR_NUL;
    return (false);
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
  bool result=false;   
  if (!SC_UNDEFINED_P(ps) && !CONTRAINTE_UNDEFINED_P(ps->egalites) 
      && CONTRAINTE_NULLE_P(ps->egalites))  {
    *pe = VECTEUR_NUL;
    return(false);
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
  return(false);
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
sc_minmax_of_pvector(Psysteme ps_prec, Pvecteur pv1, Pvecteur pv2, bool is_min)
{
  /* Automatic variables read in a CATCH block need to be declared volatile as
   * specified by the documentation*/
  Psysteme volatile ps_egal = SC_UNDEFINED;
  Psysteme volatile ps_inf = SC_UNDEFINED;
  Psysteme volatile ps_sup = SC_UNDEFINED;

  Psysteme  pt=  SC_UNDEFINED;
  Pcontrainte pc_egal = CONTRAINTE_UNDEFINED, 
    pc_inf = CONTRAINTE_UNDEFINED, 
    pc_sup = CONTRAINTE_UNDEFINED;
  Pvecteur p1, p2, p_egal, p_inf, p_sup,pvt,pv_1;
  bool  egal = false, inf = false, sup = false;
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
      sc_rational_feasibility_ofl_ctrl(ps_egal, OFL_CTRL, true);
    inf =  !SC_UNDEFINED_P(ps_inf) && 
      sc_rational_feasibility_ofl_ctrl(ps_inf, OFL_CTRL, true);
    sup =  !SC_UNDEFINED_P(ps_sup) && 
      sc_rational_feasibility_ofl_ctrl(ps_sup, OFL_CTRL, true);
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
  de la variable var sous la forme de Pvecteur pmin et pmax. Sinon, la valeur retournée est false et les Pvecteurs sont nuls.
  Dans cette fonction, il y a des appels à la fonction sc_min_max_of_pvector() pour comparer deux vecteurs. */

static bool
extract_constraint_from_inequalities(Psysteme ps, Variable var, Psysteme ps_prec, Pvecteur *pe, Pvecteur *pmin, Pvecteur *pmax)
{
  Pcontrainte pc;
  Value v_phi = VALUE_ZERO;
  Pvecteur p_var = VECTEUR_NUL, ptmp = VECTEUR_NUL, p_max = VECTEUR_NUL, p_min = VECTEUR_NUL ; 
  p_max = *pe;
  if (VECTEUR_NUL_P(*pe)) 
    p_min = vect_new(TCST, VALUE_ONE);
  else  p_min = *pe;
  
  if (!SC_UNDEFINED_P(ps) && !CONTRAINTE_UNDEFINED_P(ps->inegalites) 
      && CONTRAINTE_NULLE_P(ps->inegalites))  {
    *pmax = p_max;
    *pmin = p_min;
    return(false);
  }  
  for (pc = ps->inegalites; pc != NULL; pc = pc->succ) {
    p_var = contrainte_vecteur(pc);
    v_phi = vect_coeff(var,p_var);
    if (v_phi) {
      (void)extract_constraint_on_var(p_var,var,v_phi,&ptmp);
      
      if (value_pos_p(v_phi)) 
	p_max = sc_minmax_of_pvector(ps_prec, p_max, ptmp, false);
      else if (value_neg_p(v_phi)) {
	p_min = sc_minmax_of_pvector(ps_prec, p_min, ptmp, true);
      }
    }
  }
  *pmax = p_max;
  *pmin = p_min;
  return (true);
}
  
/*Cette fonction a été écrite pour déterminer les valeurs minimum et maximum d'une variable dans
  un système de contraintes, elle est donc la fonction principale du programme. 
  . La valeur retournée est false si le système de contraintes est infaisable ou les valeurs min, max sont
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
  bool ok2;
  assert(var!=TCST);  
  *min = vect_new(TCST, VALUE_MIN);
  *max = vect_new(TCST, VALUE_MAX);
  
  /* faire la projection sur toutes les variables sauf var, parametres formels et commons */
  for (b = ps->base; !VECTEUR_NUL_P(b); b = b->succ) {
    Variable v = var_of(b);
    if (variable_to_project_p(var, v)) {
      if (SC_EMPTY_P(ps = sc_projection_pure(ps, v))) 
	return false;
    }
  }
  if (SC_EMPTY_P(ps = sc_normalize(ps)))
    return false;
  
  if (SC_UNDEFINED_P(ps) || ( sc_nbre_inegalites(ps)==0  && sc_nbre_egalites(ps)==0))
    return(false);  
  ps_e = sc_dup(ps);
  ps_i = sc_dup(ps); 
  (void)extract_constraint_from_equalitites(ps_e, var, &pe);  
  ok2 = extract_constraint_from_inequalities(ps_i, var, ps_prec, &pe, min, max);
  if (ok2) {
    pips_debug(8, "The upper bound has been found\n");
    return (true);
  }
  vect_rm(pe);
  sc_rm(ps_e); 
  sc_rm(ps_i);
  pips_debug(8, "The upper bound has not been found\n");
  return (false);
}

static void new_array_declaration_from_region(region reg, entity e, Psysteme pre)
{  
  variable v = type_variable(entity_type(e));   
  list l_dims = variable_dimensions(v);
  int length = gen_length(l_dims);
  //  dimension last_dim =  find_ith_dimension(l_dims,length);
  entity phi = make_phi_entity(length);
  expression upper = expression_undefined;
  Pvecteur min,max;
  dimension last_dim = find_ith_dimension(l_dims,length);
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
    {
      number_of_right_array_declarations++;
      fprintf(instrument_file,"%s\t%s\t%s\t%s\t%d\t%s\t%s\n",PREFIX,file_name,
	      current_mod,entity_local_name(e),length,
	      words_to_string(words_expression(dimension_upper(last_dim), NIL)),
	      words_to_string(words_expression(upper, NIL)));
    }
  dimension_upper(last_dim) = upper;
}

/* This function finds in the list of regions the read and write regions of e.
   If there are 2 regions, it returns the union region */
static region find_union_regions(list l_regions,entity e)
{
  region reg = region_undefined;
  while (!ENDP(l_regions))
    {
      region re = REGION(CAR(l_regions));
      reference ref = effect_any_reference(re);
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

/* This phase do array resizing for all kind of arrays: formal or local, 
   unnormalized or not, depending on choosen option.*/

bool array_resizing_bottom_up(char* mod_name)
{
  /* instrument_file is used to store new array declarations and will be used by 
     a script to insert these declarations in the source code in xxx.database/Src/file_name.f

     file_name gives the file containing the current module in xxx.database/Src/ */

  entity mod_ent = module_name_to_entity(mod_name);
  list l_decl = code_declarations(entity_code(mod_ent)), l_regions = NIL; 
  statement stmt = (statement) db_get_memory_resource(DBR_CODE, mod_name, true);
  transformer mod_pre;
  Psysteme pre;
  string dir_name = db_get_current_workspace_directory();
  string instrument_file_name = strdup(concatenate(dir_name, "/BU_instrument.out", NULL));
  string user_file = db_get_memory_resource(DBR_USER_FILE,mod_name,true);
  string base_name = pips_basename(user_file, NULL);
  instrument_file = safe_fopen(instrument_file_name, "a");  
  file_name = strdup(concatenate(db_get_directory_name_for_module(WORKSPACE_SRC_SPACE), 
				 "/",base_name,NULL));
  current_mod = mod_name;  
  set_precondition_map((statement_mapping)
		       db_get_memory_resource(DBR_PRECONDITIONS,mod_name,true));
  set_rw_effects((statement_effects) 
      db_get_memory_resource(DBR_REGIONS, mod_name, true));
  regions_init(); 
  debug_on("ARRAY_RESIZING_BOTTOM_UP_DEBUG_LEVEL");
  debug(1," Begin bottom up array resizing for %s\n", mod_name);
    l_regions = load_rw_effects_list(stmt);  
  mod_pre = load_statement_precondition(stmt);
  pre = predicate_system(transformer_relation(mod_pre));

  opt = get_int_property("ARRAY_RESIZING_BOTTOM_UP_OPTION");

  /* opt in {0,1,2,3} => Do not compute new declarations for instrumented array (I_PIPS_MODULE_ARRAY)
     opt in {4,5,6,7} => Compute new declarations for instrumented array (I_PIPS_MODULE_ARRAY) 	 
     => (opt mod 8) <= 3 or not  
     
     opt in {0,1,4,5} => Compute new declarations for assumed-size and one arrays only
     opt in {2,3,6,7} => Compute new declarations for all kinds of arrays
     => (opt mod 4) <= 1 or not 
     
     opt in {0,2,4,6} => Do not compute new declarations for local array arguments
     opt in {1,3,5,7} => Compute new declarations for local array arguments also
     => (opt mod 2) = 0 or not */

  while (!ENDP(l_decl))
    {
      entity e = ENTITY(CAR(l_decl));
      if (opt%8 <= 3)
	  {
	    /* Do not compute new declarations for instrumented array (I_PIPS_MODULE_ARRAY)*/
	    if (opt%4 <= 1)
	      {
		/* Compute new declarations for assumed-size and one arrays only */
		if (opt%2 == 0)
		  {
		    /* Do not compute new declarations for local array arguments */
		    if (unnormalized_array_p(e) && formal_parameter_p(e))
		      {
			region reg = find_union_regions(l_regions,e);
			new_array_declaration_from_region(reg,e,pre);
		      }
		  }
		else 
		  {
		    /*  Compute new declarations for local array arguments also*/
		    if (unnormalized_array_p(e))
		      {
			region reg = find_union_regions(l_regions,e);
			new_array_declaration_from_region(reg,e,pre);
		      }
		  }
	      }
	    else 
	      {
		/* Compute new declarations for all kinds of arrays
		   To be modified, the whole C code: assumed-size bound if not success, ... 
		   How about multi-dimensional array ? replace all upper bounds ?
		   => different script, ...*/

		user_log("This option has not been implemented yet\n");
		
		//	if ((opt%2 == 0) && formal_parameter_p(e))
		// {
		    /* Do not compute new declarations for local array arguments */
		//  region reg = find_union_regions(l_regions,e);
		//   new_array_declaration_from_region(reg,e,pre);
		// }
		//	else 
		// {
		    /* Compute new declarations for local array arguments also*/
		//  region reg = find_union_regions(l_regions,e);
		//  new_array_declaration_from_region(reg,e,pre);
		// }

	      }
	  }
	else
	  {
	    /* Compute new declarations for instrumented array (I_PIPS_MODULE_ARRAY) 
	       Looking for arrays that contain I_PIPS in the last upper bound declaration
	       Attention: this case excludes some other cases */
	    //if (pips_instrumented_array_p(e))
	    // { 
	    //	region reg = find_union_regions(l_regions,e);
	    //	new_array_declaration_from_region(reg,e,pre);
	    // }
	    user_log("This option has not been implemented yet\n");
	  }
      l_decl = CDR(l_decl);
    }
  user_log("* Number of right array declarations replaced: %d *\n"
	   ,number_of_right_array_declarations );
  
  debug(1,"End bottom up array resizing for %s\n", mod_name);
  debug_off();  
  regions_end();
  reset_precondition_map();
  reset_rw_effects();
  safe_fclose(instrument_file,instrument_file_name);
  free(dir_name), dir_name = NULL;
  free(instrument_file_name), instrument_file_name = NULL;
  free(file_name), file_name = NULL;
  current_mod = "";
  DB_PUT_MEMORY_RESOURCE(DBR_CODE, mod_name, stmt);
  return true;
}

















