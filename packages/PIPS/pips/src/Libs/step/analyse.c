/* Copyright 2007, 2008 Alain Muller, Frederique Silber-Chaussumier

This file is part of STEP.

The program is distributed under the terms of the GNU General Public
License.
*/


/*
 - Calcul des regions SEND

 - Test de l'entrelacement des regions SEND

 - Suppression des contraintes "complexes": contraintes non paralleles
   aux axes

IN: Utilise les analyses de PIPS sur les modules outlines
   * SUMMARY_REGIONS (read, write)
   * IN_SUMMARY_REGIONS
   * OUT_SUMMARY_REGIONS

OUT: met a jour la ressource step_status pour chaque module
      - liste de regions SEND
      - pour chaque region SEND, on sait si entrelacement ou non

Creation: A. Muller, 2007-2008
Modification: F. Silber-Chaussumier, 2008  

*/

#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif

#include "defines-local.h"

/* pour step_rw_regions */
#include "effects-convex.h"
/* pour add_intermediate_value, entity_to_intermediate_value, free_value_mappings */
#include "transformer.h"
/* pour set_cumulated_rw_effects, reset_cumulated_rw_effects, w_r_combinable_p */
#include "effects-generic.h"
/* pour module_to_value_mappings */
#include "semantics.h"

#define LOCAL_DEBUG 2

static void step_print_region(char *text, list reg)
{
  printf("\n%s : %zd\n", text, gen_length(reg));
  print_rw_regions(reg);
}

static int step_nb_phi_vecteur(list phi_l, Pvecteur v)
{
  int nb_phi=0;

  pips_debug(1, "phi_l = %p, v = %p\n", phi_l, v);

  MAP(VARIABLE,phi,{
    if(vect_coeff((Variable)phi,v)!=VALUE_ZERO)
      nb_phi++;
  },
      phi_l);

  pips_debug(1, "nb_phi = %d\n", nb_phi);
  return nb_phi;
}

static void step_drop_complex_constraints(region reg)
{
  list phi_l;
  Pcontrainte c,ct;

  pips_debug(1, "reg = %p\n", reg);

  pips_debug(LOCAL_DEBUG, "Region entity %s\n", entity_name(region_entity(reg)));

  phi_l = NIL;
  MAP(EXPRESSION, exp, {
      syntax s=expression_syntax(exp);
      pips_assert("reference", syntax_reference_p(s));
      entity phi=reference_variable(syntax_reference(s));
      phi_l=CONS(ENTITY,phi,phi_l);
    },
    reference_indices(effect_any_reference(reg)));

  sc_base(region_system(reg)) = BASE_NULLE;

  pips_debug(LOCAL_DEBUG, "1 nb egalites : %i\n", sc_nbre_egalites(region_system(reg)));

  if (sc_nbre_egalites(region_system(reg)))
    {
      c = sc_egalites(region_system(reg));
      sc_egalites(region_system(reg))=(Pcontrainte) NULL;
      sc_nbre_egalites(region_system(reg)) = 0;
      for (;!CONTRAINTE_UNDEFINED_P(c); c = ct)
	{
	  ct=c->succ;
	  if (step_nb_phi_vecteur(phi_l,contrainte_vecteur(c)) == 1)
	    {
	      pips_debug(LOCAL_DEBUG, "1 add\n");
	      c->succ = NULL;
	      sc_add_eg(region_system(reg),c);
	    }
	}
    }
  
  pips_debug(LOCAL_DEBUG, "2 nb inegalites : %i\n", sc_nbre_inegalites(region_system(reg)));
  if (sc_nbre_inegalites(region_system(reg)))
    {
      c = sc_inegalites(region_system(reg));
      sc_inegalites(region_system(reg))=(Pcontrainte) NULL;
      sc_nbre_inegalites(region_system(reg)) = 0;
      for (;!CONTRAINTE_UNDEFINED_P(c); c = ct) 
	{
	  ct=c->succ;
	  if (step_nb_phi_vecteur(phi_l,contrainte_vecteur(c)) == 1)
	    {
	      pips_debug(LOCAL_DEBUG, "2 add\n");
	      c->succ = NULL;
	      sc_add_ineg(region_system(reg),c);
	    }
	}
    }
  sc_creer_base(region_system(reg));
}


/* Recherche des variables et tableaux privatisables  
   
   Calcul de PRIV a partir de LOCAL: variables ecrites non
   importees, definies dans le corps
   
   LOCAL(i) = W(i) - {les regions correspondant a des tableaux dans IN(i)}
     PRIV(i) = LOCAL(i) - {les regions corespondant a des tableaux dans OUT(i}
     
     Resultat non utilise: serait utile pour generer du OpenMP
*/
static list step_private_regions(list write_l, list in_l, list out_l)
{

  list local_l = RegionsEntitiesInfDifference(regions_dup(write_l), regions_dup(in_l), w_r_combinable_p);
  list priv_l = RegionsEntitiesInfDifference(regions_dup(local_l), regions_dup(out_l), w_w_combinable_p);

  ifdebug(LOCAL_DEBUG) step_print_region("Region priv ",priv_l);
  return priv_l;
}


/* Recherche des variables et sections de tableau privatisables 
   
   Possibilite de diviser les tableaux en plusieurs sections au lieu
   d'allouer tout le tableau sur chaque noeud
   
   LOCAL(i) = W(i) - inf IN(i)
   PRIV_SEC(i) = LOCAL(i) -inf proj_i'[IN(i')]
   
   Resultat non utilise:
   - serait utile quand allocation dynamique
   
   - pour generer du MPI avec adaptation de la taille du tableau
   pour chaque noeud
   
   - pour generer du Open MP
   
   Resultats incertains: encore a debugger...
*/
static list step_private_section_regions(list write_l, list in_l, list out_l)
{
  list local_sec = RegionsInfDifference(regions_dup(write_l),regions_dup(in_l), w_r_combinable_p);
  list priv_sec = RegionsInfDifference(regions_dup(local_sec),regions_dup(out_l), w_w_combinable_p);
  
  ifdebug(LOCAL_DEBUG) step_print_region("Region priv_sec ", priv_sec);
  return priv_sec; 
}


/* Recherche des regions copy_out
   
   Quand on a eu la possibilite de diviser un tableau en plusieurs
   sections sur differents noeuds
   
   Calcul des sections mises a jour et necessaires pour la suite des
   calculs
   
   COPY_OUT(i) = PRIV_SEC(i) inter OUT(i)
   
   Meme utilite que precedemment
   
   Resultats incertains: encore a debugger...
*/
static list step_copy_out_regions(list priv_sec,list out_l)
{
    list copy_out = RegionsIntersection(regions_dup(priv_sec),regions_dup(out_l), w_w_combinable_p);

    ifdebug(LOCAL_DEBUG) step_print_region("Region copy_out ", copy_out);
    return copy_out;
}


/* Recherche des regions SEND 
   
   Regions communiquees apres un calcul (necessaires pour les calculs
   suivants)
   
   SEND(tranche) = OUT inter W(tranche)
   
   Tout d'abord on retire les contraintes non paralleles aux axes
*/
static list step_send_regions(list write_l, list out_l)
{
  // simplification des regions (on ne garde que les contraintes ayant 1 "PHI")
  MAP(REGION, reg, {step_drop_complex_constraints(reg);}, write_l);
  MAP(REGION, reg, {step_drop_complex_constraints(reg);}, out_l);

  list send_l = RegionsIntersection(regions_dup(out_l), regions_dup(write_l), w_w_combinable_p);
  list send_final = NIL;

  MAP(REGION,r,{
      entity e=region_entity(r);
      if (strcmp(entity_module_name(e),IO_EFFECTS_PACKAGE_NAME) ==0 )
	pips_user_warning("STEP : possible IO concurrence\n");
      else
	send_final=CONS(REGION,region_dup(r),send_final);
    }
    ,send_l);
  gen_full_free_list(send_l);

  ifdebug(LOCAL_DEBUG) step_print_region("Region Send ", send_final); 
  return send_final;
}


/* Test d'entrelacement des regions SEND 
   
   Une region est "entrelacee" si elle contient des tranches
   d'indices sur des noeuds differents qui se chevauchent:
   
   - pourrait correspondre a des acces concurrents mais la norme OpenMP
   garantit que ce n'est pas le cas
   
   - correspond ici au cas ou on ne determine pas suffisamment
   precisement les acces aux regions: typiquement un noeud accede
   aux elements pairs l'aitre noeud accede aux elements impairs
   
   
   Traitement different si region entrelacee ou non
   Region non entrelacee: communication MPI directe
   
   Region entrelacee: determination a l'execution de la modification
*/
static boolean step_interlaced_iteration_regions_p(region reg, list loop_data_l)
{
  Psysteme s = sc_dup(region_system(reg));
  Psysteme  s_prime = sc_dup(s);
  pips_debug(1, "reg = %p, loop_data_l = %p\n", reg, loop_data_l);
 
  MAP(LOOP_DATA,data,
    {
      Pcontrainte c;
      entity l_prime;
      entity u_prime;
      entity low=loop_data_lower(data);
      entity up=loop_data_upper(data);

      pips_debug(LOCAL_DEBUG, "low = %s up = %s\n",entity_name(low), entity_name(up));
      
      add_intermediate_value(low);
      add_intermediate_value(up);
      l_prime = entity_to_intermediate_value(low);
      u_prime = entity_to_intermediate_value(up);
      s_prime = sc_variable_rename(s_prime, (Variable)low, (Variable)l_prime);
      s_prime = sc_variable_rename(s_prime, (Variable)up, (Variable)u_prime);

      // contrainte U<L' qui s'ecrit : U-L'+1 <= 0
      c = contrainte_make(vect_make(VECTEUR_NUL, 
				    (Variable) up, VALUE_ONE,
				    (Variable) l_prime, VALUE_MONE,
				    TCST, VALUE_ONE));
      sc_add_inegalite(s_prime, contrainte_dup(c));
    },loop_data_l);
  
  s = sc_append(s,s_prime);
  s->base = BASE_NULLE;
  sc_creer_base(s);
  
//sc_default_dump(s);
  pips_debug(1, "Fin\n");

  return sc_integer_feasibility_ofl_ctrl(s, NO_OFL_CTRL, TRUE);
}

static list step_interlaced_iteration_regions(list loop_data_l, list send_l)
{
  list interlaced_l = NIL;

  MAP(REGION,reg,{
      if (step_interlaced_iteration_regions_p(reg,loop_data_l))
	interlaced_l = CONS(REGION, reg, interlaced_l);
    }, send_l);

  ifdebug(LOCAL_DEBUG) step_print_region("Region interlaced ", interlaced_l);

  return interlaced_l;
}


/* recherche des regions RECV 
   
   Regions communiquees avant un calcul (necessaires pour le calcul
   considere)
   
     
   RECV(tranche) = IN union SEND_MAY ( union SEND(interlaced) )
   
   Resultat non utilise: les donnees sont mises a jour apres les
   calculs (donc utilisation des regions SEND)
*/
static list step_recv_regions(list send_l, list in_l)
{
  list recv_l;
  list send_may_l = NIL;

  MAP(REGION,reg,{
      if (region_may_p(reg))
	send_may_l=CONS(REGION, reg, send_may_l);
    }, send_l);

  recv_l = RegionsMustUnion(regions_dup(in_l),regions_dup(send_may_l),r_w_combinable_p);

  ifdebug(LOCAL_DEBUG) step_print_region("Region Recv ", recv_l);
  return recv_l;
}


/* Recherche des regions USED 
   Si allocation dynamique possible, determination de l'espace memoire a utiliser
   
   USED(tranche) = R union W
   
   Resultat non utilise: pas d'allocation dynamique
*/
static list step_used_regions(list read_l, list write_l)
{
  list used_l = RegionsMustUnion(regions_dup(read_l), regions_dup(write_l), r_w_combinable_p);

  ifdebug(LOCAL_DEBUG) step_print_region("Region Used ", used_l);
  return used_l;
}

static step_region_analyse step_analyse_loop_regions(list read_l, list write_l, list in_l, list out_l, list loop_data_l)
{
  // Recherche des regions SEND et de leur entrelacement
  list send_l = step_send_regions(write_l, out_l);
  list interlaced_l = step_interlaced_iteration_regions(loop_data_l,send_l);

  // Recherche des regions RECV : non utilise car les donnees sont mises a jour apres les calculs
  list recv_l = step_recv_regions(send_l, in_l);

  /* Resultat non utilise:
  // Recherche des regions USED : si allocation dynamique possible, determination de l'espace memoire a utiliser
  list used_l = step_used_regions(read_l, write_l);

  // Recherche des regions PRIV : serait utile pour generer du OpenMP
  list priv_l = step_private_regions(write_l, in_l, out_l);
  */

  return make_step_region_analyse(recv_l,send_l, interlaced_l,NIL);
}

static step_region_analyse step_analyse_master_regions(list read_l, list write_l, list in_l, list out_l)
{
  ifdebug(LOCAL_DEBUG)
  {
    step_print_region("Region read ", read_l);
    step_print_region("Region write ", write_l);
    step_print_region("Region in ", in_l);
    step_print_region("Region out ", out_l);
  }
  // Recherche des regions SEND
  list send_l = step_send_regions(write_l, out_l);
  ifdebug(LOCAL_DEBUG) step_print_region("Region master SEND ", send_l);
  return make_step_region_analyse(NIL,send_l,NIL,NIL);
}


bool step_analyse(string module_name)
{ 
  list region_l,read_l,write_l,in_l,out_l;
  entity module = local_name_to_top_level_entity(module_name);  ;
  step_region_analyse result=step_region_analyse_undefined;

  debug_on("STEP_DEBUG_LEVEL");
  pips_debug(1, "considering module %s\n", module_name);
  debug_on("STEP_ANALYSE_DEBUG_LEVEL");

  // recuperation des effects R W IN et OUT
  region_l = effects_to_list((effects)db_get_memory_resource(DBR_SUMMARY_REGIONS, entity_user_name(module), TRUE));
  in_l = effects_to_list((effects)db_get_memory_resource(DBR_IN_SUMMARY_REGIONS, entity_user_name(module), TRUE));
  out_l = effects_to_list((effects)db_get_memory_resource(DBR_OUT_SUMMARY_REGIONS, entity_user_name(module), TRUE));

  set_methods_for_convex_effects();

  read_l = regions_read_regions(region_l);
  write_l = regions_write_regions(region_l);
  ifdebug(LOCAL_DEBUG)
  {
    step_print_region("Region read ", read_l);
    step_print_region("Region write ", write_l);
    step_print_region("Region in ", in_l);
    step_print_region("Region out ", out_l);
  }

  step_load_status();
  global_directives_load();
  set_current_module_entity(module);

  set_current_module_statement((statement)db_get_memory_resource(DBR_CODE, entity_user_name(module), TRUE));
  set_cumulated_rw_effects((statement_effects)db_get_memory_resource(DBR_CUMULATED_EFFECTS, entity_user_name(module), TRUE));
  module_to_value_mappings(module);

  // STEP analysis
  if(bound_global_directives_p(module))
    {
      directive d=load_global_directives(module);
      pips_debug(1,"Directive module : %s\n%s\n", module_name,directive_txt(d));
      switch(type_directive_tag(directive_type(d)))
	{
	case is_type_directive_omp_parallel:
	  break;
	case is_type_directive_omp_parallel_do:
	  result = step_analyse_loop_regions(read_l,write_l,in_l,out_l,type_directive_omp_parallel_do(directive_type(d)));
	  break;
	case is_type_directive_omp_do:
	  result = step_analyse_loop_regions(read_l,write_l,in_l,out_l,type_directive_omp_do(directive_type(d)));
	  break;
	case is_type_directive_omp_master:
	  result = step_analyse_master_regions(read_l, write_l, in_l, out_l);
	  break;
	case is_type_directive_omp_barrier:
	  pips_debug(2,"Directive %s : no analyse to perform\n", directive_txt(d));
	  break;
	default:
	    pips_user_warning("Directive %s : analyse not yet implemented\n", directive_txt(d));
	}
      store_or_update_step_analyse_map(module,result);
    }
  else
    pips_debug(2,"Not directive module\n");

  reset_current_module_statement();
  reset_cumulated_rw_effects();
  reset_current_module_entity(); 
  free_value_mappings();

  generic_effects_reset_all_methods();
  global_directives_save();
  step_save_status();
  
  debug_off(); 
  debug_off();

  pips_debug(1, "FIN\n");
  return TRUE;
}
