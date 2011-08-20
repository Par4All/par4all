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
/* pour words_effect */
#include "effects-simple.h"

#include "dg.h"
typedef dg_arc_label arc_label;
typedef dg_vertex_label vertex_label;
#include "graph.h"
#include "ricedg.h"

#define LOCAL_DEBUG 2

GENERIC_GLOBAL_FUNCTION(global_step_analyses,map_entity_step_analyses);

void global_step_analyses_load()
{
  set_global_step_analyses((map_entity_step_analyses)db_get_memory_resource(DBR_STEP_ANALYSES, "", true));
}

void global_step_analyses_save()
{
  DB_PUT_MEMORY_RESOURCE(DBR_STEP_ANALYSES, "", get_global_step_analyses());
  reset_global_step_analyses();
}

void global_step_analyse_init()
{
  init_global_step_analyses();
  global_step_analyses_save();
}

static void step_print_region(char *text, list reg)
{
  printf("\n%s : %zd\n", text, gen_length(reg));
  print_rw_regions(reg);
}

static int step_nb_phi_vecteur(list phi_l, Pvecteur v)
{
  int nb_phi=0;

  pips_debug(1, "phi_l = %p, v = %p\n", phi_l, v);

  FOREACH(VARIABLE,phi,phi_l)
    {
      if(vect_coeff((Variable)phi,v)!=VALUE_ZERO)
	nb_phi++;
    }

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
  FOREACH(EXPRESSION, exp,reference_indices(effect_any_reference(reg)))
    {
      syntax s=expression_syntax(exp);
      pips_assert("reference", syntax_reference_p(s));
      entity phi=reference_variable(syntax_reference(s));
      phi_l=CONS(ENTITY,phi,phi_l);
    }

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

#if 0
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
#endif

/* Recherche des regions SEND 
   
   Regions communiquees apres un calcul (necessaires pour les calculs
   suivants)
   
   SEND(tranche) = OUT inter W(tranche)
   
   Tout d'abord on retire les contraintes non paralleles aux axes
*/
static list step_send_regions(list write_l, list out_l)
{
  // simplification des regions (on ne garde que les contraintes ayant 1 "PHI")
  FOREACH(REGION, reg, write_l){step_drop_complex_constraints(reg);};
  FOREACH(REGION, reg, out_l) {step_drop_complex_constraints(reg);};

  list send_l = RegionsIntersection(regions_dup(out_l), regions_dup(write_l), w_w_combinable_p);
  list send_final = NIL;

  FOREACH(REGION,r,send_l)
    {
      entity e=region_entity(r);
      if (strcmp(entity_module_name(e),IO_EFFECTS_PACKAGE_NAME) ==0 )
	pips_user_warning("STEP : possible IO concurrence\n");
      else
	send_final=CONS(REGION,region_dup(r),send_final);
    }

  gen_full_free_list(send_l);

  gen_sort_list(send_final, (int (*)(const void *,const void *)) effect_compare); 

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
static bool step_interlaced_iteration_regions_p(region reg, list loop_data_l)
{
  Psysteme s = sc_dup(region_system(reg));
  Psysteme  s_prime = sc_dup(s);
  pips_debug(1, "reg = %p, loop_data_l = %p\n", reg, loop_data_l);
 
  FOREACH(LOOP_DATA,data,loop_data_l)
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
    }
  
  s = sc_append(s,s_prime);
  s->base = BASE_NULLE;
  sc_creer_base(s);
  
//sc_default_dump(s);
  pips_debug(1, "Fin\n");

  return sc_integer_feasibility_ofl_ctrl(s, NO_OFL_CTRL, true);
}

static list step_interlaced_iteration_regions(list loop_data_l, list send_l)
{
  list interlaced_l = NIL;

  FOREACH(REGION,reg,send_l)
    {
      if (step_interlaced_iteration_regions_p(reg,loop_data_l))
	interlaced_l = CONS(ENTITY, region_entity(reg), interlaced_l);
    }

  ifdebug(LOCAL_DEBUG) step_print_region("Region interlaced ", interlaced_l);

  return interlaced_l;
}


/* recherche des regions RECV 
   
   Regions communiquees avant un calcul (necessaires pour le calcul
   considere)
   
     
   RECV(tranche) = IN union SEND_MAY ( union SEND(interlaced) )
*/
static list step_recv_regions(list send_l, list in_l)
{
  list recv_l;
  list send_may_l = NIL;

  FOREACH(REGION,r,send_l)
    {
      if (region_may_p(r))
	send_may_l=CONS(REGION, region_dup(r), send_may_l);
    }
  recv_l = RegionsMustUnion(regions_dup(in_l),regions_dup(send_may_l),r_w_combinable_p);

  gen_full_free_list(send_may_l);

  gen_sort_list(recv_l, (int (*)(const void *,const void *)) effect_compare);

  ifdebug(LOCAL_DEBUG) step_print_region("Region Recv ", recv_l);
  return recv_l;
}

#if 0
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
#endif

static step_analyses step_analyse_loop_regions(list read_l, list write_l, list in_l, list out_l, list loop_data_l)
{
  // Recherche des regions SEND et de leur entrelacement
  list send_l = step_send_regions(write_l, out_l);
  list interlaced_l = step_interlaced_iteration_regions(loop_data_l,send_l);

  // Recherche des regions RECV
  list recv_l = step_recv_regions(send_l, in_l);

  /* Resultat non utilise:
  // Recherche des regions USED : si allocation dynamique possible, determination de l'espace memoire a utiliser
  list used_l = step_used_regions(read_l, write_l);

  // Recherche des regions PRIV : serait utile pour generer du OpenMP
  list priv_l = step_private_regions(write_l, in_l, out_l);
  */

  return make_step_analyses(recv_l, send_l, interlaced_l, NIL, make_map_entity_bool());
}

static step_analyses step_analyse_master_regions(list read_l, list write_l, list in_l, list out_l)
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

  // Recherche des regions RECV
  list recv_l = step_recv_regions(send_l, in_l);

  ifdebug(LOCAL_DEBUG) step_print_region("Region master SEND ", send_l);
  return make_step_analyses(recv_l, send_l, NIL, NIL, make_map_entity_bool());
}
static step_analyses step_analyse_parallel_regions(list read_l, list write_l, list in_l, list out_l)
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

  // Recherche des regions RECV
  list recv_l = step_recv_regions(send_l, in_l);

  ifdebug(LOCAL_DEBUG) step_print_region("Region parallel RECV ", recv_l);
  return make_step_analyses(recv_l, send_l, NIL, NIL, make_map_entity_bool());
}

static step_analyses step_analyse_critical_regions(list read_l, list write_l, list in_l, list out_l)
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

  // Recherche des regions RECV
  list recv_l = step_recv_regions(send_l, in_l);

  ifdebug(LOCAL_DEBUG) step_print_region("Region critical SEND ", send_l);
  return make_step_analyses(recv_l, send_l, NIL, NIL, make_map_entity_bool());
}

bool step_analyse(string module_name)
{ 
  list region_l,read_l,write_l,in_l,out_l;
  entity module = local_name_to_top_level_entity(module_name);
  step_analyses result=step_analyses_undefined;

  debug_on("STEP_DEBUG_LEVEL");
  pips_debug(1, "considering module %s\n", module_name);
  debug_on("STEP_ANALYSE_DEBUG_LEVEL");

  // recuperation des effects R W IN et OUT
  region_l = effects_to_list((effects)db_get_memory_resource(DBR_SUMMARY_REGIONS, entity_user_name(module), true));
  in_l = effects_to_list((effects)db_get_memory_resource(DBR_IN_SUMMARY_REGIONS, entity_user_name(module), true));
  out_l = effects_to_list((effects)db_get_memory_resource(DBR_OUT_SUMMARY_REGIONS, entity_user_name(module), true));

  set_methods_for_convex_effects();
  init_convex_rw_prettyprint(module_name);

  read_l = regions_read_regions(region_l);
  write_l = regions_write_regions(region_l);

  global_step_analyses_load();
  global_directives_load();
  set_current_module_entity(module);

  ifdebug(LOCAL_DEBUG)
  {
    /*
      current_module_entity must be defined
    */
    step_print_region("Region read ", read_l);
    step_print_region("Region write ", write_l);
    step_print_region("Region in ", in_l);
    step_print_region("Region out ", out_l);
  }

  set_current_module_statement((statement)db_get_memory_resource(DBR_CODE, entity_user_name(module), true));
  set_cumulated_rw_effects((statement_effects)db_get_memory_resource(DBR_CUMULATED_EFFECTS, entity_user_name(module), true));
  module_to_value_mappings(module);

  // STEP analysis
  if(bound_global_directives_p(module))
    {
      directive d=load_global_directives(module);
      pips_debug(1,"Directive module : %s\n\t directive text: %s\n", module_name,directive_txt(d));
      switch(type_directive_tag(directive_type(d)))
	{
	case is_type_directive_omp_parallel:
	  result = step_analyse_parallel_regions(read_l, write_l, in_l, out_l);
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
	case is_type_directive_omp_critical:
	  result = step_analyse_critical_regions(read_l, write_l, in_l, out_l);
	  break;	
	case is_type_directive_omp_barrier:
	  pips_debug(2,"Directive %s : no analyse to perform\n", directive_txt(d));
	  break;
	default:
	    pips_user_warning("Directive %s : analyse not yet implemented\n", directive_txt(d));
	}
      store_or_update_global_step_analyses(module,result);
    }
  else
    pips_debug(2,"Not directive module\n");

  reset_current_module_statement();
  reset_cumulated_rw_effects();
  reset_current_module_entity(); 
  free_value_mappings();

  generic_effects_reset_all_methods();
  reset_convex_prettyprint(module_name);
  global_directives_save();
  global_step_analyses_save();
  
  debug_off(); 
  debug_off();

  pips_debug(1, "FIN\n");
  return true;
}


static bool com_optimizable_p(map_entity_bool optimizable, entity v, entity directive_module)
{
  string v_name;
  entity array;
  bool optimizable_p;

  if (io_effect_entity_p(v))
    {
      optimizable_p = false;
      goto end;
    }

  v_name = entity_local_name(v);
  pips_debug(1, "v_name = %s\n", v_name);
  
  array = gen_find_tabulated(concatenate(entity_user_name(directive_module), MODULE_SEP_STRING, v_name, NULL), entity_domain);

  /* FSC: COMMENTER ICI */
  /* AM: on verifie si l'entite 'v' n'est pas deja ete marquee comme non-optimizable dans la table 'optimizable' par des analyses precedentes
   */
  if (bound_map_entity_bool_p(optimizable, array))
    optimizable_p = apply_map_entity_bool(optimizable, array);
  else
    optimizable_p = true;

 end:
  pips_debug(1, "optimizable_p = %d\n", optimizable_p);
  return optimizable_p;
}

bool step_com_optimize_p(map_entity_bool optimizable, entity v, entity directive_module)
{
  string v_name = entity_local_name(v);
  entity array = gen_find_tabulated(concatenate(entity_user_name(directive_module), MODULE_SEP_STRING, v_name, NULL), entity_domain);
  if (io_effect_entity_p(v) ||
      !bound_map_entity_bool_p(optimizable, array))
    return false;

  return apply_map_entity_bool(optimizable, array);
}

static void set_optimizable(map_entity_bool optimizable, entity v, entity directive_module, bool is_optimizable)
{
  string v_name = entity_local_name(v);
  entity array = gen_find_tabulated(concatenate(entity_user_name(directive_module), MODULE_SEP_STRING, v_name, NULL), entity_domain);

  if (!entity_undefined_p(array))
    {
      if(!bound_map_entity_bool_p(optimizable, array))
	extend_map_entity_bool(optimizable, array, is_optimizable);
      else if(is_optimizable)
	pips_assert("already optimizable",apply_map_entity_bool(optimizable, array));
      else
	update_map_entity_bool(optimizable, array, false);
    }
}

bool step_analyse_com(string module_name)
{


  debug_on("STEP_ANALYSE_COM_DEBUG_LEVEL");
  pips_debug(1, "considering module %s\n", module_name);
  graph dg = (graph) db_get_memory_resource(DBR_DG, module_name, true);
  statement body = (statement) db_get_memory_resource(DBR_CODE, module_name, true);

  set_current_module_entity(local_name_to_top_level_entity(module_name));
  set_current_module_statement(body);
  set_ordering_to_statement(body);

  global_directives_load();
  global_step_analyses_load();
  reset_action_interpretation();

  ifdebug(1)
    {
      pips_debug(1, "Print current module: \n");
      print_text(stderr, text_module(get_current_module_entity(), get_current_module_statement()));
      pips_debug(1, "\n");
    }

  FOREACH(VERTEX, v1, graph_vertices(dg))
    {
      statement s1 = vertex_to_statement(v1);

      if (!(statement_call_p(s1) && bound_global_directives_p(call_function(statement_call(s1)))))
	{
	  pips_debug(4, "\n");
	  pips_debug(4, "from %s", safe_statement_identification(s1));
	  pips_debug(4, "\tskip (not a directive module)\n");
	  STEP_DEBUG_STATEMENT(4, "statement", s1);
	}
      else
	{
	  entity S1_directive_module = call_function(statement_call(s1));
	  string S1_directive_module_name = entity_local_name(S1_directive_module);
	  step_analyses S1_analyses = load_global_step_analyses(S1_directive_module);
	  directive d1 = load_global_directives(S1_directive_module);

	  pips_debug(1, "\n");
	  pips_debug(1, "from %s", safe_statement_identification(s1));

	  if(step_analyses_undefined_p(S1_analyses))
	    pips_debug(3, "\tAnalyse not available for %s\n", S1_directive_module_name);
	  else if (type_directive_omp_master_p(directive_type(d1))) 
	    {
	      // communication directive master non optimisable actuellement
	      pips_debug(4, "\tskip (MASTER construct)\n");
	    }
	  else
	    {
	      map_entity_bool optimizable_S1 = step_analyses_optimizable(S1_analyses);
	      pips_debug(3, "\tAnalyse available for %s\t\n",S1_directive_module_name);
	      
	      FOREACH(SUCCESSOR, su, vertex_successors(v1))
		{
		  statement s2 = vertex_to_statement(successor_vertex(su));
		  bool directive_module_S2_p = statement_call_p(s2) && bound_global_directives_p(call_function(statement_call(s2)));
		  pips_debug(1, "\tto %s", safe_statement_identification(s2));
		  bool directive_master_2_p = false;
		  if (directive_module_S2_p)  // communication directive master non optimisable actuellement
		    {
		      /* FSC: commenter ICI, à quoi sert d2? */
		      /* AM: les directives master n'etant pas actuellement optimisable,
			 on aura besoin de savoir si le statement S2 correspont a une directive master
			 d2 sert a ne pas alourdir le statement suivant.
		      */ 
		      directive d2 = load_global_directives(call_function(statement_call(s2)));
		      directive_master_2_p = type_directive_omp_master_p(directive_type(d2));
		    }

		  FOREACH(CONFLICT, c, dg_arc_label_conflicts((dg_arc_label)successor_arc_label(su)))
		    {
		      effect source = conflict_source(c);
		      /* FSC a quoi sert sink ? */
		      /* AM: a de l'affichage pips_debug */
		      effect sink = conflict_sink(c);
		      reference ref = region_any_reference(source);
		      entity e = reference_variable(ref);
		      string caller_side = entity_local_name(e);
		      int caller_side_length = strlen(caller_side);
		      

		      /* FSC: le premier test sert-il a eliminer tout ce qui n'est pas un tableau? */
		      /* a quoi sert le deuxieme test? */
		      /* AM: oui. Le second test sert a verifier que le tableau 'e' ne fait pas deja l'objet
			 d'une communication marquee comme non optimisable */
		      if (reference_indices(ref) == NIL ||
			  !com_optimizable_p(optimizable_S1, e, S1_directive_module))
			pips_debug(3, "\t\t\t%s UNOPTIMIZABLE (1)\n", caller_side);
		      else
			{
			  // on verifie que le tableau 'e' est SEND à partir de S1
			  /* ATTENTION : le test se fait sur le nom du tableau car il correspond à une entite
			     cote caller et a une autre entite cote called.
			  */
			  bool send_array_p = false;
			  list remaining_region = step_analyses_send(S1_analyses);
			  for(; !(send_array_p || ENDP(remaining_region)); POP(remaining_region))
			    {
			      region reg = REGION(CAR(remaining_region));
			      string called_side = entity_local_name(reference_variable(region_any_reference(reg)));
			      send_array_p = (strncmp(caller_side, called_side, caller_side_length)==0);
			    }
			  
			  if (send_array_p)
			    {
			      bool is_optimizable_p = (directive_module_S2_p && !directive_master_2_p);

			      pips_debug(2, "\t\t%s from\t%s", effect_to_string(source), text_to_string(text_region(source)));
			      pips_debug(2, "\t\t%s to\t%s", effect_to_string(sink), text_to_string(text_region(sink)));
			      if (directive_master_2_p)
				pips_debug(3, "\t\tMASTER construct\n");
			      pips_debug(2, "\t\t\t%s %sOPTIMIZABLE (2)\n", caller_side, is_optimizable_p?"":"UN");
			      
			      // mise a jour si la communication de ref est optimisable ou non
			      set_optimizable(optimizable_S1, reference_variable(ref), S1_directive_module, is_optimizable_p);
			    }
			  else
			    pips_debug(4, "\t\t%s skip (not a SEND array)\n", caller_side);
			}
		    }
		}
	      
	      ifdebug(1)
		{
		  pips_debug(1, "RECAPITULATIF for : %s",safe_statement_identification(s1));
		  MAP_ENTITY_BOOL_MAP(v, is_optimizable,
				      {
					pips_debug(1, "\t\t%s -> %sOPTIMIZABLE\n", entity_name(v), is_optimizable?"":"UN");
				      }, optimizable_S1);
		  pips_debug(1, "END RECAPITULATIF\n");
		}
	    }
	}
    }
  
  global_step_analyses_save();
  global_directives_save();
  
  reset_ordering_to_statement();
  reset_current_module_statement();
  reset_current_module_entity();

  debug_off(); 
  return true;
}

static list step_effect_backward_translation(statement current_stat, list comm_entities)
{
  call call_site=statement_call(current_stat);
  entity callee=call_function(call_site);
  list real_args=call_arguments(call_site);

  list real_comm_entities = NIL, r_args;
  int arg_num;

  pips_debug(5,"callee=%s\n",entity_name(callee));
  for (r_args = real_args, arg_num = 1; r_args != NIL;
       r_args = CDR(r_args), arg_num++) 
    {
      FOREACH(STEP_COMM_ENTITIES, comm_e, comm_entities)
	 {
	   entity re=step_comm_entities_real(comm_e);
	   entity fe=step_comm_entities_formal(comm_e);

	   pips_debug(3,"argnum=%i\tfe=%s\n",arg_num,entity_name(re));
	   if (ith_parameter_p(callee, re, arg_num))
	     {
	       expression real_exp = EXPRESSION(CAR(r_args));
	       syntax real_syn = expression_syntax(real_exp);
	       
	       /* test reference simple */
	       if(!syntax_reference_p(real_syn) || !ENDP(reference_indices(syntax_reference(real_syn))))
		 {
		   print_statement(current_stat);
		   pips_user_error("Unsupported syntax arg=%i : %s\n",arg_num,words_to_string(words_syntax(real_syn, NIL)));
		 }

	       /* test de reshaping */
	       /* TODO ? */


	       re=reference_variable(syntax_reference(real_syn));
	       real_comm_entities=CONS(STEP_COMM_ENTITIES, make_step_comm_entities(re, fe), real_comm_entities);
	       pips_debug(3,"add comm_entity :(%s,%s)\n",entity_name(re),entity_name(fe));
	     }
	 }
    }


  return real_comm_entities;
}

static bool in_step_analyses_com_p(entity real, list l)
{
  bool result = false;
  FOREACH(STEP_COMM_ENTITIES,c,l)
    {
      if(step_comm_entities_real(c)==real)
	result=true;
    }
  return result;
}

static void statement_to_send_recv(statement s, list *send, list *recv)
{
  *send=NIL;
  *recv=NIL;
  if (statement_call_p(s) && entity_module_p(call_function(statement_call(s))))
    {
      entity func = call_function(statement_call(s));
      string func_name = entity_local_name(func);
      step_analyses_com analyses_com=(step_analyses_com)db_get_memory_resource(DBR_STEP_ANALYSES_COM, func_name, true);
      pips_debug(3, "\tCalled module %s\n",func_name);
      
      if(!step_analyses_com_undefined_p(analyses_com))
	{
	  *send =  step_effect_backward_translation(s, step_analyses_com_send(analyses_com));
	  *recv =  step_effect_backward_translation(s, step_analyses_com_recv(analyses_com));
	}
      else
	pips_debug(3, "\tStep_analyse_com not available for %s\n", func_name);
    }
  else
    {
      pips_debug(3, "\tNot a call statement\n");
      STEP_DEBUG_STATEMENT(4, "statement", s);
    }
  ifdebug(2)
    {
      printf("RECV:\n");
      FOREACH(STEP_COMM_ENTITIES,c,*recv)
	{
	  printf("\t(%s, %s)\n", entity_name(step_comm_entities_real(c)),entity_name(step_comm_entities_formal(c)));
	}
      printf("SEND:\n");
      FOREACH(STEP_COMM_ENTITIES,c,*send)
	{
	  printf("\t(%s, %s)\n", entity_name(step_comm_entities_real(c)),entity_name(step_comm_entities_formal(c)));
	}
    }

  return;
}

bool step_analyse_com2(string module_name)
{
  debug_on("STEP_ANALYSE_COM2_DEBUG_LEVEL");
  pips_debug(1, "considering module %s\n", module_name);
  entity module = local_name_to_top_level_entity(module_name);
  graph dg = (graph) db_get_memory_resource(DBR_DG, module_name, true);
  statement body = (statement) db_get_memory_resource(DBR_CODE, module_name, true);

  set_current_module_entity(module);
  set_current_module_statement(body);
  set_ordering_to_statement(body);


  /* initialisation des listes recv (et send) a l'union des list recv (et send) des modules appelles */
  list recv = NIL;
  list send = NIL;
  list not_send = NIL;
  list not_recv = NIL;
  list l;

  /* mise a jour des listes recv (et send) */
  global_directives_load();
  global_step_analyses_load();
  reset_action_interpretation();

  if(bound_global_directives_p(module))
    {
      step_analyses analyses = load_global_step_analyses(module);
      if (step_analyses_undefined_p(analyses))
	{
	  pips_user_warning("Missing step_analyses for module directive %s\n",entity_local_name(module));
	}
      else
	{
	  step_private_before(module);
	  FOREACH(REGION, reg, step_analyses_recv(analyses))
	    {
	      if(!(io_effect_p(reg) ||
		   region_scalar_p(reg) ||
		   step_private_p(region_entity(reg))
		   ))
		recv=CONS(STEP_COMM_ENTITIES,make_step_comm_entities(region_entity(reg),region_entity(reg)),recv);
	    }
	  FOREACH(REGION, reg, step_analyses_send(analyses))
	    {
	      if(!(io_effect_p(reg) ||
		   region_scalar_p(reg) ||
		   step_private_p(region_entity(reg))
		   ))
		send=CONS(STEP_COMM_ENTITIES,make_step_comm_entities(region_entity(reg),region_entity(reg)),send);
	    }
	  step_private_after();
	}
    }


  FOREACH(VERTEX, v1, graph_vertices(dg))
    {
      list send_s1, recv_s1;
      statement s1 = vertex_to_statement(v1);
      STEP_DEBUG_STATEMENT(2, "statement s1", s1);
      pips_debug(3, "from %s", safe_statement_identification(s1));

      statement_to_send_recv(s1, &send_s1, &recv_s1);
      send=gen_nconc(gen_full_copy_list(send_s1), send);
      recv=gen_nconc(gen_full_copy_list(recv_s1), recv);

      FOREACH(SUCCESSOR, su, vertex_successors(v1))
	{
	  list send_s2, recv_s2;
	  statement s2 = vertex_to_statement(successor_vertex(su));
	  STEP_DEBUG_STATEMENT(2, "statement s2", s2);
	  pips_debug(3, "to %s", safe_statement_identification(s2));

	  statement_to_send_recv(s2, &send_s2, &recv_s2);

	  FOREACH(CONFLICT, c, dg_arc_label_conflicts((dg_arc_label)successor_arc_label(su)))
	    {
	      effect source = conflict_source(c);
	      effect sink = conflict_sink(c);
	      entity real = effect_variable(source);
	      pips_debug(2, "\t\t%s from\t%s", effect_to_string(source), text_to_string(text_region(source)));
	      pips_debug(2, "\t\t%s to  \t%s", effect_to_string(sink), text_to_string(text_region(sink)));
	      
	      if(in_step_analyses_com_p(real, send_s1) && !in_step_analyses_com_p(real, recv_s2))
		{
		  /* real n'est plus optimisable pour S1 */
		  not_send = CONS(ENTITY, real, not_send);
		  pips_debug(2,"set unoptimizable %s\n",entity_name(real));
		}

	      if(in_step_analyses_com_p(real, recv_s2) && !in_step_analyses_com_p(real, send_s1))
		{
		  /* real n'est plus remontable (produit par S1) */
		  not_recv = CONS(ENTITY, real, not_recv);
		  pips_debug(2,"set non remontable %s\n",entity_name(real));
		}
	    }
	  pips_debug(2, "statement s2 end\n");
	}
      pips_debug(2, "statement s1 end\n");      
    }

  l=recv;
  recv=NIL;
  FOREACH(STEP_COMM_ENTITIES, c, l)
    {
      entity real=step_comm_entities_real(c);
      if(!gen_in_list_p(real, not_recv))
	{
	  recv=CONS(STEP_COMM_ENTITIES, c, recv);
	  pips_debug(1,"\tadd recv (%s, %s)\n", entity_name(real),entity_name(step_comm_entities_formal(c)));
	}
      else
	{
	  pips_debug(2,"\tDROP recv %s\n",entity_name(real));
	}
    }
  
  l=send;
  send=NIL;
  FOREACH(STEP_COMM_ENTITIES, c, l)
    {
      bool is_optimizable;
      entity real = step_comm_entities_real(c);
      entity formal = step_comm_entities_formal(c);
      entity directive_module = module_name_to_entity(entity_module_name(formal));
      pips_debug(2,"directive module : %s\n", entity_name(directive_module));
      step_analyses analyses = load_global_step_analyses(directive_module);
      map_entity_bool optimizable = step_analyses_optimizable(analyses);

      if(!gen_in_list_p(real, not_send))
	{
	  is_optimizable = true;
	  send=CONS(STEP_COMM_ENTITIES, c, send);
	  pips_debug(1,"\tadd send (%s, %s)\n", entity_name(real),entity_name(step_comm_entities_formal(c)));
	}
      else
	{
	  is_optimizable = false;
	  pips_debug(2,"\tDROP send %s\n",entity_name(real));
	}

      set_optimizable(optimizable, formal, directive_module, is_optimizable);
    }
    
  
  DB_PUT_MEMORY_RESOURCE(DBR_STEP_ANALYSES_COM, module_name, make_step_analyses_com(recv, send));

  global_step_analyses_save();
  global_directives_save();

  reset_ordering_to_statement();
  reset_current_module_statement();
  reset_current_module_entity();

  debug_off(); 
  return true;
}
