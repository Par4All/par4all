/* Copyright 2007, 2008 Alain Muller, Frederique Silber-Chaussumier

This file is part of STEP.

The program is distributed under the terms of the GNU General Public
License.
*/


#include "defines-local.h"
#include "effects-convex.h"

// for :  string text_to_string(text t)...
#include "icfg.h"
#include "graph.h"
#include "ricedg.h"

GENERIC_GLOBAL_FUNCTION(initial_copy_entities, step_entity_map)
GENERIC_GLOBAL_FUNCTION(buffer_copy_entities, step_entity_map)

GENERIC_LOCAL_FUNCTION(index_copy_entities, step_entity_map)
GENERIC_LOCAL_FUNCTION(slice_array_entities, step_entity_map)
GENERIC_LOCAL_FUNCTION(lower_bounds_entities, step_entity_map)
GENERIC_LOCAL_FUNCTION(upper_bounds_entities, step_entity_map)
GENERIC_LOCAL_FUNCTION(recv_region_entities, step_entity_map)


static entity local_copy(entity module,string new_variable_name,entity original_entity)
{
  int size, offset;
  entity new=entity_undefined;
  entity area=entity_undefined;
  
  pips_debug(1, "module = %p, new_variable_name = %s, original_entity = %p\n", module, new_variable_name, original_entity);

  new=FindOrCreateEntity(entity_user_name(module),new_variable_name);
  entity_type(new)=copy_type(entity_type(original_entity));
  entity_initial(new)=entity_initial(original_entity);

  area = FindOrCreateEntity(entity_user_name(module), DYNAMIC_AREA_LOCAL_NAME);

  if (SizeOfArray(new,&size))
    {
      offset=add_variable_to_area(area, new);
    }
  else
    {
      offset = UNKNOWN_RAM_OFFSET;
      area_layout(type_area(entity_type(area))) = gen_nconc(area_layout(type_area(entity_type(area))), CONS(ENTITY, new, NIL));
    }

  entity_storage(new) = make_storage(is_storage_ram,make_ram(module,area,offset,NIL));

  pips_debug(1, "new = %p\n", new);

  return new;
}

static expression step_sizevariable_in_expression(variable v)
{
  list dim_l;
  dimension d;
  expression p;
  expression s;

  pips_debug(1, "v : %p\n", v);

  dim_l = gen_copy_seq(variable_dimensions(v));
  d = dimension_undefined;
  p = make_expression_1();
  s = expression_undefined;

  for (;dim_l;)
    {
      d=DIMENSION(CAR(dim_l));POP(dim_l);
      
      s=binary_intrinsic_expression(MINUS_OPERATOR_NAME,copy_expression(dimension_upper(d)),copy_expression(dimension_lower(d)));
      
      s=binary_intrinsic_expression(PLUS_OPERATOR_NAME,s,make_expression_1());
      
      p=binary_intrinsic_expression(MULTIPLY_OPERATOR_NAME,p,s);
    }

  pips_debug(1, "p = %p\n", p);
  return p;
}


/*############################################################################################*/
/* 
   Ajout dans la nouvelle fonction MPI de la declaration du tableau
   contenant les tranches d'indices a traiter par chaque noeud

   Ce tableau prend comme nom le nom de l'indice de la boucle prefixe par STEP_, suffixe par LOOPSLICES:

   ex: STEP_I_LOOPSLICES
*/
static entity step_declare_slice_array(entity module,entity i, entity size)
{
  entity new;
  dimension bounds, slices;
  list dimlist;
  entity area;
  int offset;

  pips_debug(1, "module = %p, i = %p, size = %p\n",  module,  i,  size);

  new = FindOrCreateEntity(entity_user_name(module),strdup(STEP_LOOPSLICES_NAME(i)));

  bounds = make_dimension(make_call_expression(step_i_slice_low,NIL),
				  make_call_expression(step_i_slice_up,NIL));
  slices = make_dimension(make_expression_1(),
				  entity_to_expression(size));

  dimlist = CONS(DIMENSION,bounds,
	      CONS(DIMENSION,slices,NIL));

  entity_type(new)=MakeTypeVariable(make_basic_int(DefaultLengthOfBasic(is_basic_int)), dimlist);
  area = FindOrCreateEntity(entity_user_name(module), DYNAMIC_AREA_LOCAL_NAME);
  offset = add_variable_to_area(area, new);

  entity_storage(new)=make_storage(is_storage_ram,make_ram(module,area,offset,NIL));
  store_slice_array_entities(i, new);
  AddEntityToDeclarations(new,module);

  pips_debug(1, "new = %p\n",  new);

  return new;
}


/*
  Declaration des tableau initial et buffer, pour gerer le cas d'entrelacement des regions SEND, 
*/
static entity step_declare_initial_copy(entity module, entity formal_entity)
{
  entity new;
  list l;
  
  pips_debug(1, "module = %p, formal_entity = %p\n", module,  formal_entity);

  new = local_copy(module,strdup(STEP_INITIAL_NAME(formal_entity)),formal_entity);
  store_initial_copy_entities(formal_entity,new);

  l=code_declarations(EntityCode(module));
  l=gen_nconc(l,CONS(ENTITY, new, NIL));

  pips_debug(1, "new = %p\n",  new);

  return new;
}

static entity step_declare_buffer_copy(entity module, entity formal_entity)
{
  entity new;
  list l;
  
  pips_debug(1, "module = %p, formal_entity = %p\n", module,  formal_entity);

  new = local_copy(module,strdup(STEP_BUFFER_NAME(formal_entity)),formal_entity);
  store_buffer_copy_entities(formal_entity,new);

  l=code_declarations(EntityCode(module));
  l=gen_nconc(l,CONS(ENTITY, new, NIL));

  pips_debug(1, "new = %p\n",  new);

  return new;
}

/*
  Declaration des variables utiliser pour le calcul des regions de tableau
*/
entity step_declare_region_computation_variable(entity mpi_module, entity i)
{
  entity new;
  entity directive_module=get_current_module_entity();

  pips_debug(1, "module = %p, i = %p\n", mpi_module, i);

  pips_debug(2, "entity_name(module) = %s, entity_user_name(i) = %s\n", entity_name(mpi_module), entity_user_name(i));

  /*
    Declaration de variables locales utilisees dans la boucle IDX, I_LOW, I_UP
   */

  new = local_copy(directive_module,strdup(STEP_INDEX_NAME),i);
  store_index_copy_entities(i,new);
  AddEntityToDeclarations(new,mpi_module);

  new = local_copy(directive_module,strdup(STEP_INDEX_LOW_NAME(i)),i);
  store_lower_bounds_entities(i,new);
  AddEntityToDeclarations(new,mpi_module);

  new = local_copy(directive_module,strdup(STEP_INDEX_UP_NAME(i)),i);
  store_upper_bounds_entities(i,new);
  AddEntityToDeclarations(new,mpi_module);

  pips_debug(1, "new = %p\n", new);

  return new;
}


/*############################################################################################*/
/*
  Generation de la boucle calculant les regions de tableau de chaque tranche d'iteration
*/
static statement build_loop_slice(entity mpi_module, entity i, entity max_i, statement body)
{
  entity i_s, i_i, i_l, i_u;
  reference low_ref, up_ref;
  statement low_statmt, up_statmt;
  range r;
  entity label;
  statement c, s; 
  instruction loop_instr;

  pips_debug(1, "mpi_module = %p, i = %p, max_i = %p, body = %p\n", mpi_module, i, max_i, body);
  if (empty_code_p(body)) return (body);

  i_i = load_index_copy_entities(i);
  i_l = load_lower_bounds_entities(i);
  i_u = load_upper_bounds_entities(i);

  pips_assert("i_i != entity_undefined", i_i != entity_undefined);
  pips_assert("i_l != entity_undefined", i_l != entity_undefined);
  pips_assert("i_u != entity_undefined", i_u != entity_undefined);


  /* I_LOW = STEP_I_LOOPSLICES(I_SLICE_LOW, IDX) */
  i_s = (entity)load_slice_array_entities(i);
  low_ref = make_reference(i_s,
		       CONS(EXPRESSION,entity_to_expression(step_i_slice_low),
			    CONS(EXPRESSION,entity_to_expression(i_i),NIL)));
  low_statmt = make_assign_statement(entity_to_expression(i_l),
			      make_expression(make_syntax_reference(low_ref),normalized_undefined));
  
  /* I_UP = STEP_I_LOOPSLICES(I_SLICE_UP, IDX) */
  up_ref = make_reference(i_s,
		       CONS(EXPRESSION,entity_to_expression(step_i_slice_up),
			    CONS(EXPRESSION,entity_to_expression(i_i),NIL)));
  up_statmt = make_assign_statement(entity_to_expression(i_u),
			    make_expression(make_syntax_reference(up_ref),normalized_undefined));

  r = make_range(int_to_expression(1),entity_to_expression(max_i),int_to_expression(1));
  pips_debug(1,"%s\n",entity_module_name(i_i));

  label = make_loop_label((int)NULL,entity_module_name(i_i));
  
  c = make_continue_statement(label);
  insert_statement(body,c,FALSE);
  insert_statement(body,up_statmt,TRUE);
  insert_statement(body,low_statmt,TRUE);
  

  loop_instr = make_instruction_loop(make_loop(i_i,r,body,label,make_execution_sequential(),NIL));

  s = make_statement(entity_empty_label(),STATEMENT_NUMBER_UNDEFINED,STATEMENT_ORDERING_UNDEFINED,strdup("\nC     Region computation\n"),loop_instr,NIL,string_undefined,empty_extensions ());

  pips_debug(1, "s = %p\n", s);
  return s;
}


static void _build_compute_region(expression exp, Psysteme sys, region reg, entity array_region, list slices, int *dim, statement *body)
{
  statement s_lower;
  statement s_upper;
  reference ref_left;
  expression call;
  int coef_phi;
  boolean existe_eq=FALSE;
  expression e;
  syntax s;
  list expr_l = NIL;
  list expr_u = NIL;
  entity phi;
  type t;
  dimension dim_d;
  Pcontrainte c;
  Pvecteur v;
  
  pips_debug(1, "exp = %p, sys = %p, reg = %p, array_region = %p, slices = %p, dim = %p, body = %p\n", exp, sys, reg, array_region, slices, dim, body);
  pips_debug(2,"region : %s\n",text_to_string(text_region(reg)));

  s = expression_syntax(exp);
  pips_assert("syntax_reference_p(s)", syntax_reference_p(s));
  phi = reference_variable(syntax_reference(s));
      
    
    /* Etape 2: on determine la valeur de l'indice extreme en fonction de L 
       Dans l'exemple, construction de la partie droite (L - 1)
       
       Traitements differents si egalite ou inegalite
    */
    
    // parcours des contraintes d'egalite a la recherche du phi courant
  c = sc_egalites(sys);

  for(;!existe_eq && !CONTRAINTE_UNDEFINED_P(c); c =c->succ)
    {
      coef_phi = VALUE_TO_INT(vect_coeff((Variable)phi, c->vecteur));
      if(coef_phi!=0)
	{
	  existe_eq = TRUE;
	  //construction des expressions d'affectation
	  v = vect_del_var(c->vecteur, (Variable)phi);
	  if(VECTEUR_NUL_P(v))
	    e=int_to_expression(0);
	  else
	    {
	      if(coef_phi > 0)
		{
		  Pvecteur coord;
		  for (coord = v; coord!=NULL; coord=coord->succ) 
		    val_of(coord) = -val_of(coord);
		  coef_phi = -coef_phi;
		}
	      e= make_vecteur_expression(v);
	      if (coef_phi != -1)
		e = make_op_exp("/", e, int_to_expression(-coef_phi));
	    }

	  expr_l = CONS(EXPRESSION,e,expr_l);
	  expr_u = CONS(EXPRESSION,copy_expression(e),expr_u);
	}
    }
  
  // generation des expressions associees aux differents Pvecteur ayant une composante selon le PHI courant
  // parcours des contraintes d'inegalite
  if(!existe_eq)
    {
      c = sc_inegalites(sys);
      
      for (;!CONTRAINTE_UNDEFINED_P(c); c =c->succ)
	{
	  coef_phi=VALUE_TO_INT(vect_coeff((Variable)phi,c->vecteur));
	  if(coef_phi != 0)
	    {
	      v = vect_del_var(c->vecteur, (Variable)phi);
	      if(VECTEUR_NUL_P(v))
		e=int_to_expression(0);
	      else
		{
		  boolean up_bound=FALSE;
		  if (coef_phi > 0) //contrainte de type : coef_phi*phi  <= "vecteur"
		    {
		      Pvecteur coord;
		      for (coord = v; coord!=NULL; coord=coord->succ) 
			val_of(coord) = -val_of(coord);
		      coef_phi = -coef_phi;
		      up_bound=TRUE;
		    }
		  e= make_vecteur_expression(v);
		  if (coef_phi != -1)
		    e = make_op_exp("/", e, int_to_expression(-coef_phi));

		  if(up_bound)
		    expr_u = CONS(EXPRESSION,e,expr_u);
		  else
		    expr_l = CONS(EXPRESSION,e, expr_l);
		}
	    }
	}
    }
  
  // generation des expressions d'initialisation (selon les bornes d'allocation du tableau)
  t = entity_type(region_entity(reg));
  pips_assert("variable",type_variable_p(t));
  
  dim_d = DIMENSION(gen_nth(*dim,variable_dimensions(type_variable(t))));
  if ( ENDP(expr_l) ) 
    expr_l=CONS(EXPRESSION,dimension_lower(dim_d),expr_l);
  if ( ENDP(expr_u) ) 
    expr_u=CONS(EXPRESSION,dimension_upper(dim_d),expr_u);
  
  /* Etape generation des statements */
  
  (*dim)++; // nb: les tableaux de region numérote les dimensions à partir de 1
  pips_assert("expression",gen_length(expr_l)&&gen_length(expr_u));
  
  //affectation "lower"
  ref_left = make_reference(array_region,
			    CONS(EXPRESSION,entity_to_expression(step_i_slice_low),
				 CONS(EXPRESSION,int_to_expression(*dim),slices)));
  
    
  e = make_expression(make_syntax_reference(ref_left),normalized_undefined);
  call = make_expression(make_syntax_call(make_call(entity_intrinsic("MAX"),expr_l)), normalized_undefined);
  
  if ( gen_length(expr_l) == 1 )
    s_lower = make_assign_statement(e,EXPRESSION(CAR(expr_l)));
  else
    s_lower = make_assign_statement(e,call);
  
  //affectation "upper"
  ref_left = make_reference(array_region,
			    CONS(EXPRESSION,entity_to_expression(step_i_slice_up),
				 CONS(EXPRESSION,int_to_expression(*dim),slices)));
  e = make_expression(make_syntax_reference(ref_left),normalized_undefined);
  call = make_expression(make_syntax_call(make_call(entity_intrinsic("MIN"),expr_u)),normalized_undefined);
  
  if(gen_length(expr_u)==1)
    s_upper = make_assign_statement(e,EXPRESSION(CAR(expr_u)));
  else
    s_upper = make_assign_statement(e,call);
  
  insert_statement(*body, s_lower, FALSE);
  insert_statement(*body, s_upper, FALSE);
}      


/*
  Traduction du systeme de contraintes en statements specifiant pour
  chaque dimension du tableau les bornes MIN et MAX d'indices pour les
  communications MPI.
  
  Ces nouveaux statements remplissent a l'execution le tableau
  STEP_SR_A par exemple.

*/
static statement step_build_compute_region(list loop_data_l, region reg2, entity array_region)
{
  statement call_statmt;
  statement body;
  string commentaire;
  region reg = region_dup(reg2);
  Psysteme sys = region_system(reg);
  list slices=NIL;
  int d = 0; // nb: gen_nth numerote les element a partir de 0 d'ou l'initialisation a 0

  pips_debug(1, "reg2 = %p, array_region = %p\n", reg2, array_region);

  /* substitution des variables formelles LU par les variables locales LU associees
  
    Remplace dans le systeme de contraintes les variables l et u de
    l'ancienne fonction outlinee par les variables i_l et i_u de la
    nouvelle fonction MPI 
  */
  MAP(LOOP_DATA,data,
      {
	entity i=loop_data_index(data);
	entity l=loop_data_lower(data);
	entity u=loop_data_upper(data);
	
	slices = CONS(EXPRESSION,entity_to_expression(load_index_copy_entities(i)),slices);
	
	sys = sc_variable_rename(sys,(Variable)l,(Variable)load_lower_bounds_entities(i));
	sys = sc_variable_rename(sys,(Variable)u,(Variable)load_upper_bounds_entities(i));
      },loop_data_l);
  slices = gen_nreverse(slices);

  body = make_block_statement(NIL);

  /* parcours des dimensions de la region
     pour rechercher pour chaque dimension les indices extremes de tableau

     ex: L <= phi1 + 1 (phi1 pour la dimension 1) sur l'indice i
     on genere le statement
     STEP_SR_A(LOW, 1, i_i) = L - 1

  */

  /* Etape 1: on cherche le phi correspondant a la dimension d */
  // d=0 : gen_nth numerote les element a partir de 0 d'ou l'initialisation a 0
  MAP(EXPRESSION, exp, {
    _build_compute_region(exp, sys, reg, array_region, slices, &d, &body);
  },
      reference_indices(region_reference(reg)));
  
  //ajout de la region en commentaire

  commentaire = text_to_string(text_region(reg));
  call_statmt = make_call_statement(CONTINUE_FUNCTION_NAME, NIL, entity_empty_label(),commentaire);
  insert_statement(body,call_statmt,TRUE);

  pips_debug(1, "body = %p\n", body);
  return body;
}

/*
  calcul des regions SEND (et des regions RECV)
*/
static void step_compute_sr_rr_arrays(list loop_data_l, step_region_analyse loop_analyse, entity mpi_module)
{
  statement compute_region;
  statement body_compute_region;
  statement assigne_region_0;
  statement statmt;

  pips_debug(1, "mpi_module = %p\n", mpi_module);

  // creation des statements initialisant les tableaux de region
  compute_region = statement_undefined;
  body_compute_region = make_block_statement(NIL);
  assigne_region_0 = make_block_statement(NIL);

  /*
  if (!get_bool_property("STEP_MPI_ALLtoALL"))
    {
      MAP(REGION,r,{ // Region RECV
	if(!region_scalar_p(r) && !bound_private_entities_p(region_entity(r)))
	  {
	    statement statmt0 = build_assigne_region0(gen_length(loop_data_l),r,load_recv_region_entities(region_entity(r)));
	    statement statmt = step_build_compute_region(loop_data_l,r,load_recv_region_entities(region_entity(r)));
	    insert_statement(assigne_region_0,statmt0,FALSE);
	    insert_statement(body_compute_region,statmt,FALSE);
	  }
      },
	step_region_analyse_recv(loop_analyse));
      
    }
  */

  MAP(REGION,r,{ // Region SEND
      if(!region_scalar_p(r) && !bound_private_entities_p(region_entity(r)))
      {
	statement statmt0=build_assigne_region0(gen_length(loop_data_l),r,load_send_region_entities(region_entity(r)));
	statement statmt=step_build_compute_region(loop_data_l,r,load_send_region_entities(region_entity(r)));
	insert_statement(assigne_region_0, statmt0, FALSE);
	insert_statement(body_compute_region, statmt, FALSE);
      }
  },
    step_region_analyse_send(loop_analyse));
  

  if (!empty_code_p(assigne_region_0))
    {
  statmt = make_continue_statement(entity_undefined);
  statement_comments(statmt) = strdup("\nC     Put array boundaries into region arrays (SR: Send region)\nC     First dimension: lower and upper bounds of each slice\nC     Second dimension: for each dimension of the original array\nC     Third dimention: store the boundaries of the local chunk. The first element stores initial boundaries, then one element for each process\n");
  step_seqlist = CONS(STATEMENT,statmt, step_seqlist);
  step_seqlist = CONS(STATEMENT,assigne_region_0, step_seqlist);

  /* Inclusion des statements de remplissage du tableau STEP_SR_A
     dans une boucle pour parcourir les tranches */
  MAP(LOOP_DATA,data,
      {
	step_seqlist = CONS(STATEMENT,build_loop_slice(mpi_module, loop_data_index(data), step_max_nb_loopslices, body_compute_region), step_seqlist);
      },loop_data_l)
    }

  pips_debug(1, "End\n");
}

/*############################################################################################*/
static void step_create_mpi_before_loop(list loop_data_l,step_region_analyse loop_analyse, entity mpi_module, entity size, entity rank)
{
  statement statmt;
  
  pips_debug(1, "mpi_module = %p, size = %p, rank = %p\n", mpi_module, size, rank);
  
  
  /* cas d'entrelacement des regions SEND: besoin de 3 tableaux
     - le tableau des donnees
     
     - avec 2 tableaux supplementaires pour faire la comparaison
     
     * tableau de valeurs initiales appele initial
     * tableau de valeurs modifiees appele buffer
     
     creation des 2 tableaux supplementaires:
     - ajout des declarations dans la nouvelle fonction MPI
  */
  
  MAP(REGION,r,{
      entity e=region_entity(r);    // e correspond au tableau (ancienne entite (dans la fonction outlinee))
      pips_debug(2,"entity region : %s\n",entity_name(e));
      if (!bound_reduction_entities_p(e) && !bound_private_entities_p(e))
	{
	  if (entity_scalar_p(e))
	    pips_user_warning("STEP : possible data race with : %s\n\n",entity_name(e));
	  else
	    {
	      // recuperation de la nouvelle entite (dans la nouvelle fonction MPI) correspondant au tableau 
	      entity interlaced=gen_find_tabulated(concatenate(entity_user_name(mpi_module), MODULE_SEP_STRING,
							       entity_user_name(region_entity(r)),NULL),
						   entity_domain);
	      pips_assert("interlaced",!entity_undefined_p(interlaced));
	      
	      // declaration des 2 nouveaux tableaux initial et buffer
	      entity initial=step_declare_initial_copy(mpi_module,interlaced);
	      entity buffer=step_declare_buffer_copy(mpi_module,interlaced);
	      pips_debug(2,"entity region interlaced : %s\nentity region initial : %s\nentity region buffer : %s\n",
			 entity_name(interlaced),entity_name(initial),entity_name(buffer));
	      
	      // Creation de l'instruction 'call STEP_InitInterlaced_I(nb_elements, interlaced, initial, buffer)' permettant d'initialiser les tableaux
	      expression nb_elements = step_sizevariable_in_expression(type_variable(entity_type(interlaced)));
	      step_seqlist=CONS(STATEMENT,
				call_STEP_subroutine(strdup(concatenate(RT_STEP_InitInterlaced,step_type_suffix(e),NULL)),
						     CONS(EXPRESSION,nb_elements,
							  CONS(EXPRESSION,entity_to_expression(interlaced),
							       CONS(EXPRESSION,entity_to_expression(initial),
								    CONS(EXPRESSION,entity_to_expression(buffer),NIL))))),
				step_seqlist);
	    }
	}
    },step_region_analyse_interlaced( loop_analyse));
  
  /* 
     declaration des constantes symboliques I_SLICE_LOW, I_SLICE_UP,
     MAX_NB_LOOPSLICES et autres variables
  */
  step_add_parameter(mpi_module);
  
  /* 
     Creation de l'instruction 'call STEP_ComputeLoopSlices(low, up,
     step, size, step_max_nb_loopslices, i_step_loopslices)' creant
     un tableau step_i_loopslices contenant les tranches d'indices a
     traiter par chaque noeud
     
     creation du tableau step_i_loopslices et des variables i_i, i_l, i_u
  */
  MAP(LOOP_DATA,data,
      {
	int step = loop_data_step(data);
	entity i = loop_data_index(data);
	entity low = loop_data_lower(data);
	entity up = loop_data_upper(data);
	entity i_s = step_declare_slice_array(mpi_module, i, step_max_nb_loopslices); //le nombre slice pourrait etre recupere dynamiquement
	
	step_declare_region_computation_variable(mpi_module,i);
	
	// statement : CALL STEP_ComputeLoopSlices(I_L, I_U, loop_step, STEP_Size, MAX_NB_LOOPSLICES, STEP_I_LOOPSLICES)
	statmt = call_STEP_subroutine(RT_STEP_ComputeLoopSlices,
				      CONS(EXPRESSION,entity_to_expression(low),
					   CONS(EXPRESSION,entity_to_expression(up),
						CONS(EXPRESSION,int_to_expression(step),
						     CONS(EXPRESSION,entity_to_expression(size),
							  CONS(EXPRESSION,entity_to_expression(step_max_nb_loopslices),
							       CONS(EXPRESSION,entity_to_expression(i_s),NIL)))))));
	statement_comments(statmt)=strdup("\n");
	step_seqlist = CONS(STATEMENT, statmt, step_seqlist);
      },loop_data_l);
  
  /*
    Declaration du tableau permettant de connaitre pour chaque noeud
    les bornes d'indices de tableau a communiquer
    
    Le tableau prend le nom du tableau initial suffixe par _STEP_RR
    
    Pour utilisation des regions RECV
    
    Non utilise pour l'instant: toujours en broadcast
  */  
  /*
  if (!get_bool_property("STEP_MPI_ALLtoALL"))
    MAP(REGION,r,{
	if(!region_scalar_p(r))
	  {
	    entity recv = region_entity(r);
	    store_or_update_recv_region_entities(region_entity(r),
						 step_create_region_array(mpi_module,strdup(concatenate(entity_user_name(recv),"_STEP_RR",NULL)),recv,TRUE));
	  }
      },step_region_analyse_recv(loop_analyse));
  */

  /*
    Declaration du tableau permettant de connaitre pour chaque noeud
    les bornes d'indices de tableau a communiquer
    
    Le tableau prend le nom du tableau initial prefixe par STEP_SR_
    
    Pour utilisation des regions SEND
    
    Utilise
  */
  MAP(REGION,r,{
      if(!region_scalar_p(r) && !bound_private_entities_p(region_entity(r))){
	entity send=region_entity(r);
	store_or_update_send_region_entities(region_entity(r),
					     step_create_region_array(mpi_module,strdup(STEP_SR_NAME(send)),send,TRUE));
      }
    },step_region_analyse_send(loop_analyse));
  
  pips_debug(2,"nb declaration :%zd\n",gen_length(code_declarations(EntityCode(mpi_module))));
  
  ifdebug(2)
    MAP(ENTITY, v, {
	pips_debug(2,"declaration  %s variable : %p\n",entity_name(v),(void*)type_tag(entity_type(v)));
      },code_declarations(EntityCode(mpi_module)));

  step_compute_sr_rr_arrays(loop_data_l,loop_analyse, mpi_module);

  //  if(!get_bool_property("STEP_MPI_ALLtoALL")){}  // communication IN
  
  pips_debug(1, "End\n");
}

static void step_create_mpi_after_loop(step_region_analyse step_analyse, entity mpi_module, entity size, entity rank)
{
  // communications OUT
  pips_debug(1,"mpi_module = %p, size = %p, rank = %p\n", mpi_module, size, rank);
  list seqlist_all2all = NIL;  
  entity requests_array = step_declare_requests_array(mpi_module,step_region_analyse_send(step_analyse));// effet de bord : initialisation de l'entite "step_nb_max_request" (constante symbolique)

  MAP(REGION,r,{
      if(!region_scalar_p(r) && !bound_private_entities_p(region_entity(r))){
	boolean interlaced_p = FALSE;
	entity array = region_entity(r);
	list interlaced_l=gen_copy_seq(step_region_analyse_interlaced(step_analyse));
	
	pips_debug(2,"region %s\n",entity_name(array));
	while(!interlaced_p && !ENDP(interlaced_l))
	  {
	    interlaced_p = array==region_entity(REGION(CAR(interlaced_l)));

	    pips_debug(2,"interlaced %i array : %s->%s %s\n",interlaced_p,entity_name(array),entity_name(load_new_entities(array)),entity_name(region_entity(REGION(CAR(interlaced_l)))));

	    POP(interlaced_l);
	  }
	
	seqlist_all2all=CONS(STATEMENT,build_call_STEP_AlltoAllRegion(interlaced_p,array,size,0),seqlist_all2all);
      }
    },step_region_analyse_send(step_analyse));
 
  if (!ENDP(seqlist_all2all))
    code_declarations(EntityCode(mpi_module)) =gen_append(code_declarations(EntityCode(mpi_module)),
							  CONS(ENTITY,step_max_nb_request,CONS(ENTITY,step_requests,NIL)));

  step_seqlist=gen_append(step_handle_comm_requests(requests_array,seqlist_all2all),step_seqlist);

  pips_debug(1, "End\n");
}


static list step_build_new_loop_data(list loop_data_l, entity mpi_module, entity rank)
{
  list new_loop_data_l=NIL;
  expression rank_expr_p1 = binary_intrinsic_expression(PLUS_OPERATOR_NAME,entity_to_expression(rank),make_expression_1()); 

  pips_debug(1, "mpi_module = %p, rank = %p\n", mpi_module, rank);

  pips_assert("length(loop_data_l)=1", gen_length(loop_data_l)==1);
  MAP(LOOP_DATA,data,{
      entity i= loop_data_index(data);
      pips_assert("i != entity_undefined", i != entity_undefined);

      // creation  et declaration des variables locales STEP_I_LOW, STEP_I_UP
      entity i_l = make_scalar_integer_entity(strdup(STEP_BOUNDS_LOW(i)),entity_user_name(mpi_module));
      entity i_u = make_scalar_integer_entity(strdup(STEP_BOUNDS_UP(i)),entity_user_name(mpi_module));
      loop_data new_data=make_loop_data(i,i_l,i_u,loop_data_step(data));
      AddEntityToDeclarations(loop_data_lower(new_data), mpi_module);
      AddEntityToDeclarations(loop_data_upper(new_data), mpi_module);
      
      new_loop_data_l = CONS(LOOP_DATA,new_data,new_loop_data_l);

      // add statement : STEP_I_LOW = STEP_I_LOOPSLICES(I_SLICE_LOW, STEP_Rank + 1)
      reference r_l = make_reference(load_slice_array_entities(i),
			   CONS(EXPRESSION,entity_to_expression(step_i_slice_low),
				CONS(EXPRESSION,rank_expr_p1,NIL)));
      step_seqlist=CONS(STATEMENT,make_assign_statement(entity_to_expression(loop_data_lower(new_data)), 
							make_expression(make_syntax_reference(r_l),normalized_undefined)),
			step_seqlist);
      
      // add statement : STEP_I_UP = STEP_I_LOOPSLICES(I_SLICE_UP, STEP_Rank + 1)
      reference r_u = make_reference(load_slice_array_entities(i),
			   CONS(EXPRESSION,entity_to_expression(step_i_slice_up),
				CONS(EXPRESSION,rank_expr_p1,NIL)));
      step_seqlist=CONS(STATEMENT,make_assign_statement(entity_to_expression(loop_data_upper(new_data)),
							make_expression(make_syntax_reference(r_u),normalized_undefined)),
			step_seqlist);

      pips_debug(2,"drop %s add %s\n",entity_name(loop_data_lower(data)),entity_name(loop_data_lower(new_data)));
      pips_debug(2,"drop %s add %s\n",entity_name(loop_data_upper(data)),entity_name(loop_data_upper(new_data)));
    },loop_data_l)
  
  pips_debug(1, "End\n");
  return gen_nreverse(new_loop_data_l);
}

static void step_call_outlined_loop(list loop_data_l,entity original_module)
{
  list arglist, exprlist;
  instruction call_instr;

  pips_debug(1, "original_module = %p\n", original_module);

  exprlist=NIL;
  arglist=NIL;

  MAP(ENTITY,e,{
      entity ne = load_new_entities(e);
      pips_debug(2,"call : entity %s -> %s\n", entity_name(e), entity_name(ne));

      exprlist = CONS(EXPRESSION,entity_to_expression(ne),exprlist);
    },
    outline_data_formal(load_outline(original_module)));
  exprlist=gen_nreverse(exprlist);
  
  pips_assert("length(loop_data_l)=1", gen_length(loop_data_l)==1);
  MAP(LOOP_DATA,data,
      {
	POP(exprlist);POP(exprlist);POP(exprlist); // suppression I,L,U
	arglist = gen_nconc(arglist,CONS(EXPRESSION,entity_to_expression(loop_data_index(data)),
					 CONS(EXPRESSION,entity_to_expression(loop_data_lower(data)),
					      CONS(EXPRESSION,entity_to_expression(loop_data_upper(data)),
						   NIL))));
      },loop_data_l);
    
  call_instr = make_instruction(is_instruction_call, make_call(original_module, gen_append(arglist,exprlist)));

  step_seqlist = CONS(STATEMENT,make_stmt_of_instr(call_instr), step_seqlist);
}

/*
 Genere la fonction incluant les communications MPI
 Appelle la fonction faisant la repartition des indices de boucles
*/
entity step_create_mpi_loop_module(entity directive_module)
{
  string new_name = NULL;
  entity mpi_module;
  entity size, rank;
  list loop_data_l;
  directive d=load_global_directives(directive_module);
  step_region_analyse loop_analyse= load_step_analyse_map(directive_module);

  pips_debug(1, "directive_module = %p\n", directive_module);
  if(type_directive_omp_parallel_do_p(directive_type(d)))
    loop_data_l = type_directive_omp_parallel_do(directive_type(d));
  else
    loop_data_l = type_directive_omp_do(directive_type(d));

  init_old_entities();
  init_new_entities();
  init_initial_copy_entities();
  init_buffer_copy_entities();
  init_index_copy_entities();
  init_slice_array_entities();
  init_lower_bounds_entities();
  init_upper_bounds_entities();
  init_recv_region_entities();
  init_send_region_entities();
  init_reduction_entities();

  new_name=step_find_new_module_name(directive_module,STEP_MPI_SUFFIX);
  mpi_module = make_empty_subroutine(new_name);

  pips_debug(2, "entity_name(mpi_module) : %s\n\n", entity_name(mpi_module));
  
  size =MakeConstant(STEP_SIZE_NAME,is_basic_string);
  rank =MakeConstant(STEP_RANK_NAME,is_basic_string);

  /* 
     ajout des variables formelles pour la nouvelle fonction MPI
     (identiques a celles de la fonction outlinee
  */
  step_add_formal_copy(mpi_module,outline_data_formal(load_outline(directive_module)));
  
  step_private_before(directive_module);
  step_reduction_before(directive_module,mpi_module);
  step_create_mpi_before_loop(loop_data_l,loop_analyse, mpi_module, size, rank); 
  
  /*
    recuperation des nouvelles bornes de boucles
    calcul des nouveaux arguments de l'appel a la fonction outlinee
  */

  statement statmt = make_continue_statement(entity_undefined);
  statement_comments(statmt) = strdup("\nC     Where work is done...\n");
  step_seqlist = CONS(STATEMENT, statmt, step_seqlist);

  step_call_outlined_loop(step_build_new_loop_data(loop_data_l, mpi_module, rank),directive_module);
  
  statmt = make_continue_statement(entity_undefined);
  statement_comments(statmt)=strdup("\nC     Communicating data to other nodes\n");
  step_seqlist = CONS(STATEMENT, statmt, step_seqlist);
  
  step_create_mpi_after_loop(loop_analyse, mpi_module, size, rank);
  step_reduction_after(directive_module);
  step_private_after();

  step_seqlist = CONS(STATEMENT, make_return_statement(mpi_module), step_seqlist);

  pips_debug(2, "entity_user_name(mpi_module) = %s\n", entity_user_name(mpi_module));

  close_old_entities();
  close_new_entities();
  close_initial_copy_entities();
  close_buffer_copy_entities();
  close_index_copy_entities();
  close_slice_array_entities();
  close_lower_bounds_entities();
  close_upper_bounds_entities();
  close_recv_region_entities();
  close_send_region_entities();
  close_reduction_entities();

  return mpi_module;
}
