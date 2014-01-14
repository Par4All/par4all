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
/*
 * build-system.c
 *
 * here should be build equations and inequations to deal
 * with the I/O in hpf programs.
 *
 * Fabien COELHO, Feb/Mar 94
 */

#include "defines-local.h"
#include "effects-generic.h"
#include "effects-simple.h"
#include "effects-convex.h"

#define ALPHA_PREFIX	"ALPHA"
#define LALPHA_PREFIX	"LALPHA"
#define THETA_PREFIX 	"THETA"
#define PSI_PREFIX	"PSI"
#define GAMMA_PREFIX	"GAMMA"
#define DELTA_PREFIX	"DELTA"
#define IOTA_PREFIX	"IOTA"
#define SIGMA_PREFIX	"SIGMA"
#define TMP_PREFIX	"TMP"

/* tags with a newgen look and feel */
#define is_entity_array		0
#define is_entity_template	1
#define is_entity_processors	2

/* Variables
 *  + array dimensions (PHIs)
 *  + template dimensions
 *  + processor dimensions
 *  + cycles and offsets
 *  + local array dimensions
 *  + indexes and others coming thru the regions
 *
 * Inequations to be defined
 *  + array declaration
 *  + template declaration
 *  + processors arrangement declaration
 *  + local offsets within a block
 *  + local declarations
 *  - regions accessed by the statement
 *
 * Equations to be defined
 *  + alignement
 *  + distribution
 *  + local <-> global?
 *  + processor linearization?
 *
 * Remarks
 *  - offset to be computed
 *  - access functions are not needed (hidden by regions)
 *  - how to be sure that something can be done?
 *  - will newgen structures be necessary to build the systems?
 *  - will I have to remove some variables (indexes ?)
 *  - one equation to be added for replicated dimensions.
 */

/* variable names:
 *
 * ALPHA{1-7}: array dimensions, 
 * THETA{1-7}: template dimensions,
 * PSI{1-7}: processor dimensions,
 * SIGMA{1-7}: auxiliary variable,
 * GAMMA{1-7}: cycles,
 * DELTA{1-7}: local offsets,
 * LALPHA{1-7}: local array dimensions, if specified...
 *
 * plus "PRIME" versions
 */

/* ------------------------------------------------------------------
 *
 * HPF CONSTRAINTS GENERATION
 */

bool entity_hpfc_dummy_p(e)
entity e;
{
  return same_string_p(entity_module_name(e), HPFC_PACKAGE);
}

/********************************************************* DUMMY VARIABLES */

#define PRIME_LETTER_FOR_VARIABLES      "p"

/* define to build the _dummy and _prime of a variable.
 */
#define GET_DUMMY_VARIABLE_ENTITY(MODULE, NAME, lname)                  \
  entity get_ith_##lname##_dummy(int i)                                 \
  {                                                                     \
    return get_ith_dummy(MODULE, NAME, i);                              \
  }                                                                     \
  entity get_ith_##lname##_prime(int i)                                 \
  {                                                                     \
    return get_ith_dummy(MODULE, NAME PRIME_LETTER_FOR_VARIABLES, i);   \
  }

GET_DUMMY_VARIABLE_ENTITY(REGIONS_MODULE_NAME,	PHI_PREFIX, 	region)
GET_DUMMY_VARIABLE_ENTITY(HPFC_PACKAGE, 	ALPHA_PREFIX, 	array)
GET_DUMMY_VARIABLE_ENTITY(HPFC_PACKAGE, 	THETA_PREFIX, 	template)
GET_DUMMY_VARIABLE_ENTITY(HPFC_PACKAGE, 	PSI_PREFIX, 	processor)
GET_DUMMY_VARIABLE_ENTITY(HPFC_PACKAGE, 	DELTA_PREFIX, 	block)
GET_DUMMY_VARIABLE_ENTITY(HPFC_PACKAGE, 	GAMMA_PREFIX, 	cycle)
GET_DUMMY_VARIABLE_ENTITY(HPFC_PACKAGE, 	LALPHA_PREFIX, 	local)
GET_DUMMY_VARIABLE_ENTITY(HPFC_PACKAGE, 	IOTA_PREFIX, 	shift)
GET_DUMMY_VARIABLE_ENTITY(HPFC_PACKAGE, 	SIGMA_PREFIX, 	auxiliary)
GET_DUMMY_VARIABLE_ENTITY(HPFC_PACKAGE,		TMP_PREFIX,	temporary)

/* shift dummy variables to prime variables.
 * systeme s is modified.
 */
GENERIC_LOCAL_FUNCTION(dummy_to_prime, entitymap)

static void put_dummy_and_prime(gen1, gen2)
entity (*gen1)(), (*gen2)();
{
  int i;
  for(i=7; i>0; i--)
    store_dummy_to_prime(gen1(i), gen2(i));
}

#define STORE(name) \
  put_dummy_and_prime(get_ith_##name##_dummy, get_ith_##name##_prime)

void hpfc_init_dummy_to_prime()
{
    init_dummy_to_prime();
    STORE(array);
    STORE(template);
    STORE(processor);
    STORE(block);
    STORE(cycle);
    STORE(local);
    STORE(shift);
    STORE(auxiliary);
}

void hpfc_close_dummy_to_prime()
{
    close_dummy_to_prime();
}

Psysteme shift_system_to_prime_variables(s)
Psysteme s;
{
    return sc_rename_variables(s, (bool (*)())bound_dummy_to_prime_p,
			       (Variable(*)()) load_dummy_to_prime);
}

/* already computed constraints
 */
#ifndef Psysteme_undefined
#define Psysteme_undefined SC_UNDEFINED
#define Psysteme_undefined_p(sc) SC_UNDEFINED_P(sc)
#endif
/* ??? used with a temporary hack to differentiate array and templates */
GENERIC_LOCAL_MAPPING(declaration_constraints, Psysteme, entity)
GENERIC_LOCAL_MAPPING(hpf_align_constraints, Psysteme, entity)
GENERIC_LOCAL_MAPPING(hpf_distribute_constraints, Psysteme, entity)
GENERIC_LOCAL_MAPPING(new_declaration_constraints, Psysteme, entity)

void make_hpfc_current_mappings()
{
    make_declaration_constraints_map();
    make_hpf_align_constraints_map();
    make_hpf_distribute_constraints_map();
    make_new_declaration_constraints_map();
}

void free_hpfc_current_mappings()
{
    free_declaration_constraints_map();
    free_hpf_align_constraints_map();
    free_hpf_distribute_constraints_map();
    free_new_declaration_constraints_map();
}

/* ------------------------------------------------------------------
 *
 * DECLARATION CONSTRAINTS GENERATION
 */

/* Psysteme compute_entity_to_constraints(ent, suffix, prefix)
 * entity ent: variable the constraints of which are computed
 * strings suffix and prefix: to be used in the dummy variables created
 *
 * computes the constraints due to the declarations.
 *! usefull
 */
Psysteme compute_entity_to_declaration_constraints(ent, suffix, prefix)
entity ent;
string suffix, prefix;
{
    list dims = variable_dimensions(type_variable(entity_type(ent)));
    int dim_number = 1;
    Psysteme new_system = sc_new();
    
    pips_assert("variable", entity_variable_p(ent));
    
    /* system may be empty for scalars ???
     */
    pips_debug(5, "entity %s, [%s,%s]\n", entity_name(ent), prefix, suffix);
    
    MAP(DIMENSION, dim,
     {
	 entity dummy = get_ith_dummy(prefix, suffix, dim_number);
	 int ilower;
	 int iupper;
	 bool blower = hpfc_integer_constant_expression_p
	     (dimension_lower(dim), &ilower);
	 bool bupper = hpfc_integer_constant_expression_p
	     (dimension_upper(dim), &iupper);

	 pips_assert("extent known", blower && bupper);
	 
	 /* now the dummy is to be used to generate two inequalities: 
	  * -dummy + lower <= 0 and dummy - upper <= 0
	  */
	 sc_add_inegalite
	     (new_system,
	      contrainte_make(vect_make(VECTEUR_NUL,
					dummy, 	VALUE_MONE,
					TCST, 	int_to_value(ilower))));
	 sc_add_inegalite
	     (new_system,
	      contrainte_make(vect_make(VECTEUR_NUL,
					dummy, 	VALUE_ONE,
					TCST, 	int_to_value(-iupper))));
	 dim_number++;
     },
	 dims);
    
    sc_creer_base(new_system);
    return new_system;
}

static Psysteme 
hpfc_compute_entity_to_declaration_constraints(
    entity e,
    tag what)
{
    string local_prefix = (what==is_entity_array? ALPHA_PREFIX:
			   what==is_entity_template? THETA_PREFIX:
			   what==is_entity_processors? PSI_PREFIX: "ERROR");

    return compute_entity_to_declaration_constraints
	   (e, local_prefix, HPFC_PACKAGE);
}

/* gives back the constraints due to the declarations.
 * Uses a demand driven approach: computed systems are stored
 * in the declaration_constraints mapping for later search.
 */
Psysteme 
entity_to_declaration_constraints(
    entity e,
    tag what)
{
    Psysteme p = load_entity_declaration_constraints(e+what);
    pips_assert("variable", entity_variable_p(e));

    if (Psysteme_undefined_p(p))
    {
	p = hpfc_compute_entity_to_declaration_constraints(e, what);
	store_entity_declaration_constraints(e+what, p);
    }

    DEBUG_SYST(9, concatenate("entity ", entity_name(e), NULL), p);

    return p;
}

/* Psysteme hpfc_compute_align_constraints(e)
 * entity e is an array
 *
 * compute the align equations:
 *
 * theta_i - a phi_j - b == 0
 */
static Psysteme hpfc_compute_align_constraints(e)
entity e;
{
    align al = load_hpf_alignment(e);
    entity template = align_template(al);
    Psysteme new_system = sc_new();
    int i;

    pips_assert("distributed array", array_distributed_p(e));

    for(i=1 ; i<=NumberOfDimension(template) ; i++)
    {
	entity theta = get_ith_template_dummy(i);
	alignment a = FindAlignmentOfTemplateDim(align_alignment(al), i);
	
	if (a!=alignment_undefined)
	{
	    int adim = alignment_arraydim(a),
		constant = HpfcExpressionToInt(alignment_constant(a));
	    Pvecteur
		v = vect_make(VECTEUR_NUL,
			      theta, 	VALUE_ONE,
			      TCST, 	int_to_value(-constant));
					       
	    if (adim==0)
	    {
		sc_add_egalite(new_system, contrainte_make(v));
	    }
	    else
	    {
		entity phi = get_ith_array_dummy(adim);
		int rate = HpfcExpressionToInt(alignment_rate(a));

		v = vect_make(v, phi, int_to_value(-rate), TCST, VALUE_ZERO);
		sc_add_egalite(new_system, contrainte_make(v));
	    }
	}
    }

    sc_creer_base(new_system);
    return(new_system);
}

/* Psysteme hpfc_compute_unicity_constraints(e)
 * entity e should be an array;
 *
 * equations for non aligned template dimensions are computed: ???
 *
 * theta_i - lower_template_i == 0
 */
Psysteme hpfc_compute_unicity_constraints(e)
entity e;
{
    align al = load_hpf_alignment(e);
    entity template = align_template(al);
    Psysteme new_system = sc_new();
    int i;

    pips_assert("distributed array", array_distributed_p(e));

    for(i=1 ; i<=NumberOfDimension(template) ; i++)
    {
	alignment
	    a = FindAlignmentOfTemplateDim(align_alignment(al), i);
	
	if (a==alignment_undefined)
	{
	    entity
		theta = get_ith_template_dummy(i);
	    int
		low = 
		    HpfcExpressionToInt
			(dimension_lower(entity_ith_dimension(template, i)));

	    sc_add_egalite
		(new_system,
		 contrainte_make(vect_make(VECTEUR_NUL,
					   theta, 	VALUE_ONE,
					   TCST, 	int_to_value(-low))));
	}
    }
    sc_creer_base(new_system);
    return(new_system);
}

/* Psysteme hpfc_compute_distribute_constraints(e)
 * entity e should be a template;
 *
 * the constraints due to the distribution are defined:
 *
 * theta_i - theta_i0 == Nj Pj gamma_j + Nj (psi_j - psi_j0) + delta_j
 * delta_j >= 0
 * delta_j < Nj
 * ??? if block distribution: gamma_j == 0 
 * ??? not distributed template dimensions are skipped...
 * ??? if cyclic(1) distribution: delta_j == 0
 */
static Psysteme hpfc_compute_distribute_constraints(e)
entity e;
{
    Psysteme new_system = sc_new();
    distribute di = load_hpf_distribution(e);
    entity proc = distribute_processors(di);
    list ld = distribute_distribution(di);
    int j, i;
    
    pips_assert("template", entity_template_p(e));

    for(j=1 ; j<=NumberOfDimension(proc) ; j++)
    {
	distribution
	    d = FindDistributionOfProcessorDim(ld, j, &i);
	entity
	    theta = get_ith_template_dummy(i),
	    psi = get_ith_processor_dummy(j),
	    gamma = get_ith_cycle_dummy(j),
	    delta = get_ith_block_dummy(j);
	int
	    param = HpfcExpressionToInt(distribution_parameter(d)),
	    theta0 = HpfcExpressionToInt
		(dimension_lower(entity_ith_dimension(e, i))),
	    psi0 = HpfcExpressionToInt
		(dimension_lower(entity_ith_dimension(proc, j))),
	    proc_size = SizeOfIthDimension(proc, j);
	style st = distribution_style(d);
	Pvecteur v = VECTEUR_UNDEFINED;

	/* -delta_j <= 0
	 */
	sc_add_inegalite(new_system, 
		contrainte_make(vect_new((Variable) delta, VALUE_MONE)));

	/* delta_j - (N_j - 1) <= 0
	 */
	sc_add_inegalite(new_system,
	   contrainte_make(vect_make(VECTEUR_NUL,
				     (Variable) delta, 	VALUE_ONE,
				     TCST, int_to_value(-param+1))));

	/* theta_i - Nj psi_j - Nj Pj gamma_j - delta_j + Nj psi_j0 - theta_i0
	 * == 0
	 */
	v = vect_make(VECTEUR_NUL,
		      (Variable) theta, VALUE_ONE,
		      (Variable) psi, 	int_to_value(-param),
		      (Variable) gamma, int_to_value(-(param*proc_size)),
		      (Variable) delta, VALUE_MONE,
		      TCST, 		int_to_value((param*psi0)-theta0));

	sc_add_egalite(new_system, contrainte_make(v));	

	/* if block distributed
	 * gamma_j == 0
	 */
	if (style_block_p(st))
	    sc_add_egalite(new_system,
	       contrainte_make(vect_new((Variable) gamma, VALUE_ONE)));

	/* if cyclic(1) distributed
	 * delta_j == 0
	 */
	if (style_cyclic_p(st) && (param==1))
	    sc_add_egalite(new_system,
		contrainte_make(vect_new((Variable) delta, VALUE_ONE)));
	    
    }
    sc_creer_base(new_system);
    return(new_system);
}

Psysteme entity_to_hpf_align_constraints(entity e)
{
    Psysteme p = load_entity_hpf_align_constraints(e);

    pips_assert("distributed variable", array_distributed_p(e));

    if (Psysteme_undefined_p(p))
    {
	p = hpfc_compute_align_constraints(e);
	store_entity_hpf_align_constraints(e, p);
    }

    return p;
}

Psysteme entity_to_hpf_distribute_constraints(entity e)
{
    Psysteme p = load_entity_hpf_distribute_constraints(e);

    pips_assert("template", entity_template_p(e));

    if (Psysteme_undefined_p(p))
    {
	p = hpfc_compute_distribute_constraints(e);
	store_entity_hpf_distribute_constraints(e, p);
    }

    return p;
}

/* effect entity_to_region(stat, ent, act)
 * statement stat; entity ent; tag act;
 *
 * gives the region of ent with action act in statement stat.
 */
effect entity_to_region(
    statement stat,
    entity ent,
    tag act)
{
    list l = load_statement_local_regions(stat);

    MAP(EFFECT, e,
	if ((reference_variable(effect_any_reference(e))==ent) &&
	    (action_tag(effect_action(e))==act)) return(e),
	l);

    return(effect_undefined);
}

/********************************************************* NEW DECLARATIONS */

Psysteme 
hpfc_compute_entity_to_new_declaration(
    entity array)
{
    int	dim = NumberOfDimension(array);
    Psysteme syst = sc_rn(NULL);

    pips_assert("distributed array", array_distributed_p(array));

    for (; dim>0; dim--)
    {
	 entity lalpha = get_ith_local_dummy(dim),
	        alpha = get_ith_array_dummy(dim);

	 switch (new_declaration_tag(array, dim))
	 {
	 case is_hpf_newdecl_none:
	     /* LALPHAi == ALPHAi
	      */
	     sc_add_egalite
		 (syst, 
		  contrainte_make(vect_make(VECTEUR_NUL,
					    alpha, 	VALUE_ONE,
					    lalpha, 	VALUE_MONE,
					    TCST,	VALUE_ZERO)));
	     break;
	 case is_hpf_newdecl_alpha:
	 {
	     /* LALPHAi = ALPHAi - ALPHAi_min + 1
	      */
	     int min = 314159;
	     int max = -314159;

	     get_ith_dim_new_declaration(array, dim, &min, &max);
	     
	     sc_add_egalite
		 (syst,
		  contrainte_make(vect_make(VECTEUR_NUL,
					    alpha, 	VALUE_ONE,
					    lalpha, 	VALUE_MONE,
					    TCST, 	int_to_value(1-min))));

	     break;
	 }
	 case is_hpf_newdecl_beta:
	 {
	     /* (|a|==1) LALPHA_i == DELTA_j + 1
	      * generalized to:
	      * (|a|!=1) |a| * (LALPHA_i - 1) + IOTA_j == DELTA_j
	      *           0 <= IOTA_j < |a|
	      */
	     entity delta, template = array_to_template(array);
	     int tdim, pdim, a, b, n;
	     
	     get_alignment(array, dim, &tdim, &a, &b);
	     pips_assert("aligned dimension", a!=0 && tdim!=0);
	     get_distribution(template, tdim, &pdim, &n);
	     
	     delta = get_ith_block_dummy(pdim);

	     if (abs(a)==1)
	     {
		 /* IOTA is not needed */
		 sc_add_egalite(syst,
		   contrainte_make(vect_make(VECTEUR_NUL,
					     delta, 	VALUE_ONE,
					     lalpha, 	VALUE_MONE,
					     TCST, 	VALUE_ONE)));
	     }
	     else
	     {
		 entity iota = get_ith_shift_dummy(pdim);
		 Pvecteur
		     v1 = vect_make(VECTEUR_NUL,
				    (Variable) lalpha, 	int_to_value(abs(a)),
				    (Variable) iota, 	VALUE_ONE,
				    (Variable) delta, 	VALUE_MONE,
				    TCST, 	int_to_value(-abs(a)));

		 sc_add_egalite(syst, contrainte_make(v1));
		 sc_add_inegalite(syst,
				  contrainte_make(vect_new((Variable) iota, 
							   VALUE_MONE)));
		 sc_add_inegalite
		     (syst,
		      contrainte_make(vect_make(VECTEUR_NUL,
			     (Variable) iota, VALUE_ONE,
			      TCST, 	int_to_value(-(abs(a)-1)))));
	     }

	     break;
	 }
	 case is_hpf_newdecl_gamma:
	 {
	     /* LALPHA_i == N* (GAMMA_j - GAMMA_0) + DELTA_j + 1
	      */
	     entity
		 gamma = entity_undefined,
		 delta = entity_undefined,
		 template = array_to_template(array),
		 processor = template_to_processors(template);
	     int gamma_0 = 0, tdim = -1, pdim = -1, a = 0, b = 0,
		 n, plow, pup, tlow, tup, alow, aup;
	     
	     get_alignment(array, dim, &tdim, &a, &b);
	     pips_assert("stride-1 aligned", abs(a)==1 && tdim!=0);
	     
	     get_distribution(template, tdim, &pdim, &n);
	     pips_assert("distributed dimension", pdim>0 && n>0);
	     
	     get_entity_dimensions(array, dim, &alow, &aup);
	     get_entity_dimensions(template, tdim, &tlow, &tup);
	     get_entity_dimensions(processor, pdim, &plow, &pup);

	     delta = get_ith_block_dummy(pdim);
	     gamma = get_ith_cycle_dummy(pdim);

	     gamma_0 = (a*alow + b - tlow) % (n * (pup - plow + 1));

	     sc_add_egalite
		 (syst,
		  contrainte_make(vect_make
		    (VECTEUR_NUL,
		     delta, 	VALUE_ONE,
		     gamma, 	int_to_value(n),
		     lalpha, 	VALUE_MONE,
		     TCST, 	int_to_value(1-(n*gamma_0)))));
	     break;
	 }
	 case is_hpf_newdecl_delta:
	 {
	     /* LALPHA_i = iceil(N,|a|) * (GAMMA_j - GAMMA_0) + SIGMA_j +1
	      * DELTA_j = |a|*SIGMA_j + IOTA_j
	      * 0 <= IOTA_j < |a|
	      */
	     entity sigma = entity_undefined,
		 iota = entity_undefined,
		 gamma = entity_undefined,
		 delta = entity_undefined,
		 template = array_to_template(array),
		 processor = template_to_processors(template);
	     int gamma_0 = 0, tdim = -1, pdim = -1, a = 0, b = 0,
		 n, icn, plow, pup, tlow, tup, alow, aup;
	     
	     get_alignment(array, dim, &tdim, &a, &b);
	     pips_assert("aligned dimension", tdim!=0);
	     
	     get_distribution(template, tdim, &pdim, &n);
	     pips_assert("distributed dimension", pdim>0 && n>0);
	     
	     get_entity_dimensions(array, dim, &alow, &aup);
	     get_entity_dimensions(template, tdim, &tlow, &tup);
	     get_entity_dimensions(processor, pdim, &plow, &pup);

	     sigma = get_ith_auxiliary_dummy(pdim);
	     iota = get_ith_shift_dummy(pdim);
	     delta = get_ith_block_dummy(pdim);
	     gamma = get_ith_cycle_dummy(pdim);

	     gamma_0 = (a*alow + b - tlow) % (n * (pup - plow + 1));
	     icn = iceil(n, abs(a));

	     sc_add_egalite
		 (syst,
		  contrainte_make
		  (vect_make(VECTEUR_NUL,
			     sigma, 	VALUE_ONE,
			     gamma, 	int_to_value(icn),
			     lalpha, 	VALUE_MONE,
			     TCST, 	int_to_value(1-(icn*gamma_0)))));

	     sc_add_egalite
		 (syst,
		  contrainte_make
		  (vect_make(VECTEUR_NUL,
			     delta,	VALUE_ONE,
			     sigma,	int_to_value(-abs(a)),
			     iota,	VALUE_MONE,
			     TCST,	VALUE_ZERO)));

	     sc_add_inegalite(syst,
			      contrainte_make(vect_new((Variable) iota, 
						       VALUE_MONE)));
	     sc_add_inegalite
		 (syst,
		  contrainte_make
		  (vect_make(VECTEUR_NUL,
			     (Variable) iota, VALUE_ONE,
			     TCST, 	int_to_value(-(abs(a)-1)))));
	     break;
	 }
	 default:
	     pips_internal_error("unexpected new declaration tag");
	 }
     }
    
    sc_creer_base(syst);
    return syst;
}

Psysteme entity_to_new_declaration(array)
entity array;
{
    Psysteme p = load_entity_new_declaration_constraints(array);

    pips_assert("distributed array", array_distributed_p(array));

    if (Psysteme_undefined_p(p))
    {
	p = hpfc_compute_entity_to_new_declaration(array);
	store_entity_new_declaration_constraints(array, p);
    }

    return p;
}

/*********************************** REGION and ARRAY link: PHIi == ALPHAi */

Psysteme 
generate_system_for_equal_variables(
    int n,
    entity (*gen1)(int),
    entity (*gen2)(int))
{
    Psysteme s = sc_rn(NULL);

    for(; n>0; n--)
	sc_add_egalite(s, contrainte_make
	     (vect_make(VECTEUR_NUL, gen1(n), VALUE_ONE, 
			gen2(n), VALUE_MONE, TCST, VALUE_ZERO)));

    sc_creer_base(s); return s;
}

Psysteme 
hpfc_unstutter_dummies(
    entity array)
{
    int ndim = variable_entity_dimension(array);

    return generate_system_for_equal_variables
	   (ndim, get_ith_region_dummy, get_ith_array_dummy);
}

/* Psysteme generate_system_for_variable(v)
 * entity v;
 *
 * what: generates a system for DISTRIBUTED variable v.
 * how: uses the declarations of v, t, p and align and distribute, 
 *      and new declarations.
 * input: entity (variable) v
 * output: the built system, which is a new allocated system.
 * side effects:
 *  - uses many functions that build and store systems...
 * bugs or features:
 */
Psysteme 
generate_system_for_distributed_variable(
    entity v)
{
    Psysteme result = sc_rn(NULL);
    entity t, p;

    pips_assert("distributed array", array_distributed_p(v));

    t = align_template(load_hpf_alignment(v)),
    p = distribute_processors(load_hpf_distribution(t));    

    result = sc_append(result, entity_to_declaration_constraints(v, 0));
    result = sc_append(result, entity_to_declaration_constraints(t, 1));
    result = sc_append(result, entity_to_declaration_constraints(p, 2));
    result = sc_append(result, entity_to_hpf_align_constraints(v));
    result = sc_append(result, entity_to_hpf_distribute_constraints(t));
    result = sc_append(result, entity_to_new_declaration(v));
 
    base_rm(sc_base(result)), sc_base(result) = NULL, sc_creer_base(result);

    return(result);
}

/* that is all
 */
