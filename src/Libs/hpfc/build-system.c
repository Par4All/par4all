/*
 * build-system.c
 *
 * here should be build equations and inequations to deal
 * with the I/O in hpf programs.
 *
 * Fabien COELHO, Feb/Mar 94
 *
 * SCCS Stuff:
 * $RCSfile: build-system.c,v $ ($Date: 1994/12/27 19:46:41 $, ) version $Revision$,
 * got on %D%, %T%
 * $Id$
 */

/*
 * Standard includes
 */
 
#include <stdio.h>
#include <string.h> 
extern fprintf();

/*
 * Psystems stuff
 */

#include "boolean.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"

/*
 * Newgen stuff
 */

#include "genC.h"

#include "ri.h" 
#include "hpf.h" 
#include "hpf_private.h"

/*
 * PIPS stuff
 */

#include "ri-util.h" 
#include "misc.h" 
#include "regions.h"
#include "semantics.h"
#include "effects.h"

/* 
 * my own local includes
 */

#include "hpfc.h"
#include "defines-local.h"

/*
 * Variables
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

/*
 * variable names:
 *
 * ALPHA{1-7}: array dimensions, 
 * THETA{1-7}: template dimensions,
 * PSI{1-7}: processor dimensions,
 * SIGMA{1-7}: auxiliary variable,
 * GAMMA{1-7}: cycles,
 * DELTA{1-7}: local offsets,
 * LALPHA{1-7}: local array dimensions, if specified...
 */

/*
 * already computed constraints
 */
#ifndef Psysteme_undefined
#define Psysteme_undefined SC_UNDEFINED
#define Psysteme_undefined_p(sc) SC_UNDEFINED_P(sc)
#endif
GENERIC_LOCAL_MAPPING(declaration_constraints, Psysteme, entity);
GENERIC_LOCAL_MAPPING(hpf_constraints, Psysteme, entity);
GENERIC_LOCAL_MAPPING(new_declaration_constraints, Psysteme, entity);

void make_hpfc_current_mappings()
{
    make_declaration_constraints_map();
    make_hpf_constraints_map();
    make_new_declaration_constraints_map();
}

void free_hpfc_current_mappings()
{
    free_declaration_constraints_map();
    free_hpf_constraints_map();
    free_new_declaration_constraints_map();
}

entity get_ith_dummy(prefix, suffix, i)
string prefix, suffix;
int i;
{
    char buffer[100];
    
    assert(i>=1 && i<=7);
    (void) sprintf(buffer, "%s%d", suffix, i);
    return(find_or_create_scalar_entity(buffer, prefix, is_basic_int));
}

/* ------------------------------------------------------------------
 *
 * DECLARATION CONSTRAINTS GENERATION
 */

/*
 * Psysteme compute_entity_to_constraints(ent, suffix, prefix)
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
    list 
	dims = variable_dimensions(type_variable(entity_type(ent)));
    int 
	dim_number = 1;
    Psysteme
	new_system = sc_new();
    
    assert(entity_variable_p(ent));
    
    /*
     * system may be empty for scalars ???
     */
    
    debug(5,"compute_entity_to_declaration_constraints",
	  "computing constraints for entity %s, prefix %s, suffix %s\n",
	  entity_name(ent), prefix, suffix);
    
    MAPL(cd,
     {
	 dimension
	     dim = DIMENSION(CAR(cd));
	 entity
	     dummy = get_ith_dummy(prefix, suffix, dim_number);
	 int ilower;
	 int iupper;
	 bool
	     blower = hpfc_integer_constant_expression_p
		 (dimension_lower(dim), &ilower);
	 bool
	     bupper = hpfc_integer_constant_expression_p
		 (dimension_upper(dim), &iupper);

	 assert(blower && bupper);
	 
	 /*
	  * now the dummy is to be used to generate two inequalities: 
	  * -dummy + lower <= 0 and dummy - upper <= 0
	  */
	 
	 sc_add_inegalite(new_system,
			  contrainte_make(vect_make(VECTEUR_NUL,
						    dummy, 	-1,
						    TCST, 	ilower)));
	 sc_add_inegalite(new_system,
			  contrainte_make(vect_make(VECTEUR_NUL,
						    dummy, 	1,
						    TCST, 	-iupper)));
	 dim_number++;
     },
	 dims);
    
    sc_creer_base(new_system);
    return(new_system);
}

static Psysteme hpfc_compute_entity_to_declaration_constraints(e)
entity e;
{
    bool
	is_darray = array_distributed_p(e),
	is_template = entity_template_p(e),
	is_processor = entity_processor_p(e),
	is_array = (!is_darray) && (!is_template) && 
	    (!is_processor) && entity_variable_p(e);
    string
	local_prefix = ((is_darray || is_array) ? ALPHA_PREFIX :
			is_template ? THETA_PREFIX :
			is_processor ? PSI_PREFIX : "ERROR");

    assert(is_darray || is_array || is_template || is_processor);

    return(compute_entity_to_declaration_constraints
	   (e, local_prefix, HPFC_PACKAGE));
}

/*
 * Psystem entity_to_declaration_constraints(entity e);
 *
 * gives back the constraints due to the declarations.
 * Uses a demand driven approach: computed systems are stored
 * in the declaration_constraints mapping for later search.
 */
Psysteme entity_to_declaration_constraints(e)
entity e;
{
    Psysteme 
	p = load_entity_declaration_constraints(e);

    assert(entity_variable_p(e));

    if (Psysteme_undefined_p(p))
    {
	p = hpfc_compute_entity_to_declaration_constraints(e);
	store_entity_declaration_constraints(e, p);
    }

    return(p);
}

/* ------------------------------------------------------------------
 *
 * HPF CONSTRAINTS GENERATION
 */

entity get_ith_array_dummy(i)
int i;
{
    return(get_ith_dummy(HPFC_PACKAGE, ALPHA_PREFIX, i));
}

entity get_ith_region_dummy(i)
int i;
{
    return(get_ith_dummy(REGIONS_MODULE_NAME, PHI_PREFIX, i));
}

entity get_ith_template_dummy(i)
int i;
{
    return(get_ith_dummy(HPFC_PACKAGE, THETA_PREFIX, i));
}

entity get_ith_processor_dummy(i)
int i;
{
    return(get_ith_dummy(HPFC_PACKAGE, PSI_PREFIX, i));
}

entity get_ith_block_dummy(i)
int i;
{
    return(get_ith_dummy(HPFC_PACKAGE, DELTA_PREFIX, i));
}

entity get_ith_cycle_dummy(i)
int i;
{
    return(get_ith_dummy(HPFC_PACKAGE, GAMMA_PREFIX, i));
}

entity get_ith_local_dummy(i)
int i;
{
    return(get_ith_dummy(HPFC_PACKAGE, LALPHA_PREFIX, i));
}

entity get_ith_shift_dummy(i)
int i;
{
    return(get_ith_dummy(HPFC_PACKAGE, IOTA_PREFIX, i));
}

entity get_ith_auxiliary_dummy(i)
int i;
{
    return(get_ith_dummy(HPFC_PACKAGE, SIGMA_PREFIX, i));
}

bool entity_hpfc_dummy_p(e)
entity e;
{
    return(!strcmp(entity_module_name(e), HPFC_PACKAGE));
}

/*
 * Psysteme hpfc_compute_align_constraints(e)
 * entity e is an array
 *
 * compute the align equations:
 *
 * theta_i - a phi_j - b == 0
 */
static Psysteme hpfc_compute_align_constraints(e)
entity e;
{
    align
	al = load_entity_align(e);
    entity
	template = align_template(al);
    Psysteme
	new_system = sc_new();
    int i;

    assert(array_distributed_p(e));

    for(i=1 ; i<=NumberOfDimension(template) ; i++)
    {
	entity
	    theta = get_ith_template_dummy(i);
	alignment
	    a = FindAlignmentOfTemplateDim(align_alignment(al), i);
	
	if (a!=alignment_undefined)
	{
	    int 
		adim = alignment_arraydim(a),
		constant = HpfcExpressionToInt(alignment_constant(a));
	    Pvecteur
		v = vect_make(VECTEUR_NUL,
			      theta, 	1,
			      TCST, 	-constant);
					       
	    if (adim==0)
	    {
		sc_add_egalite(new_system, contrainte_make(v));
	    }
	    else
	    {
		entity
		    phi = get_ith_array_dummy(adim);
		int
		    rate = HpfcExpressionToInt(alignment_rate(a));

		v = vect_make(v, phi, -rate, TCST, 0);
		
		sc_add_egalite(new_system, contrainte_make(v));
	    }
	}
    }

    sc_creer_base(new_system);
    return(new_system);
}

/*
 * Psysteme hpfc_compute_unicity_constraints(e)
 * entity e should be an array;
 *
 * equations for non aligned template dimensions are computed:
 *
 * theta_i - lower_template_i == 0
 */
Psysteme hpfc_compute_unicity_constraints(e)
entity e;
{
    align
	al = load_entity_align(e);
    entity
	template = align_template(al);
    Psysteme
	new_system = sc_new();
    int i;

    assert(array_distributed_p(e));

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

	    sc_add_egalite(new_system,
			   contrainte_make(vect_make(VECTEUR_NUL,
						     theta, 	1,
						     TCST, 	-low)));
	}
    }
    sc_creer_base(new_system);
    return(new_system);
}

/*
 * Psysteme hpfc_compute_distribute_constraints(e)
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
    Psysteme
	new_system = sc_new();
    distribute
	di = load_entity_distribute(e);
    entity
	proc = distribute_processors(di);
    list
	ld = distribute_distribution(di);
    int j, i;
    
    assert(entity_template_p(e));

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
	style
	    st = distribution_style(d);
	Pvecteur
	    v = VECTEUR_UNDEFINED;

	/*
	 * -delta_j <= 0
	 */
	sc_add_inegalite(new_system, 
			 contrainte_make(vect_new((Variable) delta, -1)));

	/*
	 * delta_j - (N_j - 1) <= 0
	 */
	sc_add_inegalite(new_system,
			 contrainte_make(vect_make(VECTEUR_NUL,
						   (Variable) delta, 	1,
						   TCST, 	-param+1)));

	/*
	 * theta_i 
	 * - Nj psi_j
	 * - Nj Pj gamma_j
	 * - delta_j
	 * + Nj psi_j0 - theta_i0
	 * == 0
	 */
	v = vect_make(VECTEUR_NUL,
		      (Variable) theta, 1,
		      (Variable) psi, 	-param,
		      (Variable) gamma, -(param*proc_size),
		      (Variable) delta, -1,
		      TCST, 		param*psi0-theta0);

	sc_add_egalite(new_system, contrainte_make(v));	

	/*
	 * if block distributed
	 * gamma_j == 0
	 */
	if (style_block_p(st))
	    sc_add_egalite(new_system,
			   contrainte_make(vect_new((Variable) gamma, 1)));

	/*
	 * if cyclic(1) distributed
	 * delta_j == 0
	 */
	if (style_cyclic_p(st) && (param==1))
	    sc_add_egalite(new_system,
			   contrainte_make(vect_new((Variable) delta, 1)));
	    
    }
    sc_creer_base(new_system);
    return(new_system);
}

static Psysteme hpfc_compute_entity_to_hpf_constraints(e)
entity e;
{
    return(array_distributed_p(e) ?
	   hpfc_compute_align_constraints(e) :
	   hpfc_compute_distribute_constraints(e));
}

/*
 * entity_to_hpf_constraints(e)
 * entity e;
 *
 * demand driven computation of constraints. e may be an
 * array, then the alignment is computed, or a template,
 * for which the distribution are computed.
 */
Psysteme entity_to_hpf_constraints(e)
entity e;
{
    Psysteme 
	p = load_entity_hpf_constraints(e);

    assert(entity_variable_p(e) && 
	   (array_distributed_p(e) || entity_template_p(e)));

    if (Psysteme_undefined_p(p))
    {
	p = hpfc_compute_entity_to_hpf_constraints(e);
	store_entity_hpf_constraints(e, p);
    }

    return(p);
}

/*
 * effect entity_to_region(stat, ent, act)
 * statement stat;
 * entity ent; 
 * tag act;
 *
 * gives the region of ent with action act in statement stat.
 *
 */
effect entity_to_region(stat, ent, act)
statement stat;
entity ent; 
tag act;
{
    list
	l = load_statement_local_regions(stat);

    MAPL(ce,
     {
	 effect 
	     e = EFFECT(CAR(ce));

	 if ((reference_variable(effect_reference(e))==ent) &&
	     (action_tag(effect_action(e))==act))
	     return(e);
     },
	 l);

    return(effect_undefined);
}

/* ---------------------------------------------------
 *
 * NEW DECLARATIONS AS CONSTRAINTS IF POSSIBLE...
 *
 */

Psysteme hpfc_compute_entity_to_new_declaration(array)
entity array;
{
    int
	dim = NumberOfDimension(array);
    Psysteme
	syst = sc_rn(NULL);

    assert(array_distributed_p(array));

    for (; dim>0; dim--)
    {
	 entity
	     lalpha = get_ith_local_dummy(dim),
	     alpha = get_ith_array_dummy(dim);

	 switch (new_declaration(array, dim))
	 {
	 case  NO_NEW_DECLARATION:
	     /*
	      * LALPHAi == ALPHAi
	      */
	     sc_add_egalite(syst, 
			    contrainte_make(vect_make(VECTEUR_NUL,
						      alpha, 	1,
						      lalpha, 	-1,
						      TCST,	0)));
	     break;
	 case ALPHA_NEW_DECLARATION:
	 {
	     /*
	      * LALPHAi = ALPHAi - ALPHAi_min + 1
	      */
	     int min = 314159;
	     int max = -314159;

	     get_ith_dim_new_declaration(array, dim, &min, &max);
	     
	     sc_add_egalite(syst,
			    contrainte_make(vect_make(VECTEUR_NUL,
						      alpha, 	1,
						      lalpha, 	-1,
						      TCST, 	1-min)));

	     break;
	 }
	 case BETA_NEW_DECLARATION:
	 {
	     /* 
	      * (|a|==1) LALPHA_i == DELTA_j + 1
	      * generalized to:
	      * (|a|!=1) |a| * (LALPHA_i - 1) + IOTA_j == DELTA_j
	      *           0 <= IOTA_j < |a|
	      */
	     entity
		 delta = entity_undefined;
	     int 
		 tdim = -1,
		 a = 0,
		 b = 0;
	     
	     get_alignment(array, dim, &tdim, &a, &b);
	     	     
	     assert(a!=0 && tdim!=0);
	     
	     delta = get_ith_block_dummy(tdim);

	     if (abs(a)==1)
	     {
		 /* IOTA is not needed */
		 sc_add_egalite(syst,
				contrainte_make(vect_make(VECTEUR_NUL,
							  delta, 	1,
							  lalpha, 	-1,
							  TCST, 	1)));
	     }
	     else
	     {
		 entity
		     iota = get_ith_shift_dummy(tdim);
		 Pvecteur
		     v1 = vect_make(VECTEUR_NUL,
				    (Variable) lalpha, 	abs(a),
				    (Variable) iota, 	1,
				    (Variable) delta, 	-1,
				    TCST, 	-abs(a));

		 sc_add_egalite(syst, contrainte_make(v1));
		 
		 sc_add_inegalite(syst,
				  contrainte_make(vect_new((Variable) iota, 
						  -1)));
		 sc_add_inegalite
		     (syst,
		      contrainte_make(vect_make(VECTEUR_NUL,
						(Variable) iota, 1,
						TCST, 	-(abs(a)-1))));
	     }

	     break;
	 }
	 case GAMMA_NEW_DECLARATION:
	 {
	     /*
	      * LALPHA_i == N* (GAMMA_j - GAMMA_0) + DELTA_j + 1
	      */
	     entity
		 gamma = entity_undefined,
		 delta = entity_undefined,
		 template = array_to_template(array),
		 processor = template_to_processors(template);
	     int 
		 gamma_0 = 0,
		 tdim = -1,
		 pdim = -1,
		 a = 0,
		 b = 0,
		 n, plow, pup, tlow, tup, alow, aup;
	     
	     get_alignment(array, dim, &tdim, &a, &b);
	     assert(abs(a)==1 && tdim!=0);
	     
	     get_distribution(template, tdim, &pdim, &n);
	     assert(pdim>0 && n>0);
	     
	     get_entity_dimensions(array, dim, &alow, &aup);
	     get_entity_dimensions(template, tdim, &tlow, &tup);
	     get_entity_dimensions(processor, pdim, &plow, &pup);

	     delta = get_ith_block_dummy(tdim);
	     gamma = get_ith_cycle_dummy(tdim);

	     gamma_0 = (a*alow + b - tlow) % (n * (pup - plow + 1));

	     sc_add_egalite
		 (syst,
		  contrainte_make(vect_make(VECTEUR_NUL,
					    delta, 	1,
					    gamma, 	n,
					    lalpha, 	-1,
					    TCST, 	1-(n*gamma_0))));
	     break;
	 }
	 case DELTA_NEW_DECLARATION:
	 {
	     /*
	      * LALPHA_i = iceil(N,|a|) * (GAMMA_j - GAMMA_0) + SIGMA_j +1
	      * DELTA_j = |a|*SIGMA_j + IOTA_j
	      * 0 <= IOTA_j < |a|
	      */
	     entity
		 sigma = entity_undefined,
		 iota = entity_undefined,
		 gamma = entity_undefined,
		 delta = entity_undefined,
		 template = array_to_template(array),
		 processor = template_to_processors(template);
	     int 
		 gamma_0 = 0,
		 tdim = -1,
		 pdim = -1,
		 a = 0,
		 b = 0,
		 n, icn, plow, pup, tlow, tup, alow, aup;
	     
	     get_alignment(array, dim, &tdim, &a, &b);
	     assert(tdim!=0);
	     
	     get_distribution(template, tdim, &pdim, &n);
	     assert(pdim>0 && n>0);
	     
	     get_entity_dimensions(array, dim, &alow, &aup);
	     get_entity_dimensions(template, tdim, &tlow, &tup);
	     get_entity_dimensions(processor, pdim, &plow, &pup);

	     sigma = get_ith_auxiliary_dummy(tdim);
	     iota = get_ith_shift_dummy(tdim);
	     delta = get_ith_block_dummy(tdim);
	     gamma = get_ith_cycle_dummy(tdim);

	     gamma_0 = (a*alow + b - tlow) % (n * (pup - plow + 1));
	     icn = iceil(n, abs(a));

	     sc_add_egalite
		 (syst,
		  contrainte_make(vect_make(VECTEUR_NUL,
					    sigma, 	1,
					    gamma, 	icn,
					    lalpha, 	-1,
					    TCST, 	1-(icn*gamma_0))));

	     sc_add_egalite
		 (syst,
		  contrainte_make(vect_make(VECTEUR_NUL,
					    delta,	1,
					    sigma,	-abs(a),
					    iota,	-1,
					    TCST,	0)));

	     sc_add_inegalite(syst,
			      contrainte_make(vect_new((Variable) iota, 
						       -1)));
	     sc_add_inegalite
		 (syst,
		  contrainte_make(vect_make(VECTEUR_NUL,
					    (Variable) iota, 1,
					    TCST, 	-(abs(a)-1))));
	     break;
	 }
	 default:
	     pips_error("hpfc_compute_entity_to_new_declaration",
			"unexpected new declaration tag\n");
	 }
     }
    
    sc_creer_base(syst);
    return(syst);
}


Psysteme entity_to_new_declaration(array)
entity array;
{
    Psysteme 
	p = load_entity_new_declaration_constraints(array);

    assert(array_distributed_p(array));

    if (Psysteme_undefined_p(p))
    {
	p = hpfc_compute_entity_to_new_declaration(array);
	store_entity_new_declaration_constraints(array, p);
    }

    return(p);
}

/* ------------------------------------------------------------
 * 
 * PHIi == ALPHAi list
 *
 */

Psysteme hpfc_unstutter_dummies(array)
entity array;
{
    Psysteme
	new_syst = sc_rn(NULL);
    int 
	i = 1,
	ndim = variable_entity_dimension(array);

    for(;i<=ndim;i++)
    {
	sc_add_egalite
	    (new_syst,
	     contrainte_make(vect_make(VECTEUR_NUL,
				       get_ith_region_dummy(i), 1,
				       get_ith_array_dummy(i),  -1,
				       TCST, 			0)));
    }

    sc_creer_base(new_syst);
    return(new_syst);
}

/*
 * that's all
 */
