/*
 * build-system.c
 *
 * here should be build equations and inequations to deal
 * with the I/O in hpf programs.
 *
 * Fabien COELHO, Feb/Mar 94
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

#include "types.h"
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
 * PHI{1-7}: array dimensions, 
 * THETA{1-7}: template dimensions,
 * PSI{1-7}: processor dimensions,
 * GAMMA{1-7}: cycles,
 * DELTA{1-7}: local offsets,
 * LPHI{1-7}: local array dimensions, if specified...
 */

/*
 * already computed constraints
 */
#ifndef Psysteme_undefined
#define Psysteme_undefined SC_UNDEFINED
#define Psysteme_undefined_p(sc) SC_UNDEFINED_P(sc)
#endif
GENERIC_CURRENT_MAPPING(declaration_constraints, Psysteme, entity);
GENERIC_CURRENT_MAPPING(hpf_constraints, Psysteme, entity);

void reset_hpfc_systems()
{
    reset_declaration_constraints_map();
    reset_hpf_constraints_map();
}

entity get_ith_dummy(prefix, suffix, i)
string prefix, suffix;
int i;
{
    char buffer[100];
    
    pips_assert("get_ith_dummy", (i>=1) && (i<=7));
    (void) sprintf(buffer, "%s%d", suffix, i);
    return(find_or_create_scalar_entity(buffer, prefix, is_basic_int));
}

/*
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
    
    pips_assert("compute_entity_to_declaration_constraints",
		entity_variable_p(ent));
    pips_assert("compute_entity_to_declaration_constraints",
		gen_length(dims)!=0);
    
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

	 pips_assert("compute_entity_to_declaration_constraints", 
		     blower && bupper);
	 
	 /*
	  * now the dummy is to be used to generate two inequalities: 
	  * -dummy + lower <= 0 and dummy - upper <= 0
	  */
	 
	 sc_add_inegalite(new_system,
			  contrainte_make(vect_add(vect_new(TCST, ilower),
						   vect_new(dummy, -1))));
	 sc_add_inegalite(new_system,
			  contrainte_make(vect_add(vect_new(TCST, -iupper),
						   vect_new(dummy, 1))));
	 sc_creer_base(new_system);
	 
	 dim_number++;
     },
	 dims);
    
    return(new_system);
}

static Psysteme hpfc_compute_entity_to_declaration_constraints(e)
entity e;
{
    bool
	is_array = array_distributed_p(e),
	is_template = entity_template_p(e),
	is_processor = entity_processor_p(e);
    string
	local_prefix = (is_array ? PHI_PREFIX :
			is_template ? THETA_PREFIX :
			is_processor ? PSI_PREFIX : "ERROR");

    pips_assert("hpfc_compute_entity_to_declaration_constraints",
		(is_array || is_template || is_processor));

    return(compute_entity_to_declaration_constraints
	   (e, HPFC_PACKAGE, local_prefix));
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

    pips_assert("entity_to_declaration_constraints", entity_variable_p(e));
    pips_assert("entity_to_declaration_constraints",
		array_distributed_p(e) ||
		entity_template_p(e) ||
		entity_processor_p(e));

    if (Psysteme_undefined_p(p))
    {
	p = hpfc_compute_entity_to_declaration_constraints(e);
	store_entity_declaration_constraints(e, p);
    }

    return(p);
}

/*
 * HPF CONSTRAINTS GENERATION
 */

entity get_ith_array_dummy(i)
int i;
{
    return(get_ith_dummy(HPFC_PACKAGE, PHI_PREFIX, i));
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
    return(get_ith_dummy(HPFC_PACKAGE, LPHI_PREFIX, i));
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
	al = (align) GET_ENTITY_MAPPING(hpfalign, e);
    entity
	template = align_template(al);
    Psysteme
	new_system = sc_new();
    int i;

    pips_assert("hpfc_compute_align_constraints", 
		array_distributed_p(e));

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
		v = vect_add(vect_new(TCST, -constant),
			     vect_new(theta, 1));
					       
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

		sc_add_egalite(new_system,
		     contrainte_make(vect_add(v, vect_new(phi, -rate))));
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
 * phi_i - lower_template_i == 0
 */
Psysteme hpfc_compute_unicity_constraints(e)
entity e;
{
    align
	al = (align) GET_ENTITY_MAPPING(hpfalign, e);
    entity
	template = align_template(al);
    Psysteme
	new_system = sc_new();
    int i;

    pips_assert("hpfc_compute_unicity_constraints", 
		array_distributed_p(e));

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
			   contrainte_make(vect_add(vect_new(theta, 1),
						    vect_new(TCST, -low))));
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
 */
static Psysteme hpfc_compute_distribute_constraints(e)
entity e;
{
    Psysteme
	new_system = sc_new();
    distribute
	di = (distribute) GET_ENTITY_MAPPING(hpfdistribute, e);
    entity
	proc = distribute_processors(di);
    list
	ld = distribute_distribution(di);
    int j, i;
    
    pips_assert("hpfc_compute_distribute_constraints",
		entity_template_p(e));

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
	bool
	    is_block = style_block_p(distribution_style(d));
	Pvecteur
	    v = VECTEUR_NUL;

	/*
	 * -delta_j <= 0
	 */
	sc_add_inegalite(new_system, 
			 contrainte_make(vect_new(delta, -1)));

	/*
	 * delta_j - (N_j - 1) <= 0
	 */
	sc_add_inegalite(new_system,
			 contrainte_make(vect_add(vect_new(delta, 1),
						  vect_new(TCST, -param+1))));

	/*
	 * theta_i 
	 * - Nj psi_j
	 * - Nj Pj gamma_j
	 * - delta_j
	 * + Nj psi_j0 - theta_i0
	 * == 0
	 */
	v = vect_new(theta, 1);
	v = vect_add(v, vect_new(psi, -param));
	v = vect_add(v, vect_new(gamma, -(param*proc_size)));
	v = vect_add(v, vect_new(delta, -1));
	v = vect_add(v, vect_new(TCST, param*psi0-theta0));
	
	sc_add_egalite(new_system, contrainte_make(v));	

	/*
	 * if block distributed
	 * gamma_j == 0
	 */
	if (is_block)
	    sc_add_egalite(new_system,
			   contrainte_make(vect_new(gamma, 1)));
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

    pips_assert("entity_to_hpf_constraints", entity_variable_p(e));
    pips_assert("entity_to_hpf_constraints",
		array_distributed_p(e) || entity_template_p(e));

    if (Psysteme_undefined_p(p))
    {
	p = hpfc_compute_entity_to_hpf_constraints(e);
	store_entity_hpf_constraints(e, p);
    }

    return(p);
}

