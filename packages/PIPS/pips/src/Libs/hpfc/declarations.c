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
/* HPFC module by Fabien COELHO
 *
 * new declarations compilation.
 * normalization of HPF declarations.
 */
 
#include "defines-local.h"

#include "effects-generic.h"
#include "effects-simple.h"
#include "effects-convex.h"

bool expression_constant_p(expression); /* in static_controlize */

/************************************************** HPF OBJECTS MANAGEMENT */

/* DISTRIBUTED ARRAYS
 */

#define STATIC_LIST_OF_HPF_OBJECTS(name, set_name, pred_name)\
static list name = NIL;\
int number_of_##name(){return(gen_length(name));}\
list list_of_##name(){return(name);}\
bool pred_name(entity e){return(gen_in_list_p(e, name));}\
void set_name(entity e){ if (!gen_in_list_p(e, name)){\
 name = CONS(ENTITY, e, name); normalize_hpf_object(e);}}

STATIC_LIST_OF_HPF_OBJECTS(distributed_arrays, set_array_as_distributed, 
			   array_distributed_p)

bool declaration_delayed_p(e)
entity e;
{
    bool
	distributed = 
	    (array_distributed_p(e) ||
	     array_distributed_p(load_old_host(e)) ||
	     array_distributed_p(load_old_node(e))),
	in_common = entity_in_common_p(e);

    return(distributed && in_common);
}

/* returns the list of entities that are 'local' to module
 */
list list_of_distributed_arrays_for_module(module)
entity module;
{
    list l = NIL;

    MAP(ENTITY, e,
    {
	if (hpfc_main_entity(e)==module) l = CONS(ENTITY, e, l);
    },
	list_of_distributed_arrays());

    return(l);
}

/************************************************** TEMPLATES and PROCESSORS */

STATIC_LIST_OF_HPF_OBJECTS(templates, set_template, entity_template_p)
STATIC_LIST_OF_HPF_OBJECTS(processors, set_processor, entity_processor_p)

void reset_hpf_object_lists()
{
    distributed_arrays = NIL,
    templates = NIL,
    processors = NIL;
}

void free_hpf_object_lists()
{
    gen_free_list(distributed_arrays),
    gen_free_list(templates),
    gen_free_list(processors);

    reset_hpf_object_lists();
}

/****************************************************  HPF NUMBER MANAGEMENT */

GENERIC_GLOBAL_FUNCTION(hpf_number, entity_int)

static int
  current_array_index = 1,
  current_template_index = 1,
  current_processors_index = 1;

static void init_currents()
{
    current_array_index = 1,
    current_template_index = 1,
    current_processors_index = 1;
}

/* STANDARS STATIC MANAGEMENT
 *
 * functions: {init,get,set,reset,close}_hpf_number_status
 */

void init_hpf_number_status()
{
    init_hpf_number();
    init_currents();
}

numbers_status get_hpf_number_status()
{
    return(make_numbers_status(get_hpf_number(),
			       current_array_index,
			       current_template_index,
			       current_processors_index));
}

void reset_hpf_number_status()
{
    reset_hpf_number();
    init_currents();
}

void set_hpf_number_status(s)
numbers_status s;
{
    set_hpf_number(numbers_status_numbermap(s));
    current_array_index = numbers_status_arrays(s);
    current_template_index = numbers_status_templates(s);
    current_processors_index = numbers_status_processors(s);
}

void close_hpf_number_status()
{
    close_hpf_number();
    init_currents();
}

/* give to hpf objects listed in distributedarrays, templates and processors
 * their number for the code generation...
 */
void GiveToHpfObjectsTheirNumber()
{
    pips_debug(7, "Here I am!\n");

    MAP(ENTITY, e,
    {
	if (!bound_hpf_number_p(e))
	    store_hpf_number(e, current_array_index++);
    },
	distributed_arrays);

    MAP(ENTITY, e,
    {
	if (!bound_hpf_number_p(e))
	    store_hpf_number(e, current_template_index++);
    },
	templates);

    MAP(ENTITY, e,
    {
	if (!bound_hpf_number_p(e))
	    store_hpf_number(e, current_processors_index++);
    },
	processors);
}

/*   returns the hpf_number parameter as a string
 *   not really needed ???
 *   ??? never called
 */
expression entity_hpf_number(e)
entity e;
{
    storage
	s = entity_storage(e);
    bool
	in_common = entity_in_common_p(e),
	in_ram = storage_ram_p(s);
    ram 
	r = (in_ram ? storage_ram(s) : ram_undefined);
	const char* suffix = entity_local_name(e),
	*prefix = 
	    (in_ram ? 
	     entity_local_name(in_common ? ram_section(r) : ram_function(r)) :
	     "DEFAULT");

    pips_assert("ram variable", entity_variable_p(e) && in_ram);

    return(MakeCharacterConstantExpression
	   (strdup(concatenate("n_", prefix, "_", suffix, NULL))));
			   
}

/****************************************************** ALIGN and DISTRIBUTE */

GENERIC_GLOBAL_FUNCTION(hpf_alignment, alignmap)
GENERIC_GLOBAL_FUNCTION(hpf_distribution, distributemap)

/********************************************************** NEW DECLARATIONS */

GENERIC_LOCAL_FUNCTION(new_declaration, newdeclmap)

tag 
new_declaration_tag(
    entity array,
    int dim)
{
    tag t;

    pips_assert("valid dimension and distributed array",
		dim>0 && dim<=7 && array_distributed_p(array));

    t = hpf_newdecl_tag
	(HPF_NEWDECL(gen_nth(dim-1, 
	 hpf_newdecls_dimensions(load_new_declaration(array)))));

    pips_debug(1, "%s[%d]: %d\n", entity_name(array), dim, t);

    return(t);
}

static void 
create_new_declaration(
    entity e)
{
    type t = entity_type(e);
    list l = NIL;
    int ndim;

    pips_assert("variable", type_variable_p(t));
    
    ndim = gen_length(variable_dimensions(type_variable(t)));

    for(; ndim>0; ndim--)
	l = CONS(HPF_NEWDECL, make_hpf_newdecl(is_hpf_newdecl_none, UU), l);

    store_new_declaration(e, make_hpf_newdecls(l));
}

static void 
store_a_new_declaration(array, dim, what)
entity array;
int dim;
tag what;
{
    hpf_newdecl n;

    pips_assert("valid dimension and distributed array",
		dim>0 && dim<=7 && array_distributed_p(array));

    if (!bound_new_declaration_p(array))
	create_new_declaration(array);

    n = HPF_NEWDECL(gen_nth(dim-1, 
	    hpf_newdecls_dimensions(load_new_declaration(array))));

    hpf_newdecl_tag(n) = what;
}

void get_ith_dim_new_declaration(array, i, pmin, pmax)
entity array;
int i, *pmin, *pmax;
{
    dimension d = entity_ith_dimension(load_new_node(array), i);

    pips_assert("distributed array",
		array_distributed_p(array) && entity_variable_p(array));

    *pmin = HpfcExpressionToInt(dimension_lower(d));
    *pmax = HpfcExpressionToInt(dimension_upper(d));
}

/************************************* DATA STATUS INTERFACE FOR HPFC STATUS */

void init_data_status()
{
    init_new_declaration();
    init_hpf_alignment();
    init_hpf_distribution();
    reset_hpf_object_lists();
}

data_status get_data_status()
{
    /* ??? previsous data_status lost: memory leak
     */
    return(make_data_status(get_new_declaration(),
			    get_hpf_alignment(), 
			    get_hpf_distribution(),			    
			    list_of_distributed_arrays(),
			    list_of_templates(),
			    list_of_processors()));
}

void reset_data_status()
{
    reset_new_declaration();
    reset_hpf_alignment();
    reset_hpf_distribution();
    reset_hpf_object_lists();
}

void set_data_status(s)
data_status s;
{
    set_new_declaration(data_status_newdeclmap(s));
    set_hpf_alignment(data_status_alignmap(s));
    set_hpf_distribution(data_status_distributemap(s));
    distributed_arrays = data_status_arrays(s);
    templates = data_status_templates(s);
    processors = data_status_processors(s);
}

void close_data_status()
{
    close_new_declaration();
    close_hpf_alignment();
    close_hpf_distribution();
    free_hpf_object_lists();
}


/********************************************************* Normalizations */

/* NormalizeOneTemplateDistribution
 */
static void NormalizeOneTemplateDistribution(d,templ,templdimp,procs,procsdimp)
distribution d;
entity templ,procs;
int *templdimp, *procsdimp;
{
    if (style_none_p(distribution_style(d)))
	(*templdimp)++;
    else
    {
	int szoftempldim = SizeOfIthDimension(templ,(*templdimp)),
	    szofprocsdim = SizeOfIthDimension(procs,(*procsdimp));
	
	if (distribution_parameter(d)==expression_undefined)
	{
	    /* compute the missing value, in case of BLOCK distribution
	     */
	    
	    switch(style_tag(distribution_style(d)))
	    {
	    case is_style_block:
		distribution_parameter(d)=
		    int_to_expression(iceil(szoftempldim, szofprocsdim));
		break;
	    default:
		pips_internal_error("undefined style tag");
		break;
	    }
	}
	else
	{
	    /* check the given value
	     */
	    
	    int paramvalue = HpfcExpressionToInt(distribution_parameter(d));
	    
	    switch(style_tag(distribution_style(d)))
	    {
	    case is_style_block:
	    {
		int minvalue = iceil(szoftempldim, szofprocsdim);
		
		if (paramvalue<minvalue) 
		    pips_user_error("block too small in %s distribution\n",
				    entity_name(templ));
		break;
	    }
	    default:
		break;
	    }
	}
	
	(*templdimp)++;
	(*procsdimp)++;
    }
}

void normalize_distribute(t, d)
entity t;
distribute d;
{
    entity p = distribute_processors(d);
    list /* of distribution */ ld = distribute_distribution(d);
    int tdim = 1, pdim = 1;

    normalize_first_expressions_of(d);
    
    MAP(DISTRIBUTION, di,
	NormalizeOneTemplateDistribution(di, t, &tdim, p, &pdim), ld);

    if ((pdim-1)!=NumberOfDimension(p))
	pips_user_error("%s not enough distributions\n", entity_name(t));
		   
}

void normalize_align(e, a)
entity e;
align a;
{
    normalize_first_expressions_of(a);
}

void normalize_hpf_object(v)
entity v;
{
    normalize_first_expressions_of(entity_type(v));
}

void NormalizeHpfDeclarations()
{
    GiveToHpfObjectsTheirNumber();
    ifdebug(8){print_hpf_dir();}
}

/********************************************************* NEW DECLARATIONS */

/*  local macros...
 */
#define normalized_dimension_p(dim) \
  (HpfcExpressionToInt(dimension_lower(dim))==1)

/* here the new size of the ith dimension of the given array is computed.
 * because the declarations are static, there is a majoration of the space
 * required on each processors to held his part of the distributed array.
 */
static int 
ComputeNewSizeOfIthDimension(
    dimension dim,
    int i,
    entity array,
    tag *newdeclp)
{
    align a = load_hpf_alignment(array);
    entity t = align_template(a);
    distribute d = load_hpf_distribution(t);
    alignment al = alignment_undefined;
    distribution di = distribution_undefined;
    int rate, param, pdim = 1, asize = 0;
    style st;

    asize=dimension_size(dim);

    pips_debug(9, "dimension %d of array %s\n", i, entity_name(array));
    ifdebug(9)
    {
	print_align(a);
	print_distribute(d);
    }

    /* default: the new declaration is the same as the old one.
     */
    (*newdeclp) = is_hpf_newdecl_none;

    /* looking for the matching alignment...
     */
    al = FindAlignmentOfDim(align_alignment(a), i);

    /* no alignment => scratching of the dimension...
     */
    if (al==alignment_undefined) 
    {
	if (!normalized_dimension_p(dim)) 
	    (*newdeclp) = is_hpf_newdecl_alpha;
	return(asize);
    }

    /* there is an alignment, but the rate is zero, so the whole
     * dimension has to be declared on every processors, despite the
     * fact that the dimension is mapped on only one element.
     */
    rate=HpfcExpressionToInt(alignment_rate(al));
    if (rate==0) 
    {
	if (!normalized_dimension_p(dim)) 
	    (*newdeclp) = is_hpf_newdecl_alpha;
	return asize;
    }
    
    /* looking for the matching distribution...
     * pdim is the corresponding dimension of  processors p
     */
    di = FindDistributionOfDim(distribute_distribution(d),
			       alignment_templatedim(al),
			       &pdim);
			       
    st=distribution_style(di);

    /* no style => scratching of the dimension...
     */
    if (style_none_p(st)) 
    {
	/* ???
	 * should delete the alignment which is not usefull...
	 */
	/* alpha case
	 */
	if (!normalized_dimension_p(dim)) 
	    (*newdeclp) = is_hpf_newdecl_alpha;
	return asize;
    }
    
    /* and now, let's look at the different cases.
     *
     * beta case
     */
    param=HpfcExpressionToInt(distribution_parameter(di));

    if (style_block_p(st))
    {
	int
	    major = iceil(param, abs(rate)),
	    choice = min(asize, major);

	if (choice==asize)
	{
	    if (normalized_dimension_p(dim)) 
		(*newdeclp) = is_hpf_newdecl_none;
	    else
		(*newdeclp) = is_hpf_newdecl_alpha;
	}
	else
	    (*newdeclp) = is_hpf_newdecl_beta;

	return choice;
    }

    /* gamma case
     *
     * ??? what about rate==-1 ?
     */
    if (style_cyclic_p(st) && (rate==1))
    {
	int
	    psize = SizeOfIthDimension(distribute_processors(d), pdim),
	    major = param * iceil(asize+param-1, param*psize),
	    choice = min(asize, major);

	if (choice==asize)
	{
	    if (normalized_dimension_p(dim)) 
		(*newdeclp) = is_hpf_newdecl_none;
	    else
		(*newdeclp) = is_hpf_newdecl_alpha;
	}
	else
	    (*newdeclp) = is_hpf_newdecl_gamma;

	return choice;
    }

    /* delta case
     */
    if (style_cyclic_p(st))
    {
	int
	    absrate = abs(rate),
	    psize = SizeOfIthDimension(distribute_processors(d),pdim),
	    major = (iceil(param, absrate)*
		     iceil(absrate*(asize-1)+param, param*psize)),
	    choice = min(asize, major);

	if (choice==asize)
	{
	    if (normalized_dimension_p(dim)) 
		(*newdeclp) = is_hpf_newdecl_none;
	    else
		(*newdeclp) = is_hpf_newdecl_alpha;
	}
	else
	    (*newdeclp) = is_hpf_newdecl_delta;

	return choice;
    }
	
    /* alpha case, if nothing matches, what shouldn't be the case :
     */
    if (!normalized_dimension_p(dim)) 
	(*newdeclp) = is_hpf_newdecl_alpha;
    return asize;
}


/* for node this are reformated, and for host these variables are
 * deleted.
 */
static void 
NewDeclarationOfDistributedArray(
    entity array)
{
    entity newarray;
    int ithdim = 1, newsz, p;
    tag newdecl;
    list ld = NIL;

    /* it may happen that no newarray is available, 
     * when a module with no distributed variables is considered...
     */
    if (!bound_new_node_p(array)) return;
    newarray = load_new_node(array);
    pips_assert("distributed array",
		array_distributed_p(array) && entity_variable_p(array));

    pips_debug(6, "considering array %s, new %s\n",
	       entity_name(array), entity_name(newarray));

    /* compute the new size for every dimension on the array,
     * then update the dimensions of the newarray. remember
     * that the dimensions are shared between the old and new arrays.
     */
    MAP(DIMENSION, dim,
    {
	if (ith_dim_distributed_p(array, ithdim, &p))
	{
	    newsz = ComputeNewSizeOfIthDimension(dim, ithdim, array, &newdecl);
	    
	    pips_debug(8, "dimension %d new size: %d\n", ithdim, newsz);
	     
	    ld = gen_nconc(ld,
			   CONS(DIMENSION, make_dimension(int_to_expression(1),
						      int_to_expression(newsz)),
				NIL));
	    
	}
	else
	{
	    pips_debug(8, "dimension %d isn't touched\n", ithdim);
	    
	    newdecl = is_hpf_newdecl_none;
	    ld = gen_nconc(ld, CONS(DIMENSION, copy_dimension(dim), NIL));
	}
	
	store_a_new_declaration(array, ithdim, newdecl);
	
	ithdim++;
    },
	variable_dimensions(type_variable(entity_type(array))));
    
    variable_dimensions(type_variable(entity_type(newarray)))=ld;
}

/* this procedure generate the new declarations of every distributed arrays
 * of the program, in order to minimize the amount of memory used.
 * The new declarations have to be suitable for the new index computation
 * which is to be done dynamically...
 */
void NewDeclarationsOfDistributedArrays()
{
    MAP(ENTITY, array,
    {
	if (!bound_new_declaration_p(array))
	    NewDeclarationOfDistributedArray(array);
	else
	    pips_debug(3, "skipping array %s\n", entity_name(array));
    },
	list_of_distributed_arrays());
}

/**************************************************************** OVERLAPS */

GENERIC_GLOBAL_FUNCTION(overlap_status, overlapsmap)

static void 
create_overlaps(entity e)
{
    type t = entity_type(e);
    list o=NIL;
    int n;

    pips_assert("variable", type_variable_p(t));

    n = gen_length(variable_dimensions(type_variable(t)));
    for(; n>=1; n--) o = CONS(OVERLAP, make_overlap(0, 0), o);

    store_overlap_status(e, o);

    pips_assert("overlap stored", bound_overlap_status_p(e));
}

/* set the overlap value for entity ent, on dimension dim,
 * dans side side to width, which must be a positive integer.
 * if necessary, the overlap is updates with the value width.
 */
void set_overlap(ent, dim, side, width)
entity ent;
int dim, side, width;
{
    overlap o;
    int current;

    pips_assert("valid dimension", dim>0);

    pips_debug(10, "%s:(DIM=%d) %s%d\n", 
	       entity_name(ent), dim, side?"+":"-", width);

    if (!bound_overlap_status_p(ent)) create_overlaps(ent);
    o = OVERLAP(gen_nth(dim-1, load_overlap_status(ent)));

    if (side) /* upper */
    {
	current = overlap_upper(o);
	if (current<width) overlap_upper(o)=width;
    }
    else /* lower */
    {
	current = overlap_lower(o);
	if (current<width) overlap_lower(o)=width;
    }
}

/* returns the overlap for a given entity, dimension and side,
 * to be used in the declaration modifications
 */
int get_overlap(ent, dim, side)
entity ent;
int dim, side;
{
    overlap o;

    pips_assert("valid dimension", dim>0);
    pips_debug(10, "%s (DIM=%d) %s\n", entity_name(ent), dim, side?"+":"-");

    if (!bound_overlap_status_p(ent)) create_overlaps(ent);
    pips_assert("overlap ok", bound_overlap_status_p(ent));

    o = OVERLAP(gen_nth(dim-1, load_overlap_status(ent)));
    return(side ? overlap_upper(o) : overlap_lower(o));
}

/* redefines the bound given the overlap which is to be included
 */
static void overlap_redefine_expression(pexpr, ov)
expression *pexpr;
int ov;
{
    expression
	copy = *pexpr;

    if (expression_constant_p(*pexpr))
    {
	*pexpr = int_to_expression(HpfcExpressionToInt(*pexpr)+ov);
	free_expression(copy); /* this avoid a memory leak */
    }
    else
	*pexpr = MakeBinaryCall(FindOrCreateEntity(TOP_LEVEL_MODULE_NAME, 
						   PLUS_OPERATOR_NAME),
				*pexpr,
				int_to_expression(ov));
}

static void declaration_with_overlaps(l)
list l;
{
    entity ent;
    int ndim, i, lower_overlap, upper_overlap;
    dimension the_dim;

    MAP(ENTITY, oldent,
     {
	 ent = load_new_node(oldent);
	 ndim = variable_entity_dimension(ent);

	 pips_assert("variable", type_variable_p(entity_type(ent)));

	 if (storage_formal_p(entity_storage(ent)))
	 {
	     /* arguments are passed the declarations from outside
	      */
	     for (i=1 ; i<=ndim ; i++)
	     {
		 if (ith_dim_overlapable_p(oldent, i))
		 {
		     the_dim = entity_ith_dimension(ent, i); 
		     dimension_lower(the_dim) = /* ??? memory leak */
			hpfc_array_bound(ent, false, i);
		     dimension_upper(the_dim) = 
			hpfc_array_bound(ent, true, i);
		 }
	     }
	 }
	 else
	 {
	     for (i=1 ; i<=ndim ; i++)
	     {
		 the_dim = entity_ith_dimension(ent, i);
		 lower_overlap = get_overlap(oldent, i, 0);
		 upper_overlap = get_overlap(oldent, i, 1);
		 
		 pips_debug(8, "%s(DIM=%d): -%d, +%d\n", 
			    entity_name(ent), i, lower_overlap, upper_overlap);
		 
		 if (lower_overlap!=0) 
		     overlap_redefine_expression(&dimension_lower(the_dim),
						 -lower_overlap);
		 
		 if (upper_overlap!=0) 
		     overlap_redefine_expression(&dimension_upper(the_dim),
						 upper_overlap);
	     }
	 }
     },
	 l);
}

void 
declaration_with_overlaps_for_module(entity  module)
{
    list l = list_of_distributed_arrays_for_module(module);
    declaration_with_overlaps(l);
    gen_free_list(l);
}

static void 
update_overlaps_of(
    entity u /* distributed variable in the caller */, 
    entity v /* formal parameter in the callee */)
{
    int ndim = NumberOfDimension(v);

    pips_assert("conformance", ndim==NumberOfDimension(u));

    pips_debug(7, "%s from %s\n", entity_name(u), entity_name(v));

    for(; ndim>0; ndim--)
    {
	set_overlap(u, ndim, 0, get_overlap(v, ndim, 0));
	set_overlap(u, ndim, 1, get_overlap(v, ndim, 1));
    }
}

/* the overlaps of the actual parameters are updated according
 * to the formal requirements. 
 */
void 
update_overlaps_in_caller(
    entity fun,                 /* the function */
    list /* of expression */ le /* call arguments in the initial code */)
{
    int len = gen_length(le), i;
    for (i=1; i<=len; i++, POP(le))
    {
	entity v = find_ith_parameter(fun, i);
	if (array_distributed_p(v))
	{
	    expression e = EXPRESSION(CAR(le));
	    entity u = expression_to_entity(e), nu;
	    pips_assert("bounded to a new var", bound_new_node_p(u));
	    nu = load_new_node(u);
	    pips_debug(5, "call to %s, %s (%s) -> %s\n", 
		       entity_name(fun), entity_name(u), 
		       entity_name(nu), entity_name(v));
	    update_overlaps_of(u, v);
	}

    }
}

/*   That is all
 */
