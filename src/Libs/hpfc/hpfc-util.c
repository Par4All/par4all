/*
 * this file describe a few functions usefull to the compiler
 * to manage the hpfc data structures.
 *
 * Fabien Coelho, May 1993.
 *
 * SCCS stuff:
 * $RCSfile: hpfc-util.c,v $ ($Date: 1994/12/23 16:30:48 $, ) version $Revision$,
 * got on %D%, %T%
 * $Id$
 */

#include <stdio.h>
#include <string.h>

extern int fprintf();

#include "genC.h"

#include "ri.h"
#include "hpf.h"
#include "hpf_private.h"

#include "ri-util.h"
#include "misc.h"
#include "effects.h"
#include "hpfc.h"
#include "defines-local.h"

/*
 * Predicates
 */

/* TRUE if there is a reference to a distributed array within obj
 *
 * ??? not very intelligent, should use the regions, the problem is
 * that I should normalize the code *before* the pips analysis...
 */
bool ref_to_dist_array_p(obj)
chunk* obj;
{
    list l = FindRefToDistArray(obj);
    bool b = (l!=NIL);

    gen_free_list(l); return(b);
}

/*
 * written_effects_to_dist_arrays_p
 */

bool written_effect_p(var, le)
entity var;
list le;
{
    effect e;

    MAPL(ce,
     {
	 e = EFFECT(CAR(ce));

	 if (reference_variable(effect_reference(e))==var &&
	     action_write_p(effect_action(e)))
	     return(TRUE);
     },
	 le);

    return(FALSE);
}

bool written_effects_to_dist_arrays_p(expr)
expression expr;
{
    list
	l,
	leffects_to_dist_arrays=DistArraysEffects(expr);

    for(l=leffects_to_dist_arrays;
	!ENDP(l);
	l=CDR(l))
	if  (action_write_p(effect_action(EFFECT(CAR(l)))))
	{
	    gen_free_list(leffects_to_dist_arrays);
	    return(TRUE);
	}

    gen_free_list(leffects_to_dist_arrays);
    return(FALSE);
}


/*
 * replicated_p
 *
 * check whether the distributed array e
 * is replicated or not.
 * if not sure, say yes.
 */
bool replicated_p(e)
entity e;
{
    int i;
    align a;
    list la, ld;
    entity template;
    distribute d;

    assert(array_distributed_p(e));

    a = load_entity_align(e);
    la = align_alignment(a);    
    template = align_template(a);

    d = load_entity_distribute(template);
    ld = distribute_distribution(d);

    for(i=1;
	i<=NumberOfDimension(template);
	i++, ld=CDR(ld))
	if (ith_dim_replicated_p(template, i, la, DISTRIBUTION(CAR(ld))))
	    return(TRUE);

    return(FALSE);
}

/*
 * ith_dim_replicated_p
 */
bool ith_dim_replicated_p(template, i, la, dist)
entity template;
int i;
list la;
distribution dist;
{
    alignment al=NULL;

    if (style_none_p(distribution_style(dist))) return(FALSE);

    /*
     * select the relevent alignment if exists.
     * could be some kind of gen_find_if()...
     */
    MAPL(ca,
     {
	 alignment ali=ALIGNMENT(CAR(ca));

	 if(alignment_templatedim(ali)==i) al=ali;
     },
	 la);

    return(al==NULL);
}

/*
 * bool ith_dim_distributed_p(array, i)
 *
 * whether a dimension is distributed or not.
 */
bool ith_dim_distributed_p(array, i, pprocdim)
entity array;
int i, *pprocdim;
{
    align
	al = load_entity_align(array);
    list
	lal = align_alignment(al);
    alignment
	alt = FindAlignmentOfDim(lal, i);
    entity
	template = align_template(al);
    distribute
	dis = load_entity_distribute(template);
    list
	ld = distribute_distribution(dis);
    distribution
	d;

    if (alignment_undefined_p(alt)) return(FALSE);
    
    d = FindDistributionOfDim(ld, alignment_templatedim(alt), pprocdim);

    return(!style_none_p(distribution_style(d)));
}

/*
 * MakeStatementLike 
 *
 * creates a new statement for the given module
 * that looks like the stat one, i.e. same (shared) comment, same 
 * label, and so on. The goto table is updated. The instruction
 * is also created. (is that really a good idea?)
 */
statement MakeStatementLike(stat, the_tag, gotos)
statement stat;
int the_tag;
statement_mapping gotos;
{
    statement 
	newstat =
	    make_statement(statement_label(stat),
			   STATEMENT_NUMBER_UNDEFINED,
			   STATEMENT_ORDERING_UNDEFINED,
			   statement_comments(stat),     /* sharing! */
			   make_instruction(the_tag, chunk_undefined));

    SET_STATEMENT_MAPPING(gotos, stat, newstat);
    return(newstat);
}


/*
 * DistArraysEffects
 *
 * effects' action in an expression are here supposed to be read one's
 * but that may not be correct?
 */
list DistArraysEffects(expr)
expression expr;
{
    list
	le=proper_effects_of_expression(expr,is_action_read),
	lde=NULL;

    MAPL(ce,
     {
	 effect e=EFFECT(CAR(ce));

	 if(array_distributed_p(e)) lde=CONS(EFFECT,e,lde);
     },
	 le);
    
    gen_free_list(le);
    return(lde);
}

/*
 * FindRefToDistArrayFromList
 *
 * these functions compute the list of syntax that are
 * references to a distributed variable.
 */
list FindRefToDistArrayFromList(lexpr)
list lexpr;
{
    list l=NULL;

    MAPL(ce,{l=gen_nconc(FindRefToDistArray(EXPRESSION(CAR(ce))),l);},lexpr);

    return(l);
}

static list
    found_syntaxes = NIL;

static void FindRefToDistArray_syntax_rewrite(s)
syntax s;
{
    if (syntax_reference_p(s))
	if (array_distributed_p
	    (reference_variable(syntax_reference(s))))
	    found_syntaxes = 
		CONS(SYNTAX, s, found_syntaxes);
}

list FindRefToDistArray(obj)
chunk* obj;
{
    list
	result = NIL,
	saved = found_syntaxes;

    found_syntaxes = NIL;

    gen_recurse(obj,
		syntax_domain,
		gen_true,
		FindRefToDistArray_syntax_rewrite);

    result = found_syntaxes, found_syntaxes = saved;

    return(result);
}

/* -------------------------------------------------------------
 *
 * New Temporary Variables MANAGEMENT
 *
 */

static int 
    unique_integer_number,
    unique_float_number,
    unique_logical_number,
    unique_complex_number;

void hpfc_init_unique_numbers()
{
    unique_integer_number=0;
    unique_float_number=0;
    unique_logical_number=0;
    unique_complex_number=0;
}

entity NewTemporaryVariable(module, base)
entity module;
basic base;
{
    char buffer[20];
    entity e;
    
    switch(basic_tag(base))
    {
    case is_basic_int:
	sprintf(buffer,"%s%d",
		HPFINTPREFIX,
		unique_integer_number++);
	break;
    case is_basic_float:
	sprintf(buffer,"%s%d",
		HPFFLOATPREFIX,
		unique_float_number++);
	break;
    case is_basic_logical:
	sprintf(buffer,"%s%d",
		HPFLOGICALPREFIX,
		unique_logical_number++);
	break;
    case is_basic_complex:
	sprintf(buffer,"%s%d",
		HPFCOMPLEXPREFIX,
		unique_complex_number++);
	break;
    default:
	pips_error("NewTemporaryVariable",
		   "basic not welcomed, %d\n",
		   basic_tag(base));
	break;
    }

    debug(9,"NewTemporaryVariable","var %s, tag %d\n", buffer, basic_tag(base));

    e = make_scalar_entity(buffer, module_local_name(module), base);

    AddEntityToDeclarations(e,module);
    
    return(e);
}

entity FindOrCreateEntityLikeModel(package, name, model)
string package, name;
entity model;
{
    string
	new_name = concatenate(package, 
			       MODULE_SEP_STRING, 
			       name, 
			       NULL);
    entity 
	new = gen_find_tabulated(new_name, entity_domain);
    area
	tmp_area = area_undefined;

    debug(3, "FindOrCreateEntityLikeModel",
	  "entity %s to be made after %s\n",
	  new_name, entity_name(model));

    assert(gen_consistent_p(model));

    return(!entity_undefined_p(new) ?
	   new :
	   make_entity(copy_string(new_name),
		       /*
			* ??? some bug in copy_type disable the possibility
			* of copying area for instance...
			*
			* moreover I do not wish to copy the layout list
			* for commons.
			*/
		       (!type_area_p(entity_type(model)) ?
			copy_type(entity_type(model)) :
			make_type(is_type_area,
				  (tmp_area = type_area(entity_type(model)),
				   make_area(area_size(tmp_area), NIL)))),
		       copy_storage(entity_storage(model)),
		       copy_value(entity_initial(model))));
}

/*   !!! caution, it may not be a module, but a common...
 */
entity AddEntityToModule(e, module)
entity e, module;
{
    entity 
	new = FindOrCreateEntityLikeModel(module_local_name(module),
					  entity_local_name(e),
					  e);

    debug(7, "AddEntityToModule", "adding %s to module %s\n",
	  entity_name(new), entity_name(module));
    
    if (entity_module_p(module))
	AddEntityToDeclarations(new, module);

    return(new);
}

/*   AddEntityToHostAndNodeModules
 */
void AddEntityToHostAndNodeModules(e)
entity e;
{
    if (entity_node_new_undefined_p(e))
	store_new_node_variable(AddEntityToModule(e, node_module), e);
    else
	AddEntityToDeclarations(load_entity_node_new(e), node_module);
    
    if (!array_distributed_p(e))
    {
	if (entity_host_new_undefined_p(e))
	    store_new_host_variable(AddEntityToModule(e, host_module), e);
	else
	    AddEntityToDeclarations(load_entity_host_new(e), host_module);
    }
}

/* The common name is changed to distinguish the current, host and
 * node instances of the common. 
 */
void AddCommonToModule(common, module, update, suffix)
entity common, module;
void (*update)();
string suffix;
{
    string
	name = strdup(concatenate(entity_local_name(common),
				  "_", suffix, NULL));
    entity
	new_common = 
	    FindOrCreateEntityLikeModel(HPFC_PACKAGE,
					name,
					common);
    list 
	lref = area_layout(type_area(entity_type(common))),
	lold = area_layout(type_area(entity_type(new_common))),
	lnew = NIL;

    free(name);
    update(new_common, common);

    /* The layout list must be updated to the right entities
     */

    MAPL(ce,
     {
	 entity 
	     e = ENTITY(CAR(ce));
	 entity
	     new_e;

	 if (local_entity_of_module_p(e, common)) /* !!! not in current  */
	 {
	     new_e = AddEntityToModule(e, new_common);

	     if (gen_find_eq(new_e, lold)==entity_undefined)
	     {
		 lnew = CONS(ENTITY, new_e, lnew);
		 update(new_e, e);
	     }
	 }
     },
	 lref);

    AddEntityToDeclarations(new_common, module);

    area_layout(type_area(entity_type(new_common))) = 
	gen_nconc(gen_nreverse(lnew), lold);
}

void AddCommonToHostAndNodeModules(common)
entity common;
{
    AddCommonToModule(common, node_module, store_new_node_variable, NODE_NAME);
    AddCommonToModule(common, host_module, store_new_host_variable, HOST_NAME); 
}

/*
 * nextline
 */
string nextline(line)
string line;
{
    while (((*line)!='\n') && ((*line)!='\0')) line++;
    return(((*line)=='\n')?(line+1):(line));
}

/*
 * FindAlignmentOfDim
 */
alignment FindAlignmentOfDim(lal, dim)
list lal;
int dim;
{
    list l=lal;

    while ((!ENDP(l)) &&
	   (alignment_arraydim(ALIGNMENT(CAR(l))) != dim))
	l = CDR(l);

    return ((l==NULL)?(alignment_undefined):(ALIGNMENT(CAR(l))));
}


/*
 * FindAlignmentOfTemplateDim
 */
alignment FindAlignmentOfTemplateDim(lal, dim)
list lal;
int dim;
{
    list l=lal;

    while ((!ENDP(l)) &&
	   (alignment_templatedim(ALIGNMENT(CAR(l))) != dim))
	l = CDR(l);

    return ((l==NULL)?(alignment_undefined):(ALIGNMENT(CAR(l))));
}

/*
 * FindDistributionOfDim
 */
distribution FindDistributionOfDim(ldi, dim, pdim)
list ldi;
int dim, *pdim;
{
    list 
	l = ldi;
    int 
	i,
	procdim = 1;

    assert(dim>=1 && dim<=gen_length(ldi));

    for (i=1; i<dim; i++) 
    {
	if (!style_none_p(distribution_style(DISTRIBUTION(CAR(l)))))
	    procdim++;
	l=CDR(l);
    }

    (*pdim) = procdim;
    return(DISTRIBUTION(CAR(l)));
}

/*
 * distribution FindDistributionOfProcessorDim(ldi, dim, tdim)
 */
distribution FindDistributionOfProcessorDim(ldi, dim, tdim)
list ldi;
int dim, *tdim;
{
    int 
	i = 1,
	procdim = 0;
    
    MAPL(cd,
     {
	 distribution 
	     d = DISTRIBUTION(CAR(cd));

	 if (!style_none_p(distribution_style(d)))
	     procdim++;

	 if (procdim==dim)
	 {
	     (*tdim) = i;
	     return(d);
	 }
	 
	 i++;
     },
	 ldi);

    pips_error("FindDistributionOfProcessorDim",
	       "dimension %d not found\n", dim);

    return(distribution_undefined);
}    

/*
 * int template_dimension_of_array_dimension(array, dim)
 *
 * the matching dimension of a distributed
 */
int template_dimension_of_array_dimension(array, dim)
entity array;
int dim;
{
    align
	a = load_entity_align(array);
    alignment
	al = FindAlignmentOfDim(align_alignment(a), dim);
    
    return((al==alignment_undefined)?
	   (-1):
	   (alignment_templatedim(al)));
}

/*
 * int DistributionParameterOfArrayDim(array, dim, pprocdim)
 *
 *
 */
int DistributionParameterOfArrayDim(array, dim, pprocdim)
entity array;
int dim, *pprocdim;
{
    entity
	template = array_to_template(array);
    distribute
	d = load_entity_distribute(template);
    distribution
	di = FindDistributionOfDim
	    (distribute_distribution(d),
	     alignment_templatedim(FindArrayDimAlignmentOfArray(array, dim)),
	     pprocdim);
    
    return(HpfcExpressionToInt(distribution_parameter(di)));
}


/*
 * int processor_number(template, tdim, tcell, pprocdim)
 *
 * the processor number of a template cell, on dimension *pprocdim
 */
int processor_number(template, tdim, tcell, pprocdim)
entity template;
int tdim, tcell, *pprocdim; /* template dimension, template cell */
{
    distribute
	d = load_entity_distribute(template);
    list
	ld = distribute_distribution(d);
    entity
	procs = distribute_processors(d);
    distribution
	di = FindDistributionOfDim(ld, tdim, pprocdim);
    style
	st = distribution_style(di);
    int
	n, tmin, pmin, psiz;

    if (style_none_p(st))
    {
	*pprocdim = -1;
	return(-1);
    }

    tmin = HpfcExpressionToInt(dimension_lower(FindIthDimension(template, tdim)));
    pmin = HpfcExpressionToInt(dimension_lower(FindIthDimension(procs, *pprocdim)));
    psiz = SizeOfIthDimension(procs, *pprocdim);
    n    = HpfcExpressionToInt(distribution_parameter(di));

    if (style_block_p(st))
	return(((tcell-tmin)/n)+pmin);

    if (style_cyclic_p(st))
	return((((tcell-tmin)/n)%psiz)+pmin);

    *pprocdim = -1; /* just to avoid a gcc warning */
    return(-1); 
}    


/*
 * int template_cell_local_mapping(array, dim, tc)
 *
 * ??? should check that it is indeed block distributed !
 * or should implement all the formulas...
 */
int template_cell_local_mapping(array, dim, tc)
entity array;
int dim, tc;
{
    alignment
	a = FindArrayDimAlignmentOfArray(array, dim);
    int
	p,
	tmin,
	n = DistributionParameterOfArrayDim(array, dim, &p);
    dimension
	d = FindIthDimension(array_to_template(array), 
			     alignment_templatedim(a));

    tmin = HpfcExpressionToInt(dimension_lower(d));

    return((tc-tmin)%n+1);
}
	
/*
 * int global_array_cell_to_local_array_cell(array, dim, acell)
 *
 * ??? not enough general a function
 */
int global_array_cell_to_local_array_cell(array, dim, acell)
entity array;
int dim, acell;
{
    alignment
	a = FindArrayDimAlignmentOfArray(array, dim);
    int
	rate, constant;

    assert(a!=alignment_undefined);

    rate     = HpfcExpressionToInt(alignment_rate(a));
    constant = HpfcExpressionToInt(alignment_constant(a));

    return(template_cell_local_mapping(array, dim, rate*acell+constant));	
}

/*
 * HpfcExpressionToInt(e)
 *
 * uses the normalized value if possible. 
 */
int HpfcExpressionToInt(e)
expression e;
{
    normalized
	n = expression_normalized(e);

    ifdebug(8) print_expression(e);

    if ((n!=normalized_undefined) && (normalized_linear_p(n)))
    {
	Pvecteur
	    v = normalized_linear(n);
	int
	    s = vect_size(v),
	    val = (int) vect_coeff(TCST, v);
	
	if (s==0) return(0);
	if ((s==1) && (val!=0)) return(val);
    }
    
    if (expression_integer_constant_p(e))
	return(ExpressionToInt(e));
    else
	pips_error("HpfcExpressionToInt", "can't return anything, sorry\n");

    return(-1); /* just to avoid a gcc warning */
}

/* -------------------------------------------------------
 *
 * a nicer interface to extract the needed informations:-)
 * FC 29/03/94
 *
 */

void get_alignment(array, dim, ptdim, pa, pb)
entity array;
int dim, *ptdim, *pa, *pb;
{ 
    align
	al = load_entity_align(array);
    alignment
	a = alignment_undefined;
    
    assert(array_distributed_p(array));
    
    *ptdim = template_dimension_of_array_dimension(array, dim);
    a = FindAlignmentOfTemplateDim(align_alignment(al), *ptdim);

    if (a==alignment_undefined)
    {
	assert(*ptdim==0);
	*pa = 0;
	*pb = 0;
    }
    else
    {
	assert(*ptdim>=1);
	*pa = HpfcExpressionToInt(alignment_rate(a));
	*pb = HpfcExpressionToInt(alignment_constant(a));
    }
}

void get_distribution(template, dim, ppdim, pn)
entity template;
int dim, *ppdim, *pn;
{
    distribution
        d = FindDistributionOfDim
	    (distribute_distribution(load_entity_distribute(template)), 
				     dim, ppdim);

    *pn = (distribution_undefined_p(d) ?
	   -1: HpfcExpressionToInt(distribution_parameter(d)));
}

void get_entity_dimensions(e, dim, plow, pup)
entity e;
int dim, *plow, *pup;
{
    dimension
	d = dimension_undefined;

    assert(entity_variable_p(e) && dim>0 && dim<=7);

    d = entity_ith_dimension(e, dim),
    *plow = ExpressionToInt(dimension_lower(d)),
    *pup = ExpressionToInt(dimension_upper(d));
}

/*   that is all
 */
