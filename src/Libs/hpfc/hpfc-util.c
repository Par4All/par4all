/*
 * this file describe a few functions usefull to the compiler
 * to manage the hpfc data structures.
 *
 * Fabien Coelho, May 1993.
 */

#include <stdio.h>
#include <string.h>

extern int fprintf();

#include "genC.h"
#include "hash.h"

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

/*
 * array_distributed_p
 */
bool array_distributed_p(ent)
entity ent;
{
    list lda=distributedarrays;

    while ((lda!=NULL) && (ENTITY(CAR(lda))!=ent)) lda=CDR(lda);
    return((lda!=NULL) && (ENTITY(CAR(lda))==ent));
}

/*
 * ref_to_dist_array_p
 */
bool ref_to_dist_array_p(expr)
expression expr;
{
    return(FindRefToDistArray(expr)!=NULL);
}

/*
 * call_ref_to_dist_array_p
 */
bool call_ref_to_dist_array_p(c)
call c;
{
    bool flag=FALSE;

    MAPL(ce,
     {
	 if (ref_to_dist_array_p(EXPRESSION(CAR(ce)))) flag=TRUE;
     },
	 call_arguments(c));

    return(flag);
}

/*
 * written_effects_to_dist_arrays_p
 */
bool written_effects_to_dist_arrays_p(expr)
expression expr;
{
    bool written=FALSE;
    list leffects_to_dist_arrays=DistArraysEffects(expr);

    MAPL(ce,
     {
	 if (action_write_p(effect_action(EFFECT(CAR(ce))))) 
	     written=TRUE;
     },
	 leffects_to_dist_arrays);

    gen_free_list(leffects_to_dist_arrays);

    return(written);
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
    bool replicated=FALSE;
    align a;
    list la,ld;
    entity template;
    distribute d;

    pips_assert("replicated_p",array_distributed_p(e));

    a=(align) GET_ENTITY_MAPPING(hpfalign,e);
    la=align_alignment(a);    
    template=align_template(a);
    d=(distribute) GET_ENTITY_MAPPING(hpfdistribute,template);
    ld=distribute_distribution(d);

    for(i=1;i<=NumberOfDimension(template);i++)
    {
	replicated = (replicated || 
		      ith_dim_replicated_p(template,i,la,DISTRIBUTION(CAR(ld))));

	ld=CDR(ld);
    }
    
    return(replicated);
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
	al = (align) GET_ENTITY_MAPPING(hpfalign, array);
    list
	lal = align_alignment(al);
    alignment
	alt = FindAlignmentOfDim(lal, i);
    entity
	template = align_template(al);
    distribute
	dis = (distribute) GET_ENTITY_MAPPING(hpfdistribute, template);
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
statement MakeStatementLike(stat,the_tag,gotos)
statement stat;
int the_tag;
statement_mapping gotos;
{
    statement newstat=make_statement(statement_label(stat),
				     STATEMENT_NUMBER_UNDEFINED,
				     STATEMENT_ORDERING_UNDEFINED,
				     statement_comments(stat),     /* sharing! */
				     make_instruction(the_tag,
						      NULL));

    SET_STATEMENT_MAPPING(gotos,stat,newstat);

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
	     

list FindRefToDistArray(expr)
expression expr;
{
    syntax the_syntax=expression_syntax(expr);

    switch(syntax_tag(the_syntax))
    {
    case is_syntax_reference:
	if (array_distributed_p(reference_variable(syntax_reference(the_syntax))))
	{
	    return(CONS(SYNTAX,the_syntax,NULL));
	}
	break;
    case is_syntax_range:
    {
	list 
	    ll=NULL,
	    lu=NULL,
	    li=NULL,
	    lt=NULL;
	
	ll=FindRefToDistArray(range_lower(syntax_range(the_syntax)));
	lu=FindRefToDistArray(range_upper(syntax_range(the_syntax)));
	li=FindRefToDistArray(range_increment(syntax_range(the_syntax)));
	lt=gen_nconc(ll,gen_nconc(lu,li));

	return(lt);
	break;
    }
    case is_syntax_call:
    {
	list
	    lt=FindRefToDistArrayFromList(call_arguments(syntax_call(the_syntax)));

	return(lt);
	break;
    }
    default:
	pips_error("FindRefToDistArray","unexpected syntax tag\n");
	break;
    }

    return(NULL);
}


/*
 * NewTemporaryVariable
 */
entity NewTemporaryVariable(module, base)
entity module;
basic base;
{
    char buffer[20];
    entity e;
    
    switch(basic_tag(base))
    {
    case is_basic_int:
	sprintf(buffer,"%s%d",HPFINTPREFIX,uniqueintegernumber++);
	break;
    case is_basic_float:
	sprintf(buffer,"%s%d",HPFFLOATPREFIX,uniquefloatnumber++);
	break;
    case is_basic_logical:
	sprintf(buffer,"%s%d",HPFLOGICALPREFIX,uniquelogicalnumber++);
	break;
    case is_basic_complex:
	sprintf(buffer,"%s%d",HPFCOMPLEXPREFIX,uniquecomplexnumber++);
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

/*
 * AddEntityToHostAndNodeModules
 */
void AddEntityToHostAndNodeModules(e)
entity e;
{
    entity 
	en;

    ifdebug(9)
    {
	fprintf(stderr,"[AddEntityToHostAndNodeModules]\nentity\n");
	print_entity_variable(e);
    }

    en = make_entity(strdup(concatenate(NODE_NAME,
					MODULE_SEP_STRING,
					entity_local_name(e),
					NULL)),
		     type_variable_dup(entity_type(e)),
		     entity_storage(e),
		     entity_initial(e));
    
    ifdebug(8)
    {
	fprintf(stderr,"[AddEntityToHostAndNodeModules]\nentity\n");
	print_entity_variable(en);
    }
    
    AddEntityToDeclarations(en,nodemodule);
    SET_ENTITY_MAPPING(oldtonewnodevar,e,en);
    SET_ENTITY_MAPPING(newtooldnodevar,en,e);
    
    if (!array_distributed_p(e))
    {
	entity eh=make_entity(strdup(concatenate(HOST_NAME,
						 MODULE_SEP_STRING,
						 entity_local_name(e),
						 NULL)),
			      type_variable_dup(entity_type(e)),
			      entity_storage(e),
			      entity_initial(e));
	
	AddEntityToDeclarations(eh,hostmodule);
	SET_ENTITY_MAPPING(oldtonewhostvar,e,eh);
	SET_ENTITY_MAPPING(newtooldhostvar,eh,e);
    }
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

    pips_assert("FindDistributionOfDim",((dim>=1) && (dim<=gen_length(ldi))));

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
int dim, *pdim;
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
 * local_index_is_different_p
 */
bool local_index_is_different_p(array,dim)
entity array;
int dim;
{
    list
	l = (list) GET_ENTITY_MAPPING(newdeclarations, array);
    int i;

    pips_assert("local_index_is_different_p", (array_distributed_p(array)));

    for (i=1; i<dim; i++) l=CDR(l);

    return(INT(CAR(l))!=NO_NEW_DECLARATION);
}

/*
 * alignment FindArrayDimAlignmentOfArray(array, dim)
 */
alignment FindArrayDimAlignmentOfArray(array, dim)
entity array;
int dim;
{
    align
	a = (align) GET_ENTITY_MAPPING(hpfalign, array);
    
    return(FindAlignmentOfDim(align_alignment(a), dim));
}

/*
 * alignment FindTemplateDimAlignmentOfArray(array, dim)
 */
alignment FindTemplateDimAlignmentOfArray(array, dim)
entity array;
int dim;
{
    align
	a = (align) GET_ENTITY_MAPPING(hpfalign, array);
    
    return(FindAlignmentOfTemplateDim(align_alignment(a), dim));
}


/*
 * entity array_to_template(array)
 *
 */
entity array_to_template(array)
entity array;
{
    pips_assert("array_to_template",
		array_distributed_p(array));

    return(align_template((align) GET_ENTITY_MAPPING(hpfalign, array)));
}

/*
 * entity template_to_processors(template)
 *
 */
entity template_to_processors(template)
entity template;
{
    pips_assert("template_to_processors",
		(gen_find_eq(template, templates)!=chunk_undefined));

    return(distribute_processors
	   ((distribute) GET_ENTITY_MAPPING(hpfdistribute, template)));
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
	a = (align) GET_ENTITY_MAPPING(hpfalign, array);
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
	d = (distribute) GET_ENTITY_MAPPING(hpfdistribute, template);
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
	d = (distribute) GET_ENTITY_MAPPING(hpfdistribute, template);
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

    pips_assert("global_array_cell_to_local_array_cell",
		(a!=alignment_undefined));

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
	return(-314); /* value returned if doesn't know */
}

