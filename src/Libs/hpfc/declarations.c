/*
 * HPFC module by Fabien COELHO
 *
 * DECLARATIONS compilation
 *
 * SCCS stuff:
 * $RCSfile: declarations.c,v $ ($Date: 1994/06/03 14:14:51 $, ) version $Revision$,
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
#include "control.h"
#include "regions.h"
#include "semantics.h"
#include "effects.h"

/* 
 * my own local includes
 */

#include "hpfc.h"
#include "defines-local.h"

/*
 *  local macros...
 */

#define normalized_dimension_p(dim) \
  (HpfcExpressionToInt(dimension_lower(dim))==1)

/* -----------------------------------------------------------------
 *
 * New Declarations
 *
 */

/*
 * ComputeNewSizeOfIthDimension
 *
 * here the new size of the ith dimension of the given array is computed.
 * because the declarations are static, there is a majoration of the space
 * required on each processors to held his part of the distributed array.
 */
static int ComputeNewSizeOfIthDimension(dim, i, array, newdeclp)
dimension dim;
int i;
entity array;
int *newdeclp;
{
    align 
	a = load_entity_align(array);
    entity 
	t = align_template(a);
    distribute 
	d = load_entity_distribute(t);

    alignment 
	al=NULL;
    distribution 
	di=NULL;
    int 
	rate,
	param,
	pdim = 1,
	asize = SizeOfDimension(dim);
    style st;

    debug(9,"ComputeNewSizeOfIthDimension",
	  "dimension %d of array %s\n",i,entity_name(array));
    ifdebug(9)
    {
	print_align(a);
	print_distribute(d);
    }

    /*
     * default: the new declaration is the same as the old one.
     */
    (*newdeclp) = NO_NEW_DECLARATION;

    /*
     * looking for the matching alignment...
     */
    al = FindAlignmentOfDim(align_alignment(a), i);

    /*
     * no alignment => scratching of the dimension...
     */
    if (al==alignment_undefined) 
    {
	if (!normalized_dimension_p(dim)) 
	    (*newdeclp) = ALPHA_NEW_DECLARATION;
	return(asize);
    }

    /*
     * there is an alignment, but the rate is zero, so the whole
     * dimension has to be declared on every processors, despite the
     * fact that the dimension is mapped on only one element.
     */
    rate=HpfcExpressionToInt(alignment_rate(al));
    if (rate==0) 
    {
	if (!normalized_dimension_p(dim)) 
	    (*newdeclp) = ALPHA_NEW_DECLARATION;
	return(asize);
    }
    
    /*
     * looking for the matching distribution...
     * pdim is the corresponding dimension of  processors p
     */
    di = FindDistributionOfDim(distribute_distribution(d),
			       alignment_templatedim(al),
			       &pdim);
			       
    st=distribution_style(di);

    /*
     * no style => scratching of the dimension...
     */
    if (style_none_p(st)) 
    {
	/*
	 * ???
	 * should delete the alignment which is not usefull...
	 */
	/*
	 * alpha case
	 */
	if (!normalized_dimension_p(dim)) 
	    (*newdeclp) = ALPHA_NEW_DECLARATION;
	return(asize);
    }
    
    /*
     * and now, let's look at the different cases.
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
		(*newdeclp) = NO_NEW_DECLARATION;
	    else
		(*newdeclp) = ALPHA_NEW_DECLARATION;
	}
	else
	    (*newdeclp) = BETA_NEW_DECLARATION;

	return(choice);
    }

    /*
     * gamma case
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
		(*newdeclp) = NO_NEW_DECLARATION;
	    else
		(*newdeclp) = ALPHA_NEW_DECLARATION;
	}
	else
	    (*newdeclp) = GAMMA_NEW_DECLARATION;

	return(choice);
    }

    /*
     * delta case
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
		(*newdeclp) = NO_NEW_DECLARATION;
	    else
		(*newdeclp) = ALPHA_NEW_DECLARATION;
	}
	else
	    (*newdeclp) = DELTA_NEW_DECLARATION;

	return(choice);
    }
	
    /*
     * alpha case, if nothing matches, what shouldn't be the case :
     */
    if (!normalized_dimension_p(dim)) 
	(*newdeclp) = ALPHA_NEW_DECLARATION;
    return(asize);
}


/*
 * NewDeclarationOfDistributedArray
 *
 * for node this are reformated, and for host these variables are
 * deleted.
 */
static void NewDeclarationOfDistributedArray(array)
entity array;
{
    entity 
	newarray = load_entity_node_new(array);
    int 
	ithdim = 1,
	newdecl = NEW_DECLARATION_UNDEFINED;
    list 
	ld=NIL;
    
    pips_assert("NewDeclarationOfDistributedArray",
		((array_distributed_p(array)) && 
		 (entity_variable_p(array))));

    debug(6,"NewDeclarationOfDistributedArray",
	  "considering array %s, new %s\n",
	  entity_name(array),
	  entity_name(newarray));

    /*
     * compute the new size for every dimension on the array,
     * then update the dimensions of the newarray. remember
     * that the dimensions are shared between the old and new arrays.
     */
    MAPL(cd,
     {
	 int newsize;
	 int p;
	 dimension
	     dim = DIMENSION(CAR(cd));

	 if (ith_dim_distributed_p(array, ithdim, &p))
	 {
	     newsize = ComputeNewSizeOfIthDimension(dim, 
						    ithdim, 
						    array,
						    &newdecl);
	     
	     debug(8, "NewDeclarationOfDistributedArray",
		   "dimension %d new size: %d\n", ithdim, newsize);
	     
	     
	     ld = gen_nconc(ld,
			    CONS(DIMENSION,
				 make_dimension(int_to_expression(1),
						int_to_expression(newsize)),
				 NULL));
	     
	 }
	 else
	 {
	     debug(8, "NewDeclarationOfDistributedArray",
		   "dimension %d isn't touched\n", ithdim);

	     newdecl = NO_NEW_DECLARATION;
	     ld = gen_nconc(ld, CONS(DIMENSION, dim, NIL)); /* sharing ! */
	 }
	 
	 store_new_declaration(array, ithdim, newdecl);

	 ithdim++;
     },
	 variable_dimensions(type_variable(entity_type(array))));

    variable_dimensions(type_variable(entity_type(newarray)))=ld;
}

/*
 * NewDeclarationsOfDistributedArrays
 *
 * this procedure generate the new declarations of every distributed arrays
 * of the program, in order to minimize the amount of memory used.
 * The new declarations have to be suitable for the new index computation
 * which is to be done dynamically...
 */
void NewDeclarationsOfDistributedArrays()
{
    MAPL(ce,
     {
	 NewDeclarationOfDistributedArray(ENTITY(CAR(ce)));
     },
	 list_of_distributed_arrays());
}

/*
 * that is all
 */

