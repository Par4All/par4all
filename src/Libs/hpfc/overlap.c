/*
 * Overlap Management Module for HPFC
 * Fabien Coelho, August 1993
 *
 * $RCSfile: overlap.c,v $ ($Date: 1995/03/14 18:34:31 $, )
 * version $Revision$
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

#include "misc.h"
#include "ri-util.h"
#include "loop_normalize.h"
#include "hpfc.h"

/* could be
 * GENERIC_GLOBAL_FUNCTION(overlap_management, overlapsmap)
 */

GENERIC_CURRENT_MAPPING(overlaps, overlaps, entity);

void init_overlap_management()
{
    make_overlaps_map();
}

void close_overlap_management()
{
    /* ??? memory leak */
    free_overlaps_map();
}

static void create_overlaps(e)
entity e;
{
    type t = entity_type(e);
    list o=NIL;
    int n;

    assert(type_variable_p(t));

    n = gen_length(variable_dimensions(type_variable(t)));
    for(; n>=1; n--) o = CONS(OVERLAP, make_overlap(0, 0), o);
    store_entity_overlaps(e, make_overlaps(o));
}

/* set_overlap(ent, dim, side, width)
 *
 * set the overlap value for entity ent, on dimension dim,
 * dans side side to width, which must be a positive integer.
 * if necessary, the overlap is updates with the value width.
 */
void set_overlap(ent, dim, side, width)
entity ent;
int dim, side, width;
{
    overlap o;
    int current;

    assert(dim>0);

    if (entity_overlaps_undefined_p(ent)) create_overlaps(ent);
    o = OVERLAP(gen_nth(dim-1, overlaps_dimensions(load_entity_overlaps(ent))));

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

/* int get_overlap(ent, dim, side)
 *
 * returns the overlap for a given entity, dimension and side,
 * to be used in the declaration modifications
 */
int get_overlap(ent, dim, side)
entity ent;
int dim, side;
{
    overlap o;

    assert(dim>0);

    if (entity_overlaps_undefined_p(ent)) create_overlaps(ent);
    o = OVERLAP(gen_nth(dim-1, overlaps_dimensions(load_entity_overlaps(ent))));
    return(side ? overlap_upper(o) : overlap_lower(o));
}

/* static void overlap_redefine_expression(pexpr, ov)
 *
 * redefine the bound given the overlap which is to be included
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
    entity oldent, ent;
    int ndim, i, lower_overlap, upper_overlap;
    dimension the_dim;

    MAPL(ce,
     {
	 oldent = ENTITY(CAR(ce));
	 ent = load_entity_node_new(oldent);
	 ndim = variable_entity_dimension(ent);

	 assert(type_variable_p(entity_type(ent)));

	 for (i=1 ; i<=ndim ; i++)
	 {
	     the_dim = entity_ith_dimension(ent, i);
	     lower_overlap = get_overlap(oldent, i, 0);
	     upper_overlap = get_overlap(oldent, i, 1);

	     debug(8, "declaration_with_overlaps", 
		   "%s(DIM=%d): -%d, +%d\n", 
		   entity_name(ent), i, lower_overlap, upper_overlap);

	     if (lower_overlap!=0) 
		 overlap_redefine_expression(&dimension_lower(the_dim),
					     -lower_overlap);
		 
	     if (upper_overlap!=0) 
		 overlap_redefine_expression(&dimension_upper(the_dim),
					     upper_overlap);
	 }
     },
	 l);
}

void declaration_with_overlaps_for_module(module)
entity  module;
{
    list l = list_of_distributed_arrays_for_module(module);

    declaration_with_overlaps(l);
    gen_free_list(l);
}

/*   That is all
 */
