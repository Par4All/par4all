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
/* Overlap Management Module for HPFC
 * Fabien Coelho, August 1993
 */

#include "defines-local.h"
bool expression_constant_p(expression); /* in static_controlize */

GENERIC_GLOBAL_FUNCTION(overlap_status, overlapsmap)

static void create_overlaps(e)
entity e;
{
    type t = entity_type(e);
    list o=NIL;
    int n;

    assert(type_variable_p(t));

    n = gen_length(variable_dimensions(type_variable(t)));
    for(; n>=1; n--) o = CONS(OVERLAP, make_overlap(0, 0), o);

    store_overlap_status(e, o);

    assert(bound_overlap_status_p(e));
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

    if (!bound_overlap_status_p(ent)) create_overlaps(ent);
    assert(bound_overlap_status_p(ent));

    o = OVERLAP(gen_nth(dim-1, load_overlap_status(ent)));
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
    entity ent;
    int ndim, i, lower_overlap, upper_overlap;
    dimension the_dim;

    MAP(ENTITY, oldent,
     {
	 ent = load_new_node(oldent);
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

/* updates overlaps for similar arrays that are going to be merged
 */
static void deal_with_similars(list le)
{
    MAP(ENTITY, array,
    {
	if (entity_dynamic_p(array) && load_similar_mapping(array)!=array)
	{
	    entity sim = load_similar_mapping(array);
	    int dim;

	    pips_debug(8, "translating overlaps from %s to %s\n",
		       entity_name(array), entity_name(sim));

	    for (dim=variable_entity_dimension(array); dim>0; dim--)
	    {
		int o;

		o = get_overlap(array, dim, 0);
		set_overlap(sim, dim, 0, o);
		o = get_overlap(array, dim, 1);
		set_overlap(sim, dim, 1, o);
	    }
	}
    },
       le);
}

void declaration_with_overlaps_for_module(module)
entity  module;
{
    list l = list_of_distributed_arrays_for_module(module);

    deal_with_similars(l);
    declaration_with_overlaps(l);
    gen_free_list(l);
}

/*   That is all
 */
