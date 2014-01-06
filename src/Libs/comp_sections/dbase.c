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
#include "all.h"

/* ---------------------------------------------------------------------- */
/* list-comp_desc conversion functions                                      */
/* ---------------------------------------------------------------------- */

comp_desc_set list_to_comp_secs(l_eff)
list l_eff;
{
    comp_desc_set res = make_comp_desc_set(l_eff);
    return res;
}

list comp_desc_set_to_list(cset)
comp_desc_set cset;
{
    list l_res = comp_desc_set_comp_descs(cset);
    return l_res;
}

statement_mapping listmap_to_compsecs_map(l_map)
statement_mapping l_map;
{
    statement_mapping comp_secs_map = MAKE_STATEMENT_MAPPING();
    
    STATEMENT_MAPPING_MAP(s,val,
      {
	hash_put((hash_table) comp_secs_map, (char *) s, (char *) list_to_comp_secs((list) val));
    }, l_map);

    return comp_secs_map;
}

statement_mapping comp_secs_map_to_listmap(compsecs_map)
statement_mapping compsecs_map;
{
    statement_mapping l_map = MAKE_STATEMENT_MAPPING();
    
    STATEMENT_MAPPING_MAP(s,val,{
	hash_put((hash_table) l_map, (char *) s, (char *) comp_desc_set_to_list((comp_desc_set) val));
    }, compsecs_map);

    return l_map;
}

/*********************************************************************************/
/* REGIONS AND LISTS OF REGIONS MANIPULATION                                     */
/*********************************************************************************/

/* list comp_regions_dup(list l_reg)
 * input    : a list of comp_regions.
 * output   : a new list of regions, in which each region of the
 *            initial list is duplicated.
 * modifies : nothing.
 */
list comp_regions_dup(list l_reg)
{
    list l_reg_dup = NIL;
    
    MAP(COMP_DESC, reg,
    {
	comp_desc reg_dup = comp_region_dup(reg);
	l_reg_dup = comp_region_add_to_regions(reg_dup, l_reg_dup);
    }, l_reg);
    
    return(l_reg_dup);
}

comp_desc comp_region_dup(comp_desc reg)
{
    comp_desc new_reg;
    
    /* debug_region_consistency(reg); */
    new_reg = copy_comp_desc(reg);
    /* work around persistency of comp_desc reference */
    comp_desc_reference(new_reg) = copy_reference(comp_desc_reference(reg)); 
    return(new_reg);
}
/* void region_add_to_regions(region reg, list l_reg)
 * input    : a region and a list of regions.
 * output   : nothing.
 * modifies : l_reg.
 * comment  : adds reg at the end of l_reg.
 */
list comp_region_add_to_regions(comp_desc reg, list l_reg)
{
    return gen_nconc(l_reg, CONS(COMP_DESC, reg, NIL));
}





