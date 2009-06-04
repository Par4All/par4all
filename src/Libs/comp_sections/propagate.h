/*

  $Id$

  Copyright 1989-2009 MINES ParisTech

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
#ifndef _PROPAGATE
#define _PRPPAGATE
/* propagate.c */
extern void CheckStride(loop Loop);
extern list CompRegionsExactUnion(list l1, list l2, bool (*union_combinable_p)(effect, effect));
extern list CompRegionsMayUnion(list l1, list l2, bool (*union_combinable_p)(effect, effect));
extern bool comp_regions(char *module_name);
extern list comp_regions_of_statement(statement s);
extern list comp_regions_of_instruction(instruction i, transformer t_inst, transformer context, list *plpropreg);
extern list comp_regions_of_block(list linst);
extern list comp_regions_of_test(test t, transformer context, list *plpropreg);
extern list comp_regions_of_loop(loop l, transformer loop_trans, transformer context, list *plpropreg);
extern list comp_regions_of_call(call c, transformer context, list *plpropreg);
extern list comp_regions_of_unstructured(unstructured u, transformer t_unst);
extern list comp_regions_of_range(range r, transformer context);
extern list comp_regions_of_syntax(syntax s, transformer context);
extern list comp_regions_of_expressions(list exprs, transformer context);
extern list comp_regions_of_expression(expression e, transformer context);
extern list comp_regions_of_read(reference ref, transformer context);
extern list comp_regions_of_write(reference ref, transformer context);

#endif


