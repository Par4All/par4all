/*

  $Id$

  Copyright 1989-2010 MINES ParisTech
  Copyright 2010 HPC Project

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

#include <stdio.h>
#include <string.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "effects.h"
#include "effects-util.h"

cell_relation make_value_of_pointer_value(cell c1, cell c2, tag app_tag, descriptor d)
{
  interpreted_cell ic1 = make_interpreted_cell(c1, make_cell_interpretation_value_of());
  interpreted_cell ic2 = make_interpreted_cell(c2, make_cell_interpretation_value_of());
	      
  cell_relation pv = make_cell_relation(ic1, ic2, make_approximation(app_tag, UU), d);
  return(pv);
}

cell_relation make_address_of_pointer_value(cell c1, cell c2, tag app_tag, descriptor d)
{
  interpreted_cell ic1 = make_interpreted_cell(c1, make_cell_interpretation_value_of());
  interpreted_cell ic2 = make_interpreted_cell(c2, make_cell_interpretation_address_of());
	      
  cell_relation pv = make_cell_relation(ic1, ic2, make_approximation(app_tag, UU), d);
  return(pv);
}
