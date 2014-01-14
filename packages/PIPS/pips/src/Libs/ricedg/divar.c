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
statement_mapping contexts_mapping_of_nest(stat, ce_map)
statement stat;
statement_mapping ce_map;
{
    pips_assert(contexts_mapping_of_nest, statement_loop_p(stat));

    ifdebug(5)  {
	STATEMENT_MAPPING_MAP(st, context, {
	    statement stp = (statement) st;

	    if (statement_call_p(stp)) {
		fprintf(stderr, "Execution context of statement %d :\n", 
			statement_number(stp));
		sc_fprint(stderr, (Psysteme) context, entity_local_name);
	    }
	}, contexts_map);
    }

    return(contexts_map);
}
