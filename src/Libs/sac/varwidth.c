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

#include "genC.h"
#include "linear.h"
#include "ri.h"

#include "resources.h"

#include "misc.h"
#include "ri-util.h"
#include "pipsdbm.h"

#include "dg.h"

typedef dg_arc_label arc_label;
typedef dg_vertex_label vertex_label;

#include "graph.h"

#include "sac.h"


static int g_varwidth;

static bool variables_width_filter(reference r)
{
   basic b;
   int width;
   type t = entity_type(reference_variable(r));

   if (!type_variable_p(t))
      return TRUE;  /* keep searching recursively */

   b = variable_basic(type_variable(t));

   /* do NOT forget to multiply the size by 8, to get it in
    * bits instead of bytes....
    */
   switch(basic_tag(b))
   {
      case is_basic_int:
	 width = 8*basic_int(b);
	 break;

      case is_basic_float:
	 width = 8*basic_float(b);
	 break;

      /*case is_basic_logical:
	 width = 8*basic_logical(b);
	 break;*/

      case is_basic_pointer:
      {
	 basic bas = get_basic_from_array_ref(r);
	 if(!basic_undefined_p(bas))
	 {
	    switch(basic_tag(bas))
	    {
	       case is_basic_int: width = 8*basic_int(bas); break;
	       case is_basic_float: width = 8*basic_float(bas); break;
           default: pips_internal_error("basic type %u not supported",basic_tag(bas));
	    }
	 }
	 else
	 {
	    return TRUE;
	 }
	 break;
      }

      default:
	 return TRUE; /* don't know what to do with this... keep searching */
   }
   
   if (width > g_varwidth)
      g_varwidth = width;

   return FALSE; /* do not search recursively */
}

int effective_variables_width(instruction i)
{
   g_varwidth = 0;

   gen_recurse( i, reference_domain, 
		variables_width_filter, gen_null);

   return g_varwidth;
}
