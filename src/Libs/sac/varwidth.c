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
#include "stdlib.h"

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"

#include "resources.h"

#include "ri-util.h"
#include "effects-util.h"

#include "sac.h"



static bool variables_width_filter(reference r,int *g_varwidth)
{
    type ut = ultimate_type(entity_type(reference_variable(r)));
   if (type_variable_p(ut))
   {
       basic b = basic_of_reference(r);

       /* do NOT forget to multiply the size by 8, to get it in
        * bits instead of bytes....
        */
       int width;
       switch(basic_tag(b))
       {
           case is_basic_int:
               width = SizeOfElements(b);
               break;

           case is_basic_float:
               width = SizeOfElements(b);
               break;

               /*case is_basic_logical:
                 width = 8*basic_logical(b);
                 break;*/

           case is_basic_pointer:
           default:
               free(b);
               return true; /* don't know what to do with this... keep searching */
       }
       free(b);
       width*=8;

       if (width > *g_varwidth)
           *g_varwidth = width;

       return false; /* do not search recursively */
   }
   else
       return true; /* keep searching recursievly*/
}

int effective_variables_width(instruction i)
{
   int g_varwidth = 0;

   gen_context_recurse( i, &g_varwidth,reference_domain, variables_width_filter, gen_null);

   return g_varwidth;
}
