
#include "genC.h"
#include "linear.h"
#include "ri.h"

#include "resources.h"

#include "misc.h"
#include "ri-util.h"
#include "pipsdbm.h"

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

      case is_basic_logical:
	 width = 8*basic_logical(b);
	 break;

      default:
	 return TRUE; /* don't know what to do with this... keep searching */
   }
   printf( "Reference %s is %i bits wide\n",
	   entity_name(reference_variable(r)), width );
   
   if (width > g_varwidth)
      g_varwidth = width;

   return FALSE; /* do not search recursively */
}

static void variables_width_rewrite(reference r)
{
   return;
}

int effective_variables_width(instruction i)
{
   g_varwidth = 0;

   gen_recurse( i, reference_domain, 
		variables_width_filter, variables_width_rewrite);

   return g_varwidth;
}
