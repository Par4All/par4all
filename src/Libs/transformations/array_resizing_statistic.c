/* This phase computes the number of pointer-type A(1) and assumed-size A(*) array declarators 
   in one program. There are two cases: these unnormalized declarations are formal array 
   parameters or not */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "database.h"
#include "pipsdbm.h"
#include "resources.h"
#include "misc.h"
#include "control.h"
#include "properties.h"
#include "semantics.h"
#include "instrumentation.h"
#include "transformations.h"

static int number_of_pointer_type_and_formal_arrays = 0;
static int number_of_assumed_size_and_formal_arrays = 0;
static int number_of_formal_arrays = 0;
static int number_of_pointer_type_and_not_formal_arrays = 0;
static int number_of_assumed_size_and_not_formal_arrays = 0;
static int number_of_not_formal_arrays = 0;

static int number_of_processed_modules = 0;

bool array_resizing_statistic(char *module_name)
{
  entity module_ent = local_name_to_top_level_entity(module_name);
  list l_decl = code_declarations(entity_code(module_ent));

  number_of_processed_modules++;

  debug_on("ARRAY_RESIZING_STATISTIC_DEBUG_LEVEL");
  ifdebug(1)
    fprintf(stderr, " \n Begin array resizing statistic for %s \n", module_name);   
   
  /* search for unnormalized array declarations in the list */  
  while(!ENDP(l_decl))
    {
      entity e = ENTITY(CAR(l_decl));
      if (entity_variable_p(e))
	{
	  variable v = type_variable(entity_type(e));   
	  list l_dims = variable_dimensions(v);
	  if (l_dims != NIL)
	    {
	      int length = gen_length(l_dims);
	      dimension last_dim =  find_ith_dimension(l_dims,length);
	      expression exp = dimension_upper(last_dim);
	      storage s = entity_storage(e);
	      if (storage_formal_p(s))   
		number_of_formal_arrays++;
	      else 
		number_of_not_formal_arrays++;
	      if (unbounded_dimension_p(last_dim)) 
		{
		  if (storage_formal_p(s))  
		    number_of_assumed_size_and_formal_arrays++;
		  else 
		    number_of_assumed_size_and_not_formal_arrays++;
		}
	      if (expression_equal_integer_p(exp,1))
		{
		  if (storage_formal_p(s)) 
		    number_of_pointer_type_and_formal_arrays++;
		  else 
		    number_of_pointer_type_and_not_formal_arrays++;
		}
	    }
	}
      l_decl = CDR(l_decl);
    }
  
  user_log(" \n Number of pointer-type A(1), formal arrays : %d \n"
	  ,number_of_pointer_type_and_formal_arrays );
  user_log(" \n Number of assumed-size A(*), formal arrays : %d \n"
	   ,number_of_assumed_size_and_formal_arrays );
  user_log(" \n Number of formal array declarators : %d\n"
	   ,number_of_formal_arrays );
  user_log(" \n Number of pointer-type A(1), local arrays : %d \n"
	  ,number_of_pointer_type_and_not_formal_arrays );
  user_log(" \n Number of assumed-size A(*), local arrays : %d \n"
	   ,number_of_assumed_size_and_not_formal_arrays );
  user_log(" \n Number of local array declarators : %d\n"
	   ,number_of_not_formal_arrays );
  user_log(" \n Total number of pointer-type arrays : %d\n"
	  ,number_of_pointer_type_and_formal_arrays + 
	   number_of_pointer_type_and_not_formal_arrays );
  user_log(" \n Total number of assumed-size arrays : %d\n"
	   ,number_of_assumed_size_and_formal_arrays + 
	   number_of_assumed_size_and_not_formal_arrays);
  user_log(" \n Total number of arrays : %d\n"
	   ,number_of_formal_arrays +
	   number_of_not_formal_arrays);

  user_log("\n Number of processed modules: %d \n"
	   ,number_of_processed_modules);

  ifdebug(1)
    fprintf(stderr, " \n End array resizing statistic for %s \n", module_name);
  debug_off();
  return TRUE;
}













