/* This phase is to calculate the number of 1 and assumed-size (*) array declarators in one program */

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


static int number_of_one_and_formal_array_declarators = 0;
static int number_of_assumed_size_and_formal_array_declarators = 0;
static int number_of_formal_array_declarators = 0;
static int number_of_one_array_declarators = 0;
static int number_of_assumed_size_array_declarators = 0;
static int number_of_array_declarators = 0;

bool adn_instrumentation(char *module_name)
{
  entity module_ent = local_name_to_top_level_entity(module_name);
  list l_decl = code_declarations(entity_code(module_ent));

  debug_on("ADN_INSTRUMENTATION_DEBUG_LEVEL");
  ifdebug(1)
    fprintf(stderr, " \n Begin  adn instrumentation for %s \n", module_name);   
   
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
	      number_of_array_declarators ++;
	      if (storage_formal_p(s))   
		number_of_formal_array_declarators++;
	      if (unbounded_dimension_p(last_dim)) 
		{
		  number_of_assumed_size_array_declarators++;
		  if (storage_formal_p(s))   
		    number_of_assumed_size_and_formal_array_declarators++;
		}
	      if (expression_equal_integer_p(exp,1))
		{
		  number_of_one_array_declarators++;
		  if (storage_formal_p(s))   
		    number_of_one_and_formal_array_declarators++;
		}
	    }
	}
      l_decl = CDR(l_decl);
    }
  
  user_log(" \n Number of 1 and formal array declarators : %d \n"
	  ,number_of_one_and_formal_array_declarators );
  user_log(" \n Number of * and formal array declarators : %d \n"
	  ,number_of_assumed_size_and_formal_array_declarators );
  user_log(" \n Number of formal array declarators : %d\n"
	  ,number_of_formal_array_declarators );
  user_log(" \n Number of 1 array declarators : %d\n"
	  ,number_of_one_array_declarators );
  user_log(" \n Number of * array declarators : %d\n"
	  , number_of_assumed_size_array_declarators  );
  user_log(" \n Number of array declarators : %d\n"
	  , number_of_array_declarators  );
  ifdebug(1)
    fprintf(stderr, " \n End  adn instrumentation for %s \n", module_name);
  debug_off();
  return TRUE;
}


