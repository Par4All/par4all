
#include "genC.h"
#include "linear.h"
#include "ri.h"

#include "resources.h"

#include "misc.h"
#include "ri-util.h"
#include "pipsdbm.h"

#include "semantics.h"
#include "effects-generic.h"
#include "transformations.h"

#include "sac.h"

typedef struct {
      expression result;
      expression operand[2];
} statement_arguments;


/* Calc the number of statements we can pack, based on the
 * arguments size 
 */
static int calc_nb_pack(int kind, int argc, list* args)
{
   int max_width = 0;

   /* for these operations, it is enough to look at the size
    * of the result: we can truncate the higher order bits of
    * the input operands.
    */
   int i;

   for(i = 0; i < argc; i++)
   {
      int width;
      type t = entity_type(reference_variable(expression_reference(EXPRESSION(CAR(args[i])))));
      basic b;
	    
      switch(basic_tag(b = variable_basic(type_variable(t))))
      {
	 case is_basic_int: width = 8*basic_int(b); break;
	 case is_basic_float: width = 8*basic_float(b); break;
	 case is_basic_logical: width = 8*basic_logical(b); break;
	 default: /* HELP: what to do ? */
	    printf("unsuppported basic type\n");
	    return 0;
      }
      if (width > max_width)
	 max_width = width;
   }

   return argc;
}

static entity get_function_entity(string name)
{
   entity e = local_name_to_top_level_entity(name);
   if (entity_undefined_p(e))
      e = make_empty_subroutine(name);
   return e;
}

typedef struct {
      entity     e;
      int        nbDimensions;
      reference  IndexVariable;
      int        IndexOffset;
} reference_info;

static bool analyse_reference(reference r, reference_info* i)
{
   syntax s;

   i->e = reference_variable(r);
   i->nbDimensions = gen_length(reference_indices(r));

   if (i->nbDimensions == 0)
      return FALSE;

   s = expression_syntax(EXPRESSION(CAR(reference_indices(r))));
   switch(syntax_tag(s))
   {
      case is_syntax_call:
      {
	 call c = syntax_call(s);

	 /* this should be a reference + a constant int offset */
	 if (call_constant_p(c))
	 {
	    constant cn;

	    cn = value_constant(entity_initial(call_function(c)));

	    if (constant_int_p(cn))
	       i->IndexOffset = constant_int(cn);
	    else
	       return FALSE;

	    i->IndexVariable = reference_undefined;
	 }
	 else if (ENTITY_PLUS_P(call_function(c)))
	 {
	    cons * arg = call_arguments(c);
	    bool order; /* TRUE means reference+constant, 
			   FALSE means constant+reference */
	    syntax e;

	    if ((arg == NIL) || (CDR(arg) == NIL))
	       return FALSE;

	    e = expression_syntax(EXPRESSION(CAR(arg)));
	    switch(syntax_tag(e))
	    {
	       case is_syntax_call:
	       {
		  call cc = syntax_call(e);
		  if (!call_constant_p(cc))  //prevent non-constant call
		     return FALSE;

		  order = FALSE;
		  i->IndexOffset = constant_int(value_constant(entity_initial(call_function(cc))));
		  break;
	       }

	       case is_syntax_reference:
		  order = TRUE;
		  i->IndexVariable = syntax_reference(e);		 
		  break;
		  
	       default:
	       case is_syntax_range:
		  return FALSE;
	    }

	    e = expression_syntax(EXPRESSION(CAR(CDR(arg))));
	    switch(syntax_tag(e))
	    {
	       case is_syntax_call:
	       {
		  call cc = syntax_call(e);
		  if ( (order == FALSE) ||    //prevent constant+call
		       !call_constant_p(cc) ) //prevent reference+call
		     return FALSE;

		  i->IndexOffset = constant_int(value_constant(entity_initial(call_function(cc))));
		  break;
	       }

	       case is_syntax_reference:
		  if (order == TRUE)  //prevent reference+reference
		     return FALSE;
		  i->IndexVariable = syntax_reference(e); 
		  break;

	       default:
	       case is_syntax_range:
		  return FALSE;
	    }	    
	 }
	 else
	    return FALSE;
	 break;
      }
	 
      case is_syntax_reference:
	 i->IndexVariable = syntax_reference(s);
	 i->IndexOffset = 0;
	 break;
	    
      default:
	 return FALSE;
   }
   return TRUE;
}

static bool consecutive_refs_p(reference_info * firstRef, int lastOffset, reference_info * cRef)
{
   return ( same_entity_p(firstRef->e, cRef->e) &&
	    (firstRef->nbDimensions == cRef->nbDimensions) &&
	    (same_entity_p(reference_variable(firstRef->IndexVariable),
			   reference_variable(cRef->IndexVariable))) &&
	    (lastOffset + 1 == cRef->IndexOffset) );
}

static statement make_loadsave_statement(int argc, list args, bool isLoad)
{
   enum {
      CONSEC_REFS,
      CONSTANT,
      OTHER
   } argsType;
   const char funcNames[3][2][20] = {
      { "simd_save",          "simd_load"          },
      { "simd_constant_save", "simd_constant_load" },
      { "simd_generic_save",  "simd_generic_load"  } };
   reference_info firstRef;
   int lastOffset = 0;
   cons * argPtr;
   expression e;
   char functionName[30];
   long long unsigned int constantValue = 0;
   int constantShift = 0;
   int bitmask = (1 << (64 / argc)) - 1;

   /* the function should not be called with an empty arguments list */
   assert((argc > 1) && (args != NIL));

   /* first, find out if the arguments are:
    *    - consecutive references to the same array
    *    - all constant
    *    - or any other situation
    */

   /* classify according to the second element
   * (first one should be the SIMD vector) */
   e = EXPRESSION(CAR(CDR(args)));
   if (expression_constant_p(e))
   {
      unsigned int val;
      argsType = CONSTANT;

      if (!isLoad)
      {
	 printf("Error: should not save something into a constant expression!"
		"\nAborting...\n");
	 abort();
      }

      val = constant_int(value_constant(entity_initial(call_function(
	 syntax_call(expression_syntax(e))))));
      
      val &= bitmask; //mask bits that would be truncated
      constantValue = val;
      constantShift = (64 / argc);
   }
   else if ( (expression_reference_p(e)) &&
	     (analyse_reference(expression_reference(e), &firstRef)) )
   {
      lastOffset = firstRef.IndexOffset;
      argsType = CONSEC_REFS;
   }
   else
      argsType = OTHER;

   /* now verify the estimation on the first element is correct, and update
    * parameters needed later */
   for( argPtr = CDR(CDR(args)); argPtr != NIL; argPtr = CDR(argPtr) )
   {
      e = EXPRESSION(CAR(argPtr));

      if (argsType == OTHER)
	 break;
      else if (argsType == CONSTANT)
      {
	 if (expression_constant_p(e))
	 {
	    unsigned int val;
      
	    val = constant_int(value_constant(entity_initial(call_function(
	       syntax_call(expression_syntax(e))))));

	    val &= bitmask; //mask bits that would be truncated
	    constantValue |= (((unsigned long long)val) << constantShift);
	    constantShift += (64 / argc);
	 }
	 else
	 {
	    argsType = OTHER;
	    break;
	 }
      }
      else if (argsType == CONSEC_REFS)
      {
	 reference_info cRef;

	 if ( (expression_reference_p(e)) &&
	      (analyse_reference(expression_reference(e), &cRef)) &&
	      (consecutive_refs_p(&firstRef, lastOffset, &cRef)) )
	 {
	    lastOffset = cRef.IndexOffset;
	 }
	 else
	 {
	    printf("*********\nfailure due to expr:\n");
	    print_expression(e);
	    argsType = OTHER;
	    break;
	 }
      }
   }

   /* Now that the analyze is done, we can generate an "optimized"
    * load instruction.
    */
   switch(argsType)
   {
      case CONSEC_REFS:
      {
	 /* build a new list of arguments */
	 args = gen_make_list( expression_domain, 
			       EXPRESSION(CAR(args)),
			       reference_to_expression(
				  make_reference(firstRef.e, NIL)),
			       reference_to_expression(
				  firstRef.IndexVariable),
			       make_integer_constant_expression(
				  firstRef.IndexOffset),
			       NULL);
	 break;
      }

      case CONSTANT:
      {
	 /* build a new list of arguments */
	 args = gen_make_list( expression_domain,
			       EXPRESSION(CAR(args)),
			       make_integer_constant_expression(
				  (int)(constantValue&0xFFFFFFFF)),
			       make_integer_constant_expression(
				  (int)(constantValue>>32)),
			       NULL);
	 break;
      }

      case OTHER:
      default:
	 break;
   }
   sprintf(functionName, "%s%i", funcNames[argsType][isLoad], argc);
   return call_to_statement(make_call(get_function_entity(functionName), 
				      args));

}

static statement make_load_statement(int argc, list args)
{
   return make_loadsave_statement(argc, args, TRUE);
}

static statement make_save_statement(int argc, list args)
{
   return make_loadsave_statement(argc, args, FALSE);
}

static statement make_exec_statement(int kind, int argc, list args)
{
   char* name;

   name = get_operation_opcode(kind, argc, 64/argc);
   if (name == NULL)
   {
      printf("ERROR: no matching operator !\n");
      name = "UnknownSIMDOperator";
   }
   return call_to_statement(make_call(get_function_entity(name),
				      args));
}

static entity make_new_simd_vector(int itemSize, int nbItems, bool isInt)
{
   extern list integer_entities, real_entities, double_entities;
  
   basic simdVector = isInt ? 
      make_basic_int(itemSize/8) :
      make_basic_float(itemSize/8);
   entity new_ent, mod_ent;
   char prefix[5], *name, *num;
   static int number = 0;
   entity dynamic_area;

   /* build the variable prefix code, which is in fact also the type */
   prefix[0] = 'v';
   prefix[1] = '0'+nbItems;
   switch(itemSize)
   {
      case 8:  prefix[2] = 'q'; break;
      case 16: prefix[2] = 'h'; break;
      case 32: prefix[2] = 's'; break;
      case 64: prefix[2] = 'd'; break;
   }
   if (isInt)
      prefix[3] = 'i';
   else
      prefix[3] = 'f';
   prefix[5] = 0;

   mod_ent = get_current_module_entity();
   num = (char*) malloc(32);
   sprintf(num, "_vec%i", number++);
   name = strdup(concatenate(entity_local_name(mod_ent),
			     MODULE_SEP_STRING, prefix, num, (char *) NULL));
   
   new_ent = make_entity(name,
			 make_type(is_type_variable,
				   make_variable(simdVector, 
						 CONS(DIMENSION, 
						      make_dimension(int_expr(0),
								     int_expr(nbItems-1)), 
						      NIL))),
			 storage_undefined,
			 make_value(is_value_unknown, UU));
   dynamic_area = global_name_to_entity(module_local_name(mod_ent),
					DYNAMIC_AREA_LOCAL_NAME);
   entity_storage(new_ent) = make_storage(is_storage_ram,
					  make_ram(mod_ent,
						   dynamic_area,
						   add_variable_to_area(dynamic_area, new_ent),
						   NIL));
   add_variable_declaration_to_module(mod_ent, new_ent);
   
   /* The new entity is stored in the list of entities of the same type. */
   switch(basic_tag(simdVector))
   {
      case is_basic_int:
      {
	 integer_entities = CONS(ENTITY, new_ent, integer_entities);
	 break;
      }
      case is_basic_float:
      {
	 if(basic_float(simdVector) == DOUBLE_PRECISION_SIZE)
	    double_entities = CONS(ENTITY, new_ent, double_entities);
	 else
	    real_entities = CONS(ENTITY, new_ent, real_entities);
	 break;
      }
      default:break;
   }
   
   return new_ent;
}

static void pack_instructions(int kind, int nb_pack, list* args, cons** instr)
{
   expression result;
   expression * operands;
   cons * argList;
   int j, nbargs;
   cons * iargs;

   /* find out the number of arguments needed */
   nbargs = get_operation(kind)->nbArgs-1;

   /* allocated memory for the operands */
   operands = (expression*)malloc(sizeof(expression) * nbargs);

   /* build the variables */
   for(j=nbargs-1; j>=0; j--)
      operands[j] = entity_to_expression(make_new_simd_vector(64/nb_pack,nb_pack,TRUE));
   result = entity_to_expression(make_new_simd_vector(64/nb_pack,nb_pack,TRUE));

   /* make the save statement(s) */
   argList = CONS(EXPRESSION, result, NIL);
   iargs = argList;
   for( j=0; j<nb_pack; j++)
   {
      CDR(iargs) = CONS(EXPRESSION, 
			copy_expression(EXPRESSION(CAR(args[j]))), 
			NIL);
      iargs = CDR(iargs);
   }
   *instr = CONS( STATEMENT, 
		  make_save_statement(nb_pack, argList),
		  *instr);

   /* make the calculation statement(s) */
   argList = CONS(EXPRESSION, result, NIL);
   iargs = argList;
   for(j=0; j<nbargs; j++)
   {
      CDR(iargs) = CONS(EXPRESSION, operands[j], NIL);
      iargs = CDR(iargs);
   }
   *instr = CONS( STATEMENT,
		  make_exec_statement(kind, nb_pack, argList),
		  *instr);

   /* make the load instruction(s) */
   {
   list * pArgs = (list*)malloc(nb_pack*sizeof(list));
   int i;

   for(i=0; i<nb_pack; i++)
      pArgs[i] = CDR(args[i]);

   for( j=0; j<nbargs; j++)
   {
      argList = CONS(EXPRESSION, operands[j], NIL);
      iargs = argList;
      for( i=0; i<nb_pack; i++)
      {
	 CDR(iargs) = CONS( EXPRESSION, 
			    copy_expression(EXPRESSION(CAR(pArgs[i]))), 
			    NIL);
	 iargs = CDR(iargs);

	 pArgs[i] = CDR(pArgs[i]);
      }

      *instr = CONS( STATEMENT,
		     make_load_statement(nb_pack, argList),
		     *instr);
   }

   free(pArgs);
   }

   free(operands);
}

#define MAX_PACK 8

cons* make_simd_statements(list kinds, cons* first, cons* last)
{
   cons * i;
   int index;
   list args[MAX_PACK];
   int type;
   cons * instr; 
   cons * all_instr;

   i = first;
   all_instr = NIL;

   type = INT(CAR(kinds));
   while(i != CDR(last))
   {
      int nb_pack;
      cons * j;

      /* get the variables */
      for( index = 0, j = i;
	   (index < MAX_PACK) && (j != CDR(last));
	   index++, j = CDR(j) )
      {
	 match m = get_statement_match_of_kind(STATEMENT(CAR(j)), type);
	 args[index] = m->args;
      }

      /* compute number of instructions to pack */
      nb_pack = calc_nb_pack(type, index, args);

      /* update the pointer to the next statement to be processed */
      for(index = 0; (index<nb_pack) && (i!=CDR(last)); index++)
	 i = CDR(i);

      /* generate the instructions */
      instr = NIL;
      pack_instructions(type, nb_pack, args, &instr);

      /* insert the new statements */
      if (all_instr == NIL)
	 all_instr = instr;
      else
	 all_instr = gen_concatenate(all_instr, instr);
   }

   return all_instr;
}
