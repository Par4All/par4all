
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


#define MAX_PACK 16

static argumentInfo* arguments = NULL;
static int nbArguments = 0;
static int nbAllocatedArguments = 0;

entity vectorElement_vector(vectorElement ve)
{
   return simdStatementInfo_vectors(vectorElement_statement(ve))[vectorElement_vectorIndex(ve)];
}

int vectorElement_vectorLength(vectorElement ve)
{
   return opcode_vectorSize(simdStatementInfo_opcode(vectorElement_statement(ve)));
}

int vectorElement_subwordSize(vectorElement ve)
{
   return opcode_subwordSize(simdStatementInfo_opcode(vectorElement_statement(ve)));
}

/* Computes the optimal opcode for simdizing 'argc' statements of the
 * 'kind' operation, applied to the 'args' arguments
 */
static opcode get_optimal_opcode(opcodeClass kind, int argc, list* args)
{
   int max_width = 0;
   int i;
   opcode best;
   list l;

   //Find out the maximum width of all variables used
   for(i = 0; i < argc; i++)
   {
      MAP(EXPRESSION,
	  arg,
      {
	 int width;
	 type t;
	 basic b;

	 if (!expression_reference_p(arg))
	    continue;

	 t = entity_type(reference_variable(expression_reference(arg)));
	    
	 switch(basic_tag(b = variable_basic(type_variable(t))))
	 {
	    case is_basic_int: width = 8*basic_int(b); break;
	    case is_basic_float: width = 8*basic_float(b); break;
	    case is_basic_logical: width = 8*basic_logical(b); break;
	    default: /* HELP: what to do ? */
	       printf("unsuppported basic type\n");
	       return opcode_undefined;
	 }
	 if (width > max_width)
	    max_width = width;
      },
	  args[i]);
   }

   /* Based on the available implementations of the operation, decide
    * how many statements to pack together
    */
   best = opcode_undefined;
   for( l = opcodeClass_opcodes(kind);
	l != NIL;
	l = CDR(l) )
   {
      opcode oc = OPCODE(CAR(l));

      if ( (opcode_subwordSize(oc) >= max_width) &&
	   (opcode_vectorSize(oc) <= argc) &&
	   ((best == opcode_undefined) || 
	    (opcode_vectorSize(oc) > opcode_vectorSize(best))) )
	 best = oc;
   }

   return best;
}

entity get_function_entity(string name)
{
   entity e = local_name_to_top_level_entity(name);
   if (entity_undefined_p(e))
      e = make_empty_subroutine(name);
   return e;
}

static bool analyse_reference(reference r, referenceInfo i)
{
   syntax s;

   referenceInfo_entity(i) = reference_variable(r);
   referenceInfo_nbDimensions(i) = gen_length(reference_indices(r));

   if (referenceInfo_nbDimensions(i) == 0)
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
	       referenceInfo_offset(i) = constant_int(cn);
	    else
	       return FALSE;

	    referenceInfo_index(i) = reference_undefined;
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
		  referenceInfo_offset(i) = constant_int(value_constant(entity_initial(call_function(cc))));
		  break;
	       }

	       case is_syntax_reference:
		  order = TRUE;
		  referenceInfo_index(i) = syntax_reference(e);		 
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

		  referenceInfo_offset(i) = constant_int(value_constant(entity_initial(call_function(cc))));
		  break;
	       }

	       case is_syntax_reference:
		  if (order == TRUE)  //prevent reference+reference
		     return FALSE;
		  referenceInfo_index(i) = syntax_reference(e); 
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
	 referenceInfo_index(i) = syntax_reference(s);
	 referenceInfo_offset(i) = 0;
	 break;
	    
      default:
	 return FALSE;
   }
   return TRUE;
}

static bool consecutive_refs_p(referenceInfo firstRef, int lastOffset, referenceInfo cRef)
{
   return ( same_entity_p(referenceInfo_entity(firstRef), 
			  referenceInfo_entity(cRef)) &&
	    (referenceInfo_nbDimensions(firstRef) == referenceInfo_nbDimensions(cRef)) &&
	    (same_entity_p(reference_variable(referenceInfo_index(firstRef)),
			   reference_variable(referenceInfo_index(cRef)))) &&
	    (lastOffset + 1 == referenceInfo_offset(cRef)) );
}

static referenceInfo make_empty_referenceInfo()
{
   return make_referenceInfo(entity_undefined,
			     0,
			     reference_undefined,
			     0);
}

static void free_empty_referenceInfo(referenceInfo ri)
{
   referenceInfo_entity(ri) = entity_undefined;
   referenceInfo_index(ri) = reference_undefined;
   free_referenceInfo(ri);
}

static statement make_loadsave_statement(int argc, list args, bool isLoad)
{
   enum {
      CONSEC_REFS,
      CONSTANT,
      OTHER
   } argsType;
   const char funcNames[3][2][20] = {
      { "SIMD_SAVE",          "SIMD_LOAD"          },
      { "SIMD_CONSTANT_SAVE", "SIMD_CONSTANT_LOAD" },
      { "SIMD_GENERIC_SAVE",  "SIMD_GENERIC_LOAD"  } };
   referenceInfo firstRef = make_empty_referenceInfo();
   referenceInfo cRef = make_empty_referenceInfo();
   int lastOffset = 0;
   cons * argPtr;
   expression e;
   char functionName[30];
   long long unsigned int constantValue = 0;
   int constantShift = 0;
   int bitmask;

   /* the function should not be called with an empty arguments list */
   assert((argc > 1) && (args != NIL));

   /* Compute the bitmask with the formula:
    *    bitmask = (1 << (64 / argc)) - 1
    * There is a bug when argc = 2 on SPARC, thus we do it in two times,
    * in order to never shift by 32 bits at a time
    */
   bitmask = 1;
   bitmask <<= ((64/argc + 1) >> 1);
   bitmask <<= ((64/argc) >> 1);
   bitmask --;

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
	     (analyse_reference(expression_reference(e), firstRef)) )
   {
      lastOffset = referenceInfo_offset(firstRef);
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
	 if ( (expression_reference_p(e)) &&
	      (analyse_reference(expression_reference(e), cRef)) &&
	      (consecutive_refs_p(firstRef, lastOffset, cRef)) )
	 {
	    lastOffset = referenceInfo_offset(cRef);
	 }
	 else
	 {
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
			       entity_to_expression(
				  referenceInfo_entity(firstRef)),
			       reference_to_expression(
				  referenceInfo_index(firstRef)),
			       make_integer_constant_expression(
				  referenceInfo_offset(firstRef)),
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

   free_empty_referenceInfo(firstRef);
   free_empty_referenceInfo(cRef);

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

static statement make_exec_statement(opcode oc, list args)
{
   return call_to_statement(make_call(get_function_entity(opcode_name(oc)),
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
						      NIL),
						 NIL)),
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
      default:
	 break;
   }
   
   return new_ent;
}

void reset_argument_info()
{
   nbArguments = 0;
}

//Get id for this "class" of expression
//If not, add the expression to the mapping.
static int get_argument_id(expression e)
{
   int id;
   int i;

   //See if the expression has already been seen
   for(i=0; i<nbArguments; i++)
      if (same_expression_p(argumentInfo_expression(arguments[i]), e))
	 return i;

   //Find an id for the operation
   id = nbArguments++;

   //Make room for the new operation if needed
   if (nbArguments > nbAllocatedArguments)
   {
      nbAllocatedArguments += 10;

      arguments = (argumentInfo*)realloc((void*)arguments, 
					 sizeof(argumentInfo)*nbAllocatedArguments);

      if (arguments == NULL)
      {
	 printf("Fatal error: could not allocate memory for arguments.\n");
	 exit(-1);
      }
   }

   //Initialize members
   arguments[id] = make_argumentInfo(e, NIL);

   return id;
}

//Get info on argument with specified id
static argumentInfo get_argument_info(int id)
{
   return ((id>=0) && (id<nbArguments)) ?
      arguments[id] : argumentInfo_undefined;
}

static statementInfo make_nonsimd_statement_info(statement s)
{
   statementInfo ssi;
   int i;

   ssi = make_statementInfo_nonsimd(s);

   /* see if some expressions are modified.
    * If that is the case, invalidate the list of their available places 
    */
   for(i=0; i<nbArguments; i++)
   {
      //for now, reset all...
      gen_free_list(argumentInfo_placesAvailable(arguments[i]));
      argumentInfo_placesAvailable(arguments[i]) = NIL;
   }

   return ssi;
}

static vectorElement make_vector_element(simdStatementInfo ssi, int i, int j)
{
   return make_vectorElement(ssi, i, j);
}

static vectorElement copy_vector_element(vectorElement ve)
{
   return make_vectorElement(vectorElement_statement(ve),
			     vectorElement_vectorIndex(ve),
			     vectorElement_element(ve));
}

static statementInfo make_simd_statement_info(opcodeClass kind, opcode oc, list* args)
{
   statementInfo si;
   simdStatementInfo ssi;
   int i,j, nbargs;

   /* find out the number of arguments needed */
   nbargs = opcodeClass_nbArgs(kind);

   /* allocate memory */
   ssi = make_simdStatementInfo(oc, 
				nbargs,
				(entity *)malloc(sizeof(entity)*nbargs),
				(statementArgument*)malloc(
				   sizeof(statementArgument) *
				   nbargs * 
				   opcode_vectorSize(oc)));
   si = make_statementInfo_simd(ssi);

   /* create the simd vector entities */
   for(j=0; j<nbargs; j++)
      simdStatementInfo_vectors(ssi)[j] = 
	 make_new_simd_vector(opcode_subwordSize(oc),
			      opcode_vectorSize(oc),
			      TRUE);

   /* Fill the matrix of arguments */
   for(j=0; j<opcode_vectorSize(oc); j++)
   {
      list l = args[j];

      for(i=nbargs-1; i>=0; i--)
      {
	 argumentInfo ai;
	 statementArgument ssa;
	 expression e = EXPRESSION(CAR(l));
	 
	 //Store it in the argument's matrix
	 ssa = make_statementArgument(e, NIL);
	 simdStatementInfo_arguments(ssi)[j + opcode_vectorSize(oc) * i] = ssa;
	 
	 l = CDR(l);

	 //Get the id of the argumet
	 //Build the dependance tree
	 ai = get_argument_info(get_argument_id(e));

	 if (i == nbargs-1)
	 {  //we write to this variable

	    //Free the list of places available. Those places are
	    //not relevant any more
	    gen_free_list(argumentInfo_placesAvailable(ai));
	    argumentInfo_placesAvailable(ai) = NIL;
	 }
	 else
	 {  //we read this variable

	    //ssa depends on all the places where the expression was
	    //used before
	    statementArgument_dependances(ssa) = 
	       gen_copy_seq(argumentInfo_placesAvailable(ai));
	 }

	 //Remember that this variable can be found here too
	 argumentInfo_placesAvailable(ai) = 
	    CONS(VECTOR_ELEMENT,
		 make_vector_element(ssi, i, j),
		 argumentInfo_placesAvailable(ai));
      }
   }

   return si;
}

list make_simd_statements(list kinds, cons* first, cons* last)
{
   list i;
   int index;
   list args[MAX_PACK];
   opcodeClass type;
   list instr; 
   list all_instr;

   if (first == last)
      return CONS(STATEMENT_INFO,
		  make_nonsimd_statement_info(STATEMENT(CAR(first))),
		  NIL);

   i = first;
   all_instr = CONS(STATEMENT_INFO, NULL, NIL);
   instr = all_instr;

   type = OPCODECLASS(CAR(kinds));
   while(i != CDR(last))
   {
      opcode oc;
      list j;

      /* get the variables */
      for( index = 0, j = i;
	   (index < MAX_PACK) && (j != CDR(last));
	   index++, j = CDR(j) )
      {
	 match m = get_statement_match_of_kind(STATEMENT(CAR(j)), type);
	 args[index] = match_args(m);
      }

      /* compute the opcode to use */
      oc = get_optimal_opcode(type, index, args);

      if (oc == opcode_undefined)
      {
	 /* No optimized opcode found... */
	 for( index = 0, j = i;
	      (index < MAX_PACK) && (j != CDR(last));
	      index++, j = CDR(j) )
	 {
	    CDR(instr) = CONS(STATEMENT_INFO, 
			      make_nonsimd_statement_info(STATEMENT(CAR(j))),
			      NIL);
	    instr = CDR(instr);
	 }
      }
      else
      {
	 /* update the pointer to the next statement to be processed */
	 for(index = 0; 
             (index<opcode_vectorSize(oc)) && (i!=CDR(last)); 
             index++)
	    i = CDR(i);

         /* generate the statement information */
	 CDR(instr) = CONS(STATEMENT_INFO, 
			   make_simd_statement_info(type, oc, args),
			   NIL);
	 instr = CDR(instr);
      }
   }

   instr = CDR(all_instr);
   CDR(all_instr) = NIL;
   gen_free_list(all_instr);

   return instr;
}

static statement generate_exec_statement(simdStatementInfo si)
{
   list args = NIL;
   int i;

   for(i = 0; i < simdStatementInfo_nbArgs(si); i++)
   {
      args = CONS(EXPRESSION,
		  entity_to_expression(simdStatementInfo_vectors(si)[i]),
		  args);
   }

   return make_exec_statement(simdStatementInfo_opcode(si), args);
}

static list merge_available_places(list l1, list l2, int element)
{
   list i;
   list j;

   list res = NIL;

   for(i = l1; i != NIL; i = CDR(i))
   {
      for(j = l2; j != NIL; j = CDR(j))
      {
	 vectorElement ei = VECTORELEMENT(CAR(i));
	 vectorElement ej = VECTORELEMENT(CAR(j));

	 if (vectorElement_vector(ei) == vectorElement_vector(ej))
	 {
	    if (element < 0)
	    {
	       vectorElement ve = copy_vector_element(ej);

	       //quite a hack here: this vectorElement represents in fact an 
	       //aggregate entity x int, where the int is the order param
	       //to be used for shuffle
	       vectorElement_element(ve) = 
		  vectorElement_element(ej) | 
		  (vectorElement_element(ei) << (-2*element));

	       res = CONS(VECTORELEMENT, ve, res);
	    }
	    else if (vectorElement_element(ei) == element)
	       res = CONS(VECTORELEMENT, ej, res);
	 }
      }
   }

   return res;
}

static statement make_shuffle_statement(entity dest, entity src, int order)
{
   list args = gen_make_list(expression_domain,
			     entity_to_expression(dest),
			     entity_to_expression(src),
			     make_integer_constant_expression(order),
			     NULL);
   return call_to_statement(make_call(get_function_entity("PSHUFW"),
				      args));
}

static statement generate_load_statement(simdStatementInfo si, int line)
{
   list args = NIL;
   int i;
   int offset = line * opcode_vectorSize(simdStatementInfo_opcode(si));
   list sourcesCopy = NIL;
   list sourcesShuffle = NIL;

   //try to see if the arguments have not already been loaded
   MAP(VECTORELEMENT,
       ve,
   {
      if (vectorElement_element(ve) == 0)
	 sourcesCopy = CONS(VECTORELEMENT, ve, sourcesCopy);
      if ( (vectorElement_subwordSize(ve) == 16) &&
	   (vectorElement_vectorLength(ve) == 4) )
      {
	 vectorElement e = copy_vector_element(ve);
	 sourcesShuffle = CONS(VECTORELEMENT, e, sourcesShuffle);
      }
   },
       statementArgument_dependances(simdStatementInfo_arguments(si)[offset]));

   for(i = 1; 
       (i<opcode_vectorSize(simdStatementInfo_opcode(si))) && 
	  ((sourcesShuffle!=NIL) || (sourcesCopy!=NIL)); 
       i++)
   {
      list new_sources;

      //update the list of places where the copy can be found
      new_sources = merge_available_places(
	 statementArgument_dependances(simdStatementInfo_arguments(si)[i + offset]), 
	 sourcesCopy, 
	 i);

      gen_free_list(sourcesCopy);
      sourcesCopy = new_sources;

      //update the list of places where shuffled copy can be found
      new_sources = merge_available_places(
	 statementArgument_dependances(simdStatementInfo_arguments(si)[i + offset]), 
	 sourcesShuffle,
	 -i);

      gen_free_list(sourcesShuffle); //we should free the elements too (but not recusively, else we would free the simdStatementInfo...)
      sourcesShuffle = new_sources;
   }

   //Best case is we already have the same thing in another register
   if (sourcesCopy != NIL)
   {
      vectorElement vec = VECTORELEMENT(CAR(sourcesCopy));
      simdStatementInfo_vectors(si)[line] = vectorElement_vector(vec);
      return statement_undefined;
   }
   //Else, maybe we can use a shuffle instruction
   else if (sourcesShuffle != NIL)
   {
      vectorElement ve = VECTORELEMENT(CAR(sourcesShuffle));
      return make_shuffle_statement(simdStatementInfo_vectors(si)[line],
				    vectorElement_vector(ve),
				    vectorElement_element(ve));
   }
   //Only choice left is to load from memory
   else
   {
      //Build the arguments list
      for(i = opcode_vectorSize(simdStatementInfo_opcode(si))-1; 
	  i >= 0; 
	  i--)
      {
	 args = CONS(EXPRESSION,
		     copy_expression(statementArgument_expression(simdStatementInfo_arguments(si)[i + offset])),
		  args);
      }
      args = CONS(EXPRESSION,
		  entity_to_expression(simdStatementInfo_vectors(si)[line]),
		  args);
      
      //Make a load statement
      return make_load_statement(
	 opcode_vectorSize(simdStatementInfo_opcode(si)), 
	 args);
   }
}
   
static statement generate_save_statement(simdStatementInfo si)
{
   list args = NIL;
   int i;
   int offset = opcode_vectorSize(simdStatementInfo_opcode(si)) * 
      (simdStatementInfo_nbArgs(si)-1);

   for(i = opcode_vectorSize(simdStatementInfo_opcode(si))-1; 
       i >= 0; 
       i--)
   {
      args = CONS(EXPRESSION,
		  copy_expression(
		     statementArgument_expression(simdStatementInfo_arguments(si)[i + offset])),
		  args);
   }

   args = CONS(EXPRESSION,
	       entity_to_expression(simdStatementInfo_vectors(si)[simdStatementInfo_nbArgs(si)-1]),
	       args);

   return make_save_statement(opcode_vectorSize(simdStatementInfo_opcode(si)),
			      args);
}

list generate_simd_code(list/* <statementInfo> */ sil)
{
   list sl_begin; /* <statement> */
   list sl; /* <statement> */

   sl = sl_begin = CONS(STATEMENT, NULL, NIL);

   for(; sil != NIL; sil=CDR(sil))
   {
      statementInfo si = STATEMENTINFO(CAR(sil));

      if (statementInfo_nonsimd_p(si))
      {
	 /* regular (non-SIMD) statement */
	 sl = CDR(sl) = CONS(STATEMENT, statementInfo_nonsimd(si), NIL);
      }
      else
      {
	 /* SIMD statement (will generate more than one statement) */
	 int i;
	 simdStatementInfo ssi = statementInfo_simd(si);

	 //First, the load statement(s)
	 for(i = 0; i < simdStatementInfo_nbArgs(ssi)-1; i++)
	 {
	    statement s = generate_load_statement(ssi, i);

	    if (s != statement_undefined)
	       sl = CDR(sl) = CONS(STATEMENT, s, NIL);
	 }

	 //Then, the exec statement
	 sl = CDR(sl) = CONS(STATEMENT, generate_exec_statement(ssi), NIL);

	 //Finally, the save statement (always generated. It is up to 
	 //latter phases (USE-DEF elimination....) to remove it, if needed
	 sl = CDR(sl) = CONS(STATEMENT, generate_save_statement(ssi), NIL);
      }
   }

   sl = CDR(sl_begin);
   CDR(sl_begin) = NIL;
   gen_free_list(sl_begin);
   
   return sl;
}
