
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

/* Computes the optimal opcode for simdizing 'argc' statements of the
 * 'kind' operation, applied to the 'args' arguments
 */
static opcode get_optimal_opcode(int kind, int argc, list* args)
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
	       return 0;
	 }
	 if (width > max_width)
	    max_width = width;
      },
	  args[i]);
   }

   /* Based on the available implementations of the operation, decide
    * how many statements to pack together
    */
   best = NULL;
   for( l = get_operation(kind)->opcodes; 
	l != NIL;
	l = CDR(l) )
   {
      opcode oc = OPCODE(CAR(l));

      if ( (oc->subwordSize >= max_width) &&
	   (oc->vectorSize <= argc) &&
	   ((best == NULL) || (oc->vectorSize > best->vectorSize)) )
	 best = oc;
   }

   return best;
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

static statement make_exec_statement(opcode oc, list args)
{
   return call_to_statement(make_call(get_function_entity(oc->name),
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
      default:
	 break;
   }
   
   return new_ent;
}

#define MAX_PACK 16

typedef struct {
      int expr;
      list deps;
} _simd_statement_arg, * simd_statement_arg;

typedef struct {
      opcode op;
      int nbArgs;
      entity * vectors;
      simd_statement_arg arguments;    //2 dimensions array.
      statement s;
} _statement_info, *statement_info;
#define STATEMENT_INFO(x) ((statement_info)((x).p))

typedef struct {
      expression e;
      list places_available;
} _argument_info, * argument_info;

typedef struct {
      statement_info ssi;
      int vector;  //index of the vector
      int element; //index of the element in the vector
} _vector_element, * vector_element;
#define VECTOR_ELEMENT(x) ((vector_element)((x).p))

static argument_info arguments = NULL;
static int nbArguments = 0;
static int nbAllocatedArguments = 0;

static void reset_argument_info()
{
   nbArguments = 0;
}

//Get id for this "class" of expression
//If not, add the expression to the mapping.
static int get_argument_id(expression e)
{
   int id;
   argument_info ai;
   int i;

   //See if the expression has already been seen
   for(i=0; i<nbArguments; i++)
      if (same_expression_p(arguments[i].e, e))
	 return i;

   //Find an id for the operation
   id = nbArguments++;

   //Make room for the new operation if needed
   if (nbArguments > nbAllocatedArguments)
   {
      nbAllocatedArguments += 10;

      arguments = (argument_info)realloc((void*)arguments, 
					 sizeof(_argument_info)*nbAllocatedArguments);

      if (arguments == NULL)
      {
	 printf("Fatal error: could not allocate memory for arguments.\n");
	 exit(-1);
      }
   }

   //Initialize members
   ai = arguments + id;
   ai->e = e;
   ai->places_available = NIL;

   return id;
}

//Get info on argument with specified id
static argument_info get_argument_info(int id)
{
   return ((id>=0) && (id<nbArguments)) ?
      &arguments[id] : NULL;
}

static vector_element make_vector_element(statement_info ssi, int vector, int element)
{
   vector_element v = (vector_element)malloc(sizeof(_vector_element));
   
   v->ssi = ssi;
   v->vector = vector;
   v->element = element;

   return v;
}

static statement_info make_statement_info(statement s)
{
   statement_info ssi;

   /* allocate memory */
   ssi = (statement_info)malloc(sizeof(_statement_info));

   /* initialize members */
   ssi->op = NULL;
   ssi->nbArgs = 0;
   ssi->vectors = NULL;
   ssi->arguments = NULL;
   ssi->s = copy_statement(s);

   /* see if some expressions are modified.
    * If that is the case, invalidate the list of its available places 
    */
   

   return ssi;
}

static statement_info make_simd_statement_info(int kind, opcode oc, list* args)
{
   statement_info ssi;
   int i,j, nbargs;

   /* find out the number of arguments needed */
   nbargs = get_operation(kind)->nbArgs;

   /* allocate memory */
   ssi = (statement_info)malloc(sizeof(_statement_info));
   ssi->vectors = (entity *)malloc(sizeof(entity)*nbargs);
   ssi->arguments = (simd_statement_arg)malloc(sizeof(_simd_statement_arg) *
					       nbargs * oc->vectorSize);

   /* initialize members */
   ssi->op = oc;
   ssi->nbArgs = nbargs;
   ssi->s = NULL;

   /* create the simd vector entities */
   for(j=0; j<nbargs; j++)
      ssi->vectors[j] = make_new_simd_vector(oc->subwordSize,oc->vectorSize,TRUE);

   /* Fill the matrix of arguments */
   for(j=0; j<oc->vectorSize; j++)
   {
      list l = args[j];

      pips_assert("2", ssi->op);

      for(i=nbargs-1; i>=0; i--)
      {
	 argument_info ai;
	 int id;
	 simd_statement_arg ssa;

	 //Get the id of the argumet
	 id = get_argument_id(EXPRESSION(CAR(l)));
	 
	 //Store it in the argument's matrix
	 ssa = &(ssi->arguments[j + oc->vectorSize * i]);
	 ssa->expr = id;
	 l = CDR(l);

	 //Build the dependance tree
	 ai = get_argument_info(id);

	 if (i == nbargs-1)
	 {  //we write to this variable

	    //Free the list of places available. Those places are
	    //not relevant any more
	    gen_free_list(ai->places_available);
	    ai->places_available = NIL;
	 }
	 else
	 {  //we read this variable
	    list l;

	    //ssi depends on all the places where the variable was
	    //used before
	    for(l = ai->places_available; l != NIL; l = CDR(l))
	    {
	       vector_element pVe = VECTOR_ELEMENT(CAR(l));
	       
	       ssa->deps = CONS(VECTOR_ELEMENT, pVe, ssa->deps);
	    }
	 }

	 //Remember that this variable can be found here too
	 ai->places_available = CONS(VECTOR_ELEMENT,
				     make_vector_element(ssi, i, j),
				     ai->places_available);
      }
   }

   if (ssi->op == NULL)
   {
      printf("in the end....\n");
      exit(-1);
   }

   return ssi;
}

list make_simd_statements(list kinds, cons* first, cons* last)
{
   list i;
   int index;
   list args[MAX_PACK];
   int type;
   list instr; 
   list all_instr;

   if (first == last)
      return CONS(STATEMENT_INFO,
		  make_statement_info(STATEMENT(CAR(first))),
		  NIL);

   reset_argument_info();

   i = first;
   all_instr = CONS(STATEMENT_INFO, NULL, NIL);
   instr = all_instr;

   type = INT(CAR(kinds));
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
	 args[index] = m->args;
      }

      /* compute the opcode to use */
      oc = get_optimal_opcode(type, index, args);

      if (oc == NULL)
      {
	 /* No optimized opcode found... */
	 for( index = 0, j = i;
	      (index < MAX_PACK) && (j != CDR(last));
	      index++, j = CDR(j) )
	 {
	    CDR(instr) = CONS(STATEMENT_INFO, 
			      make_statement_info(STATEMENT(CAR(j))),
			      NIL);
	    instr = CDR(instr);
	 }
      }
      else
      {
	 /* update the pointer to the next statement to be processed */
	 for(index = 0; (index<oc->vectorSize) && (i!=CDR(last)); index++)
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

static statement generate_exec_statement(statement_info si)
{
   list args = NIL;
   int i;

   for(i = si->nbArgs-1; i >= 0; i--)
   {
      args = CONS(EXPRESSION,
		  entity_to_expression(si->vectors[i]),
		  args);
   }

   return make_exec_statement(si->op, args);
}

static statement generate_load_statement(statement_info si, int line)
{
   list args = NIL;
   int i;
   int offset = si->op->vectorSize * line;

   for(i = si->op->vectorSize-1; i >= 0; i--)
   {
      argument_info ai = get_argument_info(si->arguments[i + offset].expr);
      args = CONS(EXPRESSION,
		  copy_expression(ai->e),
		  args);
   }

   args = CONS(EXPRESSION,
	       entity_to_expression(si->vectors[line]),
	       args);

   return make_load_statement(si->op->vectorSize, args);
}

static statement generate_save_statement(statement_info si)
{
   list args = NIL;
   int i;
   int offset = si->op->vectorSize * (si->nbArgs-1);

   for(i = si->op->vectorSize-1; i >= 0; i--)
   {
      argument_info ai = get_argument_info(si->arguments[i + offset].expr);
      args = CONS(EXPRESSION,
		  copy_expression(ai->e),
		  args);
   }

   args = CONS(EXPRESSION,
	       entity_to_expression(si->vectors[si->nbArgs-1]),
	       args);

   return make_save_statement(si->op->vectorSize, args);
}

list generate_simd_code(list/* <statement_info> */ sil)
{
   list sl_begin; /* <statement> */
   list sl; /* <statement> */

   sl = sl_begin = CONS(STATEMENT, NULL, NIL);

   printf("************************\n"
	  "generating code:\n");
   for(; sil != NIL; sil=CDR(sil))
   {
      statement_info si = STATEMENT_INFO(CAR(sil));

      if (si->s != NULL)
      {
	 /* regular (non-SIMD) statement */
	 sl = CDR(sl) = CONS(STATEMENT, si->s, NIL);
	 print_statement(STATEMENT(CAR(sl)));
      }
      else
      {
	 /* SIMD statement (will generate more than one statement) */
	 int i;

	 printf("<there>");
	 
	 //First, the load statement(s)
	 for(i = 0; i < si->nbArgs-1; i++)
	 {
	    statement s = generate_load_statement(si, i);

	    if (s != NULL)
	    {
	       sl = CDR(sl) = CONS(STATEMENT, s, NIL);
	       print_statement(s);
	    }
	 }

	 printf("<then>");

	 //Then, the exec statement
	 sl = CDR(sl) = CONS(STATEMENT, generate_exec_statement(si), NIL);
	 print_statement(STATEMENT(CAR(sl)));

	 printf("<nowhere>");

	 //Finally, the save statement (always generated. It is up to 
	 //latter phases (USE-DEF elimination....) to remove it, if needed
	 sl = CDR(sl) = CONS(STATEMENT, generate_save_statement(si), NIL);
	 print_statement(STATEMENT(CAR(sl)));

	 printf("<done>");
      }
   }

   sl = CDR(sl_begin);
   CDR(sl_begin) = NIL;
   gen_free_list(sl_begin);
   
   return sl;
}
