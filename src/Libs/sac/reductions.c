
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

#include "reductions_private.h"
#include "reductions.h"
#include "sac.h"

#include "control.h"

/* START of PIPSMAKE.RC generated aprt */

/*
  reductioInfo = persistant reduction x count:int x persistant vector:entity
 */
typedef struct {
      reduction _reductionInfo_reduction_;
      int _reductionInfo_count_;
      entity _reductionInfo_vector_;
} _reductionInfo_, * reductionInfo;
#define REDUCTIONINFO(x) ((reductionInfo)((x).p))
#define REDUCTIONINFO_TYPE reductionInfo
#define reductionInfo_reduction(x) (x->_reductionInfo_reduction_)
#define reductionInfo_count(x) (x->_reductionInfo_count_)
#define reductionInfo_vector(x) (x->_reductionInfo_vector_)

static reductionInfo make_reductionInfo(reduction r, int c, entity v)
{
   reductionInfo ri = (reductionInfo)malloc(sizeof(_reductionInfo_));
   reductionInfo_reduction(ri) = r;
   reductionInfo_count(ri) = c;
   reductionInfo_vector(ri) = v;
   return ri;
}

/* END of PIPSMAKE.RC generated aprt */


#define reduction_basic(x) (variable_basic(type_variable(entity_type(reference_variable(reduction_reference(x))))))
     
static entity make_reduction_vector_entity(reduction r)
{
   extern list integer_entities, real_entities, double_entities;
   entity var = reference_variable(reduction_reference(r));
   basic base = copy_basic(variable_basic(type_variable(entity_type(var))));
   entity new_ent, mod_ent;
   char *name, num[32];
   static int number = 0;
   entity dynamic_area;

   mod_ent = get_current_module_entity();
   sprintf(num, "_RED%i", number++);
   name = strdup(concatenate(entity_local_name(mod_ent),
			     MODULE_SEP_STRING, 
			     entity_local_name(var), 
			     num, 
			     (char *) NULL));
   
   new_ent = make_entity(name,
			 make_type(is_type_variable,
				   make_variable(base, 
						 CONS(DIMENSION, 
						      make_dimension(int_expr(0),
								     int_expr(0)), 
						      NIL),
						 NIL)),
			 storage_undefined,
			 make_value(is_value_unknown, UU));
   dynamic_area = global_name_to_entity(module_local_name(mod_ent),
					DYNAMIC_AREA_LOCAL_NAME);
   entity_storage(new_ent) = make_storage(is_storage_ram,
					  make_ram(mod_ent,
						   dynamic_area,
						   add_variable_to_area(dynamic_area,
									new_ent),
						   NIL));
   add_variable_declaration_to_module(mod_ent, new_ent);
   
   /* The new entity is stored in the list of entities of the same type. */
   switch(basic_tag(base))
   {
      case is_basic_int:
      {
	 integer_entities = CONS(ENTITY, new_ent, integer_entities);
	 break;
      }
      case is_basic_float:
      {
	 if(basic_float(base) == DOUBLE_PRECISION_SIZE)
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

static reductions get_reductions(statement s)
{
   //get the reduction associated with this statement, if any
   return load_cumulated_reductions(s);
}

static bool same_reduction_p(reduction r1, reduction r2)
{
   return ( (reference_equal_p(reduction_reference(r1),reduction_reference(r2))) &&
	    (reduction_operator_tag(reduction_op(r1)) == reduction_operator_tag(reduction_op(r2))) );
}

static reductionInfo add_reduction(list* reds, reduction r)
{
   list l;
   list prev = NIL;
   reductionInfo ri;

   //See if the reduction has already been encountered
   for(l = *reds; l != NIL; l = CDR(l))
   {
      ri = REDUCTIONINFO(CAR(l));

      if (same_reduction_p(r, reductionInfo_reduction(ri)))
      {
	 //The reduction has already been encountered: update the coun
	 reductionInfo_count(ri)++;

	 free_expression(dimension_upper(DIMENSION(CAR((variable_dimensions(type_variable(entity_type(reductionInfo_vector(ri)))))))));
	 dimension_upper(DIMENSION(CAR((variable_dimensions(type_variable(entity_type(reductionInfo_vector(ri)))))))) = int_expr(reductionInfo_count(ri)-1);

	 return ri; 
      }

      prev = l;
   }

   //First time we see this reduction: initialize a reductionInfo structure
   ri = make_reductionInfo(r, 1, make_reduction_vector_entity(r));

   //Add to the list of reductions encountered
   if (prev == NIL) // same as (*reductions == NIL)
      *reds = CONS(REDUCTIONINFO, ri, NIL);
   else
      CDR(prev) = CONS(REDUCTIONINFO, ri, NIL);

   return ri;
}

static void rename_reduction_rewrite(expression e, reductionInfo ri)
{
   syntax s = expression_syntax(e);

   if (!syntax_reference_p(s) ||
       !reference_equal_p(syntax_reference(s), reduction_reference(reductionInfo_reduction(ri))))
      return;

   syntax_reference(s) = make_reference(reductionInfo_vector(ri),
					CONS(EXPRESSION,
					     int_expr(reductionInfo_count(ri)-1),
					     NIL));
   free_normalized(expression_normalized(e));
   expression_normalized(e) = normalized_undefined;
}

static void rename_reduction_ref(statement s, reductionInfo ri)
{
   //recursively replace `reduction_reference(reductionInfo_reduction(ri))'
   //with `reductionInfo_vector(ri)[reductionInfo_count(ri)-1]' in `s'
   
   gen_context_recurse(s, ri, expression_domain, gen_true, rename_reduction_rewrite);
}

static void rename_statement_reductions(statement s, list * reds)
{
   MAP(REDUCTION,
       r,
       rename_reduction_ref(s, add_reduction(reds, r)),
       reductions_list(get_reductions(s)));
}

static expression make_maxval_expression(basic b)
{
   switch(basic_tag(b))
   {
      case is_basic_float:
	 return expression_undefined;

      case is_basic_int:
      {
	 long long max = (2 << (basic_int(b) - 2)) - 1;
	 return make_integer_constant_expression(max);
      }

      default:
	 return expression_undefined;
   }
}

static expression make_minval_expression(basic b)
{
   switch(basic_tag(b))
   {
      case is_basic_float:
	 return expression_undefined;

      case is_basic_int:
      {
	 long long min = -(2 << (basic_int(b) - 2));
	 return make_integer_constant_expression(min);
      }

      default:
	 return expression_undefined;
   }
}

static expression make_0val_expression(basic b)
{
   switch(basic_tag(b))
   {
      case is_basic_float:
	 return expression_undefined;

      case is_basic_int:
	 return make_integer_constant_expression(0);

      default:
	 return expression_undefined;
   }
}

static expression make_1val_expression(basic b)
{
   switch(basic_tag(b))
   {
      case is_basic_float:
	 return expression_undefined;

      case is_basic_int:
	 return make_integer_constant_expression(1);

      default:
	 return expression_undefined;
   }
}
static statement generate_prelude(reductionInfo ri)
{
   expression initval;
   list prelude = NIL;
   int i;

   switch(reduction_operator_tag(reduction_op(reductionInfo_reduction(ri))))
   {
      default:
      case is_reduction_operator_none:
	 return statement_undefined;
	 break;

      case is_reduction_operator_min:
	 initval = make_maxval_expression(reduction_basic(reductionInfo_reduction(ri)));
	 break;

      case is_reduction_operator_max:
	 initval = make_minval_expression(reduction_basic(reductionInfo_reduction(ri)));
	 break;

      case is_reduction_operator_sum:
	 initval = make_0val_expression(reduction_basic(reductionInfo_reduction(ri)));
	 break;

      case is_reduction_operator_prod:
	 initval = make_1val_expression(reduction_basic(reductionInfo_reduction(ri)));
	 break;

      case is_reduction_operator_and:
	 initval = make_constant_boolean_expression(TRUE);
	 break;

      case is_reduction_operator_or:
	 initval = make_constant_boolean_expression(FALSE);
	 break;
   }

   for(i=0; i<reductionInfo_count(ri); i++)
   {
      instruction is;

      is = make_assign_instruction(
	 reference_to_expression(make_reference(
	    reductionInfo_vector(ri), CONS(EXPRESSION, int_expr(i), NIL))),
	 copy_expression(initval));
      
      prelude = CONS(STATEMENT, 
		     instruction_to_statement(is),
		     prelude);
   }

   free_expression(initval);

   return instruction_to_statement(make_instruction_sequence(make_sequence(prelude)));
}

static statement generate_compact(reductionInfo ri)
{
   expression rightExpr;
   reference redVar;
   entity operator;
   instruction compact;
   int i;
   
   switch(reduction_operator_tag(reduction_op(reductionInfo_reduction(ri))))
   {
      default:
      case is_reduction_operator_none:
	 return statement_undefined;  //nothing to generate
	 break;

      case is_reduction_operator_min:
	 operator = entity_intrinsic(MIN_OPERATOR_NAME);
	 break;

      case is_reduction_operator_max:
	 operator = entity_intrinsic(MAX_OPERATOR_NAME);
	 break;

      case is_reduction_operator_sum:
	 operator = entity_intrinsic(PLUS_OPERATOR_NAME);
	 break;

      case is_reduction_operator_prod:
	 operator = entity_intrinsic(MULTIPLY_OPERATOR_NAME);
	 break;

      case is_reduction_operator_and:
	 operator = entity_intrinsic(AND_OPERATOR_NAME);
	 break;

      case is_reduction_operator_or:
	 operator = entity_intrinsic(OR_OPERATOR_NAME);
	 break;
   }

   redVar = copy_reference(reduction_reference(reductionInfo_reduction(ri)));
   rightExpr = reference_to_expression(redVar);
   for(i=0; i<reductionInfo_count(ri); i++)
   {
      call c;
      expression e;

      e = reference_to_expression(make_reference(
	     reductionInfo_vector(ri), CONS(EXPRESSION, int_expr(i), NIL)));
      c = make_call(operator, CONS(EXPRESSION, e, 
				   CONS(EXPRESSION, rightExpr, NIL)));

      rightExpr = call_to_expression(c);
   }

   compact = make_assign_instruction(
      reference_to_expression(redVar),
      rightExpr);

   return make_stmt_of_instr(compact);
}

static void reductions_rewrite(statement s)
{
   instruction i = statement_instruction(s);
   statement body;
   instruction ibody;
   list reductions = NIL;
   list preludes = NIL;
   list compacts = NIL;

   //We are only interested in loops
   switch(instruction_tag(i))
   {
      case is_instruction_loop:
	 body = loop_body(instruction_loop(i));
	 break;

      case is_instruction_whileloop:
	 body = whileloop_body(instruction_whileloop(i));
	 break;

      case is_instruction_forloop:
	 body = forloop_body(instruction_forloop(i));
	 break;

      default:
	 return;
   }

   //Lookup the reductions in the loop's body, and change the loop body accordingly
   ibody = statement_instruction(body);
   switch(instruction_tag(ibody))
   {
      case is_instruction_sequence:
	 MAP(STATEMENT,
	     s,
	     rename_statement_reductions(s, &reductions),
	     sequence_statements(instruction_sequence(ibody)));
	 break;

      case is_instruction_call:
	 rename_statement_reductions(s, &reductions);
	 break;

      default:
	 return;
   }

   //Generate prelude and compact code for each of the reductions
   MAP(REDUCTIONINFO,
       ri,
   {
      statement s;

      s = generate_prelude(ri);
      if (s != statement_undefined)
	 preludes = CONS(STATEMENT, s, preludes);

      s = generate_compact(ri);
      if (s != statement_undefined)
	 compacts = CONS(STATEMENT, s, compacts);
   },
       reductions);

   statement_instruction(s) = make_instruction_sequence(make_sequence(
      gen_concatenate(preludes, 
		      CONS(STATEMENT, copy_statement(s),
			   compacts))));

   statement_label(s) = entity_empty_label();
   statement_number(s) = STATEMENT_NUMBER_UNDEFINED;
   statement_ordering(s) = STATEMENT_ORDERING_UNDEFINED;
   statement_comments(s) = empty_comments;
   statement_declarations(s) = NIL;
   statement_decls_text(s) = string_undefined;

   gen_full_free_list(reductions);
}

bool simd_remove_reductions(char * mod_name)
{
   
   /* get the resources */
   statement mod_stmt = (statement)
      db_get_memory_resource(DBR_CODE, mod_name, TRUE);

   set_current_module_statement(mod_stmt);
   set_current_module_entity(local_name_to_top_level_entity(mod_name));
   set_cumulated_reductions((pstatement_reductions)
			    db_get_memory_resource(DBR_CUMULATED_REDUCTIONS,
						   mod_name, TRUE));

   debug_on("SIMDREDUCTION_DEBUG_LEVEL");
   /* Now do the job */
  
   gen_recurse(mod_stmt, statement_domain,
	       gen_true, reductions_rewrite);

   pips_assert("Statement is consistent after SIMDIZER", 
	       statement_consistent_p(mod_stmt));

   /* Reorder the module, because new statements have been added */  
   module_reorder(mod_stmt);
   DB_PUT_MEMORY_RESOURCE(DBR_CODE, mod_name, mod_stmt);
   DB_PUT_MEMORY_RESOURCE(DBR_CALLEES, mod_name, 
			  compute_callees(mod_stmt));
 
   /* update/release resources */
   reset_cumulated_reductions();
   reset_current_module_statement();
   reset_current_module_entity();

   debug_off();

   return TRUE;
}

