/* -----------------------------------------------------------------
 *
 *                 TOP-DOWN ARRAY BOUND CHECK VERSION
 *
 * -----------------------------------------------------------------
 *
 * This phase is based on the array region analyses 
 *  Different strategies :
 * 1. When a bounds violation is detected, we generate a STOP 
 * statement  and we don't go down anymore 
 * => difficult for debugging all the program's error
 * => finish as soon as possible 
 * 2. Generate comments for errors and warnings
 * => convenient for debugging
 * => can not detect bounds violation for dynamic tests
 *    with generated code
 * 3. Test and STOP statement every where 
 * => the {0==1} every where , not very intelligent ? 
 *
 * The first strategy is implemented for this moment 
 *
 *
 * Hypotheses : there is no write effect on the array bound expression.
 *
 * There was a test for write effect on bound here but I put it away (in 
 * effect_on_array_bound.c) because it takes time to calculate the effect
 * but in fact this case is rare */

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
#include "makefile.h"
#include "misc.h"
#include "control.h"
#include "properties.h"
#include "semantics.h"
#include "transformer.h"
#include "pipsmake.h"
#include "instrumentation.h"
#include "abc_private.h"
#include "effects-generic.h"
#include "effects-convex.h"
#include "effects-simple.h"
#include "conversion.h"

/* The following data structure is the context of top_down_abc:
   The read_marked_list marks if one bound of one array's dimension
   is tested or not, for the READ regions. 
   The write_marked_list marks if one bound of one array's dimension
   is tested or not, for the WRITE regions.
   The saved list keeps the status before go down in the tree. 
   The hash table statement_check_list associates each statement 
   with its list of checks for each bound of array's dimension  */
 
typedef struct 
{
  abc_checked read_marked_list;
  abc_checked write_marked_list;
  hash_table read_saved_list;
  hash_table write_saved_list;
  hash_table statement_check_list; 
  persistant_statement_to_control map; // to treat unstructured case
  stack uns;
} top_down_abc_context_t,
* top_down_abc_context_p;


typedef struct Bound_test
{
  bool bound;
  expression test;
} Bound_test;

static string read_or_write(bool a)
{
  if (a)
    return ", READING, ";
  else 
    return ", WRITING, ";  
}
static string bool_to_bound(bool b)
{
  if (b)
    return ", lower bound, ";
  else 
    return ", upper bound, ";  
}

/* Statistic variables: */

static int number_of_added_tests;
static int number_of_bound_violations;

static void initialize_top_down_abc_statistics()
{
    number_of_added_tests = 0;
    number_of_bound_violations = 0;
}

static void display_top_down_abc_statistics()
{
  if (number_of_added_tests > 0) 
    user_log("* There %s %d array bound check%s added *\n",
	     number_of_added_tests > 1 ? "are" : "is",
	     number_of_added_tests,
	     number_of_added_tests > 1 ? "s" : "");	

  if (number_of_bound_violations > 0) 
    user_log("* There %s %d bound violation%s *\n",
	     number_of_bound_violations > 1 ? "are" : "is",
	     number_of_bound_violations,
	     number_of_bound_violations > 1 ? "s" : "");
}

static abc_checked initiliaze_marked_list()
{
  abc_checked retour = abc_checked_undefined;
  list tmp = NIL;

  // get list of entities in the declaration part

  list ld = code_declarations(value_code(entity_initial(get_current_module_entity())));

  while (!ENDP(ld))
    {     
      entity e = ENTITY(CAR(ld));
      type t = entity_type(e);
      array_dimension_checked adc = array_dimension_checked_undefined ;     
      if (type_variable_p(t)) 
	{
	  list ldim = variable_dimensions(type_variable(t));
	  int length = gen_length(ldim);
	  int i;	  
	  if (length > 0)
	    {
	      dimension_checked dc = dimension_checked_undefined ;
	      list dc_list = NIL;
	      for(i=1; i <= length; i++ )
		{
		  dc = make_dimension_checked(i,FALSE,FALSE);	
		  dc_list = gen_nconc(dc_list,
				      CONS(DIMENSION_CHECKED,dc,NIL));	      
		}	      
	      adc = make_array_dimension_checked(e, dc_list);
	    }
	}      
      if (adc != array_dimension_checked_undefined )
	tmp = gen_nconc(tmp,CONS(ARRAY_DIMENSION_CHECKED,adc,NIL));
      ld = CDR(ld);  
    }

  retour = make_abc_checked(tmp);

  return retour;
}
  
static void set_array_dimension_checked(top_down_abc_context_p context, 
	bool action, entity array, int dim, bool bound)
{
  if (action)
    {
      // read region 
      MAP(ARRAY_DIMENSION_CHECKED, adc,
      {
	if (same_entity_p(array,array_dimension_checked_array(adc)))
	  MAP(DIMENSION_CHECKED, dc,
	  {
	    if (dimension_checked_dim(dc) == dim)
	      {
		if (bound)
		  dimension_checked_lower(dc) = TRUE;
		else
		  dimension_checked_upper(dc) = TRUE;	
	      }
	  },array_dimension_checked_dims(adc));       
      }, abc_checked_list(context->read_marked_list)); 
    }
  else 
    {
      // write region
      MAP(ARRAY_DIMENSION_CHECKED, adc,
      {
	if (same_entity_p(array,array_dimension_checked_array(adc)))
	  MAP(DIMENSION_CHECKED, dc,
	  {
	    if (dimension_checked_dim(dc) == dim)
	      {
		if (bound)
		  dimension_checked_lower(dc) = TRUE;
		else
		  dimension_checked_upper(dc) = TRUE;	
	      }
	  },array_dimension_checked_dims(adc));       
      }, abc_checked_list(context->write_marked_list));
    }
}

Psysteme my_system_projection_along_variables(Psysteme ps, Pvecteur pv)
{  
  // handle overflow by the calling procedure (FWD_OFL_CTRL)   
  CATCH(overflow_error)
    return SC_UNDEFINED;     
  TRY 
    {
      bool is_proj_exact = TRUE;   
      sc_projection_along_variables_with_test_ofl_ctrl(&ps, pv, 
				  &is_proj_exact, FWD_OFL_CTRL);
      UNCATCH(overflow_error);
      if (is_proj_exact)
	return ps;  
      else
	return SC_UNDEFINED;      
    }
}

static Psysteme my_system_remove_variables(Psysteme ps)
{
  /* Project all PHI and #init variables, ....in the system ps, 
     there are 2 cases :
     1. The result is not sure , there are over flow
        Return the SC_UNDEFINED 
     2. The result is sure, three small cases: 
        2.1 The system is always false sc_empty => no bounds violation
        2.2 The system is always true sc_rn => bounds violation
        2.3 The system is parametric => test to put*/

  if (!sc_empty_p(ps)&& !sc_rn_p(ps))
    {
      list l_phi = phi_entities_list(1,7);  
      list l_var = l_phi;
      Pvecteur pv_var = NULL; 
      Pbase b = ps->base;
       
      /* converts the phi list into a Pvecteur */
      MAP(ENTITY, e,
      {
	if (base_contains_variable_p(ps->base, (Variable) e) )
	  vect_add_elem(&pv_var, (Variable) e, VALUE_ONE);
      },l_var);

      for(; !VECTEUR_NUL_P(b);b = b->succ)
	{
	  entity e = (entity) vecteur_var(b);
	  if (strstr(entity_name(e),OLD_VALUE_SUFFIX) != NULL) 
	    vect_add_elem(&pv_var, (Variable) e, VALUE_ONE);
	}
     
      ps = my_system_projection_along_variables(ps, pv_var);       
      vect_rm(pv_var);      
      gen_free_list(l_phi);
    }
  return ps;     
}

static void top_down_abc_insert_before_statement(statement s, 
		statement s1,top_down_abc_context_p context)
{
  /* If s is in an unstructured instruction, we must pay attetion 
     when inserting s1 before s.  */
  if (bound_persistant_statement_to_control_p(context->map, s))
    {
      /* take the control qui has s as statement  */
      
      control c = apply_persistant_statement_to_control(context->map, s);
      if (stack_size(context->uns)>0)
	{	
	  /* take the unstructured correspond to the control c */
	  unstructured u = (unstructured) stack_head(context->uns);
	  control newc;

	  /* for a consistent unstructured, a test must have 2 successors, 
	     so if s1 is a test, we transform it into sequence in order 
	     to avoid this constraint. 
	  
	     Then we create a new control for it, with the predecessors 
	     are those of c and the only one successor is c. 
	     The new predecessors of c are only the new control*/
	  
	  if (statement_test_p(s1))
	    {
	      list seq = CONS(STATEMENT,s1,NIL);
	      statement s2=instruction_to_statement(make_instruction(is_instruction_sequence,
								     make_sequence(seq))); 
	      newc = make_control(s2, control_predecessors(c), CONS(CONTROL, c, NIL));
	    }
	  else 
	    newc = make_control(s1, control_predecessors(c), CONS(CONTROL, c, NIL));

	  // replace c by  newc as successor of each predecessor of c 

	  MAP(CONTROL, co,
	  {
	    MAPL(lc, 
	    {
	      if (CONTROL(CAR(lc))==c) CONTROL(CAR(lc)) = newc;
	    }, control_successors(co));
	  },control_predecessors(c));
	  
	  control_predecessors(c) = CONS(CONTROL,newc,NIL);

	  /* if c is the entry node of the correspond unstructured u, 
	     the newc will become the new entry node of u */
	      
	  if (unstructured_control(u)==c) 
	    unstructured_control(u) = newc;	 
	  
	  gen_recurse_stop(newc); // ????????????
	}
      else
	{
	  // there is no unstructured (?)
	  insert_statement(s,s1,TRUE);
	}
    }
  else
    // structured case 
    insert_statement(s,s1,TRUE);     
}


static list top_down_abc_call(call c, entity array,
		    dimension dim_i, int i, bool bound)
{
  list retour = NIL;
  list args = call_arguments(c);  
  while (!ENDP(args))
    {
      expression e = EXPRESSION(CAR(args));
      syntax s = expression_syntax(e);
      tag t = syntax_tag(s);
      switch (t){ 
      case is_syntax_call:  
	{	  
	  list tmp = top_down_abc_call(syntax_call(s),array,dim_i,i,bound);
	  if (tmp != NIL)	
	    // add tmp to retour
	    while (!ENDP(tmp))
	      {
		expression exp = EXPRESSION(CAR(tmp));
		if (!same_expression_in_list_p(exp,retour))
		  retour = gen_nconc(retour,CONS(EXPRESSION,exp,NIL));
		tmp = CDR(tmp);
	      }	     
	  break;
	}
      case is_syntax_range:
	/* There is nothing to check here*/
	break;
      case is_syntax_reference:
	{ 
	  reference ref = syntax_reference(s);
	  entity arr = reference_variable(ref);
	  if (same_entity_p(arr,array)) 
	    {	 	      
	      list arrayinds = reference_indices(ref);	     	       
	      expression ith = find_ith_argument(arrayinds,i); 
	      expression exp = expression_undefined;
	      
	      if (!expression_undefined_p(ith))
		{
		  ifdebug(2) 
		    {	  
		      fprintf(stderr, "\n The ith expression");    
		      print_expression(ith);	
		      fprintf(stderr, " \n array: %s ",entity_name(array));
		    }
		  if (bound)
		    {
		      /* Make expression : ith < lower_bound */
		      exp = lt_expression(copy_expression(ith),
					  dimension_lower(copy_dimension(dim_i)));		
		      ifdebug(2) 
			{	  
			  fprintf(stderr, "\n The lower bound test");    
			  print_expression(exp);			 
			}
		    }
		  else
		    // test if the upper bound is unbounded or not
		    if (!unbounded_dimension_p(dim_i))
		      {
			/* Make expression  : upper_bound < ith */
			exp = lt_expression(dimension_upper(copy_dimension(dim_i)),
					    copy_expression(ith));	
			ifdebug(2) 
			  {	  
			    fprintf(stderr, "\n The upper bound test");    
			    print_expression(exp);			 
			  }
		      }
		}
	      
	      /* Remark : Doesn't like the 1st version of abc,
		 we don't have to put a test for ith in the indirect case.
		 F.e : for A(B(i)) = 0.0, we have 2 regions :
		 < B(PHI1)-R-EXACT-{PHI1==I} >
		 < A(PHI1)-W-MAY-{} >
		 when we check for the inexact case of A, we don't have 
		 to check for B.

		 In case if there is another access of array B in the same 
		 statement, the region of B may be MAY => we check it for 
		 the read region of B. 

		 In case of A(A(i)) , we have the different regions for read
		 and write effect ??????? => ?????
		 ATT : example Indirection.f */

	      /* Test if exp is trivial or not 
		 + If exp is always TRUE: there is certainly bound violation, 
	             return  make_true_expression        
		 + If exp is always FALSE, we don't have to add it to retour 
		 + Otherwise, we add it to retour.*/

	      if (!expression_undefined_p(exp))
		{
		  int tr = trivial_expression_p(exp);
		  switch(tr){
		  case 1:
		    {
		      retour = CONS(EXPRESSION,make_true_expression(),NIL);
		      return retour;
		    }
		  case 0: 
		    { 
		      // test if exp is already in retour 
		      if (!same_expression_in_list_p(exp,retour))
			retour = gen_nconc(retour,CONS(EXPRESSION,exp,NIL));
		      break;
		    }
		  case -1:   
		    break;
		  }
		}	
	    }
	  break;
	}          
      }     
      args = CDR(args);
    }  
  return retour;
}

/* hack: clean all normalize fields... 
 */
static void expr_rwt(expression e)
{
  if (!normalized_undefined_p(expression_normalized(e)))
    {
      free_normalized(expression_normalized(e));
      expression_normalized(e) = normalized_undefined;
    }
}

void clean_all_normalized(expression e)
{
  gen_recurse(e, expression_domain, gen_true, expr_rwt);
}

static Bound_test top_down_abc_not_exact_case( statement s, 
	       top_down_abc_context_p context, bool action, 
               entity array, dimension dim_i,int i, bool bound)
{
  Bound_test retour;
  retour.test = expression_undefined;
  retour.bound = TRUE;
  
  /* Test if s is a call (elementary statement)*/
  if (statement_call_p(s))
    {	
      /* generate a lower/upper bound test expression for all 
	 array reference of this dimension of array*/
      call c = instruction_call(statement_instruction(s));   
      list l = top_down_abc_call(c,array,dim_i,i,bound);     
      if (l!= NIL)
	retour.test = expression_list_to_binary_operator_call(l,
	          	    entity_intrinsic(OR_OPERATOR_NAME));
      set_array_dimension_checked(context,action,array,i,bound);  
      gen_free_list(l);
    }
  else
    /* s is not a call, no conclusion for this bound,
       continue to go down */		
    retour.bound = FALSE;
  
  return retour;
}

static Bound_test top_down_abc_dimension(statement s, 
	   top_down_abc_context_p context, region re,
	   bool action, entity array, int i, bool bound)
{
  Bound_test retour;
  variable var = type_variable(entity_type(array));
  dimension dim_i = find_ith_dimension(variable_dimensions(var),i);
  retour.bound = TRUE;
  retour.test = expression_undefined;
  if (!bound && unbounded_dimension_p(dim_i))
    {
      /* unbounded dimension, we don't have to check for this bound */
      set_array_dimension_checked(context,action,array,i,bound);
    }
  else 
    {              
      Psysteme P = sc_dup(region_system(re));
      Pcontrainte con_exp = CONTRAINTE_UNDEFINED;
      normalized nexpr = normalized_undefined;      
      expression exp = expression_undefined;
      expression exp_1 = int_to_expression(1);
      if (bound)
	exp = binary_intrinsic_expression(MINUS_OPERATOR_NAME,
					  dimension_lower(copy_dimension(dim_i)),
					  exp_1);
      else    
	exp = binary_intrinsic_expression(PLUS_OPERATOR_NAME,
					  dimension_upper(copy_dimension(dim_i)),
					  exp_1);
      clean_all_normalized(exp);
      
      // fast check: PHIi<=lower-1 (or upper+1<=PHIi) is trivial redundant wrt P or not
      // transform exp to Pcontrainte con_exp

      nexpr = NORMALIZE_EXPRESSION(exp);
      
      if (normalized_linear_p(nexpr)) 
	{
	  entity phi = make_phi_entity(i);	
	  Pvecteur v1 = vect_new((Variable) phi, VALUE_ONE);
	  Pvecteur v2 = normalized_linear(nexpr);
	  Pvecteur vect;	  
	  if (bound) 
	    vect = vect_substract(v1, v2);
	  else 
	    vect = vect_substract(v2, v1);
	  vect_rm(v1);
	  con_exp = contrainte_make(vect);
	 
	  switch (sc_check_inequality_redundancy(con_exp, P)) /* try fast check */
	    {
	    case 1: /* ok, redundant => there is certainly bound violation */
	      {
		set_array_dimension_checked(context,action,array,i,bound);	       
		retour.test = make_true_expression();
		break;
	      }
	    case 2: /* ok, not feasible => there is no bound violation*/
	      {
		set_array_dimension_checked(context,action,array,i,bound);
		break;
	      }
	    case 0: /* no result, try slow version */
	      {
		/* Add the equation  PHIi <= lower-1 (upper+1 <= PHIi)
		   to the predicate of region re */

		if (sc_add_phi_equation(P,exp,i,FALSE,bound))
		  { 
		    /* Every expression is linear.
		     * Test the feasibility of P by using this function: 
		     * sc_rational_feasibility_ofl_ctrl(sc, ofl_ctrl, ofl_res) in which
		     *
		     * ofl_ctrl = OFL_CTRL means that the overflows are treated in the 
		     * called procedure (sc_rational_feasibility_ofl_ctrl())
		     *
		     * ofl_res = TRUE means that if the overflows occur, function 
		     * sc_rational_feasibility_ofl_ctrl will return the value TRUE
		     * we don't know if the system is feasible or not
		     *
		     * The function sc_rational_feasibility_ofl_ctrl() is less 
		     * expensive than the function sc_integer_feasibility_ofl_ctrl()*/
		    
		    if (!sc_rational_feasibility_ofl_ctrl(P, OFL_CTRL, TRUE))
		      /* The system is not feasible (certainly) => no violation */
		      set_array_dimension_checked(context,action,array,i,bound);
		    else 	
		      {
			/* The system is feasible or we don't know it is feasible or not
			 * Test if the region re is EXACT or MAY */	  
			if (region_exact_p(re)) 
			  {
			    /* EXACT region
			       Remove all PHI variables, useless variables such as 
			       PIPS generated variables V#init, common variable 
			       from another subroutine ......
			       
			       SUBROUTINE X
			       CALL Y(I)
			       <P(I) {I == FOO:J}
			       END
			       
			       SUBROUTINE Y(K)
			       COMMON J
			       K=J
			       END
			    */
			    
			    Psysteme ps = my_system_remove_variables(P);
			    if (ps == SC_UNDEFINED)
			      // the projection is not exact
			      retour = top_down_abc_not_exact_case (s,context,action,array,dim_i,i,bound);
			    else 
			      {
				// the projection is exact		      
				set_array_dimension_checked(context,action,array,i,bound);	       
				if (!sc_empty_p(ps) && !sc_rn_p(ps))
				  // there is a test to put 	
				  // sc_normalized or sc_elim_redon ?????
				  retour.test = Psysteme_to_expression(ps);		
				else 
				  if (sc_rn_p(ps))
				    // the system is trivial true, there are bounds violation
				    retour.test = make_true_expression();
				//else, ps=sc_empty, the system is false, no bounds violation
			      }	    
			  }
			else 
			  /* MAY region */	 
			  retour = top_down_abc_not_exact_case(s,context,action,array,dim_i,i,bound);
		      }
		  }
		else
		  // the exp is not linear, we can't add to P 
		  retour.bound = FALSE;	
	      }
	    }
	  contrainte_free(con_exp);
	}
       sc_rm(P);
    } 
  return retour;
}

static bool top_down_abc_flt(statement s, 
				 top_down_abc_context_p context)
{
  bool retour = FALSE;
  list l_rw = regions_dup(load_statement_local_regions(s));  
  statement test_sequence = statement_undefined;
  list lexp = NIL;
  
  ifdebug(3) 
    {	  
      fprintf(stderr, "\n list of regions ");    
      print_effects(l_rw);
      fprintf(stderr, "\n for the statement");    
      print_statement(s);      
    }

  hash_put(context->read_saved_list,s,copy_abc_checked(context->read_marked_list));
  hash_put(context->write_saved_list,s,copy_abc_checked(context->write_marked_list));

  while (!ENDP(l_rw))
    {
      region re = REGION(CAR(l_rw));
      reference ref = region_reference(re);
      entity array = reference_variable(ref); 

      if (array_reference_p(ref) && array_need_bound_check_p(array))
	{	  
	  list marked_list = NIL; 
	  list dc_list = NIL;	
	  bool action = region_read_p(re);
	   
	  if (action)
	    marked_list = abc_checked_list(context->read_marked_list);
	  else
	    marked_list = abc_checked_list(context->write_marked_list);
 	  
	  while (!ENDP(marked_list))
	    {
	      array_dimension_checked adc = 
		ARRAY_DIMENSION_CHECKED(CAR(marked_list));
	      if (same_entity_p(array_dimension_checked_array(adc),array))
		{
		  dc_list = array_dimension_checked_dims(adc);
		  break;
		}
	      else
		marked_list = CDR(marked_list); 
	    }
	  
	  // traverse each dimension
	  
	  while (!ENDP(dc_list))
	    {	
	      dimension_checked dc = DIMENSION_CHECKED(CAR(dc_list));
	      int i = dimension_checked_dim(dc);
	      Bound_test lower, upper;
	      lower.test = expression_undefined;
	      upper.test = expression_undefined;
	      lower.bound = TRUE;
	      upper.bound = TRUE;

	      /* if we have a region like: <A(PHI)-EXACT-{}>
	       * it means that all *declared* elements are touched, although
	       * this is implicit. this occurs with io effects of "PRINT *, A".
	       * in such a case, the declaration constraints MUST be appended
	       * before the translation, otherwise the result might be false.
	       *
	       * potential bug : if the declaration system cannot be generated,
	       *   the region should be turned to MAY for the translation?
	       */
	      
	      append_declaration_sc_if_exact_without_constraints(re);

	      if (!dimension_checked_lower(dc))  
		{
		  /* The lower bound of the dimension i is not marked TRUE*/
		  lower = top_down_abc_dimension(s,context,re,action,array,i,TRUE);	
		  if (!expression_undefined_p(lower.test))
		    {
		      statement sta;
		      test t;
		      string message = 
			strdup(concatenate("\"Bound violation:",
					   read_or_write(action)," array ",
					   entity_name(array),
					   bool_to_bound(TRUE),
					   int_to_dimension(i),"\"",NULL));
		      
		      if (true_expression_p(lower.test))
		      	{
			  // there is bounds violation !
			  // replace s by a STOP statement	

			  /*=> Bug in Examples/perma.f
			   Do not replace s by STOP, but insert STOP before s*/

			  number_of_bound_violations ++;
			  user_log("\n Bound violation !!! \n");
			  sta  = make_stop_statement(message);

			  top_down_abc_insert_before_statement(s,sta,context);

			  //  free_instruction(statement_instruction(s));
			  // statement_instruction(s) = statement_instruction(sta);
			  return FALSE;  // follow the first strategy
		      	}

		      // test if expression lower.test exists already in test_sequence
		     
		      if (!same_expression_in_list_p(lower.test,lexp))  
			{
			  ifdebug(2) 
			    {	  
			      fprintf(stderr, "\n The lower test");    
			      print_expression(lower.test);			 
			    }
			  number_of_added_tests ++;
			  lexp = gen_nconc(lexp,CONS(EXPRESSION,lower.test,NIL));
			  t = make_test(lower.test, 
					make_stop_statement(message),
					make_block_statement(NIL));
			  sta  = test_to_statement(t);
			  
			  if (statement_undefined_p(test_sequence))
			    test_sequence = copy_statement(sta);
			  else 
			    insert_statement(test_sequence,copy_statement(sta), TRUE);
			}
		    }
		}
      	      if (!dimension_checked_upper(dc))
		{
		  /* The upper bound of the dimension i is not marked TRUE*/
		  upper = top_down_abc_dimension(s,context,re,action,array,i,FALSE);
		  if (!expression_undefined_p(upper.test))
		    {
		      statement sta;
		      test t;
		      string message = 
			strdup(concatenate("\"Bound violation:",
					   read_or_write(action), " array ",
					   entity_name(array),
					   bool_to_bound(FALSE),
					   int_to_dimension(i),"\"",NULL));
		      
		      if (true_expression_p(upper.test))
		      	{
			  // there is bounds violation !
			  // replace s by a STOP statement

			  
			  /*=> Bug in Examples/perma.f
			   Do not replace s by STOP, but insert STOP before s*/

			  number_of_bound_violations ++;
			  user_log("\n Bound violation !!! \n");
			  sta  = make_stop_statement(message);

			  top_down_abc_insert_before_statement(s,sta,context);

			  //  free_instruction(statement_instruction(s));
			  // statement_instruction(s) = statement_instruction(sta);

			  return FALSE;  // follow the first strategy
		      	}

		      // test if expression lower.test exists already in test_sequence
		     
		      if (!same_expression_in_list_p(upper.test,lexp))
			{
			  ifdebug(2) 
			    {	  
			      fprintf(stderr, "\n The upper test");    
			      print_expression(upper.test);			 
			    }

			  number_of_added_tests++;
			  lexp = gen_nconc(lexp,CONS(EXPRESSION,upper.test,NIL));
			  t = make_test(upper.test, 
					make_stop_statement(message),
					make_block_statement(NIL));
			  sta  = test_to_statement(t);
			  
			  if (statement_undefined_p(test_sequence))
			    test_sequence = copy_statement(sta);
			  else 
			    insert_statement(test_sequence,copy_statement(sta),TRUE);	
			}
		    }
		}
	      /* If one bound of the dimension is marked FALSE, 
		 we have to go down*/
	      if ((!lower.bound) || (!upper.bound)) retour = TRUE;	      
	      dc_list = CDR(dc_list);
	    }	  
	}      
      l_rw = CDR(l_rw); 
    }
  
  if (!statement_undefined_p(test_sequence))
    {
      ifdebug(3) 
	{	  
	  fprintf(stderr, "\n The sequence of test");    
	  print_statement(test_sequence);
	}
      if (!retour) 
	// retour = FALSE, insert new tests for the statement s here
	top_down_abc_insert_before_statement(s,test_sequence,context); 
      else 
	// insert new tests in function rwt
	hash_put(context->statement_check_list,s,test_sequence);
    }  

  if (!retour)
    {
      context->read_marked_list = (abc_checked) hash_get(context->read_saved_list,s); 
      context->write_marked_list = (abc_checked) hash_get(context->write_saved_list,s);
    }
  gen_free_list(l_rw);
  gen_free_list(lexp);

  return retour;
}

static void top_down_abc_rwt(statement s,
				 top_down_abc_context_p context)
{    
  statement test_sequence = statement_undefined;

  context->read_marked_list = (abc_checked) hash_get(context->read_saved_list,s); 
  context->write_marked_list = (abc_checked) hash_get(context->write_saved_list,s); 

  test_sequence = (statement) hash_get(context->statement_check_list,s); 
 
  if (!statement_undefined_p(test_sequence))
    {
      ifdebug(3) 
	{
	  fprintf(stderr, "\n Rewrite : The sequence of test");    
	  print_statement(test_sequence);
	  fprintf(stderr, "\n of statement");    
	  print_statement(s);
	}      	
      // insert the new sequence of tests before the current statement  
      top_down_abc_insert_before_statement(s,test_sequence,context);
    }
}

static bool store_mapping(control c, top_down_abc_context_p context)
{
  extend_persistant_statement_to_control(context->map,
					 control_statement(c), c);
  return TRUE;
}

static bool push_uns(unstructured u, top_down_abc_context_p context)
{
  stack_push((char *) u, context->uns);
  return TRUE;
}

static void pop_uns(unstructured u, top_down_abc_context_p context)
{
  stack_pop(context->uns);
}


static void top_down_abc_statement(statement module_statement)
{
  top_down_abc_context_t context;
  context.read_marked_list = initiliaze_marked_list();
  context.write_marked_list = copy_abc_checked(context.read_marked_list);
  context.read_saved_list =  hash_table_make(hash_pointer, 0);
  context.write_saved_list = hash_table_make(hash_pointer, 0);

  context.statement_check_list = hash_table_make(hash_pointer, 0);
  context.map = make_persistant_statement_to_control();
  context.uns = stack_make(unstructured_domain,0,0);
 
  gen_context_multi_recurse(module_statement, &context,
      unstructured_domain, push_uns, pop_uns,
      control_domain, store_mapping, gen_null,
      statement_domain, top_down_abc_flt, top_down_abc_rwt,
      NULL);

  free_abc_checked(context.read_marked_list);
  free_abc_checked(context.write_marked_list);
  hash_table_free(context.read_saved_list);
  hash_table_free(context.write_saved_list);
  hash_table_free(context.statement_check_list);
  free_persistant_statement_to_control(context.map);
  stack_free(&context.uns);
}

bool top_down_array_bound_check(char *module_name)
{ 
  statement module_statement;
 
  set_current_module_entity(local_name_to_top_level_entity(module_name));

  if (!same_string_p(rule_phase(find_rule_by_resource("REGIONS")),
		     "MUST_REGIONS"))
    pips_user_warning("\n MUST REGIONS not selected - "
		      "\n Do not expect wonderful results\n");
      
  /* set and get the current properties concerning regions */
      
  set_bool_property("MUST_REGIONS", TRUE);
  set_bool_property("EXACT_REGIONS", TRUE);
  get_regions_properties();

  /* Get the code of the module. */
  module_statement= (statement) db_get_memory_resource(DBR_CODE, 
						       module_name, 
						       TRUE);
  set_current_module_statement(module_statement);
 
  initialize_ordering_to_statement(module_statement);
  
  /* Get the READ and WRITE regions of the module */
  set_rw_effects((statement_effects) 
		 db_get_memory_resource(DBR_REGIONS, module_name, TRUE)); 
  
  debug_on("TOP_DOWN_ARRAY_BOUND_CHECK_DEBUG_LEVEL");
  
  pips_debug(1, " Region based ABC, Begin for %s\n", module_name);
  pips_assert("Statement is consistent ...", 
	      statement_consistent_p(module_statement));
  
  initialize_top_down_abc_statistics();
  top_down_abc_statement(module_statement);
  display_top_down_abc_statistics();
  
  /* Reorder the module, because the bound checks have been added */
  module_reorder(module_statement);
  
  pips_debug(1, "end\n");
  debug_off();
  
  DB_PUT_MEMORY_RESOURCE(DBR_CODE, strdup(module_name), module_statement);
  reset_current_module_entity();
  reset_current_module_statement();   
  reset_rw_effects();   
  return TRUE;
}

/* END OF FILE */













