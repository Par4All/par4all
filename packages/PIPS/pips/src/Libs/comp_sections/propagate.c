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
/* {{{  banner*/
/* package complementary sections :  Manjunathaiah M , 18-4-96
 *
 * This File contains the functions computing the regions of a module :
 * proper, local and global.
 *
 *
 */
/* }}} */

#include "all.h"

#define IS_EG TRUE
#define NOT_EG FALSE

#define PHI_FIRST TRUE
#define NOT_PHI_FIRST FALSE

/* global static variable local_regions_map, and its access functions */
DEFINE_CURRENT_MAPPING(local_comp_regions, list)

/* {{{  auxilliary functions*/
void CheckStride(loop __attribute__ ((unused)) Loop)
{
    /* expression Stride = range_increment(loop_range(Loop)); */

}

/* just concatentate list for now : change later */
list CompRegionsExactUnion(list l1, list l2, bool __attribute__ ((unused)) (*union_combinable_p)(effect,effect))
{
   return(gen_nconc(l1,l2));
}

/* just concatentate list for now : change later */
list CompRegionsMayUnion(list l1, list l2, bool __attribute__ ((unused)) (*union_combinable_p)(effect,effect))
{
   return(gen_nconc(l1,l2));
}


/* }}} */

/* {{{  process the body of a procedure*/
/* =============================================================================== 
 *
 * INTRAPROCEDURAL ARRAY REGIONS ANALYSIS
 *
 * =============================================================================== */

/* {{{  intra procedural entry point "complementary_sections" calls comp_regions*/
/* {{{  comments*/
/* bool regions(const char* module_name) 
 * input    : the name of the current module
 * output   : nothing.
 * modifies : computes the local regions of a module.
 * comment  : local regions can contain local variables.
 */
/* }}} */
bool comp_regions(const char* module_name)
{
  /* {{{  code*/
  /* {{{  initialize*/
  /* regions_init(); */
  
  /* get the current properties concerning regions */
  get_regions_properties();
  
  /* Get the code of the module. */
  set_current_module_statement( (statement)
  	db_get_memory_resource(DBR_CODE, module_name, true) );
  /* }}} */
  /* {{{  transformers and preconditions*/
  /* Get the transformers and preconditions of the module. */
  set_transformer_map( (statement_mapping)
  	db_get_memory_resource(DBR_TRANSFORMERS, module_name, true) );	
  set_precondition_map( (statement_mapping) 
  	db_get_memory_resource(DBR_PRECONDITIONS, module_name, true) );
  /* }}} */
  /* {{{  predicates for purpose of debugging*/
  /* predicates defining summary regions from callees have to be 
     translated into variables local to module */
  set_current_module_entity( local_name_to_top_level_entity(module_name) );
  
  
  set_cumulated_rw_effects((statement_effects)
		db_get_memory_resource(DBR_CUMULATED_EFFECTS, module_name, true));
  module_to_value_mappings(get_current_module_entity());
  
  debug_on("COMP_REGIONS_DEBUG_LEVEL");
  pips_debug(3, "begin\n");
  
  /* }}} */
  /* }}} */
  /* Compute the regions of the module. */
  set_local_comp_regions_map( MAKE_STATEMENT_MAPPING() );
  /* {{{  for dependence analysis : currently masked */
  /* set_proper_regions_map( MAKE_STATEMENT_MAPPING() ); */
  /* }}} */
  (void)comp_regions_of_statement(get_current_module_statement()); 
  /* {{{  code*/
  /* {{{  debug stmts*/
  /* if (op_statistics_p()) print_regions_op_statistics(module_name, R_RW); */
  
  pips_debug(3, "end\n");
  
  debug_off();
  /* }}} */
  /* {{{  store in database : modify later*/
  DB_PUT_MEMORY_RESOURCE(DBR_COMPSEC, 
		  strdup(module_name),
		  (char*) listmap_to_compsecs_map(get_local_comp_regions_map()) );
		  
  
  /* DB_PUT_MEMORY_RESOURCE(DBR_PROPER_REGIONS, 
		  strdup(module_name),
		  (char*) listmap_to_effectsmap(get_proper_regions_map()));
  */
  /* }}} */
  /* {{{  finalise*/
  reset_current_module_entity();
  reset_current_module_statement();
  reset_transformer_map();
  reset_precondition_map();
  reset_cumulated_rw_effects();
  free_local_comp_regions_map();
  /* }}} */
  /* }}} */

  return(true);
}
/* }}} */
/* {{{  comp_regions_of_statement*/
/* {{{  comments*/
/* list comp_regions_of_statement(statement s)
 * input    : the current statement.
 * output   : a list of regions.
 * modifies : the local regions_map.
 * comment  : computes the local regions of a statement and stores it
 *            in the local_regions_map.
 */
/* }}} */
list comp_regions_of_statement(statement s)
{
  /* {{{  code*/
  /* {{{  inits*/
  transformer t_stat, context;
  list lreg, lpropreg = NIL;
  
  ifdebug(3)
  {
	  pips_debug(3, "begin\n\tComputation of regions of statement %03td\n", 
		 statement_number(s));
  }
  /* }}} */
  /* {{{  transformer and preconditions*/
    context = load_statement_precondition(s);

  /* compute the regions on the instruction of the statement */
    t_stat = load_statement_transformer(s);

  /* }}} */
  /* }}} */

  lreg = comp_regions_of_instruction(statement_instruction(s),
             t_stat, context, &lpropreg);

  /* {{{  code*/
  /* {{{  store the result : currently masked,  modify later*/
  /* FI: make a copy to safely store that intermediate state in the mapping */
  /* debug_regions_consistency(lreg);
  debug_regions_consistency(lpropreg);
  store_statement_proper_regions(s, lpropreg);
  */
  store_statement_local_comp_regions(s, comp_regions_dup(lreg) ); 
   /* }}} */
  /* }}} */

  return(lreg);
}

/* }}} */
/* {{{  comp_regions_of_instruction*/
/* {{{  comments*/
/* list comp_regions_of_instruction(instruction i, transformer t_inst, context, 
 *                             list *plpropreg)
 * input    : the current instruction and the corresponding transformer 
 *            and context (i.e. precondition), and a pointer that will contain
 *            the proper regions of the instruction.
 * output   : the corresponding list of regions.
 * modifies : nothing.
 */
/* }}} */
list comp_regions_of_instruction(instruction i, transformer t_inst, transformer context,
			    list *plpropreg)
{
    /* {{{  init*/
    list lreg = NIL;
    
    *plpropreg = NIL;
    
    /* }}} */
    switch(instruction_tag(i))
    {
      /* {{{  code*/
      case is_instruction_goto:
	      /* {{{  code*/
	      pips_debug(3, "goto\n");
         break;
	      /* }}} */
      case is_instruction_test:
	      /* {{{  code*/
	      ifdebug(3)
	      {
	        pips_debug(3, "test : %s\n",
		         words_to_string(words_expression
					 (test_condition(instruction_test(i)),
					  NIL)));
	      }
        lreg = comp_regions_of_test(instruction_test(i), context, plpropreg);
	      break;
	      /* }}} */
      case is_instruction_loop:
	      /* {{{  code*/
	      ifdebug(3)
	      {
	      pips_debug(3, "loop : index %s\n",
		         entity_local_name(loop_index(instruction_loop(i))));
	      }
        lreg = comp_regions_of_loop(instruction_loop(i), t_inst, context, plpropreg);
        break;
	      /* }}} */
      case is_instruction_call:
      /* {{{  code*/
      ifdebug(3)
      {
        pips_debug(3, "call : %s\n",
	         module_local_name(call_function(instruction_call(i))));
      }
      lreg = comp_regions_of_call(instruction_call(i), context, plpropreg);
      break;
      /* }}} */
      case is_instruction_unstructured: 
      /* {{{  code*/
      pips_debug(3, "unstructured\n");
      lreg = comp_regions_of_unstructured(instruction_unstructured( i ), t_inst);
      break ;
      /* }}} */
      case is_instruction_block: 
      /* {{{  code*/
      pips_debug(3, "inst block\n");
      lreg = comp_regions_of_block(instruction_block(i));
      break;
      /* }}} */
      default:
          pips_debug(3, "unexpected tag %d\n", instruction_tag(i));
      /* }}} */
    }
    
    return(lreg);
}
/* }}} */
/* {{{  comp_region_of_block*/
/* {{{  comment*/
/* list comp_regions_of_block(list linst) 
 * input    : a list of instructions, representing a sequential block 
 *            of instructions, and the context before the block.
 * output   : a list of regions
 * modifies : linst.
 * comment  : calls itself recursively to calculate the list of regions.	
 */
/* }}} */
list comp_regions_of_block(list linst)
{
    /* {{{  init*/
    statement first_statement;
    list remaining_block, first_s_regions, lres = NIL;
    
    pips_debug(3, "begin\n");
    /* }}} */

    /* {{{  Is it end of list ? */
    if (ENDP(linst))
    {
	    user_warning("regions_of_block", "empty block\n");
	    lres = NIL;
    }
    /* }}} */
    /* {{{  else process list*/
    else
    {
	    /* {{{  regions of CAR(linst)*/
	    first_statement = STATEMENT(CAR(linst));
	    remaining_block = CDR(linst);
	    	
	    first_s_regions = comp_regions_of_statement(first_statement);
	    /* }}} */
	    /* {{{  any more statements in  CDR(linst)*/
	    if (!ENDP(remaining_block))
	    {
	      /* {{{  load transformer*/
	      list r_block_regions = NIL;
	      
	      /* }}} */
	      r_block_regions = comp_regions_of_block(remaining_block);
	      /* {{{  perform union*/
	      /* {{{  don't know that this means ???*/
	      /* blocked : check later
	      list current_transformer = load_statement_transformer(first_statement);    
	      debug_regions_consistency(r_block_regions);
	       project_regions_with_transformer_inverse(r_block_regions, 
						     current_transformer, 
						     NIL); 
	      
	      debug_regions_consistency(r_block_regions);
	      */
	      /* }}} */
	      lres = CompRegionsExactUnion(first_s_regions, r_block_regions, effects_same_action_p);	      
	      /*
	      debug_regions_consistency(lres);
	      */
	      /* }}} */
	    
	    }
	    /* }}} */
      /* {{{  if not  lres = first_s_regions*/
      else 
       lres = first_s_regions;
      /* }}} */
    
    }
    /* }}} */

    pips_debug(3, "end\n");
    return lres;
}
/* }}} */
/* {{{  comp_regions of test*/
/* list regions_of_test(test t, transformer context, list *plpropreg)
 * input    : a test instruction, the context of the test, and a pointer
 *            toward a list that will contain the proper regions of the test,
 *            which are the regions of its conditionnal part.
 * output   : the corresponding list of regions.
 * modifies : nothing.
 */
list
comp_regions_of_test(test t,
		     transformer context,
		     list __attribute__ ((unused)) *plpropreg) {
    /* {{{  init*/
    list le, lt, lf, lc, lr;
    
    pips_debug(3, "begin\n");
    /* }}} */

    /* {{{  if-then-else including if-condition*/
      /* regions of the true branch */
    lt = comp_regions_of_statement(test_true(t));
      /* regions of the false branch */
    lf = comp_regions_of_statement(test_false(t));
      /* regions of the combination of both */
    le = CompRegionsMayUnion(lt, lf, effects_same_action_p);
    
      /* regions of the condition */
    lc = comp_regions_of_expression(test_condition(t), context);
    /* check later
    *plpropreg = comp_regions_dup(lc);
    */
    /* }}} */
    /* {{{  union the regions : currently just add to the list*/
    lr = CompRegionsExactUnion(le, lc, effects_same_action_p);
    pips_debug(3, "end\n");
    /* }}} */

    return(lr);
}
/* }}} */
/* {{{  comp_regions of loop*/
/* list comp_regions_of_loop(loop l, transfomer loop_trans, context, list *plpropreg)
 * input    : a loop, its transformer and its context, and a pointer toward a list,
 *            that will contain the proper regions of the loop, which are the
 *            regions of its range.
 * output   : the corresponding list of regions.
 * modifies : nothing.
 * comment  :	
 */
list
comp_regions_of_loop(loop l,
		     transformer __attribute__ ((unused)) loop_trans,
		     transformer context,
		     list __attribute__ ((unused)) *plpropreg) {
    /* {{{  init*/
    list index_reg, body_reg, le;
    /*
      list global_reg;
      entity i = loop_index(l);
    */

    pips_debug(3, "begin\n");
    /* }}} */

    /* CheckStride(l); */
    /* regions of loop index. */
    if( execution_sequential_p( loop_execution( l )))
    {
	    /* {{{  code*/
	    /* loop index is must-written but may-read because the loop might
	       execute no iterations. */
	    reference ref = make_reference(loop_index(l), NIL);
	    
	    index_reg = comp_regions_of_write(ref, context); /* the loop index is must-written */
	    
	    /* FI, RK: the may-read effect on the index variable is masked by
	     * the intial unconditional write on it (see standard page 11-7, 11.10.3);
	     * if masking is not performed, the read may prevent privatization
	     * somewhere else in the module (12 March 1993)
	     */
	    /* }}} */
    }
    else 
	    index_reg = NIL ;

    /* {{{  regions of  loop induction variable*/
    /* regions of loop bound expressions. */
    le = CompRegionsExactUnion(
           index_reg, 
           comp_regions_of_range(loop_range(l), context), 
           effects_same_action_p);
    /* }}} */

    /* *plpropreg = regions_dup(le); */

    /* regions of loop body statement. */
    body_reg = comp_regions_of_statement(loop_body(l));

    /* insert code to eliminate private variables later */
    
    /* projection of regions along the variables modified by the loop; it includes
     * the projection along the loop index */

     
    pips_debug(7, "elimination of variables modified by the loop.\n");

    /* {{{  In simple sections it is called translation !!*/
    /*
    project_regions_with_transformer_inverse(global_reg, 
					     loop_trans, 
					     CONS(ENTITY,i,NIL));
    project_regions_along_loop_index(global_reg, i, loop_range(l));
    */
    
    pips_debug(3, "Before Translations surrounding the loop index %s :\n",
             entity_local_name(loop_index(l)));

    (void) TranslateRefsToLoop(l, body_reg);
    
    le = CompRegionsExactUnion(le, body_reg, effects_same_action_p);
    
    ifdebug(3)  {
      pips_debug(3, "After Translations  surrounding the loop index %s :\n",
        entity_local_name(loop_index(l)));
      PrintCompRegions(le);
    }

    pips_debug(3, "end\n");
    /* }}} */

    return(le);
}

/* }}} */
/* {{{  comp_regions of call*/
/* list comp_regions_of_call(call c, transformer context, list *plpropreg)
 * input    : a call, which can be a call to a subroutine, but also
 *            to an function, or to an intrinsic, or even an assignement.
 *            And a pointer that will be the proper regions of the call; NIL,
 *            except for an intrinsic (assignment or real FORTRAN intrinsic).
 * output   : the corresponding list of regions.
 * modifies : nothing.
 * comment  :	
 */
list comp_regions_of_call(call c, transformer context, list *plpropreg)
{
    list le = NIL;
    entity e = call_function(c);
    tag t = value_tag(entity_initial(e));
    const char* n = module_local_name(e);
    list pc = call_arguments(c);

    *plpropreg = NIL;

    pips_debug(3, "begin\n");

    switch (t)
    {
     /* {{{  code*/
     case is_value_code:
         pips_debug(3, "external function %s\n", n);
         /* masked now : change later */
         /*
          le = comp_regions_of_external(e, pc, context);
         */
         break;
     
     case is_value_intrinsic:
         pips_debug(3, "intrinsic function %s\n", n);
         le = comp_regions_of_intrinsic(e, pc, context);
	       /* masked now : *plpropreg = regions_dup(le); */
         break;
     
     case is_value_symbolic:
	       pips_debug(3, "symbolic\n");
	       break;
     
     case is_value_constant:
	       pips_debug(3, "constant\n");
         break;
     
     case is_value_unknown:
         pips_internal_error("unknown function %s", n);
         break;
     
     default:
         pips_internal_error("unknown tag %d", t);
     /* }}} */
    }

    pips_debug(3, "end\n");

    return(le);
}

/* }}} */
/* {{{  comp_regions of unstructured*/
/* Computes the effects of the control graph. */
/* list comp_regions_of_unstructured( u , t_unst)
 * input    : an unstructured control flow graph and the corresponding
 *            transformer.
 * output   : the corresponding list of regions.
 * modifies : nothing.
 * comment  :	
 */
list comp_regions_of_unstructured(unstructured u, transformer t_unst)
{
    control ct;
    list blocs = NIL ;
    list le = NIL ;

    pips_debug(3, "begin\n");

    ct = unstructured_control( u );

    if(control_predecessors(ct) == NIL && control_successors(ct) == NIL)
    {
	    /* there is only one statement in u; no need for a fix-point */
	    pips_debug(3, "unique node\n");
	    le = comp_regions_of_statement(control_statement(ct));
    }
    else
    {	
	    CONTROL_MAP(c,
		   {
		     le = CompRegionsMayUnion(comp_regions_of_statement(control_statement(c)), 
					 le, effects_same_action_p) ;
		   },ct, blocs) ;	
	     project_regions_along_parameters(le, transformer_arguments(t_unst));
	     gen_free_list(blocs) ;
    }    

    pips_debug(3, "end\n");

    return( le ) ;
}
/* }}} */
/* {{{  comp_regions_of_range*/
/* list comp_regions_of_range(range r, context)
 * input    : a loop range (bounds and stride) and the context.
 * output   : the corresponding list of regions.
 * modifies : nothing.
 * comment  :	
 */
list comp_regions_of_range(range r, transformer context)
{
    list le;
    expression el = range_lower(r);
    expression eu = range_upper(r);
    expression ei = range_increment(r);

    pips_debug(3, "begin\n");

    le = comp_regions_of_expression(el, context);
    le = CompRegionsExactUnion(le, comp_regions_of_expression(eu, context), 
			  effects_same_action_p);
    le = CompRegionsExactUnion(le, comp_regions_of_expression(ei, context), 
			  effects_same_action_p);

    pips_debug(3, "end\n");
    return(le);
}
/* }}} */
/* {{{  comp_regions_of_syntax*/
/* list comp_regions_of_syntax(syntax s, transformer context)
 * input    : 
 * output   : 
 * modifies : 
 * comment  :	
 */
list comp_regions_of_syntax(syntax s, transformer context)
{
    list le = NIL, lpropreg = NIL;

    pips_debug(3, "begin\n");

    switch(syntax_tag(s))
    {
      /* {{{  code*/
      case is_syntax_reference:
          le = comp_regions_of_read(syntax_reference(s), context);
          break;
      case is_syntax_range:
          le = comp_regions_of_range(syntax_range(s), context);
          break;
      case is_syntax_call:
          le = comp_regions_of_call(syntax_call(s), context, &lpropreg);
	        /* comp_desc_free(lpropreg);  */
          break;
      default:
          pips_internal_error("unexpected tag %d", syntax_tag(s));
      /* }}} */
    }

    ifdebug(3)
    {
	    pips_debug(3, "Regions of expression  %s :\n",
		       words_to_string(words_syntax(s, NIL)));
	    print_regions(le);
    }

    pips_debug(3, "end\n");
    return(le);
}
/* }}} */
/* {{{  comp_regions_of_expressions*/
/* list comp_regions_of_expressions(list exprs, transformer context)
 * input    : a list of expressions and the current context.
 * output   : the correpsonding list of regions.
 * modifies : nothing.
 * comment  :
 */
list comp_regions_of_expressions(list exprs, transformer context)
{
    list le = NIL;

    pips_debug(3, "begin\n");

    MAP(EXPRESSION, exp,
    {
	    le = CompRegionsExactUnion(le, comp_regions_of_expression(exp, context), 
			      effects_same_action_p);
    },
	  exprs);

    pips_debug(5, "end\n");
    return(le);
}
/* {{{  comp_regions_of_expression*/
/* list comp_regions_of_expression(expression e, transformer context)
 * input    : an expression and the current context
 * output   : the correpsonding list of regions.
 * modifies : nothing.
 * comment  :	
 */
list comp_regions_of_expression(expression e, transformer context)
{
    return(comp_regions_of_syntax(expression_syntax(e), context));
}
/* }}} */

/* }}} */
/* {{{  comp_regions_of_read*/
/* list comp_regions_of_read(reference ref, transformer context)
 * input    : a reference that is read, and the current context.
 * output   : the corresponding list of regions.
 * modifies : nothing.
 * comment  :	
 */
list comp_regions_of_read(reference ref, transformer context)
{
  /* {{{  init*/
  list inds = reference_indices(ref);
  list le = NIL;
  comp_desc Cdesc;
  entity e = reference_variable(ref);
  
  pips_debug(3, "begin\n");
  /* }}} */
    
	   /* {{{  code*/
	   /* {{{  this read reference*/
	   if (! entity_scalar_p(e) )
	   {
	       Cdesc = InitCompDesc(ref, is_action_read);
	       le = CONS(COMP_DESC, Cdesc, le);
	   }
	    /* }}} */
	   /* {{{  rest of the references in an expression*/
	   if (! ENDP(inds)) 
	       le = CompRegionsExactUnion(le, comp_regions_of_expressions(inds, context), effects_same_action_p);

	   /* }}} */
	   /* }}} */
  
  pips_debug(3, "end\n");
  return(le);
}
/* }}} */
/* {{{  comp_regions_of_write*/
/* regions of a reference that is written */
/* list comp_regions_of_write(reference ref, transformer context)
 * input    : a reference that is written, and the current context.
 * output   : the corresponding list of regions.
 * modifies : nothing.
 * comment  :	
 */
list comp_regions_of_write(reference ref, transformer context)
{    
    /* {{{  init*/
    comp_desc Cdesc;
    list le = NIL;
    entity e = reference_variable(ref);
    list inds = reference_indices(ref);
    /* }}} */

    pips_debug(3, "begin\n");

    /* {{{  this write */
    if (! entity_scalar_p(e) )
    {
      Cdesc = InitCompDesc(ref, is_action_write);	
       le = CONS(COMP_DESC, Cdesc, le);
    }
    /* }}} */
    /* {{{  check for arrays in subscripts*/
    if (! ENDP(inds)) 
        le = CompRegionsExactUnion(le, comp_regions_of_expressions(inds, context), effects_same_action_p);

    /* }}} */

    pips_debug(3, "end\n");

    return(le);
}
/* }}} */

/* }}} */


