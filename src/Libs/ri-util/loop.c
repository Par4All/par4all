#include <stdio.h>

#include "genC.h"
#include "ri.h"

#include "misc.h"
#include "ri-util.h"

extern int Nbrdo;

DEFINE_CURRENT_MAPPING(enclosing_loops, list)

void clean_enclosing_loops(void)
{
    STATEMENT_MAPPING_MAP(s, l, gen_free_list((list)l), 
			  get_enclosing_loops_map());
    free_enclosing_loops_map();
}

static void rloops_mapping_of_statement();

static void rloops_mapping_of_unstructured(m, loops, u)
statement_mapping m;
cons *loops;
unstructured u ;
{
    cons *blocs = NIL ;
	  
    CONTROL_MAP(c, {
	rloops_mapping_of_statement(m, loops, control_statement(c));
    }, unstructured_control(u), blocs) ;

    gen_free_list(blocs) ;
}

static void rloops_mapping_of_statement(m, loops, s)
statement_mapping m;
cons *loops;
statement s;
{
    instruction i = statement_instruction(s);

    SET_STATEMENT_MAPPING(m, s, gen_copy_seq(loops));

    switch(instruction_tag(i)) {

      case is_instruction_block:
	MAPL(ps, {
	    rloops_mapping_of_statement(m, loops, STATEMENT(CAR(ps)));
	}, instruction_block(i));
	break;

      case is_instruction_loop: {
	  cons *nloops = gen_copy_seq(loops);
	  Nbrdo++;
	  rloops_mapping_of_statement(m, 
				      gen_nconc(nloops, 
						CONS(STATEMENT, s, NIL)), 
				      loop_body(instruction_loop(i)));
	  break;
      }

      case is_instruction_test:
	rloops_mapping_of_statement(m, 
				    loops, 
				    test_true(instruction_test(i)));
	rloops_mapping_of_statement(m, 
				    loops, 
				    test_false(instruction_test(i)));
	break;

      case is_instruction_call:
      case is_instruction_goto:
	break;

      case is_instruction_unstructured: {
	  rloops_mapping_of_unstructured(m, 
					 loops,
					 instruction_unstructured(i)) ;
	  break ;
      }
	
      default:
	pips_error("rloops_mapping_of_statement", 
		   "unexpected tag %d\n", instruction_tag(i));
    }
}



statement_mapping 
loops_mapping_of_statement(statement stat)
{   
    statement_mapping loops_map;
    loops_map = MAKE_STATEMENT_MAPPING();
    Nbrdo = 0;
    rloops_mapping_of_statement(loops_map, NIL, stat);

    if (get_debug_level() >= 7) {
	STATEMENT_MAPPING_MAP(stat, loops, {
	    fprintf(stderr, "statement %d in loops ", 
		    statement_number((statement) stat));
	    MAP(STATEMENT, s, 
		fprintf(stderr, "%d ", statement_number(s)),
		(list) loops);
	    fprintf(stderr, "\n");
	}, loops_map)
    }
    return(loops_map);
}

static bool 
distributable_statement_p(statement stat, set region)
{
    instruction i = statement_instruction(stat);

    switch(instruction_tag(i)) 
    {
    case is_instruction_block:
	MAPL(ps, {
	    if (distributable_statement_p(STATEMENT(CAR(ps)), 
					  region) == FALSE) {
		return(FALSE);
	    }
	}, instruction_block(i));
	return(TRUE);
	
    case is_instruction_loop:
	region = set_add_element(region, region, (char *) stat);
	return(distributable_statement_p(loop_body(instruction_loop(i)), 
					 region));
	
    case is_instruction_call:
	region = set_add_element(region, region, (char *) stat);
	return(TRUE);

    case is_instruction_goto:
    case is_instruction_unstructured:
    case is_instruction_test:
	return(FALSE);
    default:
	pips_error("distributable_statement_p", 
		   "unexpected tag %d\n", instruction_tag(i));
    }

    return((bool) NULL); /* just to avoid a gcc warning */
}

/* this functions checks if kennedy's algorithm can be applied on the
loop passed as argument. If yes, it returns a set containing all
statements belonging to this loop including the initial loop itself.
otherwise, it returns an undefined set.

Our version of kennedy's algorithm can only be applied on loops
containing no test, goto or unstructured control structures. */

set distributable_loop(l)
statement l;
{
    set r;

    pips_assert("distributable_loop", statement_loop_p(l));

    r = set_make(set_pointer);

    if (distributable_statement_p(l, r)) {
	return(r);
    }

    set_free(r);
    return(set_undefined);
}

/* returns TRUE if loop lo's index is private for this loop */
bool index_private_p(lo)
loop lo;
{
    if( lo == loop_undefined ) {
	pips_error("index_private_p", "Loop undefined\n");
    }

    return((entity) gen_find_eq(loop_index(lo), loop_locals(lo)) != 
	   entity_undefined);
}

/* this function returns the set of all statements belonging to the loop given
   even if the loop contains test, goto or unstructured control structures */
set region_of_loop(l)
statement l;
{
    set r;

    pips_assert("distributable_loop", statement_loop_p(l));

    r = set_make(set_pointer);
    region_of_statement(l,r);
    return(r);
}

void region_of_statement(stat, region)
statement stat;
set region;
{
  instruction i = statement_instruction(stat);  
    
  switch(instruction_tag(i)) {

  case is_instruction_block:
      MAPL(ps, {
	  region_of_statement(STATEMENT(CAR(ps)),region);
      }, instruction_block(i));
      break;

  case is_instruction_loop:{
      region = set_add_element(region, region, (char *) stat);
      region_of_statement(loop_body(instruction_loop(i)),region);
      break;
  }

  case is_instruction_call:
      region = set_add_element(region, region, (char *) stat);
      break;
      
  case is_instruction_goto:
      region = set_add_element(region, region, (char *) stat);
      break;
    
  case is_instruction_test:
      /* The next statement is added by Y.Q. 12/9.*/
      region = set_add_element(region, region, (char *) stat);
      region_of_statement(test_true(instruction_test(i)), region);
      region_of_statement(test_false(instruction_test(i)), region);
      break;

  case is_instruction_unstructured:{
      unstructured u = instruction_unstructured(i);
      cons *blocs = NIL;

      CONTROL_MAP(c, {
	  region_of_statement(control_statement(c), region);
      }, unstructured_control(u), blocs) ;
      
      gen_free_list(blocs) ;
      break;
  }

  default:
      pips_error("region_of_statement", 
		 "unexpected tag %d\n", instruction_tag(i));    
      
  }
}


/************************************** SORT ALL LOCALS AFTER PRIVATIZATION */

static void loop_rwt(loop l)
{
    list /* of entity */ le = loop_locals(l);
    if (le) sort_list_of_entities(le);
}

void sort_all_loop_locals(statement s)
{
    gen_recurse(s, loop_domain, gen_true, loop_rwt);
}
