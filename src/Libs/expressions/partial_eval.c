/*
 * Integer constants calculated by preconditions are replaced by their value.
 * Expressions are evaluated to (ICOEF*SUBEXPR + ISHIFT) in order to perform
 * some simplifications.
 */
/* Hypotheses pour l'implementation:

Toute fonction d'evaluation partielle retourne eformat_undefined
lorsqu'elle n'a rien evalue (ex: lors d'un appel d'une fonction
externe).

eformat.expr NE DOIT JAMAIS partager de structures avec le code. Par
contre, une expression detachee du code peut appartenir a eformat.expr.

Lorsqu'une expression est detachee du code, il faut prendre garde a la
remplacer par expression_undefined. Sinon, le free (dans
regenerate_expression()) causera des degas!

Si une information est ajoutee a eformat_undefined, alors l'expression
est RECOPIEE. Pourtant, eformat.simpler reste FALSE et l'expression
d'origine n'est pas freee, car une seule information ne permet aucune
simplification. A partir de la prise en compte de la seconde
information, de`s qu'eformat est simplife', alors eformat.simpler
devient vrai. L'expression d'origine sera alors free'e lorsque
regenerate_expression().

Des que l'evaluation n'est plus possible, il faut regenerer l'expression
 */

#include <stdio.h>
#include <string.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "text.h"

#include "text-util.h"
#include "database.h"
#include "resources.h"
#include "control.h"
#include "ri-util.h"
#include "pipsdbm.h"
#include "misc.h"

#include "effects-generic.h"
#include "effects-simple.h" /* for print_effects() */
#include "transformer.h"
#include "semantics.h" /* for module_to_value_mappings() */

#include "arithmetique.h"

#include "transformations.h"

static struct eformat  eformat_undefined = {expression_undefined, 1, 0, FALSE};
/* when formating is useless (ie. = (1 * expr + 0)) */



/* Set of enclosing loop indices
 *
 * This set is maintained to avoid useless partial evaluation of loop indices
 * which are very unlikely to be partially evaluable. Loops with only one
 * iteration can be removed by dead code elimination.
 *
 * This set is implemented as a list because the loop nest depth is expected
 * small.
 *
 * It would be nice to take inductive variables into account. Their evaluation
 * is quite long in ocean. We could use the loop transformer instead of the
 * loop index to populate live_loop_indices. But it would be harder to regenerate
 * the set when leaving the loop, unless a copy is made on entrance.
 *
 * A stack is not too attractive, as it should be all visited to make sure
 * a variable is not a live loop index or inductive variable.
 *
 * A multiset might make loop exit easier, but each membership test will be
 * longer.
 */

static list live_loop_indices = list_undefined;

static void
set_live_loop_indices()
{
    pips_assert("set_live_loop_indices", live_loop_indices==list_undefined);
    live_loop_indices = NIL;
}

static void
reset_live_loop_indices()
{
    /* The index set should be empty when leaving partial eval */
    pips_assert("reset_live_loop_indices", ENDP(live_loop_indices));
    if(!ENDP(live_loop_indices)) {
	free_arguments(live_loop_indices);
    }
    live_loop_indices = list_undefined;
}

void dump_live_loop_indices()
{
    dump_arguments(live_loop_indices);
}

static bool
live_loop_index_p(entity i)
{
    return entity_is_argument_p(i, live_loop_indices);
}

static void
add_live_loop_index(entity i)
{
    pips_assert("add_live_index",!live_loop_index_p(i))
	live_loop_indices = gen_nconc(live_loop_indices,
				      CONS(ENTITY, i, NIL));
}

static void
rm_live_loop_index(entity i)
{
    live_loop_indices = arguments_rm_entity(live_loop_indices, i);
}


void init_use_proper_effects(char *module_name)
{
    set_proper_rw_effects((statement_effects)
	db_get_memory_resource(DBR_PROPER_EFFECTS, module_name, TRUE));
    pips_assert("init_use_proper_effects", !proper_rw_effects_undefined_p());
}

/* returns proper effects associated to statement stmt */
effects stmt_to_fx(statement stmt, statement_effects fx_map)
{
    effects fx;

    pips_assert("stmt_prec", stmt != statement_undefined);

    debug(9, "stmt_to_fx", 
	  "Look for effects for statement at %p (ordering %d, number %d):\n", 
	  stmt, statement_ordering(stmt), statement_number(stmt));

    fx = apply_statement_effects(fx_map, stmt);
    ifdebug(5) 
	{
	    print_effects(effects_effects(fx));
	}

    return(fx);
}

bool entity_written_p(entity ent, effects fx)
{
    if(fx==effects_undefined)
	pips_error("entity_written_p", "effects undefined\n");

    MAPL(ftl, {
	effect ft = EFFECT(CAR(ftl));
	if( ENDP(reference_indices(effect_reference(ft)))
	   && same_entity_p(ent, reference_variable(effect_reference(ft)))
	   && action_write_p(effect_action(ft)) )
	    return(TRUE);
    }, effects_effects(fx));

    return(FALSE);
}


void init_use_preconditions(char *module_name)
{
    set_precondition_map( (statement_mapping)
	db_get_memory_resource(DBR_PRECONDITIONS, module_name, TRUE) );
    pips_assert("init_use_preconditions",
		get_precondition_map() != hash_table_undefined);
    if(get_debug_level()==9) {
	transformer_map_print();
    }
}

/*
  cf. load_statement_transformer() in semantics/dbm_interface.c
  */
Psysteme stmt_prec(statement stmt)
{
    transformer t;

    pips_assert("stmt_prec", stmt != statement_undefined);

    debug(9, "stmt_prec", 
	  "Look for preconditions for statement at %p (ordering %d, number %d):\n", 
	  stmt, statement_ordering(stmt), statement_number(stmt));

    t = load_statement_precondition(stmt);

    if(t==(transformer) HASH_UNDEFINED_VALUE) t = transformer_undefined;

    /* pips_assert("stmt_prec", t != transformer_undefined);*/
    
    return(t==transformer_undefined ? 
	   SC_UNDEFINED :
	   (Psysteme)predicate_system(transformer_relation(t)) );

}

void transformer_map_print(void)
{
    FILE * f =stderr;
    hash_table htp = get_precondition_map();

    hash_table_print_header (htp,f);

    HASH_MAP(k, v, {
	fprintf(f, "\nFor statement at %p (ordering %d, number %d):\n", 
		k,
		statement_ordering((statement) k), 
		statement_number((statement) k));
	print_transformer((transformer) v);
    },
	     htp);
}

bool eformat_equivalent_p(struct eformat ef1, struct eformat ef2)
{
     /* should not require anything about expr */
    return( ef1.expr == ef2.expr /* ie expression_eq(ef1.expr, ef2.expr) */
	   && ef1.icoef == ef2.icoef
	   && ef1.ishift == ef2.ishift );
}

void print_eformat(struct eformat ef, char *name)
{
    (void) printf("eformat %s = %d x EXPR + %d, %ssimpler, with EXPR:\n", 
	   name, ef.icoef, ef.ishift, (ef.simpler ? "" : "NOT "));
    print_expression(ef.expr);
}

void partial_eval_expression_and_regenerate(expression *ep, Psysteme ps, effects fx)
{
    struct eformat ef;

    ef = partial_eval_expression(*ep, ps, fx);

    if (get_debug_level()>=5)
	print_eformat(ef, "before regenerate");

    regenerate_expression(&ef, ep);

    if(get_debug_level()>=5 && !expression_consistent_p(*ep)) {
	pips_error("partial_eval_expression_and_regenerate", "bad evaluation");
    }
}

struct eformat partial_eval_expression_and_copy(expression expr, Psysteme ps, effects fx)
{
    struct eformat ef;

    ef= partial_eval_expression(expr, ps, fx);

    if(eformat_equivalent_p(ef,eformat_undefined)) {
	ef.expr = copy_expression(expr);
    }

    return(ef);
}

struct eformat partial_eval_expression(expression e, Psysteme ps, effects fx)
{
    struct eformat ef = partial_eval_syntax(e, ps, fx);

    return ef;
}

struct eformat partial_eval_syntax(expression e, Psysteme ps, effects fx)
{
    struct eformat ef;
    syntax s = expression_syntax(e);

    switch (syntax_tag(s)) {
      case is_syntax_reference:
	ef = partial_eval_reference(e, ps, fx);
	break;
      case is_syntax_range:
	ef = eformat_undefined;
	break;
      case is_syntax_call:
	ef = partial_eval_call(e, ps, fx);
	break;
      default:
	pips_error( "partial_eval_syntax", "case default\n");
	abort();
    }

    if (get_debug_level()==9)
	print_eformat(ef, "after partial_eval_syntax");

    return(ef);
}

struct eformat partial_eval_reference(expression e, Psysteme ps, effects fx)
{
    reference r;
    entity var;
    Pbase base_min = BASE_UNDEFINED;
    
    pips_assert("partial_eval_reference", 
		syntax_reference_p(expression_syntax(e)));
    r = syntax_reference(expression_syntax(e));
    var = reference_variable(r);

    if(reference_indices(r) != NIL) {
	MAPL(li, {
	    expression expr = EXPRESSION(CAR(li));

	    partial_eval_expression_and_regenerate(&expr, ps, fx);
	    EXPRESSION(CAR(li)) = expr;
	}, reference_indices(r));

	debug(9, "partial_eval_reference", "Array elements not evaluated\n");
	return(eformat_undefined);
    }

    if(!type_variable_p(entity_type(var)) || 
       !basic_int_p(variable_basic(type_variable(entity_type(var)))) ) {
	debug(9, "partial_eval_reference",
	      "Reference to a non-scalar-integer variable %s cannot be evaluated\n",
	      entity_name(var));
	return(eformat_undefined);
    }

    if (SC_UNDEFINED_P(ps)) {
	debug(9, "partial_eval_reference", "No precondition information\n");
	pips_error("partial_eval_reference", "Probably corrupted precondition\n");
	return(eformat_undefined);
    }

    if(entity_written_p(var, fx)) {
	/* entity cannot be replaced */
	debug(9, "partial_eval_reference",
	      "Write Reference to variable %s cannot be evaluated\n",
	      entity_name(var));
	return(eformat_undefined);
    }

    if(live_loop_index_p(var)) {
	debug(9, "partial_eval_reference",
	      "Index %s cannot be evaluated\n",
	      entity_name(var));
	return(eformat_undefined);
    }

    /* faire la Variable */
    /* verification de la presence de la variable dans ps */
    base_min = sc_to_minimal_basis(ps);
    if(base_contains_variable_p(base_min, (Variable) var)) {
	    bool feasible;
	    Value min, max;
	    Psysteme ps1 = sc_dup(ps);

	    /* feasible = sc_minmax_of_variable(ps1, (Variable)var, &min, &max); */
	    feasible = sc_minmax_of_variable2(ps1, (Variable)var, &min, &max);
	    if (! feasible) {
		user_warning("partial_eval_reference", 
			     "Not feasible system:"
			     " module contains some dead code.\n");
	    }
	    if ( value_eq(min,max) ) {
		struct eformat ef;

		/* var is constant and has to be replaced */
		if ( get_debug_level() == 9) {
		    debug(9, "partial_eval_reference", 
			  "Constant to replace: \n");
		    print_expression(e);
		}

		ef.icoef = 0;
		ef.ishift = VALUE_TO_INT(min);
		ef.expr = expression_undefined;
		ef.simpler = TRUE;
		return(ef);

		/*		new_expr=int_expr((int)min); */		
		/* replace expression_normalized(e) with 
		   expression_normalized(new_expr) */
		   /*		free_normalized(expression_normalized(e));
				expression_normalized(e) = expression_normalized(new_expr);
				expression_normalized(new_expr) = normalized_undefined; */

		       /* replace expression_syntax(e) with 
			  expression_syntax(new_expr) */
			  /*
			    free_syntax(expression_syntax((e)));
			    expression_syntax(e) = expression_syntax(new_expr);
			    expression_syntax(new_expr) = syntax_undefined;
			    
			    free_expression(new_expr);
			    
			    if ( get_debug_level() == 9) {
			    debug(9, "partial_eval_reference", 
			    "Constant replaced by expression: \n");
			    print_expression(e);
			    expression_consistent_p(e);
			    pips_assert("partial_eval_reference", 
			    syntax_call_p(expression_syntax(e)));
			    } */
			  }
	    /*	    return(entity_initial(call_function(syntax_call(expression_syntax(e)))));
	     */
	    return(eformat_undefined);
	}
    base_rm(base_min);
    return(eformat_undefined);
}

void partial_eval_call_and_regenerate(call ca, Psysteme ps, effects fx)
{
    pips_assert("partial_eval_call_and_regenerate", 
		ca!= call_undefined);

    MAPL(le, {
	expression exp = EXPRESSION(CAR(le));

	partial_eval_expression_and_regenerate(&exp, ps, fx);
	EXPRESSION(CAR(le))= exp;
    }, call_arguments(ca));
}


struct eformat partial_eval_call(expression exp, Psysteme ps, effects fx)
{
    call ec;
    entity func;
    value vinit;
    struct eformat ef;
    
    pips_assert("partial_eval_call", 
		syntax_call_p(expression_syntax(exp)));
    ec = syntax_call(expression_syntax(exp));

    func = call_function(ec);
    vinit = entity_initial(func);
	
    switch (value_tag(vinit)) {
      case is_value_intrinsic:
      case is_value_unknown: {
	  /* it might be an intrinsic function */
	  cons *la = call_arguments(ec);
	  int token;

	  if ((token = IsUnaryOperator(func)) > 0)
	      ef = partial_eval_unary_operator(func, la, ps, fx);
	  else if ((token = IsBinaryOperator(func)) > 0)
	      ef = partial_eval_binary_operator(func, la, ps, fx);
	  else if ((token = IsNaryOperator(func)) > 0 && gen_length(la)==2)
	      ef = partial_eval_binary_operator(func, la, ps, fx);
	  else {
	      MAPL(le, {
		  expression expr = EXPRESSION(CAR(le));

		  partial_eval_expression_and_regenerate(&expr, ps, fx);
	      }, call_arguments(ec) );
	      ef = eformat_undefined;
	  }
      }
	break;
      case is_value_constant:
	if(integer_constant_p(func, &ef.ishift)) {
	    ef.icoef = 0;
	    ef.expr = expression_undefined;
	    ef.simpler = FALSE;
	}
	else ef = eformat_undefined;
	break;
      case is_value_symbolic:
	if(integer_symbolic_constant_p(func, &ef.ishift)) {
	    ef.icoef = 0;
	    ef.expr = expression_undefined;
	    ef.simpler = TRUE;
	}
	else ef = eformat_undefined;
	break;
      case is_value_code:
	ef = eformat_undefined;
	break;
      default:
	pips_error("partial_eval_call", "case default\n");
    }
    return(ef);
}

struct eformat partial_eval_unary_operator(entity func, cons *la, Psysteme ps, effects fx)
{
    struct eformat ef;
    expression *sub_ep;

    pips_assert("partial_eval_unary_operator", gen_length(la)==1);
    sub_ep = /*&EXPRESSION(CAR(la));*/ (expression*) REFCAR(la);

    if (strcmp(entity_local_name(func), UNARY_MINUS_OPERATOR_NAME) == 0) {
	ef = partial_eval_expression_and_copy(*sub_ep, ps, fx);

	if(ef.icoef==0
	   || ((ef.icoef<0 || ef.icoef>1) 
	       && (ef.ishift<=0)) 
	   ) {
	    ef.simpler= TRUE;
	}

	ef.icoef= -(ef.icoef);
	ef.ishift= -(ef.ishift);
    }
    else {
	/* operator is unknown */
	partial_eval_expression_and_regenerate(sub_ep, ps, fx);
	ef= eformat_undefined;
    }
    return(ef);
}


#define PERFORM_ADDITION 1
#define PERFORM_SUBTRACTION 2
#define PERFORM_MULTIPLICATION 3
#define PERFORM_DIVISION 4
#define PERFORM_POWER 5
#define PERFORM_MODULO 6
#define PERFORM_MINIMUM 7
#define PERFORM_MAXIMUM 8

struct eformat partial_eval_mult_operator(expression *ep1,
					  expression *ep2,
					  Psysteme ps,
					  effects fx)
{
    struct eformat ef, ef1, ef2;

	ef1 = partial_eval_expression_and_copy(*ep1, ps, fx);
	ef2 = partial_eval_expression_and_copy(*ep2, ps, fx);

	if(ef1.icoef==0 && ef2.icoef==0) {
	    ef.icoef=0;
	    ef.expr=expression_undefined;
	    ef.ishift= ef1.ishift * ef2.ishift;
	    ef.simpler= TRUE;
	}
	else if(ef1.icoef!=0 && ef2.icoef!=0) {
	    if(ef2.icoef!=1 && ef2.ishift==0) {
		expression *ep;
		/* exchange ef1 and ef2 (see later) */
		ef=ef2; ef2=ef1; ef1=ef; ef= eformat_undefined;
		ep=ep2; ep2=ep1; ep1=ep;
	    }
	    if(ef1.icoef!=1 && ef1.ishift==0) {
		ef.simpler= ef1.simpler;
		ef.icoef= ef1.icoef;
		regenerate_expression(&ef2, ep2);
		ef.expr= MakeBinaryCall(entity_intrinsic(MULTIPLY_OPERATOR_NAME),
					ef1.expr, *ep2);
		ef.ishift= 0;
	    }
	    else { /* cannot optimize */
		regenerate_expression(&ef1, ep1);
		regenerate_expression(&ef2, ep2);
		
		ef= eformat_undefined;
	    }
	}
	else {
	    if(ef2.icoef==0) {
		expression *ep;
		/* exchange ef1 and ef2 (see later) */
		ef=ef2; ef2=ef1; ef1=ef; ef= eformat_undefined;
		ep=ep2; ep2=ep1; ep1=ep;
	    }
	    /* here we know that ef1.ecoef==0 and ef2.ecoef!=0 */
	    if(ef1.ishift==0) {
		ef.icoef= 0;
		ef.expr= expression_undefined;
		ef.ishift= 0;
		ef.simpler= TRUE;
		regenerate_expression(&ef2, ep2);
	    }
	    else {
		ef.icoef= ef1.ishift * ef2.icoef;
		ef.expr= ef2.expr;
		ef.ishift= ef1.ishift * ef2.ishift;
		ef.simpler= (ef1.ishift==1 || ef2.icoef!=1 
			     || ef1.simpler || ef2.simpler);
	    }
	}

    return ef;
}

struct eformat partial_eval_plus_or_minus_operator(int token,
						   expression *ep1,
						   expression *ep2,
						   Psysteme ps,
						   effects fx)
{
    struct eformat ef, ef1, ef2;

	ef1 = partial_eval_expression_and_copy(*ep1, ps, fx);
	ef2 = partial_eval_expression_and_copy(*ep2, ps, fx);

	/* generate ef.icoef and ef.expr */
	if( (ef1.icoef==ef2.icoef || ef1.icoef==-ef2.icoef)
	   && (ef1.icoef<-1 || ef1.icoef>1) ) {
	    /* factorize */
	    ef.simpler=TRUE;
	    if( (token==PERFORM_ADDITION && ef1.icoef==ef2.icoef)
	       || (token==PERFORM_SUBTRACTION && ef1.icoef==-ef2.icoef) ) {
		/* addition */
		ef.expr= MakeBinaryCall(entity_intrinsic(PLUS_OPERATOR_NAME),
					ef1.expr, 
					ef2.expr);
		ef.icoef= ef1.icoef;
	    }
	    else if( (ef1.icoef>1)
		    && (token==PERFORM_SUBTRACTION ? (ef2.icoef>0) : (ef2.icoef<0)) ) {
		/* substraction e1-e2 */
		ef.expr= MakeBinaryCall(entity_intrinsic(MINUS_OPERATOR_NAME),
					ef1.expr, 
					ef2.expr);
		ef.icoef= ef1.icoef;
	    }
	    else {
		/* substraction e2-e1 */
		ef.expr= MakeBinaryCall(entity_intrinsic(MINUS_OPERATOR_NAME),
					ef2.expr, 
					ef1.expr);
		ef.icoef= -ef1.icoef;
	    }
	}
	else if(ef1.icoef!=0 && ef2.icoef!=0) {
	    int c1 = ef1.icoef;
	    int c2 = (token==PERFORM_SUBTRACTION ? -ef2.icoef : ef2.icoef);
	    expression e1= generate_monome((c1>0 ? c1: -c1), ef1.expr);
	    expression e2= generate_monome((c2>0 ? c2: -c2), ef2.expr);
	    /* generate without factorize */
	    ef.simpler= (ef1.simpler || ef2.simpler); /* not precise ?? */
	    if(c1*c2>0) {
		ef.expr= MakeBinaryCall(entity_intrinsic(PLUS_OPERATOR_NAME),
					e1, e2);
		ef.icoef= (c1>0 ? 1 : -1);
	    }
	    else if(c1>0) {
		ef.expr= MakeBinaryCall(entity_intrinsic(MINUS_OPERATOR_NAME),
					e1, e2);
		ef.icoef= 1;
	    }
	    else {
		ef.expr= MakeBinaryCall(entity_intrinsic(MINUS_OPERATOR_NAME),
					e2, e1);
		ef.icoef= 1;
	    }
	}
	else {
	    ef.simpler= (ef1.simpler || ef2.simpler);
	    if(ef1.icoef==0) {
		/* CA (9/9/97) condition <0 added in order to simplify 
		   also expression like (J)+(-1) in (J-1)    */
		if(ef1.ishift<=0) ef.simpler=TRUE;
		ef.expr=ef2.expr;
		ef.icoef=(token==PERFORM_SUBTRACTION ? -ef2.icoef : ef2.icoef);
	    }
	    else {
		if(ef2.ishift<=0) ef.simpler=TRUE;
		ef.expr=ef1.expr;
		ef.icoef=ef1.icoef;
	    }
	}

	/* generate ef.ishift */
	if  ((ef1.icoef==0 || ef1.ishift!=0)
	    && (ef2.icoef==0 || ef2.ishift!=0))
        {
	    /* simplify shifts */
	    ef.simpler= TRUE;
	}
	
	ef.ishift= (token==PERFORM_SUBTRACTION ? 
		    ef1.ishift-ef2.ishift : ef1.ishift+ef2.ishift);

    return ef;
}

struct eformat partial_eval_plus_operator(expression *ep1,
					  expression *ep2,
					  Psysteme ps,
					  effects fx)
{
    struct eformat ef;

    ef = partial_eval_plus_or_minus_operator(PERFORM_ADDITION,
					     ep1, ep2, ps, fx);

    return ef;
}

struct eformat partial_eval_minus_operator(expression *ep1,
					  expression *ep2,
					  Psysteme ps,
					  effects fx)
{
    struct eformat ef;

    ef = partial_eval_plus_or_minus_operator(PERFORM_SUBTRACTION,
					     ep1, ep2, ps, fx);

    return ef;
}

struct eformat partial_eval_div_or_mod_operator(int token,
						   expression *ep1,
						   expression *ep2,
						   Psysteme ps,
						   effects fx)
{
    struct eformat ef, ef1, ef2;

    ef1 = partial_eval_expression_and_copy(*ep1, ps, fx);
    ef2 = partial_eval_expression_and_copy(*ep2, ps, fx);

    if( ef2.icoef==0 && ef2.ishift == 0 ) 
	user_error("partial_eval_div_or_mod_operator", 
		   "division by zero!\n");
    if( token==PERFORM_DIVISION && ef2.icoef==0 
       && (ef1.ishift % ef2.ishift)==0 
       && (ef1.icoef % ef2.ishift)==0 ) {
	/* integer division does NOT commute with in any */
	/* multiplication -> only performed if "exact" */
	ef.simpler= TRUE;
	ef.icoef= ef1.icoef / ef2.ishift;
	ef.ishift= ef1.ishift / ef2.ishift;
	ef.expr= ef1.expr;
    }
    else if(ef1.icoef==0 && ef2.icoef==0) {
	ef.simpler= TRUE;
	ef.icoef= 0;
	ef.expr= expression_undefined;
	if (token==PERFORM_DIVISION) { /* refer to Fortran77 chap 6.1.5 */
	    ef.ishift= FORTRAN_DIV(ef1.ishift, ef2.ishift);
	}
	else { /* tocken==PERFORM_MODULO */
	    ef.ishift= FORTRAN_MOD(ef1.ishift, ef2.ishift);
	}
    }
    else {
	regenerate_expression(&ef1, ep1);
	regenerate_expression(&ef2, ep2);
	ef= eformat_undefined;
    }
    return ef;
}

struct eformat partial_eval_div_operator(expression *ep1,
					  expression *ep2,
					  Psysteme ps,
					  effects fx)
{
    struct eformat ef;

    ef = partial_eval_div_or_mod_operator(PERFORM_DIVISION,
					     ep1, ep2, ps, fx);

    return ef;
}

struct eformat partial_eval_mod_operator(expression *ep1,
					  expression *ep2,
					  Psysteme ps,
					  effects fx)
{
    struct eformat ef;

    ef = partial_eval_div_or_mod_operator(PERFORM_MODULO,
					     ep1, ep2, ps, fx);

    return ef;
}

struct eformat partial_eval_min_or_max_operator(int token,
						   expression *ep1,
						   expression *ep2,
						   Psysteme ps,
						   effects fx)
{
    struct eformat ef, ef1, ef2;

    ef1 = partial_eval_expression_and_copy(*ep1, ps, fx);
    ef2 = partial_eval_expression_and_copy(*ep2, ps, fx);

    if( ef1.icoef == 0 && ef2.icoef == 0 ) {
	ef.icoef = 0;
	ef.ishift = (token==PERFORM_MAXIMUM)? MAX(ef1.ishift,ef2.ishift):
	    MIN(ef1.ishift,ef2.ishift);
	ef.expr = expression_undefined;
	ef.simpler = TRUE;
    }
    else {
	regenerate_expression(&ef1, ep1);
	regenerate_expression(&ef2, ep2);
	ef = eformat_undefined;
    }

    return ef;
}

struct eformat partial_eval_min_operator(expression *ep1,
					  expression *ep2,
					  Psysteme ps,
					  effects fx)
{
    struct eformat ef;

    ef = partial_eval_min_or_max_operator(PERFORM_MINIMUM,
					     ep1, ep2, ps, fx);

    return ef;
}

struct eformat partial_eval_max_operator(expression *ep1,
					  expression *ep2,
					  Psysteme ps,
					  effects fx)
{
    struct eformat ef;

    ef = partial_eval_min_or_max_operator(PERFORM_MAXIMUM,
					  ep1, ep2, ps, fx);

    return ef;
}

struct eformat partial_eval_power_operator(expression *ep1,
					   expression *ep2,
					   Psysteme ps,
					   effects fx)
{
    struct eformat ef, ef1, ef2;

    ef1 = partial_eval_expression_and_copy(*ep1, ps, fx);
    ef2 = partial_eval_expression_and_copy(*ep2, ps, fx);

    if( ef1.icoef == 0 && ef2.icoef == 0 && ef2.ishift >= 0) {
	ef.icoef = 0;
	ef.ishift = ipow(ef1.ishift, ef2.ishift);
	ef.expr = expression_undefined;
	ef.simpler = TRUE;
    }
    else {
	regenerate_expression(&ef1, ep1);
	regenerate_expression(&ef2, ep2);
	ef = eformat_undefined;
    }

    return ef;
}

static struct perform_switch {
    string operator_name;
    struct eformat (*binary_operator)(expression *, expression *, Psysteme, effects);
} binary_operator_switch[] = {
    {PLUS_OPERATOR_NAME, partial_eval_plus_operator},
    {MINUS_OPERATOR_NAME, partial_eval_minus_operator},
    {MULTIPLY_OPERATOR_NAME, partial_eval_mult_operator},
    {DIVIDE_OPERATOR_NAME, partial_eval_div_operator},
    {POWER_OPERATOR_NAME, partial_eval_power_operator},
    {MODULO_OPERATOR_NAME, partial_eval_mod_operator},
    {MIN0_OPERATOR_NAME, partial_eval_min_operator},
    {MIN_OPERATOR_NAME, partial_eval_min_operator},
    {MAX0_OPERATOR_NAME, partial_eval_max_operator},
    {MAX_OPERATOR_NAME, partial_eval_max_operator},
    {0 , 0}
};

struct eformat partial_eval_binary_operator(entity func,
						cons *la,
						Psysteme ps,
						effects fx)
{
    struct eformat ef;
    expression *ep1, *ep2;
    int i = 0;
    struct eformat (*binary_partial_eval_operator)(expression *, 
				      expression *, 
				      Psysteme, 
				      effects) = 0;

    pips_assert("partial_eval_binary_operator", gen_length(la)==2);
    ep1= (expression*) REFCAR(la);
    ep2= (expression*) REFCAR(CDR(la));

    while (binary_operator_switch[i].operator_name!=NULL) {
	if (strcmp(binary_operator_switch[i].operator_name,
		   entity_local_name(func))==0) {
	    binary_partial_eval_operator = 
		binary_operator_switch[i].binary_operator;
	    break;
	}
	i++;
    }

    if (binary_partial_eval_operator!=0)
	ef = binary_partial_eval_operator (ep1, ep2, ps, fx);
    else {
	partial_eval_expression_and_regenerate(ep1, ps, fx);
	partial_eval_expression_and_regenerate(ep2, ps, fx);
	ef = eformat_undefined;
    }

    return ef;
}

struct eformat partial_eval_binary_operator_old(entity func,
						cons *la,
						Psysteme ps,
						effects fx)
{
    struct eformat ef, ef1, ef2;
    expression *ep1, *ep2;
    int token= -1;

    pips_assert("partial_eval_binary_operator", gen_length(la)==2);
    ep1= (expression*) REFCAR(la);
    ep2= (expression*) REFCAR(CDR(la));

    if (strcmp(entity_local_name(func), MINUS_OPERATOR_NAME) == 0) {
	token = PERFORM_SUBTRACTION;
    }
    if (strcmp(entity_local_name(func), PLUS_OPERATOR_NAME) == 0) {
	token = PERFORM_ADDITION;
    }
    if (strcmp(entity_local_name(func), MULTIPLY_OPERATOR_NAME) == 0) {
	token = PERFORM_MULTIPLICATION;
    }
    if (strcmp(entity_local_name(func), DIVIDE_OPERATOR_NAME) == 0) {
	token = PERFORM_DIVISION;
    }
    if (strcmp(entity_local_name(func), "MOD") == 0) {
	token = PERFORM_MODULO;
    }

    if ( token==PERFORM_ADDITION || token==PERFORM_SUBTRACTION ) {
	ef1 = partial_eval_expression_and_copy(*ep1, ps, fx);
	ef2 = partial_eval_expression_and_copy(*ep2, ps, fx);

	/* generate ef.icoef and ef.expr */
	if( (ef1.icoef==ef2.icoef || ef1.icoef==-ef2.icoef)
	   && (ef1.icoef<-1 || ef1.icoef>1) ) {
	    /* factorize */
	    ef.simpler=TRUE;
	    if( (token==PERFORM_ADDITION && ef1.icoef==ef2.icoef)
	       || (token==PERFORM_SUBTRACTION && ef1.icoef==-ef2.icoef) ) {
		/* addition */
		ef.expr= MakeBinaryCall(entity_intrinsic(PLUS_OPERATOR_NAME),
					ef1.expr, 
					ef2.expr);
		ef.icoef= ef1.icoef;
	    }
	    else if( (ef1.icoef>1)
		    && (token==PERFORM_SUBTRACTION ? (ef2.icoef>0) : (ef2.icoef<0)) ) {
		/* substraction e1-e2 */
		ef.expr= MakeBinaryCall(entity_intrinsic(MINUS_OPERATOR_NAME),
					ef1.expr, 
					ef2.expr);
		ef.icoef= ef1.icoef;
	    }
	    else {
		/* substraction e2-e1 */
		ef.expr= MakeBinaryCall(entity_intrinsic(MINUS_OPERATOR_NAME),
					ef2.expr, 
					ef1.expr);
		ef.icoef= -ef1.icoef;
	    }
	}
	else if(ef1.icoef!=0 && ef2.icoef!=0) {
	    int c1 = ef1.icoef;
	    int c2 = (token==PERFORM_SUBTRACTION ? -ef2.icoef : ef2.icoef);
	    expression e1= generate_monome((c1>0 ? c1: -c1), ef1.expr);
	    expression e2= generate_monome((c2>0 ? c2: -c2), ef2.expr);
	    /* generate without factorize */
	    ef.simpler= (ef1.simpler || ef2.simpler); /* not precise ?? */
	    if(c1*c2>0) {
		ef.expr= MakeBinaryCall(entity_intrinsic(PLUS_OPERATOR_NAME),
					e1, e2);
		ef.icoef= (c1>0 ? 1 : -1);
	    }
	    else if(c1>0) {
		ef.expr= MakeBinaryCall(entity_intrinsic(MINUS_OPERATOR_NAME),
					e1, e2);
		ef.icoef= 1;
	    }
	    else {
		ef.expr= MakeBinaryCall(entity_intrinsic(MINUS_OPERATOR_NAME),
					e2, e1);
		ef.icoef= 1;
	    }
	}
	else {
	    ef.simpler= (ef1.simpler || ef2.simpler);
	    if(ef1.icoef==0) {
		if(ef1.ishift==0) ef.simpler=TRUE;
		ef.expr=ef2.expr;
		ef.icoef=(token==PERFORM_SUBTRACTION ? -ef2.icoef : ef2.icoef);
	    }
	    else {
		if(ef2.ishift==0) ef.simpler=TRUE;
		ef.expr=ef1.expr;
		ef.icoef=ef1.icoef;
	    }
	}

	/* generate ef.ishift */
	if ( (ef1.icoef==0 || ef1.ishift!=0)
	    && (ef2.icoef==0 || ef2.ishift!=0) ) {
	    /* simplify shifts */
	    ef.simpler= TRUE;
	}
	ef.ishift= (token==PERFORM_SUBTRACTION ? 
		    ef1.ishift-ef2.ishift : ef1.ishift+ef2.ishift);
    }
    else if( token==PERFORM_MULTIPLICATION ) {
	ef1 = partial_eval_expression_and_copy(*ep1, ps, fx);
	ef2 = partial_eval_expression_and_copy(*ep2, ps, fx);

	if(ef1.icoef==0 && ef2.icoef==0) {
	    ef.icoef=0;
	    ef.expr=expression_undefined;
	    ef.ishift= ef1.ishift * ef2.ishift;
	    ef.simpler= TRUE;
	}
	else if(ef1.icoef!=0 && ef2.icoef!=0) {
	    if(ef2.icoef!=1 && ef2.ishift==0) {
		expression *ep;
		/* exchange ef1 and ef2 (see later) */
		ef=ef2; ef2=ef1; ef1=ef; ef= eformat_undefined;
		ep=ep2; ep2=ep1; ep1=ep;
	    }
	    if(ef1.icoef!=1 && ef1.ishift==0) {
		ef.simpler= ef1.simpler;
		ef.icoef= ef1.icoef;
		regenerate_expression(&ef2, ep2);
		ef.expr= MakeBinaryCall(entity_intrinsic(MULTIPLY_OPERATOR_NAME),
					ef1.expr, *ep2);
		ef.ishift= 0;
	    }
	    else { /* cannot optimize */
		regenerate_expression(&ef1, ep1);
		regenerate_expression(&ef2, ep2);
		
		ef= eformat_undefined;
	    }
	}
	else {
	    if(ef2.icoef==0) {
		expression *ep;
		/* exchange ef1 and ef2 (see later) */
		ef=ef2; ef2=ef1; ef1=ef; ef= eformat_undefined;
		ep=ep2; ep2=ep1; ep1=ep;
	    }
	    /* here we know that ef1.ecoef==0 and ef2.ecoef!=0 */
	    if(ef1.ishift==0) {
		ef.icoef= 0;
		ef.expr= expression_undefined;
		ef.ishift= 0;
		ef.simpler= TRUE;
		regenerate_expression(&ef2, ep2);
	    }
	    else {
		ef.icoef= ef1.ishift * ef2.icoef;
		ef.expr= ef2.expr;
		ef.ishift= ef1.ishift * ef2.ishift;
		ef.simpler= (ef1.ishift==1 || ef2.icoef!=1 
			     || ef1.simpler || ef2.simpler);
	    }
	}
    }
    else if(token==PERFORM_DIVISION || token==PERFORM_MODULO) {
	ef1 = partial_eval_expression_and_copy(*ep1, ps, fx);
	ef2 = partial_eval_expression_and_copy(*ep2, ps, fx);

	if( ef2.icoef==0 && ef2.ishift == 0 ) 
	    user_error("partial_eval_binary_operator", 
		       "division by zero!\n");
	if( token==PERFORM_DIVISION && ef2.icoef==0 
	   && (ef1.ishift % ef2.ishift)==0 
	   && (ef1.icoef % ef2.ishift)==0 ) {
	    /* integer division does NOT commute with in any */
	    /* multiplication -> only performed if "exact" */
	    ef.simpler= TRUE;
	    ef.icoef= ef1.icoef / ef2.ishift;
	    ef.ishift= ef1.ishift / ef2.ishift;
	    ef.expr= ef1.expr;
	}
	else if(ef1.icoef==0 && ef2.icoef==0) {
	    ef.simpler= TRUE;
	    ef.icoef= 0;
	    ef.expr= expression_undefined;
	    if (token==PERFORM_DIVISION) { /* refer to Fortran77 chap 6.1.5 */
		ef.ishift= FORTRAN_DIV(ef1.ishift, ef2.ishift);
	    }
	    else { /* tocken==PERFORM_MODULO */
		ef.ishift= FORTRAN_MOD(ef1.ishift, ef2.ishift);
	    }
	}
	else {
	    regenerate_expression(&ef1, ep1);
	    regenerate_expression(&ef2, ep2);
	    ef= eformat_undefined;
	}
    }
    else {
	partial_eval_expression_and_regenerate(ep1, ps, fx);
	partial_eval_expression_and_regenerate(ep2, ps, fx);
	ef= eformat_undefined;
    }
    return(ef);
}

/* in order to regenerate expression from eformat.;
 * optimized so that it can be called for any compatible ef and *ep;
 * result in *ep.
 */
void regenerate_expression(struct eformat *efp, expression *ep)
{
    if(eformat_equivalent_p(*efp,eformat_undefined)) {
	/* nothing to do because expressions are the same */
    }
    else if(!efp->simpler) {
	/* simply free efp->expr */
	/* ?? ******commented out for debug******* */
	/*free_expression(efp->expr);*/
	efp->expr= expression_undefined; /* useless */
    }
    else {
	expression tmp_expr;

	/* *ep must be freed */
	/* ?? ******commented out for debug******* */
	/* free_expression(*ep); */

	if(efp->icoef != 0) {
	    /* check */
	    pips_assert("regenerate_expression", 
			efp->expr != expression_undefined);

	    if(efp->icoef == 1) {
		tmp_expr= efp->expr;
	    }
	    else if(efp->icoef == -1) {
		/* generate unary_minus */
		tmp_expr= MakeUnaryCall(entity_intrinsic(UNARY_MINUS_OPERATOR_NAME),
					 efp->expr);
	    }
	    else {
		/* generate product */
		tmp_expr= MakeBinaryCall(entity_intrinsic(MULTIPLY_OPERATOR_NAME),
					 int_expr(efp->icoef), 
					 efp->expr);
	    }

	    if(efp->ishift != 0) {
		/* generate addition or substraction */
		string operator = (efp->ishift>0) ? PLUS_OPERATOR_NAME 
		    : MINUS_OPERATOR_NAME;
		tmp_expr= MakeBinaryCall(entity_intrinsic(operator),
					 tmp_expr, 
					 int_expr(ABS(efp->ishift)));
	    }
	}
	else {
	    /* check */
	    pips_assert("regenerate_expression", 
			efp->expr == expression_undefined);
	    /* final expression is constant efp->ishift */
	    tmp_expr= int_expr(efp->ishift);
	}

	/* replace *ep by tmp_expr */
	*ep = tmp_expr;
    }
}

expression generate_monome(int coef, expression expr)
{
    if(coef==0) {
	pips_assert("generate_monome", expr==expression_undefined);
	return(int_expr(0));
    }
    pips_assert("generate_monome", expr!=expression_undefined);
    if(coef==1) {
	return(expr);
    }
    if(coef==-1) {
	return(MakeUnaryCall(entity_intrinsic(UNARY_MINUS_OPERATOR_NAME),
			     expr));
    }
    return(MakeBinaryCall(entity_intrinsic(MULTIPLY_OPERATOR_NAME),
			  int_expr(coef),
			  expr));
}


void recursiv_partial_eval(statement stmt)
{
    instruction inst = statement_instruction(stmt);
    statement_effects fx_map = get_proper_rw_effects();

    debug(8, "recursiv_partial_eval", "begin with tag %d\n", 
	  instruction_tag(inst));

    switch(instruction_tag(inst)) {
      case is_instruction_block :
	MAPL( sts, {
	    statement s = STATEMENT(CAR(sts));

	    recursiv_partial_eval(s);
	}, instruction_block(inst));
	break;
      case is_instruction_test : {
	  test t = instruction_test(inst);
	  partial_eval_expression_and_regenerate(&test_condition(t), 
						 stmt_prec(stmt), 
						 stmt_to_fx(stmt,fx_map));
	  recursiv_partial_eval(test_true(t));
	  recursiv_partial_eval(test_false(t));
	  if(get_debug_level()>=9) {
	      print_text(stderr, text_statement(entity_undefined, 0, stmt));
	      pips_assert("recursiv_partial_eval", statement_consistent_p(stmt));
	  }
	  break;
      }
      case is_instruction_loop : {
	  loop l = instruction_loop(inst);
	  range r = loop_range(l);
	  partial_eval_expression_and_regenerate(&range_lower(r), 
						 stmt_prec(stmt), 
						 stmt_to_fx(stmt,fx_map));
	  partial_eval_expression_and_regenerate(&range_upper(r), 
						 stmt_prec(stmt), 
						 stmt_to_fx(stmt,fx_map));
	  partial_eval_expression_and_regenerate(&range_increment(r), 
						 stmt_prec(stmt), 
						 stmt_to_fx(stmt,fx_map));
	  add_live_loop_index(loop_index(l));
	  recursiv_partial_eval(loop_body(l));
	  rm_live_loop_index(loop_index(l));

	  if(get_debug_level()>=9) {
	      print_text(stderr, text_statement(entity_undefined, 0, stmt));
	      pips_assert("recursiv_partial_eval", statement_consistent_p(stmt));
	  }
	  break;
      }      
    case is_instruction_whileloop : {
	  whileloop l = instruction_whileloop(inst);
	  partial_eval_expression_and_regenerate(&whileloop_condition(l), 
						 stmt_prec(stmt), 
						 stmt_to_fx(stmt,fx_map));
	
	 recursiv_partial_eval(whileloop_body(l));
	 
	  if(get_debug_level()>=9) {
	      print_text(stderr, text_statement(entity_undefined, 0, stmt));
	      pips_assert("recursiv_partial_eval", statement_consistent_p(stmt));
	  }
	  break;
      }
      case is_instruction_call : {
	  partial_eval_call_and_regenerate(instruction_call(inst), 
					   stmt_prec(stmt), 
					   stmt_to_fx(stmt,fx_map));
	  break;
      }
      case is_instruction_goto :
	break;
      case is_instruction_unstructured : {
	  /* ?? What should I do? */
	  /* pips_error("recursiv_partial_eval", "?? :-(\n"); */
	  /* FI: I do not understand why Bruno (?) had metaphysical
	   * problems here
	   */
	  list blocs = NIL;

	  CONTROL_MAP(ctl, {
	      statement st = control_statement(ctl);

	      debug(5, "partial_eval", "will eval in statement number %d\n",
		    statement_number(st));

	      recursiv_partial_eval(st);	
	  }, unstructured_control(instruction_unstructured(inst)), blocs);

	  gen_free_list(blocs);
      }
	break;
    default : 
	pips_error("recursiv_partial_eval", 
		   "Bad instruction tag %d", instruction_tag(inst));
    }
}

/* Top-level function
 */

bool 
partial_eval(char *mod_name)
{
    entity module;
    statement mod_stmt;
    instruction mod_inst;
    cons *blocs = NIL;

    /* be carrefull not to get any mapping before the code */
    /* DBR_CODE will be changed: argument "pure" is TRUE because 
       partial_eval() *modifies* DBR_CODE. */
    /* still bugs in dbm because effects are stored on disc after this phase */

    set_current_module_entity(local_name_to_top_level_entity(mod_name));
    module = get_current_module_entity();

    set_current_module_statement(
		(statement) db_get_memory_resource(DBR_CODE, mod_name, TRUE));
    mod_stmt = get_current_module_statement();

    init_use_proper_effects(mod_name); /* uses set_proper_effects_map */

    /* preconditions may need to print preconditions for debugging purposes */
    set_cumulated_rw_effects((statement_effects) 
	db_get_memory_resource(DBR_CUMULATED_EFFECTS, mod_name, TRUE));

    module_to_value_mappings(module);

    init_use_preconditions(mod_name);

    set_live_loop_indices();

    mod_inst = statement_instruction(mod_stmt);

    debug_on("PARTIAL_EVAL_DEBUG_LEVEL");

    switch (instruction_tag (mod_inst)) {

    case is_instruction_block:
	MAP(STATEMENT, stmt, {recursiv_partial_eval (stmt);}, 
	    instruction_block (mod_inst));
	break;

    case is_instruction_unstructured:
	/* go through unstructured and apply recursiv_partial_eval */
	CONTROL_MAP(ctl, {
	    statement st = control_statement(ctl);

	    debug(5, "partial_eval", "will eval in statement number %d\n",
		  statement_number(st));

	    recursiv_partial_eval(st);	
	}, unstructured_control(instruction_unstructured(mod_inst)), blocs);

	gen_free_list(blocs);
	break;
    case is_instruction_call:
	if (return_statement_p(mod_stmt) || continue_statement_p(mod_stmt))
	    break;
    default:
	pips_error("partial_eval", "Non-acceptable instruction tag %d\n",
		   instruction_tag (mod_inst));
    }

    debug_off();

    /* Reorder the module, because new statements may have been generated. */
    module_reorder(mod_stmt);

    DB_PUT_MEMORY_RESOURCE(DBR_CODE, strdup(mod_name), mod_stmt);

    reset_live_loop_indices();
    reset_precondition_map();
    reset_cumulated_rw_effects();
    reset_proper_rw_effects();
    reset_current_module_entity();
    reset_current_module_statement();
    free_value_mappings();

    return TRUE;
}

