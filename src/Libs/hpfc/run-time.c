/*
 * Runtime Support Functions Management
 *
 * Fabien Coelho, May and June 1993
 *
 * SCCS stuff:
 * $RCSfile: run-time.c,v $ ($Date: 1995/03/14 14:43:17 $, ) version $Revision$,
 * got on %D%, %T%
 * $Id$
 */

#include <stdio.h>
extern int fprintf();

#include "genC.h"
#include "ri.h"

#include "misc.h"
#include "ri-util.h"
#include "bootstrap.h"
#include "properties.h"
#include "hpfc.h"
#include "defines-local.h"

extern entity CreateIntrinsic(string name);                 /* in syntax.h */
extern entity MakeExternalFunction(entity e, type r);       /* idem */

/* entity MakeRunTimeSupportSubroutine(local_name, number_of_arguments)
 *
 * modify 27/09/93, in order not to attempt to redeclare an already declared
 * subroutine. 
 */
entity MakeRunTimeSupportSubroutine(local_name, number_of_arguments)
string local_name;
int number_of_arguments;
{
    string 
	full_name = concatenate(TOP_LEVEL_MODULE_NAME, 
				MODULE_SEP_STRING, local_name, NULL);
    entity
	e = gen_find_tabulated(full_name, entity_domain);

    return((e==entity_undefined) ? make_empty_module(full_name) : e);
}

/* entity MakeRunTimeSupportFunction
 *   (local_name, number_of_arguments, return_type)
 *
 * this function can be used even if the function is already declared
 * ??? an integer shouldn't always be returned
 */
entity MakeRunTimeSupportFunction(local_name, number_of_arguments, return_type)
string local_name;
int number_of_arguments;
tag return_type;
{
    return(MakeExternalFunction(FindOrCreateEntity(TOP_LEVEL_MODULE_NAME,
						   local_name),
				(return_type==is_basic_int ? /* ??? rough */
				 MakeIntegerResult() :
				 MakeOverloadedResult())));
}

expression pvm_what_option_expression(v)
entity v;
{
    assert(entity_variable_p(v));

    return(MakeCharacterConstantExpression
	       (strdup(pvm_what_options(entity_basic(v)))));
}

/* string pvm_what_options(b)
 *
 * the pvm what option is given back as a string, fellowing the basic given.
 */
string pvm_what_options(b)
basic b;
{
    switch (basic_tag(b))
    {
    case is_basic_int:
	switch (basic_int(b))
	{
	case 2: return("INTEGER2");
	case 4: return("INTEGER4");
	default:
	    pips_error("pvm_what_options", 
		       "unexpected integer length (%d)\n",
		       basic_int(b));
	}
    case is_basic_float:
	switch (basic_float(b))
	{
	case 4: return("REAL4");
	case 8: return("REAL8");
	default:
	    pips_error("pvm_what_options", 
		       "unexpected float length (%d)\n",
		       basic_float(b));
	}
    case is_basic_logical:
	switch (basic_logical(b))
	{
	case 2: return("INTEGER2");
	case 4: return("INTEGER4");
	default:
	    pips_error("pvm_what_options", 
		       "unexpected logical length (%d)\n",
		       basic_logical(b));
	}
    case is_basic_overloaded:
	pips_error("pvm_what_options", "overloaded not welcomed\n");
    case is_basic_complex:
	switch (basic_complex(b))
	{
	case  8: return("COMPLEX8");
	case 16: return("COMPLEX16");
	default:
	    pips_error("pvm_what_options", 
		       "unexpected complex length (%d)\n",
		       basic_complex(b));
	}
    case is_basic_string:
	return("STRING");
    default:
	pips_error("pvm_what_options", "unexpected basic tag\n");
    }
    return("ERROR");
}


/******************************************************************************/
/*
 *                       mere statements generation
 */

/******************************************************************************/
/*
 * Sends
 */

statement st_call_send_or_receive(f, r)
entity f;
reference r;
{
    return
	(hpfc_make_call_statement(f,
	   CONS(EXPRESSION, pvm_what_option_expression(reference_variable(r)),
	   CONS(EXPRESSION, reference_to_expression(r),
		NIL))));
}

/******************************************************************************/
/*
 * Computes
 */

/*
 *
 */
statement st_compute_current_computer(ref)
reference ref;
{
    if (get_bool_property("HPFC_EXPAND_COMPUTE_COMPUTER"))
    {
	list 
	    linds = reference_indices(ref),
	    largs = make_list_of_constant(0, 7-gen_length(linds));
	int
	    narray = load_entity_hpf_number(reference_variable(ref));
	
	largs = gen_nconc(CONS(EXPRESSION, int_to_expression(narray), 
			       NIL),
			  gen_nconc(lUpdateExpr(node_module, linds), largs));
	
	return(hpfc_make_call_statement(hpfc_name_to_entity(CMP_COMPUTER), 
				      largs));
    }
    else
	return(hpfc_make_call_statement(hpfc_name_to_entity(CMP_COMPUTER),
		     CONS(EXPRESSION, reference_to_expression(ref),
			  NIL)));
}

/*
 * statement st_compute_current_owners(ref)
 */
statement st_compute_current_owners(ref)
reference ref;
{
    if (get_bool_property("HPFC_EXPAND_COMPUTE_OWNER"))
    {
	list 
	    linds = reference_indices(ref),
	    largs = make_list_of_constant(0, 7-gen_length(linds));
	int
	    narray = load_entity_hpf_number(reference_variable(ref));
	
	largs = gen_nconc(CONS(EXPRESSION, int_to_expression(narray), NIL),
			  gen_nconc(lUpdateExpr(node_module,  linds), largs));
	
	return(hpfc_make_call_statement(hpfc_name_to_entity(CMP_OWNERS), 
				      largs));
    }
    else
	return(hpfc_make_call_statement(hpfc_name_to_entity(CMP_OWNERS),
		     CONS(EXPRESSION, reference_to_expression(ref),
			  NIL)));
}

/*
 * expr_compute_local_index
 *
 * new index computation formula, derived from the new declarations
 * made for the given dimension.
 */
expression expr_compute_local_index(array, dim, expr)
entity array;
int dim;
expression expr;
{
    if (get_bool_property("HPFC_EXPAND_COMPUTE_LOCAL_INDEX"))
    {
	tag newdecl = new_declaration(array, dim);
	dimension the_dim = entity_ith_dimension(array, dim);
        
	switch(newdecl)
	{
	case is_hpf_newdecl_none:
	    return(expr);
	case is_hpf_newdecl_alpha:
	{
	    int
		dl = HpfcExpressionToInt(dimension_lower(the_dim));
	    expression
		shift = int_to_expression(1 - dl);
	    
	    return(MakeBinaryCall(CreateIntrinsic(PLUS_OPERATOR_NAME), 
				  expr, shift));
	}
	case is_hpf_newdecl_beta:
	{
	    align
		a = load_entity_align(array);
	    entity
		template = align_template(a);
	    distribute
		d = load_entity_distribute(template);
	    alignment
		al = FindAlignmentOfDim(align_alignment(a), dim);
	    int
		tempdim = alignment_templatedim(al),
		procdim;
	    dimension
		template_dim = FindIthDimension(template,tempdim);
	    distribution
		di = FindDistributionOfDim(distribute_distribution(d), 
					   tempdim, 
					   &procdim);
	    expression
		parameter = distribution_parameter(di),
		rate = alignment_rate(al),
		prod,
		t1,
		the_mod,
		t2;
	    int
		iabsrate = abs(HpfcExpressionToInt(rate)),
		ishift = (HpfcExpressionToInt(alignment_constant(al)) - 
			  HpfcExpressionToInt(dimension_lower(template_dim)));
	    
	    
	    prod = ((HpfcExpressionToInt(rate)==1)?
		    (expr):
		    ((iabsrate==1)?
		     (MakeUnaryCall(CreateIntrinsic(UNARY_MINUS_OPERATOR_NAME),
				    expr)):
		     (MakeBinaryCall(CreateIntrinsic(MULTIPLY_OPERATOR_NAME),
				     rate,
				     expr))));
	    
	    t1 = ((ishift==0)?
		  (prod):
		  ((ishift>0)?
		   (MakeBinaryCall(CreateIntrinsic(PLUS_OPERATOR_NAME),prod,
				   int_to_expression(ishift))):
		   (MakeBinaryCall(CreateIntrinsic(MINUS_OPERATOR_NAME),prod,
				   int_to_expression(abs(ishift))))));
	    
	    the_mod = MakeBinaryCall(CreateIntrinsic(MOD_INTRINSIC_NAME),
				     t1,
				     parameter);
	    
	    t2 = ((iabsrate==1)?
		  (the_mod):
		  MakeBinaryCall(CreateIntrinsic(DIVIDE_OPERATOR_NAME),
				 the_mod,
				 int_to_expression(iabsrate)));
	    
	    return(MakeBinaryCall(CreateIntrinsic(PLUS_OPERATOR_NAME), 
			      t2,
				  int_to_expression(1)));
	}
	case is_hpf_newdecl_gamma:
	{
	    expression
		expr1 = 
		    int_to_expression(load_entity_hpf_number(array)),
		    expr2 = int_to_expression(dim);
	    
	    return(MakeTernaryCallExpr(hpfc_name_to_entity(LOCAL_IND_GAMMA), 
				       expr1, expr2, expr));
	}
	case is_hpf_newdecl_delta:
	{
	    expression
		expr1 = int_to_expression(load_entity_hpf_number(array)),
		expr2 = int_to_expression(dim);
	    
	    return(MakeTernaryCallExpr(hpfc_name_to_entity(LOCAL_IND_DELTA), 
				       expr1, expr2, expr));
	}
	default:
	    pips_error("expr_compute_local_index",
		       "unexpected new declaration tag\n");
	}
	
    }
    else
    {
	expression
	    expr1 = int_to_expression(load_entity_hpf_number(array)),
	    expr2 = int_to_expression(dim);
	
	return(MakeTernaryCallExpr(hpfc_name_to_entity(LOCAL_IND), 
				   expr1, expr2, expr));
    }

    return(expression_undefined); /* just to avoid a gcc warning */
}

/******************************************************************************/

/*
 * statement hpfc_make_call_statement(e, l) 
 * generate a call statement to function e, with expression list l 
 * as an argument. 
 */
statement hpfc_make_call_statement(e, l)
entity e;
list l;
{
    assert(!entity_undefined_p(e));

    return(make_stmt_of_instr(make_instruction(is_instruction_call,
					       make_call(e, l))));
}

/*
 * void add_pvm_init_and_end(phs, pns)
 */
void add_pvm_init_and_end(phs, pns)
statement *phs, *pns;
{
    (*phs) = make_block_statement(CONS(STATEMENT,
				       st_init_host(),
				       CONS(STATEMENT,
					    (*phs),
					    CONS(STATEMENT,
						 st_host_end(),
						 NIL))));

    (*pns) = make_block_statement(CONS(STATEMENT,
				       st_init_node(),
				       CONS(STATEMENT,
					    (*pns),
					    CONS(STATEMENT,
						 st_node_end(),
						 NIL))));

}

/*
 * statement st_compute_neighbour(d)
 *
 * call to the runtime support function HPFC_CMPNEIGHBOUR(d)
 */
statement st_compute_neighbour(d)
int d;
{
    return(hpfc_make_call_statement(hpfc_name_to_entity(CMP_NEIGHBOUR), 
				  CONS(EXPRESSION,
				       int_to_expression(d),
				       NIL)));
}

/*
 * statement st_generate_packing_and_passing(array, content, bsend)
 *
 * dimension bounds are refered to as parameters, since we do not
 * know yet what is the lower and upper of each dimension...
 */
statement st_generate_packing(array, content, bsend)
entity array;
list content;
bool bsend;
{
    int
	len = gen_length(content);
    list
	larg = NIL;

    assert(len==NumberOfDimension(array) && (len<=4) && (len>=1));

    larg = array_lower_upper_bounds_list(array);

    MAPL(cr,
     {
	 range r = RANGE(CAR(cr));

	 larg = gen_nconc(larg,
			  CONS(EXPRESSION, range_lower(r),
			  CONS(EXPRESSION, range_upper(r),
			  CONS(EXPRESSION, range_increment(r),
			       NIL))));
     },
	 content);

    larg = CONS(EXPRESSION, entity_to_expression(array),
		larg);
    
    /* larg content:
     *
     * array, dim [lower, upper]*len, range [lower, upper, increment]*len
     */

    return(hpfc_make_call_statement
	   (make_packing_function("HPFC", 
				  len, 
				  bsend, 
				  entity_basic(array), 
				  1+5*len),
	    larg));

}

/* returns the entity to which e is attached,
 * that is first a common, then a function...
 */
entity hpfc_main_entity(e)
entity e;
{
    storage
	s = entity_storage(e);
    bool
	in_common = entity_in_common_p(e),
	in_ram = storage_ram_p(s);
    ram 
	r = (in_ram ? storage_ram(s) : ram_undefined);

    assert(!storage_rom_p(s));

    return(in_ram ?
	   (in_common ? ram_section(r) : ram_function(r)):
	   (storage_formal_p(s) ? formal_function(storage_formal(s)) :
	    (storage_return_p(s) ? storage_return(s) : entity_undefined)));
}

/* returns the name of the entity e belongs too (common, function...)
 */
string hpfc_main_entity_name(e)
entity e;
{
    return(module_local_name(hpfc_main_entity(e)));
}

/*
 * string bound_parameter_name(array, side, dim)
 *
 * returns a name for the bound of the declaration 
 * of array array, side side and dimension dim.
 */
string bound_parameter_name(array, side, dim)
entity array;
string side;
int dim;
{
    char buffer[100];

    return(strdup(sprintf(buffer, "%s_%s_%s%d",
			  hpfc_main_entity_name(array),
			  entity_local_name(array),
			  side,
			  dim)));
}

/*
 * list array_lower_bounds_list(array)
 */
list array_lower_bounds_list(array)
entity array;
{
    int
	i = -1,
	ndim = NumberOfDimension(array);
    list
	result = NIL;

    for (i=ndim ; i>=1 ; i--)
    {
	char
	    *buf = bound_parameter_name(array, LOWER, i);

	result = 
	    CONS(EXPRESSION,
		 MakeCharacterConstantExpression(buf),
		 result);

	free(buf);
    }

    return(result);
}

/*
 * list array_lower_upper_bounds_list(array)
 */
list array_lower_upper_bounds_list(array)
entity array;
{
    int
	i = -1,
	ndim = NumberOfDimension(array);
    list
	result = NIL;

    for (i=ndim ; i>=1 ; i--)
    {
	char
	    *lbuf = bound_parameter_name(array, LOWER, i),
	    *ubuf = bound_parameter_name(array, UPPER, i);

	result = 
	    CONS(EXPRESSION,
		 MakeCharacterConstantExpression(lbuf),
	    CONS(EXPRESSION,
		 MakeCharacterConstantExpression(ubuf),
		 result));

	free(ubuf);
	free(lbuf);
    }

    return(result);
}
	
/*
 * entity make_packing_function(prefix, ndim, kind, base, nargs)
 *
 * find or create an entity for the packing function...
 */
entity make_packing_function(prefix, ndim, kind, base, nargs)
string prefix;
int ndim;
bool kind;
basic base;
int nargs;
{
    char 
	buffer[100],
	*buf = buffer;

    buf += strlen(sprintf(buf, "%s_%s_%s_%d", 
			  prefix, 
			  (kind ? "PACK" : "UNPACK"),
			  pvm_what_options(base),
			  ndim));

    return(MakeRunTimeSupportSubroutine(buffer, nargs));
}

/*
 * that is all
 */

