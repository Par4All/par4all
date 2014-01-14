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
/* Runtime Support Functions Management
 *
 * Fabien Coelho, May and June 1993
 */

#include "defines-local.h"

#include "effects-generic.h"
#include "effects-simple.h"
#include "effects-convex.h"

/* entity MakeRunTimeSupportSubroutine(local_name, number_of_arguments)
 *
 * modify 27/09/93, in order not to attempt to redeclare an already declared
 * subroutine. 
 */
entity 
MakeRunTimeSupportSubroutine(
    string local_name,
    int number_of_arguments)
{
    entity res = module_name_to_entity(local_name);
    if (entity_undefined_p(res))
	res = make_empty_subroutine(local_name,make_language_fortran());
    return res;
}

/* entity MakeRunTimeSupportFunction
 *   (local_name, number_of_arguments, return_type)
 *
 * this function can be used even if the function is already declared
 * ??? an integer shouldn't always be returned
 */
entity 
MakeRunTimeSupportFunction(
    string local_name,
    int number_of_arguments,
    tag return_type)
{
    entity f = make_empty_function(local_name,
				   (return_type==is_basic_int ? /* ??? rough */
				    MakeIntegerResult() :
				    MakeOverloadedResult()),make_language_fortran());
    return f;
}

expression 
pvm_what_option_expression(entity v)
{
    pips_assert("variable", entity_variable_p(v));

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
	case 2: return(PVM_INTEGER2);
	case 4: return(PVM_INTEGER4);
	default:
	    pips_internal_error("unexpected integer*%d", basic_int(b));
	}
    case is_basic_float:
	switch (basic_float(b))
	{
	case 4: return(PVM_REAL4);
	case 8: return(PVM_REAL8);
	default:
	    pips_internal_error("unexpected real*%d", basic_float(b));
	}
    case is_basic_logical:
	switch (basic_logical(b))
	{
	case 2: return(PVM_INTEGER2);
	case 4: return(PVM_INTEGER4);
	default:
	    pips_internal_error("unexpected logical*%d", basic_logical(b));
	}
    case is_basic_overloaded:
	pips_internal_error("overloaded not welcomed");
    case is_basic_complex:
	switch (basic_complex(b))
	{
	case  8: return(PVM_COMPLEX8);
	case 16: return(PVM_COMPLEX16);
	default:
	    pips_internal_error("unexpected complex*%d", basic_complex(b));
	}
    case is_basic_string:
	return(PVM_STRING);
    default:
	pips_internal_error("unexpected basic tag");
    }
    return("ERROR");
}


/****************************************************** STATEMENTS GENERATION */

/* Sends
 */
statement st_call_send_or_receive(f, r)
entity f;
reference r;
{
    return
	 hpfc_make_call_statement(f,
	   CONS(EXPRESSION, pvm_what_option_expression(reference_variable(r)),
	   CONS(EXPRESSION, reference_to_expression(r),
		NIL)));
}

/* Computes
 */
statement st_compute_current_computer(ref)
reference ref;
{
    if (get_bool_property("HPFC_EXPAND_COMPUTE_COMPUTER"))
    {
	list linds = reference_indices(ref),
	     largs = make_list_of_constant(0, 7-gen_length(linds));
	int narray = load_hpf_number(reference_variable(ref));
	
	largs = gen_nconc(CONS(EXPRESSION, int_to_expression(narray), 
			       NIL),
			  gen_nconc(lUpdateExpr(node_module, linds), largs));
	
	return hpfc_make_call_statement(hpfc_name_to_entity(CMP_COMPUTER), 
				      largs);
    }
    else
	return hpfc_make_call_statement(hpfc_name_to_entity(CMP_COMPUTER),
		     CONS(EXPRESSION, reference_to_expression(ref),
			  NIL));
}

statement st_compute_current_owners(ref)
reference ref;
{
    if (get_bool_property("HPFC_EXPAND_COMPUTE_OWNER"))
    {
	list linds = reference_indices(ref),
	     largs = make_list_of_constant(0, 7-gen_length(linds));
	int narray = load_hpf_number(reference_variable(ref));
	
	largs = gen_nconc(CONS(EXPRESSION, int_to_expression(narray), NIL),
			  gen_nconc(lUpdateExpr(node_module,  linds), largs));
	
	return hpfc_make_call_statement(hpfc_name_to_entity(CMP_OWNERS), 
				      largs);
    }
    else
	return hpfc_make_call_statement(hpfc_name_to_entity(CMP_OWNERS),
		     CONS(EXPRESSION, reference_to_expression(ref),
			  NIL));
}

/* new index computation formula, derived from the new declarations
 * made for the given dimension.
 */
expression 
expr_compute_local_index(
    entity array,
    int dim,
    expression expr)
{
    if (get_bool_property("HPFC_EXPAND_COMPUTE_LOCAL_INDEX"))
    {
	tag newdecl = new_declaration_tag(array, dim);
	dimension the_dim = entity_ith_dimension(array, dim);
        
	switch(newdecl)
	{
	case is_hpf_newdecl_none:
	    return(expr);
	case is_hpf_newdecl_alpha:
	{
	    int	dl = HpfcExpressionToInt(dimension_lower(the_dim));
	    expression shift = int_to_expression(1 - dl);
	    
	    return(MakeBinaryCall(entity_intrinsic(PLUS_OPERATOR_NAME), 
				  expr, shift));
	}
	case is_hpf_newdecl_beta:
	{
	    align a = load_hpf_alignment(array);
	    entity template = align_template(a);
	    distribute d = load_hpf_distribution(template);
	    alignment al = FindAlignmentOfDim(align_alignment(a), dim);
	    int tempdim = alignment_templatedim(al), procdim;
	    dimension template_dim = FindIthDimension(template,tempdim);
	    distribution
		di = FindDistributionOfDim(distribute_distribution(d), 
					   tempdim, 
					   &procdim);
	    expression
		parameter = distribution_parameter(di),
		rate = alignment_rate(al),
		prod, t1, the_mod, t2;
	    int
		iabsrate = abs(HpfcExpressionToInt(rate)),
		ishift = (HpfcExpressionToInt(alignment_constant(al)) - 
			  HpfcExpressionToInt(dimension_lower(template_dim)));
	    
	    
	    prod = 
		((HpfcExpressionToInt(rate)==1)?
		 (expr):
		 ((iabsrate==1)?
		  (MakeUnaryCall(entity_intrinsic(UNARY_MINUS_OPERATOR_NAME),
				 expr)):
		  (MakeBinaryCall(entity_intrinsic(MULTIPLY_OPERATOR_NAME),
				  rate,
				  expr))));
	    
	    t1 = ((ishift==0)?
		  (prod):
		  ((ishift>0)?
		   (MakeBinaryCall(entity_intrinsic(PLUS_OPERATOR_NAME),prod,
				   int_to_expression(ishift))):
		   (MakeBinaryCall(entity_intrinsic(MINUS_OPERATOR_NAME),prod,
				   int_to_expression(abs(ishift))))));
	    
	    the_mod = MakeBinaryCall(entity_intrinsic(MOD_INTRINSIC_NAME),
				     t1,
				     parameter);
	    
	    t2 = ((iabsrate==1)?
		  (the_mod):
		  MakeBinaryCall(entity_intrinsic(DIVIDE_OPERATOR_NAME),
				 the_mod,
				 int_to_expression(iabsrate)));
	    
	    return(MakeBinaryCall(entity_intrinsic(PLUS_OPERATOR_NAME), 
			      t2,
				  int_to_expression(1)));
	}
	case is_hpf_newdecl_gamma:
	{
	    expression
		expr1 = 
		    int_to_expression(load_hpf_number(array)),
		    expr2 = int_to_expression(dim);
	    
	    return(MakeTernaryCall(hpfc_name_to_entity(LOCAL_IND_GAMMA), 
				       expr1, expr2, expr));
	}
	case is_hpf_newdecl_delta:
	{
	    expression
		expr1 = int_to_expression(load_hpf_number(array)),
		expr2 = int_to_expression(dim);
	    
	    return(MakeTernaryCall(hpfc_name_to_entity(LOCAL_IND_DELTA), 
				       expr1, expr2, expr));
	}
	default:
	    pips_internal_error("unexpected new declaration tag");
	}
	
    }
    else
    {
	expression
	    expr1 = int_to_expression(load_hpf_number(array)),
	    expr2 = int_to_expression(dim);
	
	return(MakeTernaryCall(hpfc_name_to_entity(LOCAL_IND), 
				   expr1, expr2, expr));
    }

    return(expression_undefined); /* just to avoid a gcc warning */
}

/*****************************************************************************/

/* statement hpfc_make_call_statement(e, l) 
 * generate a call statement to function e, with expression list l 
 * as an argument. 
 */
statement 
hpfc_make_call_statement(entity e, list l)
{
    pips_assert("defined", !entity_undefined_p(e));
    return instruction_to_statement(make_instruction(is_instruction_call,
					       make_call(e, l)));
}

/************************************************************** SUBSTITUTION */

static entity 
    sub_call_o = entity_undefined, 
    sub_call_n = entity_undefined,
    sub_ret_label = entity_undefined;

static void rwt(call c)
{
    if (call_function(c)==sub_call_o) 
	call_function(c) = sub_call_n;
}
static void srwt(statement s)
{
    if (entity_return_label_p(statement_label(s))) {
	sub_ret_label = statement_label(s);
	statement_label(s) = entity_empty_label();
    }
}
static void 
substitute_return(entity o, entity n, statement s)
{
    sub_call_o = o;
    sub_call_n = n;

    gen_multi_recurse(s,
		      statement_domain, gen_true, srwt,
		      call_domain, gen_true, rwt,
		      expression_domain, gen_false, gen_null,
		      NULL);

    sub_call_n = entity_undefined;
    sub_call_o = entity_undefined;
}

/* this is for the main.
 * also subs CALL RETURN -> CALL HPFC {HOST|NONE} END...
 */
void 
add_pvm_init_and_end(statement *phs, statement *pns)
{
    entity
	rete = entity_intrinsic("RETURN"),
	hhe = hpfc_name_to_entity(HOST_END),
	hne = hpfc_name_to_entity(NODE_END);
    statement ret = hpfc_make_call_statement(rete, NIL);

    substitute_return(rete, hhe, *phs);
    substitute_return(rete, hne, *pns);


    if (sub_ret_label!=entity_undefined) {
	statement_label(ret) = sub_ret_label;
	sub_ret_label = entity_undefined;
    }

    (*phs) = make_block_statement(CONS(STATEMENT, st_init_host(),
				  CONS(STATEMENT, (*phs),
				  CONS(STATEMENT, ret,
				       NIL))));

    (*pns) = make_block_statement(CONS(STATEMENT, st_init_node(),
				  CONS(STATEMENT, (*pns),
				  CONS(STATEMENT, copy_statement(ret),
				       NIL))));
}

/* call to the runtime support function HPFC_CMPNEIGHBOUR(d)
 */
statement st_compute_neighbour(int d)
{
    return hpfc_make_call_statement(hpfc_name_to_entity(CMP_NEIGHBOUR), 
				  CONS(EXPRESSION, int_to_expression(d),
				       NIL));
}

/* find or create an entity for the packing function...
 */
static entity make_packing_function(ndim, kind, base, nargs)
int ndim;
bool kind;
basic base;
int nargs;
{
    char buffer[100], *buf = buffer;
    sprintf(buf, "%s %s %d", 
	    pvm_what_options(base), (kind ? "PACK" : "UNPACK"), ndim);
    buf += strlen(buf);

    return MakeRunTimeSupportSubroutine(buffer, nargs);
}

/* statement st_generate_packing_and_passing(array, content, bsend)
 *
 * dimension bounds are refered to as parameters, since we do not
 * know yet what is the lower and upper of each dimension...
 */
statement st_generate_packing(array, content, bsend)
entity array;
list content;
bool bsend;
{
    int len = gen_length(content);
    list larg = NIL;

    pips_assert("valid number of dimensions",
		len==NumberOfDimension(array) && (len<=4) && (len>=1));

    larg = array_lower_upper_bounds_list(array);

    MAP(RANGE, r,
     {
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

    return hpfc_make_call_statement
	   (make_packing_function(len, bsend, entity_basic(array), 1+5*len),
	    larg);

}

/* returns the entity to which e is attached,
 * that is first a common, then a function...
 */
entity hpfc_main_entity(entity e)
{
    storage s = entity_storage(e);
    bool in_common = entity_in_common_p(e),
	 in_ram = storage_ram_p(s);
    ram r = (in_ram ? storage_ram(s) : ram_undefined);

    pips_assert("not in rom", !storage_rom_p(s));

    return(in_ram ?
	   (in_common ? ram_section(r) : ram_function(r)):
	   (storage_formal_p(s) ? formal_function(storage_formal(s)) :
	    (storage_return_p(s) ? storage_return(s) : entity_undefined)));
}

/* returns the name of the entity e belongs too (common, function...)
 */
const char* hpfc_main_entity_name(entity e)
{
    return(module_local_name(hpfc_main_entity(e)));
}

/* returns a name for the bound of the declaration 
 * of array array, side side and dimension dim.
 */
string 
bound_parameter_name(
    entity array,
    string side,
    int dim)
{
    return strdup(concatenate(hpfc_main_entity_name(array), " ",
			      entity_local_name(array), " ",
			      side, i2a(dim), NULL));
}

entity 
argument_bound_entity(
    entity module,
    entity array,
    bool upper,
    int dim)
{
    entity result;
    string name = bound_parameter_name(array, upper? UPPER: LOWER, dim);

    result = find_or_create_typed_entity(name, module_local_name(module), 
					 is_basic_int);

    free(name);
    return result;
}

expression 
hpfc_array_bound(entity array, bool upper, int dim)
{
    /*
    return entity_to_expression
	(argument_bound_entity(node_module, array, upper, dim)); 
	*/
    return MakeCharacterConstantExpression(bound_parameter_name(array, 
			 upper?UPPER:LOWER, dim));
}

static list /* of expressions */
array_bounds_list(entity array, bool upper)
{
    int	i = -1,	ndim = NumberOfDimension(array);
    list result = NIL;

    for (i=ndim ; i>=1 ; i--)
	result = CONS(EXPRESSION, hpfc_array_bound(array, upper, i), result);

    return result;
}

list /* of expressions */
array_lower_upper_bounds_list(entity array)
{
    list /* of expression */ lb = array_bounds_list(array, false),
    			     lu = array_bounds_list(array, true),
    			     l = lb, lnb, lnu;

    if (!l) return NIL;
    
    /* interleave both lists (just for fun:-)
     */
    for(lnb=CDR(lb), lnu=CDR(lu); lb; 
	CDR(lb)=lu, CDR(lu)=lnb, 
	lb=lnb, lnb=lnb?CDR(lnb):NIL, lu=lnu, lnu=lnu?CDR(lnu):NIL);

    return l;
}

/************************************************* HPFC ENTITIES MANAGEMENT */

/* this file stores the table that describes run time functions and
 * variables that may be called or referenced by the generated code.
 * the information needed (name, arity, type...) is stored in a static
 * table here. The table is scanned to create the corresponding entities
 * for once. Then the entities are quickly returned on demand thru the
 * hpfc_name_to_entity function. It was inspired to me by some static
 * table here around in PIPS, that deal with intrinsics for instance.
 */

/*  local defines
 */

#define is_end 0
#define is_sub 1
#define is_fun 2
#define is_var 3
#define is_int 4
#define is_ifn 5 /* intrinsic like function */
#define is_iof 6 /* I/O like function */

#define no_basic	is_basic_overloaded
#define no_entity	entity_undefined

typedef struct 
{
    char    name[30];
    int     arity;
    int     what; /* function or subroutine or variable or ...*/
    tag     basic;/* basic tag if necessary */
    entity  object;
} RunTimeSupportDescriptor;

static bool RTSTable_initialized_p = false;

static RunTimeSupportDescriptor RTSTable[] =
{
    { SND_TO_C,		2, is_sub, no_basic, no_entity },
    { SND_TO_H,		2, is_sub, no_basic, no_entity },
    { SND_TO_A,		2, is_sub, no_basic, no_entity },
    { SND_TO_A_BY_H, 	2, is_sub, no_basic, no_entity },
    { SND_TO_O,		2, is_sub, no_basic, no_entity },
    { SND_TO_OS, 	2, is_sub, no_basic, no_entity },
    { SND_TO_OOS, 	2, is_sub, no_basic, no_entity },
    { SND_TO_HA, 	2, is_sub, no_basic, no_entity },
    { SND_TO_NO, 	2, is_sub, no_basic, no_entity },
    { SND_TO_HNO, 	2, is_sub, no_basic, no_entity },
    { RCV_FR_S,		2, is_sub, no_basic, no_entity },
    { RCV_FR_H,		2, is_sub, no_basic, no_entity },
    { RCV_FR_C,		2, is_sub, no_basic, no_entity },
    { RCV_FR_mCS, 	2, is_sub, no_basic, no_entity },
    { RCV_FR_mCH, 	2, is_sub, no_basic, no_entity },
/*    { CMP_COMPUTER, 	1, is_sub, no_basic, no_entity },*/
    { CMP_COMPUTER, 	8, is_sub, no_basic, no_entity },
/*    { CMP_OWNERS, 	1, is_sub, no_basic, no_entity }, */
    { CMP_OWNERS, 	8, is_sub, no_basic, no_entity },
    { CMP_NEIGHBOUR, 	1, is_sub, no_basic, no_entity },
    { CMP_LID,		8, is_fun, no_basic, no_entity },
    { TWIN_P,		2, is_fun, no_basic, no_entity },
    { CND_SENDERP, 	0, is_fun, no_basic, no_entity },
    { CND_OWNERP, 	0, is_fun, no_basic, no_entity },
    { CND_COMPUTERP, 	0, is_fun, no_basic, no_entity },
    { CND_COMPINOWNP, 	0, is_fun, no_basic, no_entity },
    { LOCAL_IND, 	3, is_fun, no_basic, no_entity },
    { LOCAL_IND_GAMMA, 	3, is_fun, no_basic, no_entity },
    { LOCAL_IND_DELTA, 	3, is_fun, no_basic, no_entity },
    { IDIVIDE,		2, is_fun, no_basic, no_entity },
    { INIT_HOST, 	0, is_sub, no_basic, no_entity },
    { INIT_NODE, 	0, is_sub, no_basic, no_entity },
    { HOST_END, 	0, is_sub, no_basic, no_entity },
    { NODE_END, 	0, is_sub, no_basic, no_entity },
    { HPFC_STOP,	0, is_sub, no_basic, no_entity },
    { LOOP_BOUNDS, 	0, is_sub, no_basic, no_entity },
    { SYNCHRO, 		0, is_sub, no_basic, no_entity },
    { SND_TO_N, 	0, is_sub, no_basic, no_entity },
    { RCV_FR_N, 	0, is_sub, no_basic, no_entity },

/* HOST/NODE MESSAGES
 */
    { HPFC_HCAST,	0, is_sub, no_basic, no_entity },
    { HPFC_NCAST,	0, is_sub, no_basic, no_entity },
    { HPFC_sN2H,	0, is_sub, no_basic, no_entity },
    { HPFC_sH2N,	1, is_sub, no_basic, no_entity },
    { HPFC_rN2H,	1, is_sub, no_basic, no_entity },
    { HPFC_rH2N,	0, is_sub, no_basic, no_entity },

/*  PVM 3 stuff
 */
    { PVM_INITSEND,	2, is_sub, no_basic, no_entity },
    { PVM_SEND,		3, is_sub, no_basic, no_entity },
    { PVM_RECV,		3, is_sub, no_basic, no_entity },
    { PVM_CAST,		4, is_sub, no_basic, no_entity },
    { PVM_PACK,		5, is_sub, no_basic, no_entity },
    { PVM_UNPACK,	5, is_sub, no_basic, no_entity },

/*  Variables: the overloaded ones *must* be kept overloaded, otherwise
 *             they may be added in the declarations, while there are 
 *             in the included commons...
 */
    { MYPOS,		2, is_var, no_basic, no_entity },
    { MYLID,		0, is_var, no_basic, no_entity },
    { MSTATUS,		1, is_var, no_basic, no_entity },
    { LIVEMAPPING,	1, is_var, no_basic, no_entity },
    { INFO,		0, is_var, is_basic_int, 	no_entity },
    { BUFID,		0, is_var, is_basic_int,	no_entity },
    { NBTASKS, 		0, is_var, no_basic, no_entity },
    { T_LID,		0, is_var, is_basic_int, 	no_entity },
    { T_LIDp,		0, is_var, is_basic_int, 	no_entity },
    { NODETIDS, 	1, is_var, no_basic, no_entity },
    { SEND_CHANNELS, 	1, is_var, no_basic, no_entity },
    { RECV_CHANNELS, 	1, is_var, no_basic, no_entity },
    { MCASTHOST, 	0, is_var, no_basic, no_entity },
    { HOST_TID, 	0, is_var, no_basic, no_entity },
    { HOST_SND_CHAN, 	0, is_var, no_basic, no_entity },
    { HOST_RCV_CHAN, 	0, is_var, no_basic, no_entity },

/* common /hpfc_buffers/
 */
    { LAZY_SEND, 	0, is_var, no_basic, no_entity },
    { LAZY_RECV,	0, is_var, no_basic, no_entity },
    { SND_NOT_INIT,	0, is_var, no_basic, no_entity },
    { RCV_NOT_PRF,	0, is_var, no_basic, no_entity },
    { BUFFER_SIZE,	0, is_var, no_basic, no_entity },
    { BUFFER_INDEX, 	0, is_var, no_basic, no_entity },
    { BUFFER_RCV_SIZE,	0, is_var, no_basic, no_entity },
    /* { BUFFER_ENCODING,	0, is_var, no_basic, no_entity }, */
  
    /* typed buffers 
     */
    { PVM_BYTE1 BUFFER,			1, is_var, no_basic, no_entity },
    { PVM_STRING BUFFER,		1, is_var, no_basic, no_entity },
    { PVM_INTEGER2 BUFFER,		1, is_var, no_basic, no_entity },
    { PVM_INTEGER4 BUFFER,		1, is_var, no_basic, no_entity },
    { PVM_REAL4 BUFFER,			1, is_var, no_basic, no_entity },
    { PVM_REAL8 BUFFER,			1, is_var, no_basic, no_entity },
    { PVM_COMPLEX8 BUFFER,		1, is_var, no_basic, no_entity },
    { PVM_COMPLEX16 BUFFER,		1, is_var, no_basic, no_entity },

    /* typed buffer sizes
     */
    { PVM_BYTE1 BUFSZ,			1, is_var, no_basic, no_entity },
    { PVM_STRING BUFSZ,			1, is_var, no_basic, no_entity },
    { PVM_INTEGER2 BUFSZ,		1, is_var, no_basic, no_entity },
    { PVM_INTEGER4 BUFSZ,		1, is_var, no_basic, no_entity },
    { PVM_REAL4 BUFSZ,			1, is_var, no_basic, no_entity },
    { PVM_REAL8 BUFSZ,			1, is_var, no_basic, no_entity },
    { PVM_COMPLEX8 BUFSZ,		1, is_var, no_basic, no_entity },
    { PVM_COMPLEX16 BUFSZ,		1, is_var, no_basic, no_entity },

    /* typed pack/unpack hpfc functions, for buffer management.
     */
    { PVM_BYTE1 BUFPCK,			0, is_sub, no_basic, no_entity },
    { PVM_STRING BUFPCK,		0, is_sub, no_basic, no_entity },
    { PVM_INTEGER2 BUFPCK,		0, is_sub, no_basic, no_entity },
    { PVM_INTEGER4 BUFPCK,		0, is_sub, no_basic, no_entity },
    { PVM_REAL4 BUFPCK,			0, is_sub, no_basic, no_entity },
    { PVM_REAL8 BUFPCK,			0, is_sub, no_basic, no_entity },
    { PVM_COMPLEX8 BUFPCK,		0, is_sub, no_basic, no_entity },
    { PVM_COMPLEX16 BUFPCK,		0, is_sub, no_basic, no_entity },

    { PVM_BYTE1 BUFUPK,			0, is_sub, no_basic, no_entity },
    { PVM_STRING BUFUPK,		0, is_sub, no_basic, no_entity },
    { PVM_INTEGER2 BUFUPK,		0, is_sub, no_basic, no_entity },
    { PVM_INTEGER4 BUFUPK,		0, is_sub, no_basic, no_entity },
    { PVM_REAL4 BUFUPK,			0, is_sub, no_basic, no_entity },
    { PVM_REAL8 BUFUPK,			0, is_sub, no_basic, no_entity },
    { PVM_COMPLEX8 BUFUPK,		0, is_sub, no_basic, no_entity },
    { PVM_COMPLEX16 BUFUPK,		0, is_sub, no_basic, no_entity },

/* special FCD target calls.
 */
    { HOST_TIMEON,			0, is_sub, no_basic, no_entity },
    { NODE_TIMEON,			0, is_sub, no_basic, no_entity },
    { HOST_TIMEOFF,			0, is_sub, no_basic, no_entity },
    { NODE_TIMEOFF,			0, is_sub, no_basic, no_entity },
    { HPFC_NTELL,			0, is_sub, no_basic, no_entity },
    { HPFC_HTELL,			0, is_sub, no_basic, no_entity },

/* special FCD calls needed for translation...
 */
    { HPF_PREFIX SYNCHRO_SUFFIX, 	0, is_sub, no_basic, no_entity },
    { HPF_PREFIX TIMEON_SUFFIX,  	0, is_sub, no_basic, no_entity },
    { HPF_PREFIX TIMEOFF_SUFFIX, 	0, is_sub, no_basic, no_entity },
    { HPF_PREFIX RENAME_SUFFIX,  	0, is_sub, no_basic, no_entity },
    { HPF_PREFIX HOSTSECTION_SUFFIX,	0, is_sub, no_basic, no_entity },
    { HPF_PREFIX TELL_SUFFIX,		0, is_sub, no_basic, no_entity },

/* End
 */
    { "", 				0, is_end, -1, NULL },
};

/* to be seen from outside of this file
 */
void hpfc_init_run_time_entities()
{
    RunTimeSupportDescriptor *current;
    int i=0;
    list l=NIL;

    if (RTSTable_initialized_p) return;
    RTSTable_initialized_p = true;

    for(current=RTSTable;
	current->what!=is_end;
	current++)
    {
	pips_debug(6, "initializing %s, %d\n", current->name, current->arity);

	switch(current->what)
	{
	case is_fun:
	    current->object = MakeRunTimeSupportFunction(current->name, 
							 current->arity,
							 current->basic);
	    break;
	case is_sub:
	    current->object = MakeRunTimeSupportSubroutine(current->name, 
							   current->arity);
	    break;
	case is_ifn: 
	    /* they are declared as variables to avoid redefinitions 
	     * ??? this fools pips typing (functions/variables and so)
	     * just okay for the pretty printer...
	     */
	case is_iof:
	case is_var:
	    current->object = 
		find_or_create_typed_entity(current->name,
					    HPFC_PACKAGE, /* why not */
					    current->basic);
	    /* dimensions are updated
	     */
	    l = NIL;
	    for(i=1; i<=current->arity; i++)
		l = CONS(DIMENSION,
			 make_dimension(int_to_expression(1),
					int_to_expression(1)),
			 l);
	    variable_dimensions
		(type_variable(entity_type(current->object))) =	l;
	    break;
	default:
	    pips_internal_error("unexpected what field in Descriptor");
	}    
    }
}

static RunTimeSupportDescriptor *find_entry_by_name(name)
string name;
{
    RunTimeSupportDescriptor *current;

    for(current=RTSTable; current->what!=is_end; current++)
	if (!strcmp(current->name, name)) 
	    return current;
    
    return (RunTimeSupportDescriptor *) NULL;
}

entity hpfc_name_to_entity(name)
string name;
{
    RunTimeSupportDescriptor *entry = find_entry_by_name(name);

    if (entry) return entry->object;

    pips_internal_error("%s not found", name);
    
    return entity_undefined; /* just to avoid a gcc warning */
}

bool hpfc_intrinsic_like_function(e)
entity e;
{
    RunTimeSupportDescriptor *entry = find_entry_by_name(entity_local_name(e));

    return entry ? entry->what==is_ifn : false;
}

bool hpfc_io_like_function(e)
entity e;
{
    RunTimeSupportDescriptor *entry = find_entry_by_name(entity_local_name(e));

    return entry ? entry->what==is_iof : false;
}

/* that is all
 */
