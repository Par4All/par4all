/*
 * Runtime Support Functions Management
 *
 * Fabien Coelho, May and June 1993
 */

#include <stdio.h>
extern int fprintf();

#include "genC.h"
#include "ri.h"

#include "misc.h"
#include "ri-util.h"
#include "bootstrap.h"
#include "hpfc.h"
#include "compiler_parameters.h"
#include "defines-local.h"

extern instruction MakeAssignInst(syntax l, expression r);
extern entity CreateIntrinsic(string name);                 /* in syntax.h */
extern entity MakeExternalFunction(entity e, type r);       /* idem */

entity 
    e_MYPOS,
    e_SendToC, 
    e_SendToH, 
    e_SendToA,
    e_HSendToA,
    e_SendToO, 
    e_SendToOs, 
    e_SendToOOs, 
    e_SendToHA, 
    e_SendToNO, 
    e_SendToHNO, 
    e_ReceiveFromS, 
    e_ReceiveFromH, 
    e_ReceiveFromC, 
    e_ReceiveFrommCS, 
    e_ReceiveFrommCH, 
    e_ComputeComputer, 
    e_ComputeOwners,
    e_CompNeighbour,
    e_SenderP, 
    e_OwnerP, 
    e_ComputerP,
    e_CompInOwnersP,
    e_LocalInd,
    e_LocalIndGamma,
    e_LocalIndDelta,
    e_InitHost,
    e_InitNode,
    e_HostEnd,
    e_NodeEnd,
    e_LoopBounds,
    e_SendToNeighb,
    e_ReceiveFromNeighb;

#define SND_TO_C 	"HPFC_SNDTO_C"
#define SND_TO_H 	"HPFC_SNDTO_H"
#define SND_TO_A 	"HPFC_SNDTO_A"
#define SND_TO_A_BY_H 	"HPFC_HSNDTO_A"
#define SND_TO_O 	"HPFC_SNDTO_O"
#define SND_TO_OS 	"HPFC_SNDTO_OS"
#define SND_TO_OOS 	"HPFC_SNDTO_OOS"
#define SND_TO_HA 	"HPFC_SNDTO_HA"
#define SND_TO_NO 	"HPFC_SNDTO_NO"
#define SND_TO_HNO 	"HPFC_SNDTO_HNO"

#define RCV_FR_S 	"HPFC_RCVFR_S"
#define RCV_FR_H 	"HPFC_RCVFR_H"
#define RCV_FR_C 	"HPFC_RCVFR_C"
#define RCV_FR_mCS 	"HPFC_RCVFR_mCS"
#define RCV_FR_mCH 	"HPFC_RCVFR_mCH"

#define CMP_COMPUTER 	"HPFC_CMPCOMPUTER"
#define CMP_OWNERS 	"HPFC_CMPOWNERS"
#define CMP_NEIGHBOUR	"HPFC_CMPNEIGHBOUR"

#define CND_SENDERP 	"HPFC_SENDERP"
#define CND_OWNERP 	"HPFC_OWNERP"
#define CND_COMPUTERP 	"HPFC_COMPUTERP"
#define CND_COMPINOWNP 	"HPFC_COMPUTERINOWNERSP"

#define LOCAL_IND 	"HPFC_LOCALIND"
#define LOCAL_IND_GAMMA	"HPFC_LOCALINDGAMMA"
#define LOCAL_IND_DELTA "HPFC_LOCALINDDELTA"

#define INIT_NODE	"HPFC_INIT_NODE"
#define INIT_HOST	"HPFC_INIT_HOST"
#define NODE_END	"HPFC_NODE_END"
#define HOST_END	"HPFC_HOST_END"

#define LOOP_BOUNDS	"HPFC_LOOP_BOUNDS"

#define SND_TO_N	"HPFC_SNDTO_N"
#define RCV_FR_N	"HPFC_RCVFR_N"

/*
 *
 */
void init_pvm_based_intrinsics()
{
    e_SendToC		= MakeRunTimeSupportSubroutine(SND_TO_C, 2); 
    e_SendToH		= MakeRunTimeSupportSubroutine(SND_TO_H, 2); 
    e_SendToA		= MakeRunTimeSupportSubroutine(SND_TO_A, 2); 
    e_HSendToA		= MakeRunTimeSupportSubroutine(SND_TO_A_BY_H, 2); 
    e_SendToO		= MakeRunTimeSupportSubroutine(SND_TO_O, 2); 
    e_SendToOs		= MakeRunTimeSupportSubroutine(SND_TO_OS, 2); 
    e_SendToOOs		= MakeRunTimeSupportSubroutine(SND_TO_OOS, 2); 
    e_SendToHA		= MakeRunTimeSupportSubroutine(SND_TO_HA, 2); 
    e_SendToNO		= MakeRunTimeSupportSubroutine(SND_TO_NO, 2); 
    e_SendToHNO		= MakeRunTimeSupportSubroutine(SND_TO_HNO, 2);

    e_ReceiveFromS	= MakeRunTimeSupportSubroutine(RCV_FR_S, 2); 
    e_ReceiveFromH	= MakeRunTimeSupportSubroutine(RCV_FR_H, 2); 
    e_ReceiveFromC	= MakeRunTimeSupportSubroutine(RCV_FR_C, 2); 
    e_ReceiveFrommCS	= MakeRunTimeSupportSubroutine(RCV_FR_mCS, 2); 
    e_ReceiveFrommCH	= MakeRunTimeSupportSubroutine(RCV_FR_mCH, 2); 

#ifndef HPFC_EXPAND_COMPUTE_COMPUTER
    e_ComputeComputer	= MakeRunTimeSupportSubroutine(CMP_COMPUTER, 1); 
#else
    e_ComputeComputer	= MakeRunTimeSupportSubroutine(CMP_COMPUTER, 8); 
#endif

#ifndef HPFC_EXPAND_COMPUTE_OWNERS
    e_ComputeOwners	= MakeRunTimeSupportSubroutine(CMP_OWNERS, 1);
#else
    e_ComputeOwners	= MakeRunTimeSupportSubroutine(CMP_OWNERS, 8);
#endif

    e_CompNeighbour	= MakeRunTimeSupportSubroutine(CMP_NEIGHBOUR, 1);

    e_SenderP		= MakeRunTimeSupportFunction(CND_SENDERP, 0); 
    e_OwnerP		= MakeRunTimeSupportFunction(CND_OWNERP, 0); 
    e_ComputerP		= MakeRunTimeSupportFunction(CND_COMPUTERP, 0);
    e_CompInOwnersP	= MakeRunTimeSupportFunction(CND_COMPINOWNP, 0);

    e_LocalInd		= MakeRunTimeSupportFunction(LOCAL_IND, 3);
    e_LocalIndGamma	= MakeRunTimeSupportFunction(LOCAL_IND_GAMMA, 3);
    e_LocalIndDelta	= MakeRunTimeSupportFunction(LOCAL_IND_DELTA, 3);

    e_InitHost		= MakeRunTimeSupportSubroutine(INIT_HOST, 0);
    e_InitNode		= MakeRunTimeSupportSubroutine(INIT_NODE, 0);
    e_HostEnd		= MakeRunTimeSupportSubroutine(HOST_END, 0);
    e_NodeEnd		= MakeRunTimeSupportSubroutine(NODE_END, 0);

    e_LoopBounds	= MakeRunTimeSupportSubroutine(LOOP_BOUNDS, 7);

    e_SendToNeighb	= MakeRunTimeSupportSubroutine(SND_TO_N, 0);
    e_ReceiveFromNeighb = MakeRunTimeSupportSubroutine(RCV_FR_N, 0);

    /* and so on. all these functions must be defined somewhere. */
}

/*
 * entity MakeRunTimeSupportSubroutine(local_name, number_of_arguments)
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

/*
 * entity MakeRunTimeSupportFunction(local_name, number_of_arguments)
 *
 * this function can be used even if the function is already declared
 * ??? an integer shouldn't always be returned
 */
entity MakeRunTimeSupportFunction(local_name, number_of_arguments)
string local_name;
int number_of_arguments;
{
    return(MakeExternalFunction(FindOrCreateEntity(TOP_LEVEL_MODULE_NAME,
						   local_name),
				MakeIntegerResult()));
}

/*
 * string pvm_what_options(b)
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

/*
 *
 */
statement st_call_send_or_receive(e, r)
entity e;
reference r;
{
    return
	(mere_statement
	 (make_instruction
	  (is_instruction_call, 			
	   make_call(e, 				
		     CONS(EXPRESSION,
			  MakeCharacterConstantExpression
			    (pvm_what_options(entity_basic(reference_variable(r)))),
		     CONS(EXPRESSION,
			  reference_to_expression(r),
			  NIL)))))); 
}

/*
 *
 */
statement st_send_to_computer(val)
reference val;
{
    return(st_call_send_or_receive(e_SendToC, val));
}

/*
 *
 */
statement st_send_to_host(val)
reference val;
{
    return(st_call_send_or_receive(e_SendToH, val));
}

/*
 *
 */
statement st_send_to_all_nodes(val)
reference val;
{
    return(st_call_send_or_receive(e_SendToA, val));
}

/*
 *
 */
statement st_host_send_to_all_nodes(val)
reference val;
{
    return(st_call_send_or_receive(e_HSendToA, val));
}

/*
 *
 */
statement st_send_to_neighbour()
{
    return(my_make_call_statement(e_SendToNeighb, NIL));
}

/*
 *
 */
statement st_send_to_owner(val)
reference val;
{
    return(st_call_send_or_receive(e_SendToO, val));
}

/*
 *
 */
statement st_send_to_owners(val)
reference val;
{
    return(st_call_send_or_receive(e_SendToOs, val));
}

/*
 *
 */
statement st_send_to_other_owners(val)
reference val;
{
    return(st_call_send_or_receive(e_SendToOOs, val));
}

/*
 *
 */
statement st_send_to_host_and_all_nodes(val)
reference val;
{
    return(st_call_send_or_receive(e_SendToHA, val));
}

/*
 *
 */
statement st_send_to_not_owners(val)
reference val; 
{
    return(st_call_send_or_receive(e_SendToNO, val));
}

/*
 *
 */
statement st_send_to_host_and_not_owners(val)
reference val; 
{
    return(st_call_send_or_receive(e_SendToHNO, val));
}

/******************************************************************************/
/*
 * Receives
 */

/*
 *
 */
statement st_receive_from_sender(goal)
reference goal;
{
    return(st_call_send_or_receive(e_ReceiveFromS, goal));
}

/*
 *
 */
statement st_receive_from_neighbour()
{
    return(my_make_call_statement(e_ReceiveFromNeighb, NIL));
}

/*
 *
 */
statement st_receive_from_host(goal)
reference goal;
{
    return(st_call_send_or_receive(e_ReceiveFromH, goal));
}

/*
 *
 */
statement st_receive_from_computer(goal)
reference goal;
{
    return(st_call_send_or_receive(e_ReceiveFromC, goal));
}

/*
 *
 */
statement st_receive_mcast_from_sender(goal)
reference goal;
{
    return(st_call_send_or_receive(e_ReceiveFrommCS, goal));
}

/*
 *
 */
statement st_receive_mcast_from_host(goal)
reference goal;
{
    return(st_call_send_or_receive(e_ReceiveFrommCH, goal));
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
#ifndef HPFC_EXPAND_COMPUTE_COMPUTER
    return(mere_statement(MakeUnaryCallInst(e_ComputeComputer,
					    reference_to_expression(ref))));
#else
    list 
	largs=NIL,
	linds = reference_indices(ref);
    int
	narray = (int) GET_ENTITY_MAPPING(hpfnumber, reference_variable(ref)),
	arity  = gen_length(linds);

    switch(arity)
    {
    case 0: largs = CONS(EXPRESSION, int_to_expression(0), largs);
    case 1: largs = CONS(EXPRESSION, int_to_expression(0), largs);
    case 2: largs = CONS(EXPRESSION, int_to_expression(0), largs);
    case 3: largs = CONS(EXPRESSION, int_to_expression(0), largs);
    case 4: largs = CONS(EXPRESSION, int_to_expression(0), largs);
    case 5: largs = CONS(EXPRESSION, int_to_expression(0), largs);
    case 6: largs = CONS(EXPRESSION, int_to_expression(0), largs);
    case 7: break;
    default: pips_error("st_compute_current_computer", "too many indices (%d)\n", arity);
    }
	
    largs = gen_nconc(CONS(EXPRESSION, int_to_expression(narray), NIL),
		      gen_nconc(lUpdateExpr(oldtonewnodevar, linds), largs));

    return(mere_statement(make_instruction(is_instruction_call,
					   make_call(e_ComputeComputer, largs))));
#endif
}

/*
 * statement st_compute_current_owners(ref)
 */
statement st_compute_current_owners(ref)
reference ref;
{
#ifndef HPFC_EXPAND_COMPUTE_OWNERS
    return(mere_statement(MakeUnaryCallInst(e_ComputeOwners,
					reference_to_expression(ref))));
#else
    list 
	largs=NIL,
	linds = reference_indices(ref);
    int
	narray = (int) GET_ENTITY_MAPPING(hpfnumber, reference_variable(ref)),
	arity  = gen_length(linds);

    switch(arity)
    {
    case 0: largs = CONS(EXPRESSION, int_to_expression(0), largs);
    case 1: largs = CONS(EXPRESSION, int_to_expression(0), largs);
    case 2: largs = CONS(EXPRESSION, int_to_expression(0), largs);
    case 3: largs = CONS(EXPRESSION, int_to_expression(0), largs);
    case 4: largs = CONS(EXPRESSION, int_to_expression(0), largs);
    case 5: largs = CONS(EXPRESSION, int_to_expression(0), largs);
    case 6: largs = CONS(EXPRESSION, int_to_expression(0), largs);
    case 7: break;
    default: pips_error("st_compute_current_computer", "too many indices (%d)\n", arity);
    }
	
    largs = gen_nconc(CONS(EXPRESSION, int_to_expression(narray), NIL),
		      gen_nconc(lUpdateExpr(oldtonewnodevar,  linds), largs));

    return(mere_statement(make_instruction(is_instruction_call,
					   make_call(e_ComputeOwners, largs))));
#endif
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
#ifndef HPFC_EXPAND_COMPUTE_LOCAL_INDEX
    expression
	expr1 = int_to_expression((int) GET_ENTITY_MAPPING(hpfnumber, array)),
	expr2 = int_to_expression(dim);
    
    return(MakeTernaryCall(e_LocalInd, expr1, expr2, expr));
#else
    list
	l    = (list) GET_ENTITY_MAPPING(newdeclarations, array),
	ldim = variable_dimensions(type_variable(entity_type(array)));
    int 
	i,
	newdecl;
    dimension 
	the_dim;

    /* ??? gen_find_ith */
    for (i=1; i<dim; i++)
    {
	ldim=CDR(ldim);
	l=CDR(l);
    }

    newdecl = INT(CAR(l));
    the_dim = DIMENSION(CAR(ldim));
        
    switch(newdecl)
    {
    case NO_NEW_DECLARATION:
	return(expr);
    case ALPHA_NEW_DECLARATION:
    {
	expression
	    shift = 
		int_to_expression(1 - HpfcExpressionToInt(dimension_lower(the_dim)));
	
	return(MakeBinaryCall(CreateIntrinsic(PLUS_OPERATOR_NAME), expr, shift));
    }
    case BETA_NEW_DECLARATION:
    {
	align
	    a = (align) GET_ENTITY_MAPPING(hpfalign, array);
	entity
	    template = align_template(a);
	distribute
	    d = (distribute) GET_ENTITY_MAPPING(hpfdistribute, template);
	alignment
	    al = FindAlignmentOfDim(align_alignment(a),dim);
	int
	    tempdim = alignment_templatedim(al),
	    procdim;
	dimension
	    template_dim = FindIthDimension(template,tempdim);
	distribution
	    di = FindDistributionOfDim(distribute_distribution(d), tempdim, &procdim);
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
		(expr):((iabsrate==1)?
			(MakeUnaryCall(CreateIntrinsic(UNARY_MINUS_OPERATOR_NAME),
				       expr)):
			(MakeBinaryCall(CreateIntrinsic(MULTIPLY_OPERATOR_NAME),
					rate,
					expr))));
	
	t1 = ((ishift==0)?
	      (prod):((ishift>0)?
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
    case GAMMA_NEW_DECLARATION:
    {
	expression
	    expr1 = int_to_expression((int) GET_ENTITY_MAPPING(hpfnumber, array)),
	    expr2 = int_to_expression(dim);
	
	return(MakeTernaryCall(e_LocalIndGamma, expr1, expr2, expr));
    }
    case DELTA_NEW_DECLARATION:
    {
	expression
	    expr1 = int_to_expression((int) GET_ENTITY_MAPPING(hpfnumber, array)),
	    expr2 = int_to_expression(dim);
	
	return(MakeTernaryCall(e_LocalIndDelta, expr1, expr2, expr));
    }
    default:
	pips_error("expr_compute_local_index","unexpected new declaration tag\n");
    }

    return(expression_undefined);
#endif			   
}

/******************************************************************************/
/*
 * Conditions
 */

/*
 *
 */
expression condition_senderp()
{
    return(MakeNullaryCall(e_SenderP));
}

/*
 *
 */
expression condition_ownerp()
{
    return(MakeNullaryCall(e_OwnerP));
}

/*
 *
 */
expression condition_computerp()
{
    return(MakeNullaryCall(e_ComputerP));
}

/*
 *
 */
expression condition_computer_in_owners()
{
    return(MakeNullaryCall(e_CompInOwnersP));
}

/*
 *
 */
expression condition_not_computer_in_owners()
{
    return(MakeUnaryCall(CreateIntrinsic(NOT_OPERATOR_NAME),
			 condition_computer_in_owners()));
}

/*
 *
 */
instruction MakeUnaryCallInst(f,e)
entity f;
expression e;
{
    return(make_instruction(is_instruction_call,
			    make_call(f, CONS(EXPRESSION, e, NIL))));
}

/*
 *
 */
statement st_init_host()
{
    return(my_make_call_statement(e_InitHost, NIL));
}

/*
 *
 */
statement st_init_node()
{
    return(my_make_call_statement(e_InitNode, NIL));
}

/*
 *
 */
statement st_host_end()
{
    return(my_make_call_statement(e_HostEnd, NIL));
}

/*
 *
 */
statement st_node_end()
{
    return(my_make_call_statement(e_NodeEnd, NIL));
}

/*
 * statement my_make_call_statement(e, l)
 *
 * generate a call statement to function e, with expression list l as an argument.
 */
statement my_make_call_statement(e, l)
entity e;
list l;
{
    return(mere_statement(make_instruction(is_instruction_call,
					   make_call(e, l))));
}

/*
 * void add_pvm_init_and_end(phs, pns)
 */
void add_pvm_init_and_end(phs, pns)
statement *phs, *pns;
{
    statement
	stinithost = st_init_host();
    
    statement_comments(stinithost) = 
	strdup("c initializes host and spawns nodes\n");

    (*phs) = make_block_statement(CONS(STATEMENT,
				       stinithost,
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
    return(my_make_call_statement(e_CompNeighbour, 
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
	larg = NIL,
	lind = NIL;

    pips_assert("st_generate_packing_and_passing",
		(len==NumberOfDimension(array)));

    pips_assert("st_generate_packing_and_passing", ((len<=4) && (len>=1)));

    lind = array_lower_bounds_list(array);
    larg = array_lower_upper_bounds_list(array);

    MAPL(cr,
     {
	 range
	     r = RANGE(CAR(cr));

	 larg = gen_nconc(larg,
			  CONS(EXPRESSION,
			       range_lower(r),
			  CONS(EXPRESSION,
			       range_upper(r),
			  CONS(EXPRESSION,
			       range_increment(r),
			       NIL))));
     },
	 content);

/*    larg = CONS(EXPRESSION,
		MakeCharacterConstantExpression
		(pvm_what_options(entity_basic(array))), ...) */

    larg = CONS(EXPRESSION,
		reference_to_expression(make_reference(array, lind)),
		larg);
    
    /*
     * larg content:
     *
     * // what, 
     * array([dimension lower]*len), 
     * dimension [lower, upper]*len, 
     * range [lower, upper, increment]*len
     */

    return(my_make_call_statement
	   (make_packing_function("HPFC", len, bsend, entity_basic(array), 1+5*len),
	    larg));

}

/*
 * string bound_parameter_name(array, side, dim)
 *
 * returns a name for the bound of the declaration 
 * of array array, side side and dimension dim.
 */
string bound_parameter_name(array, side, dim)
entity array;
int side, dim;
{
    char
	buffer[100];

    return(strdup(sprintf(buffer, "%s_%s%d",
			  local_name(entity_name(array)),
			  (side)?("UP"):("LO"),
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

