/*
 * Management of Reductions in hpfc
 *
 * Fabien Coelho, May 1993.
 */

#include <stdio.h>
#include <strings.h>
extern int fprintf();

#include "genC.h"
#include "ri.h"
#include "hpf.h"
#include "hpf_private.h"
#include "message.h"

#include "ri-util.h"
#include "misc.h"
#include "hpfc.h"
#include "defines-local.h"

#define MAX_REDUCTION 1
#define MIN_REDUCTION 2
#define SUM_REDUCTION 3

typedef struct t_reduction
{
    char *name;
    int  kind;
    int  ndim;
} t_reduction;

static t_reduction reductions[] =
{
  {"REDMIN",  MIN_REDUCTION, -1},
  {"REDMIN1", MIN_REDUCTION, 1},
  {"REDMIN2", MIN_REDUCTION, 2},
  {"REDMIN3", MIN_REDUCTION, 3},
  {"REDMAX",  MAX_REDUCTION, -1},
  {"REDMAX1", MAX_REDUCTION, 1},
  {"REDMAX2", MAX_REDUCTION, 2},
  {"REDMAX3", MAX_REDUCTION, 3},
  {0,         0,             0}
};

/*
 * static t_reduction *find_reduction(c)
 *
 * look for the presence of the reduction in the reduction list.
 */
static t_reduction *find_reduction(s)
string s;
{
     t_reduction
	 *red = reductions;
     
     while(red->kind != 0)
     {
	 if (!strcmp(s, red->name)) return(red);
	 red++;
     }
     
     return(NULL);
 }

/*
 * bool call_reduction_p(c)
 *
 * true if a given call is a call to reduction function.
 * ??? a generic function name should be managed here?
 */
bool call_reduction_p(c)
call c;
{
    return(find_reduction(module_local_name(call_function(c)))!=NULL);
}

/*
 * void reduction_parameters(c, pred, pb, pdim)
 *
 * extract the informations needed to generate the distributed code.
 */
static void reduction_parameters(c, pred, pb, pdim, pe, pl)
call c;
t_reduction **pred;
basic *pb;
int *pdim;
entity *pe;
list *pl;
{
    expression
	arg = EXPRESSION(CAR(call_arguments(c)));

    pips_assert("reduction_parameters",
		syntax_reference_p(expression_syntax(arg)));

    *pe   = reference_variable(syntax_reference(expression_syntax(arg)));
    *pl   = CDR(call_arguments(c));
    *pdim = NumberOfDimension(*pe);
    *pred = find_reduction(module_local_name(call_function(c)));
    *pb   = entity_basic(*pe);
}

static char *reduction_name(kind)
int kind;
{
    static char *reduction_names[] = {"MAX", "MIN", "SUM"};

    pips_assert("reduction_name", (kind>=0) && (kind<3));

    return(reduction_names[kind-1]);
}

/*
 * entity make_reduction_function(prefix, ndim, kind, base, nargs)
 *
 * find or create an entity for the reduction function...
 */
entity make_reduction_function(prefix, ndim, kind, base, nargs)
string prefix;
int ndim, kind;
basic base;
int nargs;
{
    char 
	buffer[100],
	*buf = buffer;

    buf += strlen(sprintf(buf, "%sRED%d_", prefix, ndim));
    buf += strlen(sprintf(buf, pvm_what_options(base)));
    buf += strlen(sprintf(buf, "_%s", reduction_name(kind)));

    return(MakeRunTimeSupportFunction(buffer, nargs));
}

/*
 * bool compile_reduction(initial, phost, pnode)
 *
 * true is the compiler succeeded in compiling the reduction that way.
 * ??? many conditions are not yet checked...
 */
bool compile_reduction(initial, phost, pnode)
statement initial, *phost, *pnode;
{
    instruction
	i = statement_instruction(initial);
    list
	args = NIL;
    expression
	ref = expression_undefined,
	cll = expression_undefined;
    call
	reduction = call_undefined;
    t_reduction 
	*red ;
    basic b;
    int	
	dim = 0,
	arraynum = -1;
    list largs = NIL;
    entity
	array = entity_undefined,
	hostfunction = entity_undefined, 
	nodefunction = entity_undefined;

    pips_assert("compile_reduction",
		(instruction_call_p(i) && 
		 ENTITY_ASSIGN_P(call_function(instruction_call(i))) &&
		 (gen_length(call_arguments(instruction_call(i)))==2)));

    args = call_arguments(instruction_call(i));
    ref  = EXPRESSION(CAR(args));
    cll  = EXPRESSION(CAR(CDR(args)));

    pips_assert("compile_reduction",
		(syntax_reference_p(expression_syntax(ref)) &&
		 syntax_call_p(expression_syntax(cll))));

    reduction = syntax_call(expression_syntax(cll));
    
    debug(7, "compile_reduction", "call to %s\n",
	  entity_name(call_function(reduction)));

    pips_assert("compile_reduction", call_reduction_p(reduction));

    reduction_parameters(reduction, &red, &b, &dim, &array, &largs);

    /*
     * the array must be distributed accross the processors, not replicated,
     * and the reference variable mustn't be distributed.
     */

    if (!array_distributed_p(array) || 
/*	array_replicated_p(array) || ??? (to be coded for every dimensions:-) */
	(array_distributed_p
	 (reference_variable(syntax_reference(expression_syntax(ref))))))
	return(FALSE);

    arraynum = get_hpf_number(array);

    hostfunction = make_reduction_function("H", dim, red->kind, b, 0);
    nodefunction = make_reduction_function("N", dim, red->kind, b, 4*dim+2);

    *phost = make_assign_statement((expression) gen_copy_tree(ref),
				   make_call_expression(hostfunction, NIL));
    *pnode = 
	make_assign_statement
	    ((expression) gen_copy_tree(ref),
	     make_call_expression
	     (nodefunction,
	      CONS(EXPRESSION,
		   reference_to_expression
		   (make_reference(array,
				   array_lower_bounds_list(array))),
	      CONS(EXPRESSION,
		   int_to_expression(arraynum),
		   gen_nconc(array_lower_upper_bounds_list(array),
			     largs)))));
    
    return(TRUE);
}

