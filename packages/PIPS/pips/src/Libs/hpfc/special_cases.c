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
/* HPFC module, Fabien Coelho, May 1993.
 */

#include "defines-local.h"

#include "conversion.h"
#include "effects-generic.h"
#include "effects-simple.h"
#include "effects-convex.h"

/************************************************************** REDUCTIONS */

#define MAX_REDUCTION 1
#define MIN_REDUCTION 2
#define SUM_REDUCTION 3

typedef struct t_reduction
{
    char *name;
    int  kind;
    int  ndim;
} t_reduction;

static t_reduction all_reductions[] =
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

/* static t_reduction *find_reduction(c)
 *
 * look for the presence of the reduction in the reduction list.
 */
static t_reduction *find_reduction(s)
string s;
{
     t_reduction *red;
     
     for (red=all_reductions; red->kind; red++)
	 if (same_string_p(s, red->name)) return red;
     
     return NULL;
 }

/* bool call_reduction_p(c)
 *
 * true if a given call is a call to reduction function.
 * ??? a generic function name should be managed here?
 */
bool hpfc_entity_reduction_p(e)
entity e;
{
    return(find_reduction(module_local_name(e))!=NULL);
}

bool call_reduction_p(c)
call c;
{
    return(hpfc_entity_reduction_p(call_function(c)));
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

    pips_assert("reference", syntax_reference_p(expression_syntax(arg)));

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
    pips_assert("valid kind", kind>=0 && kind<3);
    return(reduction_names[kind-1]);
}

/* find or create an entity for the reduction function...
 */
static entity 
make_reduction_function(
    string prefix,
    int ndim,
    int kind,
    basic base,
    int nargs)
{
    char buffer[100];

    (void) sprintf(buffer, "%sRED %d %s %s",
		   prefix, ndim, pvm_what_options(base), reduction_name(kind));

    return MakeRunTimeSupportFunction(buffer, nargs, basic_tag(base));
}

/* bool compile_reduction(initial, phost, pnode)
 *
 * true is the compiler succeeded in compiling the reduction that way.
 * ??? many conditions are not yet checked...
 */
bool compile_reduction(initial, phost, pnode)
statement initial, *phost, *pnode;
{
    instruction	i = statement_instruction(initial);
    list args = NIL;
    expression ref, cll;
    call reduction;
    t_reduction *red ;
    basic b;
    int	dim = 0, arraynum = -1;
    list largs = NIL;
    entity array, hostfunction, nodefunction;

    pips_assert("assignment",
		(instruction_call_p(i) && 
		 ENTITY_ASSIGN_P(call_function(instruction_call(i))) &&
		 (gen_length(call_arguments(instruction_call(i)))==2)));

    args = call_arguments(instruction_call(i));
    ref  = EXPRESSION(CAR(args));
    cll  = EXPRESSION(CAR(CDR(args)));

    pips_assert("reference", (syntax_reference_p(expression_syntax(ref)) &&
	    syntax_call_p(expression_syntax(cll))));

    reduction = syntax_call(expression_syntax(cll));
    
    pips_debug(7, "call to %s\n", entity_name(call_function(reduction)));

    pips_assert("reduction call", call_reduction_p(reduction));

    reduction_parameters(reduction, &red, &b, &dim, &array, &largs);

    /*
     * the array must be distributed accross the processors, not replicated,
     * and the reference variable mustn't be distributed.
     */

    if (!array_distributed_p(array) || 
	(array_distributed_p
	 (reference_variable(syntax_reference(expression_syntax(ref))))))
	return(false);

    arraynum = load_hpf_number(array);

    hostfunction = make_reduction_function("H", dim, red->kind, b, 0);
    nodefunction = make_reduction_function("N", dim, red->kind, b, 4*dim+2);

    *phost = make_assign_statement( copy_expression(ref),
				   make_call_expression(hostfunction, NIL));
    *pnode = 
	make_assign_statement
	    ( copy_expression(ref),
	     make_call_expression
	     (nodefunction,
	      CONS(EXPRESSION, entity_to_expression(array),
	      CONS(EXPRESSION, int_to_expression(arraynum),
		   gen_nconc(array_lower_upper_bounds_list(array),
			     largs)))));
    
    return(true);
}

/******************************************** REDUCTION DIRECTIVE HANDLING */
/* hpfc_reductions =
 *     initial:entity x replacement:entity x operator:reduction ;
 * reduction = { min , max , sum , prod , and , or } ; 
 */

/* looking for the reduction operator is a basic recursion. no check. 
 */
DEFINE_LOCAL_STACK(current_call, call)
static entity searched_variable;
static tag found_operator;

static bool ref_filter(reference r)
{
    entity fun;

    if (searched_variable!=reference_variable(r)) return false;

    /* get the call just above -- well, what about intermediate casts ??? */
    fun = call_function(current_call_head()); 

    if (ENTITY_PLUS_P(fun)||ENTITY_MINUS_P(fun)) 
	found_operator=is_reduction_operator_sum;
    else if (ENTITY_MULTIPLY_P(fun)) 
	found_operator=is_reduction_operator_prod;
    else if (ENTITY_MIN_P(fun)||ENTITY_MIN0_P(fun))
	found_operator=is_reduction_operator_min;
    else if (ENTITY_MAX_P(fun)||ENTITY_MAX0_P(fun))
	found_operator=is_reduction_operator_max;
    else if (ENTITY_AND_P(fun))
	found_operator=is_reduction_operator_and;
    else if (ENTITY_OR_P(fun))
	found_operator=is_reduction_operator_or;

    return false;
}

static reduction_operator get_operator(entity e, statement s)
{
    make_current_call_stack();
    found_operator = tag_undefined;
    searched_variable = e;
    gen_multi_recurse(s,
		      call_domain, current_call_filter, current_call_rewrite,
		      reference_domain, ref_filter, gen_null,
		      NULL);
    free_current_call_stack();

    pips_assert("some operator found", found_operator!=tag_undefined);
    return make_reduction_operator(found_operator, UU);
}

/* finally, I can do without replacement:
 * the host keeps and contributes the initial value! 
 */
static hpfc_reductions reduction_of_in(entity e, statement s)
{
    pips_debug(5, "considering %s in %p\n", entity_name(e), s);

    return make_hpfc_reductions(e, entity_undefined, get_operator(e, s));
}

list /* of hpfc_reductions */
handle_hpf_reduction(statement s)
{
    list /* of hpfc_reductions */ lr = NIL;

    MAP(ENTITY, e, 
	lr = CONS(HPFC_REDUCTIONS, reduction_of_in(e, s), lr),
	entities_list(load_hpf_reductions(s)));

    return lr;
}

/* for reduction directive:
 */
static string new_reduction_name(reduction_operator op)
{
    if (reduction_operator_sum_p(op)) 
	return "SUM";
    else if (reduction_operator_prod_p(op))
	return "PROD";
    else if (reduction_operator_min_p(op))
	return "MIN";
    else if (reduction_operator_max_p(op))
	return "MAX";
    else if (reduction_operator_and_p(op))
	return "AND";
    else if (reduction_operator_or_p(op))
	return "OR";
    else
	pips_internal_error("unexpected reduction_operator tag (%d)",
			    reduction_operator_tag(op));

    return "unknown";
}

/* name is {H,N}{PRE,POST}_{SUM,PROD,MIN,...}_{REAL4,INTERGER4,...}
 */
static entity 
make_new_reduction_function(
    reduction_operator op,
    bool prolog,
    bool host,
    basic base)
{
    char buffer[100];

    (void) sprintf(buffer, "%s", concatenate
	     (host? "H": "N", prolog? "PRE": "POST", " ",
	      new_reduction_name(op), " ", pvm_what_options(base), NULL));

    return MakeRunTimeSupportSubroutine(buffer, 1);
}

static statement 
compile_one_reduction(
    hpfc_reductions red,
    bool prolog,
    bool host)
{
    entity var = hpfc_reductions_initial(red);

    return call_to_statement
	(make_call(make_new_reduction_function
	    (hpfc_reductions_operator(red), prolog, host, entity_basic(var)),
	     CONS(EXPRESSION, entity_to_expression(var), 
	     CONS(EXPRESSION, int_to_expression(element_number(
		 variable_basic(type_variable(entity_type(var))),
		 variable_dimensions(type_variable(entity_type(var))))),
		  NIL))));
}

/* 
 */
list /* of statement */ 
compile_hpf_reduction(
    list /* of hpfc_reductions */ lr,
    bool prolog,
    bool host)
{
    list /* of statement */ ls = NIL;
    MAP(HPFC_REDUCTIONS, r, 
	ls = CONS(STATEMENT, compile_one_reduction(r, prolog, host), ls),
	lr);
    return ls;
}

/******************************************************** SUB ARRAY SHIFTS */

/* bool subarray_shift_p(s, pe, plvect)
 * statement s;
 * entity *pe;
 * list *plvect;
 *
 * checks whether a loop nest is a subarray shift,
 * that is a parallel loop nest with one assign in the body 
 * which is a 1D shift of a locally constant value.
 * returns false if this dimension is not distributed, because
 * the overlap analysis will make a better job. 
 * should also check that the accessed subarray is a section.
 * and that it is a block distribution.
 * ??? should also check for a 1 alignment
 *
 * the entity returned *pe is the shifted array 
 * the list returned is the list of the vector shifts for each dimension
 */

#define expression_complex_p(e) \
    (normalized_complex_p(expression_normalized(e)))

void free_vector_list(l)
list l;
{
    gen_map((gen_iter_func_t)vect_rm, l);
    gen_free_list(l);
}

bool vecteur_nul_p(v)
Pvecteur v;
{
    return VECTEUR_NUL_P(v) || (!v->succ && var_of(v)==TCST && val_of(v)==0);
}



/* some static variables used for the detection...
 */
static bool subarray_shift_ok = true;
static entity array = entity_undefined;
static list current_regions = NIL, lvect = NIL;

static bool locally_constant_vector_p(Pvecteur v)
{
    for(; v!=NULL; v=v->succ)
	if (var_of(v)!=TCST && written_effect_p((entity) var_of(v), 
						current_regions))
	    return false;

    return true;
}

static bool subarray_shift_assignment_p(call c)
{
    list lhs_ind, rhs_ind, args = call_arguments(c);
    expression
	lhs = EXPRESSION(CAR(args)),
	rhs = EXPRESSION(CAR(CDR(args)));
    reference lhs_ref, rhs_ref;
    bool shift_was_seen = false;
    int dim;

    /*  LHS *must* be a reference
     */ 
    pips_assert("reference", expression_reference_p(lhs));

    /* is RHS a reference?
     */
    if (!expression_reference_p(rhs)) return false;

    lhs_ref = syntax_reference(expression_syntax(lhs)),
    rhs_ref = syntax_reference(expression_syntax(rhs));
    array = reference_variable(lhs_ref);

    /*  same array, and a block-distributed one ?
     */
    if (array!=reference_variable(rhs_ref) ||
	!array_distributed_p(array) || 
	!block_distributed_p(array))
	return false;
    
    lhs_ind = reference_indices(lhs_ref),
    rhs_ind = reference_indices(rhs_ref);
    
    pips_assert("same arity", gen_length(lhs_ind)==gen_length(rhs_ind));

    /*  compute the difference of every indices, if possible.
     */
    for(;
	!ENDP(lhs_ind);
	lhs_ind=CDR(lhs_ind), rhs_ind=CDR(rhs_ind))
    {
	expression
	    il = EXPRESSION(CAR(lhs_ind)),
	    ir = EXPRESSION(CAR(rhs_ind));

	if (expression_complex_p(il) || expression_complex_p(ir))
	{
	    free_vector_list(lvect), lvect = NIL;
	    return false;
	}

	/* else compute the difference */
	lvect = 
	    CONS(PVECTOR, (VECTOR)
		 vect_substract(normalized_linear(expression_normalized(il)),
				normalized_linear(expression_normalized(ir))),
		 lvect);
    }

    lvect = gen_nreverse(lvect);

    /* now checks for a constant shift on a distributed dimension
     * and that's all.
     * ??? I could deal with other constant shifts on non distributed dims
     */
    dim = 0;
    MAPL(cv,
     {
	 Pvecteur v = (Pvecteur) PVECTOR(CAR(cv));
	 int p;

	 dim++;

	 if (vecteur_nul_p(v)) continue;

	 /* else the vector is not null
	  */
	 if (shift_was_seen || 
	     !ith_dim_distributed_p(array, dim, &p) ||
	     !locally_constant_vector_p(v))
	 {
	     pips_debug(7, "false on array %s, dimension %d\n",
			entity_local_name(array), dim);
	     free_vector_list(lvect);
	     lvect = NIL;
	     return false;
	 }

	 shift_was_seen = true;
     },
	 lvect);

    return true;
}

static bool loop_filter(loop l)
{
    int stride = HpfcExpressionToInt(range_increment(loop_range(l)));
    return subarray_shift_ok = subarray_shift_ok && (stride==1 || stride==-1);
}

static bool call_filter(call c)
{
    entity e = call_function(c);

    pips_debug(9, "function: %s\n", entity_name(e));

    if (ENTITY_CONTINUE_P(e)) return false;
    if (ENTITY_ASSIGN_P(e)) 
    {
	if (ref_to_dist_array_p(c))
	  subarray_shift_ok = subarray_shift_ok && 
	    (entity_undefined_p(array))? subarray_shift_assignment_p(c): false;
	/* else: private variable assigned in a parallel loop, ok?! */
    }
    else
	subarray_shift_ok = false; 
    
    return false;
}

static bool cannot_be_a_shift(void * x)
{
    return subarray_shift_ok = false;
}

bool subarray_shift_p(s, pe, plvect)
statement s;
entity *pe;
list *plvect;
{
    array = entity_undefined;
    subarray_shift_ok = true;
    lvect = NIL;
    current_regions = load_statement_local_regions(s);

    DEBUG_STAT(8, "considering statement", s);

    gen_multi_recurse(s,
		      call_domain, call_filter, gen_null,
		      loop_domain, loop_filter, gen_null,
		      test_domain, cannot_be_a_shift, gen_null,
		      expression_domain, gen_false, gen_null,
		      NULL);

    subarray_shift_ok &= !entity_undefined_p(array) &&
	rectangular_must_region_p(array, s);

    if (subarray_shift_ok)
	*pe = array,
	*plvect = lvect;
    else
	free_vector_list(lvect), lvect = NIL;
    current_regions = NIL;

    pips_debug(8, "returning %s\n", subarray_shift_ok? "TRUE": "FALSE");
    return subarray_shift_ok;
}

/* generates the needed subroutine 
 */
static entity make_shift_subroutine(entity var)
{
    char buffer[100];
    type t;
    variable v;
    int ndim;
    
    pips_assert("defined", !entity_undefined_p(var)); t = entity_type(var);
    pips_assert("variable", type_variable_p(t)); v = type_variable(t);

    ndim = gen_length(variable_dimensions(v));

    (void) sprintf(buffer, "%s SHIFT %d",
		   pvm_what_options(variable_basic(v)),
		   ndim);

    return MakeRunTimeSupportSubroutine(buffer, ndim*4+4);
}

Psysteme get_read_effect_area(list le, entity var)
{
    MAP(EFFECT, e,
     {
	 if (action_read_p(effect_action(e)) &&
	     reference_variable(effect_any_reference(e))==var)
	     return effect_system(e);
     },
	 le);

    return SC_UNDEFINED;
}

list make_rectangular_area(statement st, entity var)
{
    list l = NIL;
    Psysteme
	s = sc_dup(get_read_effect_area(load_statement_local_regions(st),
					 var));
    Pcontrainte	c, lower = CONTRAINTE_UNDEFINED, upper = CONTRAINTE_UNDEFINED;
    Variable idx;
    entity div = hpfc_name_to_entity(IDIVIDE);
    int dim = gen_length(variable_dimensions(type_variable(entity_type(var))));

    sc_transform_eg_in_ineg(s);
    c = sc_inegalites(s);
    sc_inegalites(s) = (Pcontrainte)NULL, 
    sc_rm(s);

    for(; dim>0; dim--)
    {
	idx = (Variable) get_ith_region_dummy(dim);

	constraints_for_bounds(idx, &c, &lower, &upper);

	l = CONS(EXPRESSION, constraints_to_loop_bound(lower, idx, true, div),
	    CONS(EXPRESSION, constraints_to_loop_bound(upper, idx, false, div),
		 l));

	(void) contraintes_free(lower), lower=NULL;
	(void) contraintes_free(upper), upper=NULL;
    }

    (void) contraintes_free(c);
    
    return l;
}

/* statement generate_subarray_shift(s, var, lshift)
 * statement s;
 * entity var;
 * list lshift;
 *
 * generates a call to the runtime library shift subroutine
 * for the given array, the given shift and the corresponding
 * region in statement s.
 * this function assumes that subarray_shift_p was true on s,
 * and does not checks the conditions again.
 */
statement generate_subarray_shift(statement s, entity var, list lshift)
{
    entity subroutine = make_shift_subroutine(var);
    int array_number = load_hpf_number(array), shifted_dim = 0;
    expression shift = expression_undefined;
    list ldecl = array_lower_upper_bounds_list(array), largs;
    Pvecteur v;

    /*  gets the shifted dimension and the shift
     */
    MAPL(cv,
     {
	 v = (Pvecteur) PVECTOR(CAR(cv));
	 shifted_dim++;

	 if (!vecteur_nul_p(v)) 
	 {
	     shift = make_vecteur_expression(v);
	     break;
	 }
     },
	 lshift);

    pips_assert("shift defined", shift!=expression_undefined);
    free_vector_list(lshift);
    
    /*  all the arguments
     */
    largs = 
	CONS(EXPRESSION, entity_to_expression(array),
	CONS(EXPRESSION, int_to_expression(array_number),
	CONS(EXPRESSION, int_to_expression(shifted_dim),
        CONS(EXPRESSION, shift,
	     gen_nconc(ldecl, make_rectangular_area(s, var))))));

    return hpfc_make_call_statement(subroutine, largs);
}

/************************************************************* FULL COPIES */

/* tests whether a loop nest is a full copy loop nest, 
 * i.e. it copies an array into another, both being aligned.
 * ??? it should share some code with the subarray shift detection...
 * ??? another quick hack for testing purposes (remappings).
 */

/* statement simple_statement(statement s)
 *
 * what: tells whether s is (after simplification) a simple statement, 
 *       i.e. a call, and returns it (may be hidden by blocks and continues).
 * how: recursion thru the IR.
 * input: statement s
 * output: returned statement, or NULL if not simple...
 * side effects:
 *  - uses some static data
 * bugs or features:
 */

static statement simple_found;
static bool ok;
DEFINE_LOCAL_STACK(current_stmt, statement)

void hpfc_special_cases_error_handler()
{
    error_reset_current_stmt_stack();
    error_reset_current_call_stack();
}

static bool not_simple(void * x)
{
    ok = false;
    gen_recurse_stop(NULL);
    return false;
}

static bool check_simple(call c)
{
    if (!same_string_p(entity_local_name(call_function(c)),
		       CONTINUE_FUNCTION_NAME))
    {
	if (simple_found) 
	{
	    ok = false;
	    gen_recurse_stop(NULL);
	}
	else
	    simple_found = current_stmt_head();
    }

    return false;
}

statement simple_statement(statement s)
{
    ok = true;
    simple_found = (statement) NULL;
    make_current_stmt_stack();

    gen_multi_recurse
	(s,
	 statement_domain,  current_stmt_filter, current_stmt_rewrite,
	 test_domain,       not_simple,          gen_null,
	 loop_domain,       not_simple,          gen_null,
	 call_domain,       check_simple,        gen_null,
	 expression_domain, gen_false,           gen_null,
	 NULL);

    free_current_stmt_stack();

    return (ok && simple_found) ? simple_found : NULL;
}

/* bool full_define_p (reference r, list ll)
 *
 * what: true if the loop nest ll fully scans r
 * how: not very beautiful. 
 * input: r and ll
 * output: the bool result
 * side effects: none
 * bugs or features:
 */

static loop find_loop(entity index, list /* of loops */ ll)
{
    MAP(LOOP, l, if (loop_index(l)==index) return l, ll);
    return (loop) NULL; /* if not found */
}

bool full_define_p(reference r, list /* of loops */ ll)
{
    entity array = reference_variable(r);
    list /* of expression */ li = reference_indices(r),
         /* of dimension */ ld,
         /* of entity */ lseen = NIL;
    int ndim;
    
    pips_assert("variable", entity_variable_p(array));

    ld = variable_dimensions(type_variable(entity_type(array)));
    pips_assert("same arity", gen_length(ll)==gen_length(ld));

    /* checks that the indices are *simply* the loop indexes,
     * and that the whole dimension is scanned. Also avoids (i,i)...
     */
    for(ndim=1; li; POP(li), POP(ld), ndim++)
    {
	syntax s = expression_syntax(EXPRESSION(CAR(li)));
	entity index;
	dimension dim;
	range rg;
	loop l;
	expression inc;

	if (!syntax_reference_p(s)) 
	{
	    gen_free_list(lseen);
	    return false;
	}

	index = reference_variable(syntax_reference(s));

	if (gen_in_list_p(index, lseen))
	{
	    gen_free_list(lseen);
	    return false;
	}

	lseen = CONS(ENTITY, index, lseen);

	l = find_loop(index, ll);
	
	if (!l) 
	{
	    gen_free_list(lseen);
	    return false;
	}

	/* checks that the loop range scans the whole dimension */

	rg = loop_range(l);
	inc = range_increment(rg);
	dim = DIMENSION(CAR(ld));
	
	/* increment must be 1.
	 * lowers and uppers must be equal.
	 */
	if (!integer_constant_expression_p(inc) ||
	    integer_constant_expression_value(inc)!=1 ||
	    !same_expression_p(range_lower(rg), dimension_lower(dim)) ||
	    !same_expression_p(range_upper(rg), dimension_upper(dim)))
	{
	    pips_debug(9, "incomplete scan of %s[dim=%d]\n", 
		       entity_name(array), ndim);
	    gen_free_list(lseen);
	    return false;
	}
    }

    gen_free_list(lseen);
    return true;
}

/* bool full_copy_p(statement s, reference * pleft, reference * pright)
 *
 * what: tells whether a loop nest is a 'full copy' one, that is it fully
 *       define an array from another, with perfect alignment.
 * how: pattern matching of what we are looking for...
 * input: the statement
 * output: the bool result, plus both references.
 * side effects:
 *  - uses some static data
 * bugs or features:
 *  - pattern matching done this way is just a hack...
 */
#define XDEBUG(msg) \
  pips_debug(6, "statement %p: " msg "\n", (void*) s)

static int 
number_of_non_empty_statements(
    list /* of statement */ ls)
{
    int n = 0;
    MAP(STATEMENT, s, if (!empty_code_p(s)) n++, ls);
    return n;
}

bool full_copy_p(statement s, reference * pleft, reference * pright)
{
    list /* of lists */ lb = NIL, l,
         /* of loops */ ll = NIL,
         /* of expressions */ la = NIL;
    statement body = parallel_loop_nest_to_body(s, &lb, &ll), simple;
    expression e;
    reference left, right;
    int len;

    DEBUG_STAT(6, "considering statement", s);

    /* the loop nest must be perfect... !!!
     * should check for continues?
     */
    for (l=lb; l; POP(l)) 
	if (number_of_non_empty_statements(CONSP(CAR(l)))>1) 
	    {
		XDEBUG("non perfectly nested");
		gen_free_list(lb), gen_free_list(ll); return false;
	    }

    gen_free_list(lb), lb = NIL;

    /* the body must be a simple assignment
     */
    simple = simple_statement(body);

    if (!simple || !instruction_assign_p(statement_instruction(simple)))
    { 
	XDEBUG("body not simple"); gen_free_list(ll); return false;
    }

    la = call_arguments(instruction_call(statement_instruction(simple)));
    pips_assert("2 arguments to assign", gen_length(la)==2);

    left = expression_reference(EXPRESSION(CAR(la)));
    e = EXPRESSION(CAR(CDR(la)));

    if (!expression_reference_p(e))
    {
	XDEBUG("rhs not a reference"); gen_free_list(ll); return false;
    }

    right = expression_reference(e);

    /* compatible arities 
     */
    len = gen_length(ll); /* number of enclosing loops */

    if (gen_length(reference_indices(left))!=len ||
	gen_length(reference_indices(right))!=len)
    {
	XDEBUG("incompatible arities"); gen_free_list(ll); return false;
    }

    /* the lhs should be fully defined by the loop nest...
     * but it does not matter for the rhs!
     */
    if (!full_define_p(left, ll))
    {
	XDEBUG("lhs not fully defined"); gen_free_list(ll); return false;
    }

    gen_free_list(ll); 

    /* both lhs and rhs references must be aligned
     */
    if (!references_aligned_p(left, right))
    {
	XDEBUG("references not aligned"); return false;
    }

    /* ??? should check the new declarations...
     */

    XDEBUG("ok");
    *pleft = left;
    *pright = right;
    return true;
}

/* statement generate_full_copy(reference left, reference right)
 *
 * what: copies directly right into left, that must conform...
 * how: direct loop nest on local data
 * input: both entities
 * output: the returned statement
 * side effects: 
 * bugs or features:
 *  - assumes that the references are ok, that is they come from
 *    a full_copy_p execution.
 *  - code generated based on the original entities. thus the 
 *    code cleaning pass is trusted.
 */
statement generate_full_copy(reference left, reference right)
{
    statement body;
    int ndim, i;
    list /* of entity */ lindexes, l,
         /* of dimension */ ld;
    entity array = reference_variable(left), 
           new_array = load_new_node(array);

    ndim = gen_length(variable_dimensions(type_variable(entity_type(array))));

    body = make_assign_statement(reference_to_expression(copy_reference(left)),
			      reference_to_expression(copy_reference(right)));
    
    /* indexes are reused. bounds are taken from the node entity declaration.
     */
    lindexes = expressions_to_entities(reference_indices(left));
    ld = variable_dimensions(type_variable(entity_type(new_array)));

    for(i=1, l=lindexes; i<=ndim; i++, POP(lindexes), POP(ld))
    {
	dimension d = DIMENSION(CAR(ld));

	body = loop_to_statement(make_loop
	   (ENTITY(CAR(lindexes)),
	    make_range(copy_expression(dimension_lower(d)),
		       copy_expression(dimension_upper(d)),
		       int_to_expression(1)),
	    body,
	    entity_empty_label(),
	    make_execution(is_execution_sequential, UU),
	    NIL));
    }

    gen_free_list(l);
    return body;    
}

/*    That is all
 */
