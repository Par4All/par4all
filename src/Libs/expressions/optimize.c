/*
 * $Id$
 *
 * $Log: optimize.c,v $
 * Revision 1.34  1999/07/15 20:36:13  coelho
 * ac_cse expermimental phase added.
 *
 * Revision 1.33  1999/07/15 13:33:19  coelho
 * binary to nary stuff added...
 *
 * Revision 1.32  1999/05/28 12:57:58  coelho
 * missing break fixed.
 *
 * Revision 1.31  1999/05/28 12:04:43  coelho
 * nary pattern simplifies added.
 *
 * Revision 1.30  1999/05/27 14:56:21  ancourt
 * fma moved after icm association/atomization.
 *
 * Revision 1.29  1999/05/27 14:48:18  ancourt
 * 2 eole passes. does not work yet.
 *
 * Revision 1.28  1999/05/12 12:53:17  zory
 * reorder module statement after transformation
 *
 * Revision 1.27  1999/05/06 07:06:00  zory
 * test strategy modified !!
 *
 * Revision 1.26  1998/12/30 16:52:50  zory
 * atomize transformations added in strategies
 *
 * Revision 1.25  1998/12/09 09:51:47  zory
 * apply gen_nreverse to the list of indices to have i,j,k,.... in outermost
 * to inermost order
 *
 * Revision 1.24  1998/12/08 16:48:36  coelho
 * new structure to send data to eole.
 *
 * Revision 1.23  1998/11/26 14:26:45  coelho
 * fixed static variables. more comments.
 *
 * Revision 1.22  1998/11/26 10:22:26  coelho
 * davinci expression dump into $database/$module directory.
 *
 * Revision 1.21  1998/11/25 16:32:10  coelho
 * dump expressions in davinci format.
 *
 * Revision 1.20  1998/11/25 14:52:19  coelho
 * test/R10K...
 *
 * Revision 1.19  1998/11/24 17:22:37  coelho
 * simplify implemented.
 *
 * Revision 1.18  1998/11/24 14:15:28  coelho
 * more comments.
 *
 * Revision 1.17  1998/11/24 14:12:59  coelho
 * nary to binary makes use of the strategy.
 *
 * Revision 1.16  1998/11/24 14:09:07  coelho
 * strategies...
 *
 * Revision 1.15  1998/11/24 13:01:42  zory
 * debug manipulations
 *
 * Revision 1.14  1998/11/18 22:46:52  coelho
 * huffman included.
 *
 * Revision 1.13  1998/11/18 22:43:06  coelho
 * switch_nary_to_binary and optimize_simplify_patterns are called.
 *
 * Revision 1.12  1998/11/09 14:50:35  zory
 * use make_constant function instead of specific float and integer
 * make_float/integer_constant_entity funcions
 *
 * Revision 1.11  1998/11/04 09:42:57  coelho
 * use properties for eole and its options.
 *
 * Revision 1.10  1998/11/04 08:59:06  zory
 * double and float format updated
 *
 * Revision 1.9  1998/10/22 11:30:29  zory
 * double type for const values added
 *
 * Revision 1.8  1998/10/20 14:51:23  zory
 * move the free statement for all strings inside the if statement
 *
 * Revision 1.7  1998/10/20 14:48:47  zory
 * free the list of unoptimized expressions only when there are some
 * expressions in the module ! (if statement)
 *
 * Revision 1.6  1998/09/17 12:08:43  zory
 * taking into account new entity from eole
 *
 * Revision 1.5  1998/09/14 12:50:15  coelho
 * more comments.
 *
 * Revision 1.4  1998/09/14 12:34:11  coelho
 * added import from eole and substitution in module code.
 *
 * Revision 1.3  1998/09/11 12:18:39  coelho
 * new version thru a reference (suggested by PJ).
 *
 * Revision 1.2  1998/09/11 09:42:49  coelho
 * write equalities...
 *
 * Revision 1.1  1998/04/29 09:07:42  coelho
 * Initial revision
 *
 *
 * expression optimizations by Julien.
 */

#include <stdio.h>
#include <stdlib.h>

#include "linear.h"

#include "genC.h"
#include "ri.h"
#include "ri-util.h"

#include "misc.h"
#include "properties.h"

#include "resources.h"
#include "pipsdbm.h"

#include "eole_private.h"

#define DEBUG_NAME "TRANSFORMATION_OPTIMIZE_EXPRESSIONS_DEBUG_LEVEL"

extern void module_reorder(statement);
extern void perform_icm_association(string, statement);
extern void perform_ac_cse(string, statement);

/****************************************************************** STRATEGY */

/* this structure defines a strategy for eole.
 */
typedef struct
{
  string name;

  /* HUFFMAN.
   */
  bool apply_balancing;
  double (*huffman_cost)(expression);
  bool huffman_mode;
  
  /* EOLE. 2 passes.
   */
  string eole_strategy;

  bool apply_eole1;
  string apply_eole1_flags; /* not used yet */

  bool apply_eole2;
  string apply_eole2_flags;

  /* SIMPLIFY.
   */
  bool apply_nary_simplify;
  bool apply_simplify;

  /* GCM CSE 
   */
  bool apply_gcm_cse;

} optimization_strategy, *poptimization_strategy;

/* current strategy.
 */
static poptimization_strategy strategy = NULL;

/************************************************************** EOLE PROCESS */

/* file name prefixes to deal with eole.
 * /tmp should be fast (may be mapped into memory).
 */
#define OUT_FILE_NAME 	"/tmp/pips_to_eole"
#define IN_FILE_NAME	"/tmp/eole_to_pips"

/* property names.
 */
#define EOLE		"EOLE"		/* eole binary */
#define EOLE_FLAGS	"EOLE_FLAGS"	/* default options */
#define EOLE_OPTIONS	"EOLE_OPTIONS"	/* additionnal options */

/* returns the eole command to be executed in an allocated string.
 */
static string 
get_eole_command
  (string in, /* input file from eole. */
   string out,/* output file to eole. */
   string flags)
{
  return strdup(concatenate(get_string_property(EOLE), " ", 
			    flags, " ",
			    get_string_property(EOLE_OPTIONS), 
			    " -S ", strategy->eole_strategy,
			    " -o ", in, " ", out, NULL));
}


/********************************************************* INTERFACE TO EOLE */

/* the list of right hand side expressions.
 */
static list /* of expressionwithlevel */ rhs;

/* the current list of loop indices for the expression (not used yet).
 */
static list /* of entity */ indices; 

static void add_one_more_expression(expression e)
{
  expressionwithlevel ewl = make_expressionwithlevel(gen_nreverse(gen_copy_seq(indices)), e);
  rhs = CONS(EXPRESSIONWITHLEVEL, ewl, rhs);
}

static bool loop_flt(loop l)
{
  entity index = loop_index(l);
  indices = CONS(ENTITY, index, indices);
  return TRUE; /* keep on. */
}

static void loop_rwt(loop l)
{
  list tmp = indices;
  pips_assert("same index", ENTITY(CAR(indices))==loop_index(l));
  indices = CDR(indices);
  CDR(tmp) = NIL;
  gen_free_list(tmp);
}

/* rhs expressions of assignments.
 */
static bool call_filter(call c)
{
  if (ENTITY_ASSIGN_P(call_function(c)))
  {
    expression e = EXPRESSION(CAR(CDR(call_arguments(c))));
    add_one_more_expression(e);
  }
  return FALSE;
}

/* other expressions may be found in loops and so?
 */
static bool expr_filter(expression e)
{
  add_one_more_expression(e);
  return FALSE;
}

static list /* of expressionwithlevel */ 
get_list_of_rhs(statement s)
{
  list result;

  rhs = NIL;
  indices = NIL;
  gen_multi_recurse(s,
		    expression_domain, expr_filter, gen_null,
		    call_domain, call_filter, gen_null,
		    loop_domain, loop_flt, loop_rwt,
		    NULL);
  
  result = gen_nreverse(rhs);
  rhs = NIL;
  indices = NIL;
  return result;
}

/* export a list of expression of the current module.
 * done thru a convenient reference.
 */
static void write_list_of_rhs(FILE * out, list /* of expressionwithlevel */ le)
{
  /* reference astuce = make_reference(get_current_module_entity(), le);*/
  lexpressionwithlevel lewl = make_lexpressionwithlevel(le);
  write_lexpressionwithlevel(out, lewl);
  lexpressionwithlevel_list(lewl) = NIL;
  free_lexpressionwithlevel(lewl);
}

/* export expressions to eole thru the newgen format.
 * both entities and rhs expressions are exported. 
 */
static void write_to_eole(string module, list le, string file_name)
{
  FILE * toeole;
  
  pips_debug(3, "writing to eole for module %s\n", module);
  
  toeole = safe_fopen(file_name, "w");
  write_tabulated_entity(toeole);
  write_list_of_rhs(toeole, le);

  safe_fclose(toeole, file_name);
}

#define SIZE_OF_BUFFER 100

static string
read_and_allocate_string_from_file(FILE * file)
{
  int test; 
  char buffer[SIZE_OF_BUFFER];

  test = fscanf(file, "%s",buffer);
  pips_assert("fscanf - read string from file \n",(test==1));

  return strdup(buffer);
}


/* import a list of entity that have been created during the eole
 * transformations and create them
 */
static void 
read_new_entities_from_eole(FILE * file, string module){
  int num = 0;
  int i, test;
  string ent_type;
  string const_type;
  int const_size = 0;
  string const_value;
  
  entity e; 
  
  /* read the number of new entities to create */
  test = fscanf(file,"%d\n",&num);
  pips_assert("fscanf - read number of entity \n",(test==1));
  pips_debug(3,"reading %d new entity from module %s\n", num, module);
  for (i=0;i<num;i++){
    
    ent_type = read_and_allocate_string_from_file(file);
    
    if (!strcmp(ent_type,"constant")) { /* constant */
      
      const_type = read_and_allocate_string_from_file(file);
      
      test = fscanf(file," %d\n", &const_size);
      pips_assert("fscanf - read entity basic type size\n",(test==1));

      const_value = read_and_allocate_string_from_file(file);

      if (!strcmp(const_type,"int")) {/* int */
	
	/* create integer entity */
	e = make_constant_entity(const_value, is_basic_int, const_size);
	pips_assert("make integer constant entity\n",
		    entity_consistent_p(e));  
      }
      else 
	if (!strcmp(const_type,"float")) {/* float */
	  
	  /* create float entity */
	  e = make_constant_entity(const_value, is_basic_float, const_size);
	  pips_assert("make float constant entity\n",
		      entity_consistent_p(e));  
	}
      else 
	pips_error("read_new_entities_from_eole", 
		   "can't create this kind of constant entity : %s",
		   const_type);

      free(const_type);
      free(const_value);
    }
    else 
      pips_error("read_new_entities_from_eole", 
		 "can't create this kind of entity : %s",
		 ent_type);
    
    free(ent_type);
  }
}   


/* import expressions from eole.
 */
static list /* of expression */
read_from_eole(string module, string file_name)
{
    FILE * fromeole;
    reference astuce;
    list result;

    pips_debug(3, "reading from eole for module %s\n", module);
    
    fromeole = safe_fopen(file_name, "r");

    /* read entities to create... 
     * should use some newgen type to do so (to share buffers...)
     */
    read_new_entities_from_eole(fromeole, module);

    astuce = read_reference(fromeole);
    result = reference_indices(astuce);
    reference_indices(astuce) = NIL;
    free_reference(astuce);

    return result;
}

/* swap term to term syntax field in expression list, as a side effect...
 */
static void 
swap_syntax_in_expression(  list /* of expressionwithlevel */ lcode,
			    list /* of expression */ lnew)
{
  pips_assert("equal length lists", gen_length(lcode)==gen_length(lnew));
  
  for(; lcode; lcode=CDR(lcode), lnew=CDR(lnew))
    {
      expression old, new;
      syntax tmp;

      old = expressionwithlevel_expression(EXPRESSIONWITHLEVEL(CAR(lcode)));
      new = EXPRESSION(CAR(lnew));
      
      tmp = expression_syntax(old);
      expression_syntax(old) = expression_syntax(new);
      expression_syntax(new) = tmp;	
    }
}

/* apply eole on all expressions in s.
 */
static void 
apply_eole_on_statement(string module_name, statement s, string flags)
{
  list /* of expressionwithlevel/expression */ le, ln;

  ln = NIL;
  le = get_list_of_rhs(s);
  
  if (gen_length(le)) /* not empty list */
  {
    string in, out, cmd;

    /* create temporary files */
    in = safe_new_tmp_file(IN_FILE_NAME);
    out = safe_new_tmp_file(OUT_FILE_NAME);
    
    /* write informations in out file for EOLE */
    write_to_eole(module_name, le, out);
    
    /* run eole (Evaluation Optimization for Loops and Expressions) 
     * as a separate process.
     */
    cmd = get_eole_command(in, out, flags);
    
    pips_debug(2, "executing: %s\n", cmd);
    
    safe_system(cmd);
    
    /* read optimized expressions from eole */
    ln = read_from_eole(module_name, in);
    
    /* replace the syntax values inside le by the syntax values from ln */
    swap_syntax_in_expression(le, ln);
    
    /* must now free the useless expressions */
    
    
    /* remove temorary files and free allocated memory.
     */
    safe_unlink(out);
    safe_unlink(in);
    
    /* free strings */
    free(out), out = NULL;
    free(in), in = NULL;
    free(cmd), cmd = NULL;
  }
  else 
    pips_debug(3, "no expression for module %s\n", module_name);

  pips_debug(3,"EOLE transformations ... Done for module %s\n", module_name);

  /* free lists */
  gen_free_list(ln);
  gen_free_list(le);
}


/************************************************************* SOME PATTERNS */

/* A + (--B) -> A - B
 */
static entity
  bplus  = NULL,
  uminus = NULL,
  bminus = NULL;

/* returns B if uminus(B), else 0
 */
static expression is_uminus(expression e)
{
  syntax s = expression_syntax(e);
  if (syntax_call_p(s))
  {
    call c = syntax_call(s);
    if (call_function(c)==uminus && gen_length(call_arguments(c))==1)
      return EXPRESSION(CAR(call_arguments(c)));
  }
  return NULL;
}

static void call_simplify_rwt(call c)
{
  if (call_function(c)==bplus)
  {
    list la = call_arguments(c);
    expression e1, e2, me;
    
    pips_assert("2 args to binary plus", gen_length(la)==2);

    e1 = EXPRESSION(CAR(la));
    e2 = EXPRESSION(CAR(CDR(la)));

    me = is_uminus(e2);
    if (me)
    {
      EXPRESSION(CAR(CDR(la))) = me; /* memory leak */
      call_function(c) = bminus;
      return;
    }

    me = is_uminus(e1);
    if (me)
    {
      EXPRESSION(CAR(CDR(la))) = me;
      EXPRESSION(CAR(la)) = e2; /* memory leak */
      call_function(c) = bminus;
      return;
    }
  }
}

static void generate_bminus(statement s)
{
  bplus  = entity_intrinsic(PLUS_OPERATOR_NAME);
  uminus = entity_intrinsic(UNARY_MINUS_OPERATOR_NAME);
  bminus = entity_intrinsic(MINUS_OPERATOR_NAME);

  gen_recurse(s, call_domain, gen_true, call_simplify_rwt);

  bplus  = NULL;
  uminus = NULL;
  bminus = NULL;
}

/* look for some expressions in s and simplify some patterns.
 */
static void optimize_simplify_patterns(statement s)
{
  /* not implemented yet. */

  /* a + (-b)       -> a - b */
  /* (-b) + a       -> a - b */
  generate_bminus(s);

  /* a * (1/ b)     -> a / b */
  /* (1/ b) * a     -> a / b */
  /* a + (-b * c)   -> a - (b * c) */
}


/************************************************************ N-ARY SIMPLIFY */

static entity
  multiply = NULL,
  inverse = NULL,
  divide = NULL;

static bool is_inverse(expression e)
{
  syntax s = expression_syntax(e);
  if (!syntax_call_p(s)) return FALSE;
  return call_function(syntax_call(s)) == inverse;
}

static void call_nary_rwt(call c)
{
  entity func = call_function(c);
  list numerator = NIL, denominator = NIL;
  int nnum, nden;
  expression enu, eden;
  
  if (func != multiply) return;
  /* it is a multiply */
  
  MAP(EXPRESSION, e,
  {
    if (is_inverse(e))
      denominator = 
	CONS(EXPRESSION, 
	  EXPRESSION(CAR(call_arguments(syntax_call(expression_syntax(e))))),
	     denominator);
    else
      numerator = CONS(EXPRESSION, e, numerator);
  },
      call_arguments(c));

  nden = gen_length(denominator);
  nnum = gen_length(numerator);

  switch (nden)
  {
  case 0: /* nothing to change */
    gen_free_list(denominator);
    gen_free_list(numerator);
    return;
  case 1:
    eden = EXPRESSION(CAR(denominator));
    gen_free_list(denominator);
    break;
  default:
    eden = call_to_expression(make_call(multiply, denominator));
    break;
  }

  switch (nnum) 
  {
  case 0:
    enu = int_to_expression(1);
    break;
  case 1:
    enu = EXPRESSION(CAR(numerator));
    gen_free_list(numerator);
    break;
  default:
    enu = call_to_expression(make_call(multiply, numerator));
    break;
  }

  call_function(c) = divide;
  gen_free_list(call_arguments(c));
  call_arguments(c) = CONS(EXPRESSION, enu, CONS(EXPRESSION, eden, NIL));
}

static void optimize_simplify_nary_patterns(statement s)
{
  /* N-ARY * and 1/ -> N-ARY* / N-ARY*
   */
  multiply = entity_intrinsic(EOLE_PROD_OPERATOR_NAME);
  inverse  = entity_intrinsic("INV");
  divide   = entity_intrinsic("/");

  gen_recurse(s, call_domain, gen_true, call_nary_rwt);

  multiply = NULL;
  inverse  = NULL;
  divide   = NULL;
}


/* forward substitute if only there once. not implemented.
 */

/*********************************************************** EXPRESSION COST */

/* computes the expression weight.
 */
static double expression_weight(expression e)
{
  double cost = 0.0E0;
  syntax s = expression_syntax(e);
  if (syntax_call_p(s))
  {
    list args = call_arguments(syntax_call(s));
    cost = gen_length(args) - 1.0E0;
    MAP(EXPRESSION, e, cost += expression_weight(e), args);
  }
  return cost;
}

/* computes the expression depth.
 */
static double expression_depth(expression e)
{
  double cost = 0.0E0;
  syntax s = expression_syntax(e);
  if (syntax_call_p(s))
  {
    MAP(EXPRESSION, e, 
	{ double w = expression_depth(e); cost = cost>w? cost: w; },
	call_arguments(syntax_call(s)));
    cost += 1.0E0; 
    /* too simple... may depend on the function called. complexity tables? */
  }
  return cost;
}

/* WG */
static double expression_gravity_rc(expression e, double depth)
{
  double cost = 0.0E0;
  syntax s = expression_syntax(e);
  if (syntax_call_p(s))
  {
    MAP(EXPRESSION, e, 
	cost += expression_gravity_rc(e, depth+1.0E0), /* ? */
	call_arguments(syntax_call(s)));
    cost += 1.0E0; /* too simple? */
  }
  return cost;
}

static double expression_gravity(expression e)
{
  return expression_gravity_rc(e, 0.0E0);
}

static double expression_gravity_inv(expression e)
{
  return -expression_gravity(e);
}

/************************************************************* NARY-FICATION */
/* switch binary to nary expressions where possible.
   this is normally done in eole anyway.
*/

typedef enum 
{ 
  not_asocom, /* 0 */
  is_plus, is_mult, 
  is_max, is_min, 
  is_and, is_or 
} 
  asocom_operator_t;

typedef struct
{
  char * binary;
  char * nary;
} 
  binary_to_nary_t;

#define MAX_NAME	"MAX"
#define MIN_NAME	"MIN"

static binary_to_nary_t bton[] = 
{
  { "+", EOLE_SUM_OPERATOR_NAME },
  { "*", EOLE_PROD_OPERATOR_NAME },

  { "MAX",   MAX_NAME },
  { "MIN",   MIN_NAME },

  { "MAX0",  MAX_NAME },
  { "DMAX1", MAX_NAME },
  { "AMAX1", MAX_NAME },
  { "AMAX0", MAX_NAME },

  { "MIN0",  MIN_NAME },
  { "DMIN1", MIN_NAME },
  { "AMIN1", MIN_NAME },
  { "AMIN0", MIN_NAME },
  
  /* ??? .AND. .OR.
   */
  { NULL, NULL }
};

static entity binary_to_nary(entity e)
{
  string lname = entity_local_name(e);
  binary_to_nary_t * pbn;

  for (pbn = bton; pbn->binary; pbn++)
    if (same_string_p(pbn->binary, lname))
      return entity_intrinsic(pbn->nary);

  return e;
}

static bool nary_operator_p(entity e)
{
  binary_to_nary_t * pbn;
  string lname = entity_local_name(e);

  for (pbn = bton; pbn->nary; pbn++)
    if (same_string_p(pbn->nary, lname))
      return TRUE;

  return FALSE;
}

/* top-down: switch calls to nary form.
 */
static bool nary_call_flt(call c)
{
  call_function(c) = binary_to_nary(call_function(c));
  return TRUE;
}

/* bottom-up: + + -> +
 */
static void nary_call_rwt(call c)
{
  entity called = call_function(c);
  list newl = NIL, oldl = call_arguments(c);

  if (nary_operator_p(called))
    {
      MAP(EXPRESSION, e, 
      {
	syntax s = expression_syntax(e);
	if (syntax_call_p(s) && call_function(syntax_call(s))==called)
	  {
	    newl = gen_nconc(newl, call_arguments(syntax_call(s)));
	    call_arguments(syntax_call(s)) = NIL;
	    free_expression(e);
	  }
	else
	  {
	    newl = gen_nconc(newl, CONS(EXPRESSION, e, NIL));
	  }
      },
          oldl);
    }

  call_arguments(c) = newl;
  gen_free_list(oldl);
}

void naryfication_of_expressions(statement s)
{
  gen_recurse(s, call_domain, nary_call_flt, nary_call_rwt);
}

/*
  top down...

  -    -> + --
  /    -> * 1/ [not on integers]
  -- + -> + -- --
  1/ * -> * 1/ 1/
*/

typedef struct 
{
  string asym;
  string syme;
  string inv;
  bool not_ints;
} symetric_opertor_t;

static symetric_opertor_t symop[] =
{ { "-", "+", "--", FALSE },
  { "/", "*", "INV", TRUE },
  { NULL, NULL, NULL, FALSE }
};

static symetric_opertor_t * what_operator(entity e, int which_one)
{
  string lname = entity_local_name(e);
  symetric_opertor_t * sot;
  for (sot = symop; sot->asym; sot++)
    {
      if ((which_one==1 && same_string_p(lname, sot->asym)) ||
	  (which_one==0 && same_string_p(lname, sot->inv)) ||
	  (which_one==2 && same_string_p(lname, sot->syme)))
	return sot;
    }
  return NULL;
}

entity inverse_operator_of(entity e)
{
  symetric_opertor_t * sot = what_operator(e, 2);
  if (sot) return entity_intrinsic(sot->inv);
  else return NULL;
}

static bool inv_call_flt(call c)
{
  entity op = call_function(c);
  symetric_opertor_t * so;

  /* switch asymetric operation if needed */
  so = what_operator(op, TRUE);
  if (so)
    {
      bool doit = TRUE;

      if (so->not_ints)
	{
	  expression tmp = call_to_expression(c);
	  basic b =  please_give_me_a_basic_for_an_expression(tmp);
	  if (basic_int_p(b)) doit = FALSE;
	  syntax_call(expression_syntax(tmp)) = call_undefined;
	  free_expression(tmp);
	}
      
      if (doit) /* we have to substitute. */
	{
	  list largs = call_arguments(c);
	  expression second;
	  pips_assert("binary call", gen_length(largs)==2);
	  second = EXPRESSION(CAR(CDR(largs)));

	  call_function(c) = entity_intrinsic(so->syme);
	  expression_syntax(second) = 
	    make_syntax(is_syntax_call,
			make_call(entity_intrinsic(so->inv),
		CONS(EXPRESSION, 
		     make_expression(expression_syntax(second), 
				     normalized_undefined),
		     NIL)));
	}
    }

  /* push down inverse operators if needed. */     
  so = what_operator(op, FALSE);
  if (so)
    {
      list largs = call_arguments(c);
      syntax s;
      pips_assert("one arg to inverse op", gen_length(largs)==1);
      s = expression_syntax(EXPRESSION(CAR(largs)));
      if (syntax_call_p(s))
	{
	  call called = syntax_call(s);
	  if (same_string_p(so->syme, 
			    entity_local_name(call_function(called))))
	    {
	      /* memory leak... */
	      entity invop = call_function(c);
	      call_function(c) = call_function(called);
	      call_arguments(c) = call_arguments(called);
	      /* now insert invop */
	      MAP(EXPRESSION, e, 
	      {
		expression_syntax(e) = 
		  make_syntax(is_syntax_call,
		      make_call(invop, 
				CONS(EXPRESSION, 
				     make_expression(expression_syntax(e),
						     normalized_undefined),
				     NIL)));
	      },
	          call_arguments(called));
	    }
	}
    }
  
  return TRUE;
}

void inverse_normalization_of_expressions(statement s)
{
  gen_recurse(s, call_domain, inv_call_flt, gen_null);
}

/**************************************** HUFFMAN ALGORITHM ON AN EXPRESSION */

/* switch nary to binary with huffman algorithm.
 */ 
static entity huffman_nary_operator = NULL;
static entity huffman_binary_operator = NULL;
static double (*huffman_cost)(expression) = NULL;
/* TRUE:  Huffman.
   FALSE: rateau.
*/
static boolean huffman_mode = TRUE;

typedef struct
{
  expression expr;
  double cost;
} cost_expression;

/* comparison function for qsort. descending order.
 */
static int cost_expression_cmp(const void * v1, const void * v2)
{
  cost_expression * c1 = (cost_expression *) v1;
  cost_expression * c2 = (cost_expression *) v2;
  if (c1->cost==c2->cost) 
    /* when they are equals another criterion may be used, so as 
       to favor double loads (with neighbor references).
     */
    return 0;
  return 2*(c1->cost<c2->cost) - 1;
}

/* debug function */
/* print informations from an cost_expressions array */
static void
debug_cost_expression_array(string s,
			    cost_expression * tce,
			    int size)
{
  int i;
  pips_debug(9,"%s \n", s);
  pips_debug(9," %d elements in cost_expression array \n", size);

  for (i=0; i<size; i++) {
    pips_debug(9," - %d - expression : ", i); 
    print_expression(tce[i].expr);
    pips_debug(9,"\n - %d - cost : %f \n", i, tce[i].cost);
  }
}

/* build an cost_expression array from an expression list.
 */
static cost_expression * 
list_of_expressions_to_array(
    list /* of expression */ le,
    double (*cost)(expression))
{
  int len = gen_length(le), i=0;
  cost_expression * tce = malloc(len * sizeof(cost_expression));
  pips_assert("enough memory", tce!=NULL);
  
  MAP(EXPRESSION, e, 
  {
    tce[i].expr = e;
    tce[i].cost = cost(e);
    i++;
  },
    le);

  qsort(tce, len, sizeof(cost_expression), cost_expression_cmp);

  ifdebug(3) debug_cost_expression_array("qsort output", tce, len);

  return tce;
}

/* insert ce in tce[0..n-1] by decreassing order.
 */
static void 
insert_sorted_into_array(cost_expression * tce, int n, cost_expression ce)
{
  int i=0;
  while (i<n && ce.cost<tce[i].cost) i++; /* find where to insert. */
  while (n>=i) tce[n]=tce[n-1], n--;      /* shift tail. */
  tce[i] = ce;                            /* insert. */
}

/* simply insert.
 */
static void
insert_last_into_array(cost_expression * tce, int n, cost_expression ce)
{
  tce[n] = ce;
}

/* apply huffman balancing algorithm if it is a call to 
 * huffman_nary_operator.
 *
 * the prettyprint may only reflect this if the
 * PRETTYPRINT_ALL_PARENTHESES property is set to TRUE.
 */
static void call_rwt(call c)
{
  if (call_function(c)==huffman_nary_operator) 
  {
    /* let us switch to a binary tree. */
    list args = call_arguments(c);
    int nargs = gen_length(args);
    cost_expression * tce;

    pips_debug(4, "dealing with operator %s\n", 
	       entity_name(huffman_nary_operator));

    pips_assert("several arguments to nary operator", nargs>=2);

    if (nargs==2) /* the call is already binary. */
    {
      call_function(c) = huffman_binary_operator;
      return;
    }
    /* else */

    tce = list_of_expressions_to_array(args, huffman_cost);
    /* drop initial list: */
    gen_free_list(args), args = NIL, call_arguments(c) = NIL;

    while (nargs>2)
    {
      cost_expression ce;
      ce.expr = MakeBinaryCall(huffman_binary_operator,
			       tce[nargs-1].expr, tce[nargs-2].expr);
      ce.cost = huffman_cost(ce.expr);

      if (huffman_mode)	insert_sorted_into_array(tce, nargs-2, ce);
      else      	insert_last_into_array(tce,  nargs-2, ce);

      ifdebug(3) debug_cost_expression_array("insert done", tce, nargs-1);

      nargs--;
    }

    /* last one is done in place. */
    call_function(c) = huffman_binary_operator;
    call_arguments(c) = CONS(EXPRESSION, tce[0].expr,
			     CONS(EXPRESSION, tce[1].expr, NIL));
    free(tce), tce = NULL;
  }
  else 
    pips_debug(3,"non huffman operator : %s\n ", 
	       entity_local_name(call_function(c)));
}

/* apply the huffman balancing on every call to nary_operator
 * and build calls to binary_operator instead, with cost to chose.
 */
static void 
build_binary_operators_with_huffman(
    statement s,
    entity nary_operator,
    entity binary_operator,
    double (*cost)(expression),
    boolean mode)
{
  huffman_nary_operator = nary_operator;
  huffman_binary_operator = binary_operator;
  huffman_cost = cost;
  huffman_mode = mode;

  ifdebug(8) print_statement(s);

  gen_multi_recurse(s,
		    call_domain, gen_true, call_rwt,
		    NULL);

  huffman_nary_operator = NULL;
  huffman_binary_operator = NULL;
  huffman_cost = NULL;  
  huffman_mode = TRUE;
}

/* switch nary operators to binary ones.
 * n+ -> +, n* -> *
 * missing: & | and so on.
 */
static void
switch_nary_to_binary(statement s)
{
  entity
    plus = entity_intrinsic(PLUS_OPERATOR_NAME),
    nplus = entity_intrinsic(EOLE_SUM_OPERATOR_NAME),
    mult = entity_intrinsic(MULTIPLY_OPERATOR_NAME),
    nmult = entity_intrinsic(EOLE_PROD_OPERATOR_NAME);

  /* options: another expression cost function could be used.
   * for instance, defining -expression_gravity would build
   * the most unbalanced possible expression tree wrt WG.
   */
  build_binary_operators_with_huffman
    (s, nplus, plus, strategy->huffman_cost, strategy->huffman_mode);
  build_binary_operators_with_huffman
    (s, nmult, mult, strategy->huffman_cost, strategy->huffman_mode);
}

/***************************************************** PREDEFINED STRATEGIES */

/* predefined optimization strategies.
 */
static optimization_strategy 
  strategies[] = 
{
  { 
    /* name */ "P2SC", 
    /* huff */ TRUE, expression_gravity, TRUE,
    /* eole */ "0", TRUE, "", TRUE, "-m",
    /* simp */ TRUE, TRUE,
    /* gcm cse */ TRUE
	       
  },
  {
    "test",
    TRUE, expression_gravity, TRUE,
    "0", TRUE, "", TRUE, "-m",
    TRUE, TRUE,
    FALSE
  },
  {
    "R10K",
    TRUE, expression_gravity_inv, FALSE,
    "1", TRUE, "", TRUE, "-m",
    TRUE, TRUE,
    TRUE
  },
  {
    "EOLE",
    FALSE, NULL, FALSE,
    "0", TRUE, "", TRUE, "-m",
    FALSE, FALSE, 
    FALSE
  },

  /* this one MUST be the last one! */
  {
    NULL, /* default similar to P2SC. */
    TRUE, expression_gravity, TRUE,
    "0", TRUE, "", TRUE, "-m",
    TRUE, TRUE,
    TRUE
  }
};

static void set_current_optimization_strategy(void)
{
  string name = get_string_property("EOLE_OPTIMIZATION_STRATEGY");
  for (strategy = strategies; strategy->name!=NULL; strategy++)
  {
    if (same_string_p(name, strategy->name))
      return;
  }
  pips_user_warning("'%s' strategy not found, default assumed.\n", name);
}

static void reset_current_optimization_strategy(void)
{
  strategy = NULL;
}


/************************************************** DAVINCI DUMP EXPRESSIONS */

#define GRAPH_PREFIX "optimize_expressions_"
#define GRAPH_SUFFIX ".daVinci"

/* dump all expressions in s as davinci graphs. 
 */
static void davinci_dump_expressions(
   string module_name, string phase, statement s)
{
  string dir, filename;
  FILE * out;

  /* filename: $current.database/$module/$prefix_$phase.$suffix
   */
  dir = db_get_current_workspace_directory();
  filename = strdup(concatenate
      (dir, "/", module_name, "/", GRAPH_PREFIX, phase, GRAPH_SUFFIX, NULL));
  free(dir), dir = NULL;

  out = safe_fopen(filename, "w"); /* directory MUST exist */
  davinci_dump_all_expressions(out, s);
  safe_fclose(out, filename);

  free(filename), filename = NULL;
}

/*************************************************** INTERFACE FROM PIPSMAKE */

/* pipsmake interface.
 */
bool optimize_expressions(string module_name)
{
    statement s;

    debug_on(DEBUG_NAME);

    /* get needed stuff.
     */
    set_current_module_entity(local_name_to_top_level_entity(module_name));
    set_current_module_statement((statement)
        db_get_memory_resource(DBR_CODE, module_name, TRUE));
    set_current_optimization_strategy();

    s = get_current_module_statement();

    /* check consistency before optimizations */
    pips_assert("consistency checking before optimizations",
		statement_consistent_p(s));

    ifdebug(1) davinci_dump_expressions(module_name, "initial", s);

    /* do something here.
     */

    /* Could perform more optimizations here...
     */

    /* EOLE Stuff
     */
    if (strategy->apply_eole1)
      apply_eole_on_statement(module_name, s, get_string_property(EOLE_FLAGS));

    /* Could perform more optimizations here...
     */
    /* CSE/ICM + atom
     */

    if (strategy->apply_gcm_cse)
    {
      perform_icm_association(module_name, s); 
      perform_ac_cse(module_name, s);
    }

    /* EOLE Stuff, second pass for FMA.
     */
    if (strategy->apply_eole2)
      apply_eole_on_statement(module_name, s, strategy->apply_eole2_flags);

    if (strategy->apply_nary_simplify)
      optimize_simplify_nary_patterns(s);

    if (strategy->apply_balancing)
      switch_nary_to_binary(s);
    
    if (strategy->apply_simplify)
      optimize_simplify_patterns(s);

    /* others?
     */
          
    /* check consistency after optimizations */
    pips_assert("consistency checking after optimizations",
		statement_consistent_p(s));

    ifdebug(1) davinci_dump_expressions(module_name, "final", s);

    module_reorder(s);

    /* return result to pipsdbm
     */    
    DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name, s);

    reset_current_module_entity();
    reset_current_module_statement();
    reset_current_optimization_strategy();

    debug_off();

    return TRUE; /* okay ! */
}
