#include "genC.h"
#include "linear.h"
#include "ri.h"

#include "resources.h"

#include "misc.h"
#include "ri-util.h"
#include "pipsdbm.h"

#include "sac.h"
#include "patterns.tab.h"

typedef struct {
      int class;
      list args;        /* <int*> */
} _pattern, * pattern;

typedef struct {
      list patterns;    /* <pattern> */
      hash_table sons;  /* <int -> match_tree> */
} _match_tree, * match_tree;

#define PATTERN(x) ((pattern)((x).p))
#define TOKEN(x) ((int)((x).p))
#define ARGUMENT(x) ((int*)((x).p))

static bool patterns_initialized = FALSE;

static hash_table operation_ids;   /* <string->int> */
static operation operations = NULL;
static int nbOperations = 0;
static int nbAllocatedOperations = 0;

static match_tree patterns_tree = NULL;

static pattern make_pattern(int id, list args)
{
   pattern n = (pattern)malloc(sizeof(_pattern));

   n->class = id;
   n->args = args;

   return n;
}

static match make_match(int type, list args)
{
   match n = (match)malloc(sizeof(_match));

   n->type = type;
   n->args = args;

   return n;
}

static match_tree make_tree()
{
   match_tree n = (match_tree)malloc(sizeof(_match_tree));

   n->patterns = NIL;
   n->sons = hash_table_make(hash_int, 0);

   return n;
}

static void insert_tree_branch(match_tree t, int token, match_tree n)
{
   hash_put(t->sons, (void*)token, (void*)n);
}

static match_tree select_tree_branch(match_tree t, int token)
{
   match_tree res = (match_tree)hash_get(t->sons, (void*)token);
   
   return (res == HASH_UNDEFINED_VALUE) ? NULL : res;
}

/* Warning: list of arguments is built in reversed order
 * (the head is in fact the last argument) */
static match_tree match_call(call c, match_tree t, list *args)
{
   if (!top_level_entity_p(call_function(c)) || 
        call_constant_p(c))
      return NULL; /* no match */

   t = select_tree_branch(t, get_operator_id(call_function(c)));
   if (t == NULL)
      return NULL; /* no match */

   MAP(EXPRESSION,
       arg,
   {
      syntax s = expression_syntax(arg);

      switch(syntax_tag(s))
      {
	 case is_syntax_call:
	 {
	    call c = syntax_call(s);

	    if (call_constant_p(c))
	    {
	       t = select_tree_branch(t, CONSTANT_TOK);
	       *args = CONS(EXPRESSION, arg, *args);
	    }
	    else
	       t = match_call(c, t, args);
	    break;
	 }
	    
	 case is_syntax_reference:
	 {
	    t = select_tree_branch(t, REFERENCE_TOK);
	    *args = CONS(EXPRESSION, arg, *args);
	    break;
	 }

	 case is_syntax_range:
	 default:
	    return NULL; /* unexpected token !! -> no match */
      }

      if (t == NULL)
	 return NULL;
   },
       call_arguments(c));

   return t;
}

/* merge the 2 lists. Warning: list gets reversed. */
static list merge_lists(list l, list format)
{
   list res = NIL;

   /* merge according to the format specifier list */
   for( ; format != NIL; format = CDR(format))
   {
      int* param = ARGUMENT(CAR(format));

      if (param == NULL)
      {
	 if (l != NIL)
	 {
	    res = CONS(EXPRESSION, EXPRESSION(CAR(l)), res);
	    l = CDR(l);
	 }
	 else
	    printf("Trying to use a malformed rule. Ignoring missing parameter...\n");
      }
      else
      {
	 expression e = make_integer_constant_expression(*param);

	 res = CONS(EXPRESSION, e, res);
      }
   }

   /* append remaining arguments, if any */
   for( ; l != NIL; l = CDR(l) )
      res = CONS(EXPRESSION, EXPRESSION(CAR(l)), res);

   return res;
}

/* return a list of matching statements */
list match_statement(statement s)
{
   match_tree t;
   list args = NULL;
   list matches = NIL;
   list i;

   if (!statement_call_p(s))
      return NIL;

   /* find the matching patterns */
   t = match_call(statement_call(s), patterns_tree, &args);
   if (t == NULL)
   {
      gen_free_list(args);
      return NIL;
   }

   /* build the matches */
   for(i = t->patterns; i != NIL; i = CDR(i)) 
   {
      pattern p = PATTERN(CAR(i));
      match m = make_match(p->class, 
		           merge_lists(args, p->args));

      matches = CONS(MATCH, m, matches);
   }

   return matches;
}

void insert_operation(char * s, int nbArgs, list opcodes)
{
   int id;
   operation op;

   //Find an id for the operation
   id = nbOperations++;

   //Add the id and name in the map
   hash_put(operation_ids, (void *)s, (void *)id);

   //Make room for the new operation if needed
   if (nbOperations > nbAllocatedOperations)
   {
      nbAllocatedOperations += 10;

      operations = (operation)realloc((void*)operations, 
				      sizeof(_operation)*nbAllocatedOperations);

      if (operations == NULL)
      {
	 printf("Fatal error: could not allocate memory for operations.\n");
	 exit(-1);
      }
   }

   //Initialize members
   op = operations + id;
   op->nbArgs = nbArgs;
   op->opcodes = opcodes;
}

operation get_operation(int kind)
{
   return ((kind>=0) && (kind<nbOperations)) ?
      &operations[kind] : NULL;
}

char * get_operation_opcode(int kind, int vecSize, int subwordSize)
{
   operation op = get_operation(kind);
   list opcodes;

   if (!op)
      return NULL;

   for(opcodes = op->opcodes; opcodes != NIL; opcodes = CDR(opcodes))
   {
      opcode oc = OPCODE(CAR(opcodes));

      if ( (oc->vectorSize == vecSize) &&
	   (oc->subwordSize == subwordSize) )
	 return oc->name;
   }

   return NULL;
} 

int get_operation_id(char * s)
{
   int id = (int)hash_get(operation_ids, (void*)s);

   return (id == (int)HASH_UNDEFINED_VALUE) ? -1 : id;
}

void insert_pattern(char * s, list tokens, list args)
{
   int c = get_operation_id(s);
   pattern p = make_pattern(c, args);
   match_tree m = patterns_tree;

   if (c < 0)
   {
      printf("Warning: defining pattern for an undefined operation (%s).",s);
      free(p);
      return;
   }

   for( ; tokens != NIL; tokens = CDR(tokens) )
   {
      int token = TOKEN(CAR(tokens));
      match_tree next = select_tree_branch(m, token);

      if (next == NULL)
      {
	 /* no such branch -> create a new branch */
	 next = make_tree();
	 insert_tree_branch(m, token, next);
      }

      m = next;
   }

   m->patterns = CONS(PATTERN, p, m->patterns);
}

void patterns_yyparse();

void init_tree_patterns()
{
   extern FILE * patterns_yyin;

   if (!patterns_initialized)
   {
      patterns_initialized = TRUE;

      patterns_tree = make_tree();
      operation_ids = hash_table_make(hash_string, 0);

      patterns_yyin = fopen("patterns.def", "r");
      patterns_yyparse();
      fclose(patterns_yyin);
   }
}
