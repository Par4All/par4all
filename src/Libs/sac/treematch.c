#include "genC.h"
#include "linear.h"
#include "ri.h"

#include "resources.h"

#include "misc.h"
#include "ri-util.h"
#include "pipsdbm.h"

#include "sac.h"
#include "patterns.tab.h"


static bool patterns_initialized = FALSE;

static hash_table opcodeClass_ids;   /* <string->int> */
static opcodeClass* opcodeClasses = NULL;
static int nbOpcodeClasses = 0;
static int nbAllocatedOpcodeClasses = 0;

static matchTree patterns_tree = NULL;

static matchTree make_tree()
{
   matchTree n = make_matchTree(NIL, make_matchTreeSons());

   return n;
}

static void insert_tree_branch(matchTree t, int token, matchTree n)
{
   extend_matchTreeSons(matchTree_sons(t), token, n);
}

static matchTree select_tree_branch(matchTree t, int token)
{
   return apply_matchTreeSons(matchTree_sons(t), token);
}

/* Warning: list of arguments is built in reversed order
 * (the head is in fact the last argument) */
static matchTree match_call(call c, matchTree t, list *args)
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
      patternArg param = PATTERNARG(CAR(format));

      if (patternArg_dynamic_p(param))
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
	 expression e = make_integer_constant_expression(patternArg_static(param));

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
   matchTree t;
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
   for(i = matchTree_patterns(t); i != NIL; i = CDR(i)) 
   {
      patternx p = PATTERNX(CAR(i));
      match m = make_match(patternx_class(p), 
		           merge_lists(args, patternx_args(p)));

      matches = CONS(MATCH, m, matches);
   }

   return matches;
}

void insert_opcodeClass(char * s, int nbArgs, list opcodes)
{
   int id;

   //Find an id for the opcodeClass
   id = nbOpcodeClasses++;

   //Add the id and name in the map
   hash_put(opcodeClass_ids, (void *)s, (void *)id);

   //Make room for the new opcodeClass if needed
   if (nbOpcodeClasses > nbAllocatedOpcodeClasses)
   {
      nbAllocatedOpcodeClasses += 10;

      opcodeClasses = (opcodeClass*)realloc((void*)opcodeClasses, 
				      sizeof(opcodeClass)*nbAllocatedOpcodeClasses);

      if (opcodeClasses == NULL)
      {
	 printf("Fatal error: could not allocate memory for opcodeClasses.\n");
	 exit(-1);
      }
   }

   //Initialize members
   opcodeClasses[id] = make_opcodeClass(nbArgs, opcodes);
}

opcodeClass get_opcodeClass(int kind)
{
   return ((kind>=0) && (kind<nbOpcodeClasses)) ?
      opcodeClasses[kind] : NULL;
}

char * get_opcodeClass_opcode(int kind, int vecSize, int subwordSize)
{
   opcodeClass op = get_opcodeClass(kind);
   list opcodes;

   if (!op)
      return NULL;

   for(opcodes = opcodeClass_opcodes(op); opcodes != NIL; opcodes = CDR(opcodes))
   {
      opcode oc = OPCODE(CAR(opcodes));

      if ( (opcode_vectorSize(oc) == vecSize) &&
	   (opcode_subwordSize(oc) == subwordSize) )
	 return opcode_name(oc);
   }

   return NULL;
} 

int get_opcodeClass_id(char * s)
{
   int id = (int)hash_get(opcodeClass_ids, (void*)s);

   return (id == (int)HASH_UNDEFINED_VALUE) ? -1 : id;
}

void insert_pattern(char * s, list tokens, list args)
{
   int c = get_opcodeClass_id(s);
   patternx p;
   matchTree m = patterns_tree;

   if (c < 0)
   {
      printf("Warning: defining pattern for an undefined opcodeClass (%s).",s);
      return;
   }

   p = make_patternx(c, args);

   for( ; tokens != NIL; tokens = CDR(tokens) )
   {
      int token = INT(CAR(tokens));
      matchTree next = select_tree_branch(m, token);

      if (next == NULL)
      {
	 /* no such branch -> create a new branch */
	 next = make_tree();
	 insert_tree_branch(m, token, next);
      }

      m = next;
   }

   matchTree_patterns(m) = CONS(PATTERN, p, matchTree_patterns(m));
}

void patterns_yyparse();

void init_tree_patterns()
{
   extern FILE * patterns_yyin;

   if (!patterns_initialized)
   {
      patterns_initialized = TRUE;

      patterns_tree = make_tree();
      opcodeClass_ids = hash_table_make(hash_string, 0);

      patterns_yyin = fopen("patterns.def", "r");
      patterns_yyparse();
      fclose(patterns_yyin);
   }
}
