#include "genC.h"
#include "linear.h"
#include "ri.h"

#include "resources.h"

#include "misc.h"
#include "ri-util.h"

#include "sac.h"
#include "patterns.tab.h"

typedef struct {
      char* name;
      int id;
} oper_id_mapping;

static oper_id_mapping operators[] =
{
   { ASSIGN_OPERATOR_NAME,            ASSIGN_OPERATOR_TOK },
   { PLUS_OPERATOR_NAME,              PLUS_OPERATOR_TOK },
   { MINUS_OPERATOR_NAME,             MINUS_OPERATOR_TOK },
   { UNARY_MINUS_OPERATOR_NAME,       UNARY_MINUS_OPERATOR_TOK },
   { MULTIPLY_OPERATOR_NAME,          MULTIPLY_OPERATOR_TOK },
   { DIVIDE_OPERATOR_NAME,            DIVIDE_OPERATOR_TOK },
   { INVERSE_OPERATOR_NAME,           INVERSE_OPERATOR_TOK },
   { POWER_OPERATOR_NAME,             POWER_OPERATOR_TOK },
   { MODULO_OPERATOR_NAME,            MODULO_OPERATOR_TOK },
   { MIN_OPERATOR_NAME,               MIN_OPERATOR_TOK },
   { MIN0_OPERATOR_NAME,              MIN0_OPERATOR_TOK },
   { AMIN1_OPERATOR_NAME,             AMIN1_OPERATOR_TOK },
   { DMIN1_OPERATOR_NAME,             DMIN1_OPERATOR_TOK },
   { MAX_OPERATOR_NAME,               MAX_OPERATOR_TOK },
   { MAX0_OPERATOR_NAME,              MAX0_OPERATOR_TOK },
   { AMAX1_OPERATOR_NAME,             AMAX1_OPERATOR_TOK },
   { DMAX1_OPERATOR_NAME,             DMAX1_OPERATOR_TOK },
   { ABS_OPERATOR_NAME,               ABS_OPERATOR_TOK },
   { IABS_OPERATOR_NAME,              IABS_OPERATOR_TOK },
   { DABS_OPERATOR_NAME,              DABS_OPERATOR_TOK },
   { CABS_OPERATOR_NAME,              CABS_OPERATOR_TOK },

   { AND_OPERATOR_NAME,               AND_OPERATOR_TOK },
   { OR_OPERATOR_NAME,                OR_OPERATOR_TOK },
   { NOT_OPERATOR_NAME,               NOT_OPERATOR_TOK },
   { NON_EQUAL_OPERATOR_NAME,         NON_EQUAL_OPERATOR_TOK },
   { EQUIV_OPERATOR_NAME,             EQUIV_OPERATOR_TOK },
   { NON_EQUIV_OPERATOR_NAME,         NON_EQUIV_OPERATOR_TOK },

   { TRUE_OPERATOR_NAME,              TRUE_OPERATOR_TOK },
   { FALSE_OPERATOR_NAME,             FALSE_OPERATOR_TOK },

   { GREATER_OR_EQUAL_OPERATOR_NAME,  GREATER_OR_EQUAL_OPERATOR_TOK },
   { GREATER_THAN_OPERATOR_NAME,      GREATER_THAN_OPERATOR_TOK },
   { LESS_OR_EQUAL_OPERATOR_NAME,     LESS_OR_EQUAL_OPERATOR_TOK },
   { LESS_THAN_OPERATOR_NAME,         LESS_THAN_OPERATOR_TOK },
   { EQUAL_OPERATOR_NAME,             EQUAL_OPERATOR_TOK },

   { NULL }
};

typedef struct {
      int id;
      hash_table sons; /* char -> operator_id_tree*/
} operator_id_tree;

static operator_id_tree * mappings = NULL;

static operator_id_tree* make_operator_id_tree()
{
   operator_id_tree* n;

   n = (operator_id_tree*)malloc(sizeof(operator_id_tree));
   n->id = UNKNOWN_TOK;
   n->sons = hash_table_make(hash_int, 0);

   return n;
}

static void insert_mapping(oper_id_mapping* item)
{
   char * s;
   operator_id_tree* t;

   t = mappings;
   for(s = item->name; *s != 0; s++)
   {
      operator_id_tree * next;
      char c = *s;

      next = (operator_id_tree *)hash_get(t->sons, (void*)c);
      if (next == HASH_UNDEFINED_VALUE)
      {
	 next = make_operator_id_tree();
	 hash_put(t->sons, (void *)(c), (void*)next);
      }

      t = next;
   }
   
   if (t->id != UNKNOWN_TOK)
      printf("WARNING: overwriting previous mapping...\n");

   t->id = item->id;
}

void init_operator_id_mappings()
{
   int i;

   if (mappings != NULL)
      return;

   mappings = make_operator_id_tree();
   for(i=0; operators[i].name != NULL; i++)
      insert_mapping(&operators[i]);
}

int get_operator_id(entity e)
{
   char * s;
   operator_id_tree* t = mappings;

   for(s = entity_local_name(e); *s != 0; s++)
   {
      operator_id_tree * next;
      char c = *s;

      next = (operator_id_tree *)hash_get(t->sons, (void*)c);
      if (next == HASH_UNDEFINED_VALUE)
	 return UNKNOWN_TOK;

      t = next;
   }

   return t->id;
}
