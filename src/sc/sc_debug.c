/* 
 * $Id$
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"

#include "sc-private.h"

/************************************************************* DEBUG */

#define SC_DEBUG_LEVEL "SC_DEBUG_LEVEL"
#define SC_PRINT_FLAG "SC_PRINT_FLAG"
#define TIMEOUT_FOR_SIMPLEX "TIMEOUT_FOR_SIMPLEX"
#define TIMEOUT_FOR_FM "TIMEOUT_FOR_FM"
#define TIMEOUT_FOR_SC_CONVEX_HULL "TIMEOUT_FOR_SC_CONVEX_HULL"

int sc_debug_level = 0;
int sc_print_flag = 0;
int timeout_for_S = 180;
int timeout_for_FM = 180;
int timeout_for_scCH = 180;

void set_sc_debug_level(int l)
{ 
    sc_debug_level = l ;
}

void set_sc_print_flag(int l)
{ 
    sc_print_flag = l ;
}
void set_timeout_for_S(int l)
{ 
    timeout_for_S = l ;
}

void set_timeout_for_FM(int l)
{ 
   timeout_for_FM = l ;
}
void set_timeout_for_sc_convex_hull(int l)
{ 
   timeout_for_scCH = l ;
}
/***************************************************** VARIABLE NAME STACK */

typedef char * (*var_name_t)(Variable);

static var_name_t * var_name_stack = NULL;
static int var_name_stack_index = 0; /* next available chunck */
static int var_name_stack_size = 0;

void sc_variable_name_push(char * (*fun)(Variable))
{
  if (var_name_stack_index==var_name_stack_size)
  {
    var_name_stack_size+=10;
    var_name_stack = (var_name_t*) 
      realloc(var_name_stack, sizeof(var_name_t)*var_name_stack_size);
  }
  var_name_stack[var_name_stack_index++] = fun;
}

static void sc_variable_name_init(void)
{
  assert(!var_name_stack);

  var_name_stack_index = 0;
  var_name_stack_size = 10;
  var_name_stack = (var_name_t*) 
    malloc(sizeof(var_name_t)*var_name_stack_size);
  sc_variable_name_push(variable_default_name);
}

void sc_variable_name_pop(void)
{
  assert(var_name_stack_index>0);
  var_name_stack_index--;
}

char * default_variable_to_string(Variable v)
{
  assert(var_name_stack_index>0);
  return (*(var_name_stack[var_name_stack_index-1]))(v);
}

/********************************************************** INITIALIZATION */

void initialize_sc(char *(*var_to_string)(Variable))
{
  char * l;
  char * f;

  /* sc print */
  f = getenv(SC_PRINT_FLAG);  
  if (f) set_sc_print_flag(atoi(f));

  /* sc debug */
  l = getenv(SC_DEBUG_LEVEL);
  if (l) set_sc_debug_level(atoi(l));

  /* timeout */
  l = getenv(TIMEOUT_FOR_SIMPLEX);
  if (l) set_timeout_for_S(atoi(l));

  l = getenv(TIMEOUT_FOR_FM);
  if (l) set_timeout_for_FM(atoi(l));

  l = getenv(TIMEOUT_FOR_SC_CONVEX_HULL);
  if (l) set_timeout_for_sc_convex_hull(atoi(l));

  /* variable name stuff */
  sc_variable_name_init();
  sc_variable_name_push(var_to_string);
  
  ifscdebug(1)
    fprintf(stderr, "[initialize_sc] Value: " LINEAR_VALUE_STRING "\n");  
}
