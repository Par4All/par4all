/*

  $Id$

  Copyright 1989-2014 MINES ParisTech

  This file is part of Linear/C3 Library.

  Linear/C3 Library is free software: you can redistribute it and/or modify it
  under the terms of the GNU Lesser General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  any later version.

  Linear/C3 Library is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.

  See the GNU Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public License
  along with Linear/C3 Library.  If not, see <http://www.gnu.org/licenses/>.

*/

#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"

#include "sc-private.h"

#ifdef FILTERING
#include "signal.h"
#endif

/************************************************************* DEBUG */
#define SC_DEBUG_LEVEL "SC_DEBUG_LEVEL"
#define SC_SWITCH_HEURISTIC_FLAG "SC_SWITCH_HEURISTIC_FLAG"

/************************************************************* CONTROL */
//Note: CONTROL and FILTER are different.

//OVERFLOW_CONTROL is implemented. 
//SIZE_CONTROL is to be implemented
//TIMEOUT_CONTROL is not completed yet.

#ifdef FILTERING

#define FLAG_CONTROL_DIMENSION_CONVEX_HULL "CONTROL_DIMENSION_CONVEX_HULL"
#define FLAG_CONTROL_NUMBER_CONSTRAINTS_CONVEX_HULL "CONTROL_NUMBER_CONSTRAINTS_CONVEX_HULL"
#define FLAG_CONTROL_DIMENSION_PROJECTION "CONTROL_DIMENSION_PROJECTION"
#define FLAG_CONTROL_NUMBER_CONSTRAINTS_PROJECTION "CONTROL_NUMBER_CONSTRAINTS_PROJECTION"


/************************************************************* FILTERING TIMEOUT*/
#define FLAG_FILTERING_TIMEOUT_JANUS "FILTERING_TIMEOUT_JANUS"
#define FLAG_FILTERING_TIMEOUT_LINEAR_SIMPLEX "FILTERING_TIMEOUT_LINEAR_SIMPLEX"
#define FLAG_FILTERING_TIMEOUT_FM "FILTERING_TIMEOUT_FM"
#define FLAG_FILTERING_TIMEOUT_CONVEX_HULL "FILTERING_TIMEOUT_CONVEX_HULL"
#define FLAG_FILTERING_TIMEOUT_PROJECTION "FILTERING_TIMEOUT_PROJECTION"

/************************************************************* FILTERING SIZE*/
#define FLAG_FILTERING_DIMENSION_FEASIBILITY "FILTERING_DIMENSION_FEASIBILITY"
#define FLAG_FILTERING_NUMBER_CONSTRAINTS_FEASIBILITY "FILTERING_NUMBER_CONSTRAINTS_FEASIBILITY"
#define FLAG_FILTERING_DENSITY_FEASIBILITY "FILTERING_DENSITY_FEASIBILITY"
#define FLAG_FILTERING_MAGNITUDE_FEASIBILITY "FILTERING_MAGNITUDE_FEASIBILITY"

#define FLAG_FILTERING_DIMENSION_PROJECTION "FILTERING_DIMENSION_PROJECTION"
#define FLAG_FILTERING_NUMBER_CONSTRAINTS_PROJECTION "FILTERING_NUMBER_CONSTRAINTS_PROJECTION"
#define FLAG_FILTERING_DENSITY_PROJECTION "FILTERING_DENSITY_PROJECTION"
#define FLAG_FILTERING_MAGNITUDE_PROJECTION "FILTERING_MAGNITUDE_PROJECTION"

#define FLAG_FILTERING_DIMENSION_CONVEX_HULL "FILTERING_DIMENSION_CONVEX_HULL"
#define FLAG_FILTERING_NUMBER_CONSTRAINTS_CONVEX_HULL "FILTERING_NUMBER_CONSTRAINTS_CONVEX_HULL"
#define FLAG_FILTERING_DENSITY_CONVEX_HULL "FILTERING_DENSITY_CONVEX_HULL"
#define FLAG_FILTERING_MAGNITUDE_CONVEX_HULL "FILTERING_MAGNITUDE_CONVEX_HULL"

#endif

int sc_debug_level = 0;
int sc_switch_heuristic_flag = 0;

#ifdef FILTERING

int filtering_timeout_J = 0;
int filtering_timeout_S = 0;
int filtering_timeout_FM = 0;
int filtering_timeout_CH = 0;
int filtering_timeout_projection = 0;

int filtering_dimension_feasibility = 0;
int filtering_number_constraints_feasibility = 0;
int filtering_density_feasibility = 0;
long int filtering_magnitude_feasibility = 0;// need to cast to Value

int filtering_dimension_projection = 0;
int filtering_number_constraints_projection = 0;
int filtering_density_projection = 0;
long int filtering_magnitude_projection = 0;// need to cast to Value

int filtering_dimension_convex_hull = 0;
int filtering_number_constraints_convex_hull = 0;
int filtering_density_convex_hull = 0;
long int filtering_magnitude_convex_hull = 0;// need to cast to Value

#endif

/* SET FUNCTIONS*/
/* Let's change variables directly here, except sc_debug_level?*/
/* or use functions returning values, within private variables? DN*/

void set_sc_debug_level(int l)
{ 
    sc_debug_level = l ;
}

static void set_sc_switch_heuristic_flag(int l)
{ 
    sc_switch_heuristic_flag = l ;
}


#ifdef FILTERING

/*TIMEOUT*/
static void set_filtering_timeout_J(int l)
{
  filtering_timeout_J = l;
}
static void set_filtering_timeout_S(int l)
{ 
    filtering_timeout_S = l ;
}
static void set_filtering_timeout_FM(int l)
{ 
   filtering_timeout_FM = l ;
}
static void set_filtering_timeout_convex_hull(int l)
{ 
   filtering_timeout_CH = l ;
}
static void set_filtering_timeout_projection(int l)
{ 
   filtering_timeout_projection = l ;
}

/*SIZE - dimension, number of constraints, density and magnitude*/

static void set_filtering_dimension_feasibility(int l)
{
  filtering_dimension_feasibility = l ;
}
static void set_filtering_number_constraints_feasibility(int l)
{
  filtering_number_constraints_feasibility = l ;
}
static void set_filtering_density_feasibility(int l)
{
  filtering_density_feasibility = l ;
}
static void set_filtering_magnitude_feasibility(long int l)
{
  filtering_magnitude_feasibility = l ;
}

static void set_filtering_dimension_projection(int l)
{
  filtering_dimension_projection = l ;
}
static void set_filtering_number_constraints_projection(int l)
{
  filtering_number_constraints_projection = l ;
}
static void set_filtering_density_projection(int l)
{
  filtering_density_projection = l ;
}
static void set_filtering_magnitude_projection(long int l)
{
  filtering_magnitude_projection = l ;
}

static void set_filtering_dimension_convex_hull(int l)
{
  filtering_dimension_convex_hull = l ;
}
static void set_filtering_number_constraints_convex_hull(int l)
{
  filtering_number_constraints_convex_hull = l ;
}
static void set_filtering_density_convex_hull(int l)
{
  filtering_density_convex_hull = l ;
}
static void set_filtering_magnitude_convex_hull(long int l)
{
  filtering_magnitude_convex_hull = l ;
}
#endif

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
  char * tmp;

  /* sc debug */
  tmp = getenv(SC_DEBUG_LEVEL);
  if (tmp) set_sc_debug_level(atoi(tmp));

  /* sc switch heuristic */
  tmp = getenv(SC_SWITCH_HEURISTIC_FLAG);
  if (tmp) set_sc_switch_heuristic_flag(atoi(tmp));

#ifdef  FILTERING
  
  /* timeout filtering*/
  tmp = getenv(FLAG_FILTERING_TIMEOUT_JANUS);
  if (tmp) set_filtering_timeout_J(atoi(tmp));

  tmp = getenv(FLAG_FILTERING_TIMEOUT_LINEAR_SIMPLEX);
  if (tmp) set_filtering_timeout_S(atoi(tmp));

  tmp = getenv(FLAG_FILTERING_TIMEOUT_FM);
  if (tmp) set_filtering_timeout_FM(atoi(tmp));

  tmp = getenv(FLAG_FILTERING_TIMEOUT_CONVEX_HULL);
  if (tmp) set_filtering_timeout_convex_hull(atoi(tmp));

  tmp = getenv(FLAG_FILTERING_TIMEOUT_PROJECTION);
  if (tmp) set_filtering_timeout_projection(atoi(tmp));

  /* size filtering*/
  tmp = getenv(FLAG_FILTERING_DIMENSION_FEASIBILITY);
  if (tmp) set_filtering_dimension_feasibility(atoi(tmp));

  tmp = getenv(FLAG_FILTERING_NUMBER_CONSTRAINTS_FEASIBILITY);
  if (tmp) set_filtering_number_constraints_feasibility(atoi(tmp));

  tmp = getenv(FLAG_FILTERING_DENSITY_FEASIBILITY);
  if (tmp) set_filtering_density_feasibility(atoi(tmp));

  tmp = getenv(FLAG_FILTERING_MAGNITUDE_FEASIBILITY);
  if (tmp) set_filtering_magnitude_feasibility(atol(tmp));

  tmp = getenv(FLAG_FILTERING_DIMENSION_PROJECTION);
  if (tmp) set_filtering_dimension_projection(atoi(tmp));

  tmp = getenv(FLAG_FILTERING_NUMBER_CONSTRAINTS_PROJECTION);
  if (tmp) set_filtering_number_constraints_projection(atoi(tmp));

  tmp = getenv(FLAG_FILTERING_DENSITY_PROJECTION);
  if (tmp) set_filtering_density_projection(atoi(tmp));

  tmp = getenv(FLAG_FILTERING_MAGNITUDE_PROJECTION);
  if (tmp) set_filtering_magnitude_projection(atol(tmp));

  tmp = getenv(FLAG_FILTERING_DIMENSION_CONVEX_HULL);
  if (tmp) set_filtering_dimension_convex_hull(atoi(tmp));

  tmp = getenv(FLAG_FILTERING_NUMBER_CONSTRAINTS_CONVEX_HULL);
  if (tmp) set_filtering_number_constraints_convex_hull(atoi(tmp));

  tmp = getenv(FLAG_FILTERING_DENSITY_CONVEX_HULL);
  if (tmp) set_filtering_density_convex_hull(atoi(tmp));

  tmp = getenv(FLAG_FILTERING_MAGNITUDE_CONVEX_HULL);
  if (tmp) set_filtering_magnitude_convex_hull(atol(tmp));

#endif
 
  /* variable name stuff */
  sc_variable_name_init();
  sc_variable_name_push(var_to_string);
  
  ifscdebug(1)
    fprintf(stderr, "[initialize_sc] Value: " LINEAR_VALUE_STRING "\n");  
}
