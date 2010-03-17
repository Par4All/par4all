#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "misc.h"
#include "text-util.h"
#include "newgen_set.h"
#include "points_to_private.h"
#include "properties.h"
#include "preprocessor.h"

#include "effects-generic.h"
#include "effects-simple.h"
#include "effects-convex.h"

/* define the anywhere area, the TOP-LRVRL*MEMORY should be removed
 * from ri-util.h*/
#define ANYWHERE "*ANYWHERE*"
#define NOWHERE "*NOWHERE*"
#define NULL_POINTER "*NULL_POINTER*"

/*
The ANYWHERE lattice is the following:
               *ANYWHERE*:*ANYWHERE*
                      |
                      |
                TOP-LEVEL:*MEMORY*

                /     |      \
               /			|				\
              /       |        \
					*HEAP*   *STACK*    *STATIC*...
					    \       |         /
               \      |        /
                \     |       /
							 *NOWHERE*NOWHERE*
								

it's used to modelize the anywhere abstract locations. At the moment
all the functions return an entity, but the API can be easily
translated at the reference or the effect level.

*/
/*return anywhere(the top of the lattice)*/
entity entity_all_locations()
{
  entity anywhere = entity_undefined;
	string any_name = strdup(concatenate(ANYWHERE,MODULE_SEP_STRING, ANYWHERE ,NULL));
  anywhere = find_or_create_entity(ANYWHERE);
  if(entity_undefined_p(anywhere)) {
    area a = make_area(0,NIL); /* Size and layout are unknown */
    type t = make_type_area(a);
    anywhere = make_entity(any_name,
			   t, make_storage_rom(), make_value_unknown());
  }
 
 
  return anywhere;
}

/* test if an entity is the top of the lattice*/
bool entity_all_locations_p(entity e)
{

  bool anywhere_p;
  anywhere_p = same_string_p(entity_name(e), ANYWHERE);

  return anywhere_p;
}

 
/* return the bottom of the anywhere lattice= anywhere*/
entity  entity_nowhere_locations()
{
  entity nowhere = entity_undefined;
	string any_name = strdup(concatenate(NOWHERE, MODULE_SEP_STRING,NOWHERE ,NULL));
  nowhere = find_or_create_entity(any_name);
  if(entity_undefined_p(nowhere)) {
    area a = make_area(0,NIL); /* Size and layout are unknown */
    type t = make_type_area(a);
    nowhere = make_entity(any_name,
			   t, make_storage_rom(), make_value_unknown());
  }
  
   return nowhere;
}

/* test if an entity is the bottom of the lattice*/
bool entity_nowhere_p(entity e)
{

  bool nowhere_p;
  nowhere_p = same_string_p(entity_name(e), NOWHERE);
	
  return nowhere_p;
}

/* return the NULL/UNDEFINED POINTER...*/
entity entity_null_locations()
{
  entity null_pointer = entity_undefined;
	string any_name = strdup(concatenate(ANYWHERE, MODULE_SEP_STRING, NULL_POINTER, NULL));
  null_pointer = find_or_create_entity(strdup(any_name));
  if(entity_undefined_p(null_pointer)) {
    area a = make_area(0,NIL); /* Size and layout are unknown */
    type t = make_type_area(a);
    null_pointer = make_entity(any_name,
			   t, make_storage_rom(), make_value_unknown());
  }
 
	
  return null_pointer;
}

/* test if an entity is the NULL POINTER*/
bool entity_null_locations_p(entity e)
{

  bool null_pointer_p;
  null_pointer_p = same_string_p(entity_name(e), NULL_POINTER);

  return null_pointer_p;
}


entity entity_all_module_locations()
{
  entity anywhere = entity_undefined;
 
  anywhere = find_or_create_entity(ALL_MEMORY_ENTITY_NAME);
  if(entity_undefined_p(anywhere)) {
    area a = make_area(0,NIL); /* Size and layout are unknown */
    type t = make_type_area(a);
    anywhere = make_entity(ALL_MEMORY_ENTITY_NAME,
			   t, make_storage_rom(), make_value_unknown());
  }
 
  return anywhere;
}

/* test if an entity is the all module locations*/
bool entity_all_module_locations_p(entity e)
{

  bool null_pointer_p;
  null_pointer_p = same_string_p(entity_name(e),ALL_MEMORY_ENTITY_NAME );

  return null_pointer_p;
}



entity entity_all_module_heap_locations()
{
  entity heap = entity_undefined;
	string any_name = strdup(concatenate(get_current_module_name(),MODULE_SEP_STRING,HEAP_AREA_LOCAL_NAME,ANYWHERE,NULL));
  heap = find_or_create_entity(any_name);
  if(entity_undefined_p(heap)) {
    area a = make_area(0,NIL); /* Size and layout are unknown */
    type t = make_type_area(a);
    heap = make_entity(strdup(any_name),
			   t, make_storage_rom(), make_value_unknown());
  }
    
  return heap;
}

/* test if an entity is the a heap area*/
bool entity_all_module_heap_locations_p(entity e)
{

  bool heap_p;
  string s = concatenate(HEAP_AREA_LOCAL_NAME,ANYWHERE, NULL);
  heap_p = same_string_p(entity_name(e),s);
  free(s);

  return heap_p;
}


entity entity_all_module_stack_locations()
{
  entity stack = entity_undefined;
	string any_name = strdup(concatenate(get_current_module_name(),MODULE_SEP_STRING,STACK_AREA_LOCAL_NAME, ANYWHERE,NULL));
  stack = find_or_create_entity(any_name);
  if(entity_undefined_p(stack)) {
    area a = make_area(0,NIL); /* Size and layout are unknown */
    type t = make_type_area(a);
    stack = make_entity(strdup(any_name),
			   t, make_storage_rom(), make_value_unknown());
  }
 
  return stack;
}

/* test if an entity is the a stack area*/
bool entity_all_module_stack_locations_p(entity e)
{
  bool stack_p;
  string s = concatenate(STACK_AREA_LOCAL_NAME,ANYWHERE);
  stack_p = same_string_p(entity_name(e), s);
  free (s);
  return stack_p;
}


entity entity_all_module_static_locations()
{
  entity static_ent = entity_undefined;
	string any_name = strdup(concatenate(get_current_module_name() ,MODULE_SEP_STRING,STATIC_AREA_LOCAL_NAME,ANYWHERE,NULL));
  static_ent = find_or_create_entity(any_name);
  if(entity_undefined_p(static_ent)) {
    area a = make_area(0,NIL); /* Size and layout are unknown */
    type t = make_type_area(a);
    static_ent = make_entity(strdup(any_name),
			   t, make_storage_rom(), make_value_unknown());
  }

  return static_ent;
}

/* test if an entity is the a static area*/
bool entity_all_module_static_locations_p(entity e)
{

  bool static_p;
	string s = concatenate(STATIC_AREA_LOCAL_NAME,ANYWHERE);
	static_p = same_string_p(entity_name(e), s);
	free (s);
  return static_p;
}


entity entity_all_module_dynamic_locations()
{
  entity dynamic = entity_undefined;
	string any_name =strdup( concatenate(get_current_module_name() ,MODULE_SEP_STRING, DYNAMIC_AREA_LOCAL_NAME,ANYWHERE ,NULL));
  dynamic = find_or_create_entity(any_name);
  if(entity_undefined_p(dynamic)) {
    area a = make_area(0,NIL); /* Size and layout are unknown */
    type t = make_type_area(a);
    dynamic = make_entity(any_name,
			   t, make_storage_rom(), make_value_unknown());
  }
  
  return dynamic;
}


/* test if an entity is the a dynamic area*/
bool entity_all_module_dynamic_locations_p(entity e)
{
  bool dynamic_p;
  string s = concatenate(DYNAMIC_AREA_LOCAL_NAME,ANYWHERE,NULL);
  dynamic_p = same_string_p(entity_name(e), s);
  free (s);
  return dynamic_p;
}


/*returns the smallest abstract location greater than al1 and al2(greater==includes)*/
entity entity_locations_max(entity al1, entity al2)
{
  entity e = entity_undefined;
	if(entity_all_module_dynamic_locations_p(al1)||entity_all_module_dynamic_locations_p(al2)||
		 entity_all_module_static_locations_p(al1)||entity_all_module_static_locations_p(al2)||
		 entity_all_module_stack_locations_p(al1)||entity_all_module_stack_locations_p(al2)||
		 entity_all_module_heap_locations_p(al1)||entity_all_module_heap_locations_p(al2))
		e = entity_all_module_locations();

	if(entity_all_module_locations_p(al1)||entity_all_module_locations_p(al2))
		e = entity_all_locations();

	

	if(entity_nowhere_p(al1) && (entity_all_module_dynamic_locations_p(al2) ||entity_all_module_static_locations_p(al2) ||
															 entity_all_module_stack_locations_p(al2)|| entity_all_module_heap_locations_p(al2)))
		 e = copy_entity(al2);

	if(entity_nowhere_p(al2) && (entity_all_module_dynamic_locations_p(al1) ||entity_all_module_static_locations_p(al1) ||
															 entity_all_module_stack_locations_p(al1)|| entity_all_module_heap_locations_p(al1)))
		 e = copy_entity(al1);

	if(entity_all_module_locations_p(al1) && (entity_all_module_dynamic_locations_p(al2) ||entity_all_module_static_locations_p(al2) ||
																						entity_all_module_stack_locations_p(al2)|| entity_all_module_heap_locations_p(al2)))
		e = entity_all_module_locations(al1);
	
	if(entity_all_module_locations_p(al2) && (entity_all_module_dynamic_locations_p(al1) ||entity_all_module_static_locations_p(al1) ||
																						entity_all_module_stack_locations_p(al1)|| entity_all_module_heap_locations_p(al1)))
		e = entity_all_module_locations(al2);

	if((entity_nowhere_p(al2)&&entity_nowhere_p(al1))||(entity_nowhere_p(al1)&&entity_nowhere_p(al2)))
		 e = entity_all_module_locations(al1);
  return e;
}

/* in case we need to evaluate sigma(al), return */
entity entity_locations_dereference(entity al)
{
  entity e = entity_undefined;
  e = entity_all_locations();
  return e;
}
