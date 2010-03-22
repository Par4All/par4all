#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "misc.h"
//#include "text-util.h"
//#include "newgen_set.h"
//#include "points_to_private.h"
#include "properties.h"
//#include "preprocessor.h"

#include "alias-classes.h"

//#include "effects-generic.h"
//#include "effects-simple.h"
//#include "effects-convex.h"

/*
The ANYWHERE lattice is shown in the figure:

                            *ANYWHERE*:*ANYWHERE*
                                      |
                      --------------------------------------------------
                      |                                 |
                module:*ANYWHERE*                module2:*ANYWHERE*
                /     |      \
               /      |       \
              /       |        \
  module:*HEAP* module:*STACK*  module:*STATIC*...
              \       |         /
               \      |        /
                \     |       /
                 \    |      /
                  --- --------------------------------------------------
                                      |
		             *NOWHERE*:*NOWHERE*

It is used to modelize the anywhere abstract locations, but some
lements are missing such as *ANYWHERE*:*HEAP*, *ANYWHERE*:*STACK*, etc.

All the generating functions for this lattice return an
entity, but the API can be easily translated at the reference or the
effect level.

To merge abstract locations linked to two different modules, the
module ANY_MODULE_NAME (*ANYWHERE*) is used. For instance, FOO:*HEAP*
and BAR:*HEAP* are included into *ANYWHERE*:*HEAP*, their smallest
upper bound.

To merge different areas for the same module, such as FOO:*HEAP* and
FOO:*STACK*, the abstract location *ANYWHERE* is used.

If the modules are different and the areas are different, then the
resulting abstract location is *ANYWHERE*:*ANYWHERE*.

*/

/*return *ANY_MODULE*:*ANYWHERE* (the top of the lattice)*/
entity entity_all_locations()
{
  entity anywhere = entity_undefined;
  string any_name = strdup(concatenate(ANY_MODULE_NAME,
				       MODULE_SEP_STRING,
				       ANYWHERE_LOCATION,
				       NULL));
  anywhere = find_or_create_entity(any_name);
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
  anywhere_p = same_string_p(entity_local_name(e), ANYWHERE_LOCATION);
  anywhere_p = anywhere_p
    && same_string_p(entity_module_name(e), ANY_MODULE_NAME);

  return anywhere_p;
}

/* return the bottom of the anywhere lattice = nowhere, i.e. the
   unitialized pointer. */
entity entity_nowhere_locations()
{
  entity nowhere = entity_undefined;
  string any_name = strdup(concatenate(ANY_MODULE_NAME,
				       MODULE_SEP_STRING,
				       NOWHERE_LOCATION,
				       NULL));
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
bool entity_nowhere_locations_p(entity e)
{

  bool nowhere_p;
  nowhere_p = same_string_p(entity_local_name(e), NOWHERE_LOCATION);
  if(nowhere_p)
    pips_assert("defined in any module name",
		same_string_p(entity_module_name(e), ANY_MODULE_NAME));
  return nowhere_p;
}

/* return the NULL/UNDEFINED POINTER...*/
entity entity_null_locations()
{
  entity null_pointer = entity_undefined;
  string any_name = strdup(concatenate(ANY_MODULE_NAME,
				       MODULE_SEP_STRING,
				       NULL_POINTER_NAME,
				       NULL));
  null_pointer = find_or_create_entity(any_name);
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
  null_pointer_p = same_string_p(entity_local_name(e), NULL_POINTER_NAME);
  if(null_pointer_p)
    pips_assert("defined in any module name",
		same_string_p(entity_module_name(e), ANY_MODULE_NAME));

  return null_pointer_p;
}

/* Set of all memory locations related to one module.

   FI: This may prove useless unless the compilation unit is taken into
   account.
 */
entity entity_all_module_locations(entity m)
{
  entity anywhere = entity_undefined;
  string mn = entity_local_name(m);
  string any_name = strdup(concatenate(mn,
				       MODULE_SEP_STRING,
				       ANYWHERE_LOCATION,
				       NULL));

  anywhere = find_or_create_entity(any_name);
  if(entity_undefined_p(anywhere)) {
    area a = make_area(0,NIL); /* Size and layout are unknown */
    type t = make_type_area(a);
    anywhere = make_entity(ALL_MEMORY_ENTITY_NAME,
			   t, make_storage_rom(), make_value_unknown());
  }

  return anywhere;
}

/* test if an entity is the set of locations defined in a module */
bool entity_all_module_locations_p(entity e)
{

  bool null_pointer_p;
  null_pointer_p = same_string_p(entity_local_name(e), ANYWHERE_LOCATION);

  return null_pointer_p;
}

/* Generic set of functions for all kinds of areas */

entity entity_all_module_xxx_locations(entity m, string xxx)
{
  entity dynamic = entity_undefined;
  string any_name = strdup(concatenate(entity_local_name(m),
				       MODULE_SEP_STRING,
				       xxx,
				       ANYWHERE_LOCATION,
				       NULL));
  dynamic = find_or_create_entity(any_name);
  if(storage_undefined_p(entity_storage(dynamic))) {
    area a = make_area(0,NIL); /* Size and layout are unknown */
    type t = make_type_area(a);
    /*FI: more work to be done here... */
    entity_type(dynamic) = t;
    entity_storage(dynamic) = make_storage_rom();
    entity_initial(dynamic) = make_value_unknown();
  }

  return dynamic;
}


/* test if an entity is the set of all memory locations in the xxx
   area of a module. The module is not checked, so it can be the set
   of all modules... */
bool entity_all_module_xxx_locations_p(entity e, string xxx)
{
  bool dynamic_p;
  string s = concatenate(xxx, ANYWHERE_LOCATION, NULL);

  pips_assert("e is defined", !entity_undefined_p(e));

  dynamic_p = same_string_p(entity_local_name(e), s);

  return dynamic_p;
}

entity entity_all_xxx_locations(string xxx)
{
  entity dynamic = entity_undefined;
  string any_name = strdup(concatenate(ANY_MODULE_NAME,
				       MODULE_SEP_STRING,
				       xxx,
				       ANYWHERE_LOCATION,
				       NULL));
  dynamic = find_or_create_entity(any_name);
  if(storage_undefined_p(entity_storage(dynamic))) {
    area a = make_area(0,NIL); /* Size and layout are unknown */
    type t = make_type_area(a);
    /*FI: more work to be done here... */
    entity_type(dynamic) = t;
    entity_storage(dynamic) = make_storage_rom();
    entity_initial(dynamic) = make_value_unknown();
  }

  return dynamic;
}

/* test if an entity is the set of all memory locations in the xxx
   area of any module. */
bool entity_all_xxx_locations_p(entity e, string xxx)
{
  bool dynamic_p;
  string s = concatenate(xxx, ANYWHERE_LOCATION, NULL);

  dynamic_p = same_string_p(entity_local_name(e), s);
  dynamic_p = dynamic_p
    && same_string_p(entity_module_name(e), ANY_MODULE_NAME);

  return dynamic_p;
}


entity entity_all_module_heap_locations(entity m)
{
  return entity_all_module_xxx_locations(m, HEAP_AREA_LOCAL_NAME);
}

/* test if an entity is the a heap area*/
bool entity_all_module_heap_locations_p(entity e)
{
  return entity_all_module_xxx_locations_p(e, HEAP_AREA_LOCAL_NAME);
}

entity entity_all_heap_locations()
{
  return entity_all_xxx_locations(HEAP_AREA_LOCAL_NAME);
}

/* test if an entity is the set of all heap locations */
bool entity_all_heap_locations_p(entity e)
{
  return entity_all_xxx_locations_p(e, HEAP_AREA_LOCAL_NAME);
}


entity entity_all_module_stack_locations(entity m)
{
  return entity_all_module_xxx_locations(m, STACK_AREA_LOCAL_NAME);
}

/* test if an entity is the a stack area*/
bool entity_all_module_stack_locations_p(entity e)
{
  return entity_all_module_xxx_locations_p(e, STACK_AREA_LOCAL_NAME);
}

entity entity_all_stack_locations()
{
  return entity_all_xxx_locations(STACK_AREA_LOCAL_NAME);
}

/* test if an entity is the set of all stack locations */
bool entity_all_stack_locations_p(entity e)
{
  return entity_all_xxx_locations_p(e, STACK_AREA_LOCAL_NAME);
}


entity entity_all_module_static_locations(entity m)
{
  return entity_all_module_xxx_locations(m, STATIC_AREA_LOCAL_NAME);
}

/* test if an entity is the a static area*/
bool entity_all_module_static_locations_p(entity e)
{
  return entity_all_module_xxx_locations_p(e, STATIC_AREA_LOCAL_NAME);
}

entity entity_all_static_locations()
{
  return entity_all_xxx_locations(STATIC_AREA_LOCAL_NAME);
}

/* test if an entity is the set of all static locations*/
bool entity_all_static_locations_p(entity e)
{
  return entity_all_xxx_locations_p(e, STATIC_AREA_LOCAL_NAME);
}


entity entity_all_module_dynamic_locations(entity m)
{
  return entity_all_module_xxx_locations(m, DYNAMIC_AREA_LOCAL_NAME);
}

/* test if an entity is the a dynamic area*/
bool entity_all_module_dynamic_locations_p(entity e)
{
  return entity_all_module_xxx_locations_p(e, DYNAMIC_AREA_LOCAL_NAME);
}

entity entity_all_dynamic_locations()
{
  return entity_all_xxx_locations(DYNAMIC_AREA_LOCAL_NAME);
}

/* test if an entity is the set of all dynamic locations */
bool entity_all_dynamic_locations_p(entity e)
{
  return entity_all_xxx_locations_p(e, DYNAMIC_AREA_LOCAL_NAME);
}


bool entity_abstract_locations_p(entity al)
{
  bool abstract_p = FALSE;
  string mn = entity_module_name(al);

  if(strcmp(mn, ANY_MODULE_NAME)==0) {
    /* FI: this may change in the future and may not be a strong
       enough condition */
    abstract_p = TRUE;
  }
  else {
    string ln = entity_local_name(al);
    string found = strstr(ln, ANYWHERE_LOCATION);
    abstract_p = (found!=NULL);
  }

  return abstract_p;
}

/* returns the smallest abstract locations containing the location of
   variable v*/
entity variable_to_abstract_location(entity v)
{
  entity al = entity_undefined;
  if(entity_variable_p(v)) {
    storage s = entity_storage(v);
    ram r = storage_ram(s);
    entity f = ram_function(r);
    entity a = ram_section(r);
    string mn = entity_local_name(f);
    string ln = string_undefined;

    if(static_area_p(a))
      ln = STATIC_AREA_LOCAL_NAME;
    else if(dynamic_area_p(a))
      ln = DYNAMIC_AREA_LOCAL_NAME;
    else if(stack_area_p(a))
      ln = STACK_AREA_LOCAL_NAME;
    else if(heap_area_p(a))
      ln = HEAP_AREA_LOCAL_NAME;
    else
      pips_internal_error("unexpected area\n");
    al = FindOrCreateEntity(mn, ln);
  }
  else
    pips_internal_error("arg. not in definition domain\n");
  return al;
}


/*returns the smallest abstract location set greater than or equalt to
  al1 and al2.

  If al1 or al2 is nowhere, then return al2 or al1.

  If al1 and al2 are related to the same module, the module can be
  preserved. Else the anywhere module must be used.

  If al1 and al2 are related to the same area, then the area is
  preserved. Else, the *anywhere* area is used.

  FI: we are in trouble with the NULL pointer...
*/
/* here al1 and al2 must be abstract locations */
entity abstract_locations_max(entity al1, entity al2)
{
  entity e = entity_undefined;
  string ln1 = entity_local_name(al1);
  string ln2 = entity_local_name(al2);
  string mn1 = entity_module_name(al1);
  string mn2 = entity_module_name(al2);
  string ln;
  string mn;

  if(strcmp(ln1, ln2)==0)
    ln = ln1;
  else
    ln = ANYWHERE_LOCATION;

  if(strcmp(mn1, mn2)==0)
    mn = mn1;
  else
    mn = ANY_MODULE_NAME;
  e = FindOrCreateEntity(mn, ln);
  return e;
}

/* Here, entity al1 and entity al2 can be program variables */
entity entity_locations_max(entity al1, entity al2)
{
  entity e = entity_undefined;
  string mn1 = entity_module_name(al1);
  string mn2 = entity_module_name(al2);
  //string ln1 = entity_local_name(al1);
  //string ln2 = entity_local_name(al2);

  if(al1==al2) {
    e = al1;
  }
  else {
    string mn = string_undefined;

  /* Can we preserve information about the module? */
  if(strcmp(mn1, mn2)==0)
    mn = mn1;
  else
    mn = ANY_MODULE_NAME;

  if(entity_abstract_locations_p(al1))
    if(entity_abstract_locations_p(al2)) {
      /* Both al1 and al2 are abstract locations and they are
	 different */
      e = abstract_locations_max(al1, al2);
    }
    else {
      entity al = variable_to_abstract_location(al2);
      e = abstract_locations_max(al1, al);
    }
  else
    if(entity_abstract_locations_p(al2)) {
      entity al = variable_to_abstract_location(al1);
      e = abstract_locations_max(al, al2);
    }
    else {
      /* al1 and al2 are assumed to be variables */
      storage s1 = entity_storage(al1);
      storage s2 = entity_storage(al2);

      if(storage_ram_p(s1) && storage_ram_p(s2)) {
	ram r1 = storage_ram(s1);
	ram r2 = storage_ram(s2);
	entity f1 = ram_function(r1);
	entity f2 = ram_function(r2);
	entity a1 = ram_section(r1);
	entity a2 = ram_section(r2);
	string mn = string_undefined;
	string ln = string_undefined;

	if(f1==f2)
	  mn = entity_local_name(f1);
	else
	  mn = ANY_MODULE_NAME;

	if(static_area_p(a1) && static_area_p(a2))
	  ln = STATIC_AREA_LOCAL_NAME;
	else if(dynamic_area_p(a1) && dynamic_area_p(a2))
	  ln = DYNAMIC_AREA_LOCAL_NAME;
	else if(stack_area_p(a1) && stack_area_p(a2))
	  ln = STACK_AREA_LOCAL_NAME;
	else if(heap_area_p(a1) && heap_area_p(a2))
	  ln = HEAP_AREA_LOCAL_NAME;
	else
	  ln = ANYWHERE_LOCATION;
	e = FindOrCreateEntity(mn, ln);
      }
      else
	pips_internal_error("not implemented\n");
    }
  }
  return e;
}

/* in case we need to evaluate sigma(al), i.e. the locations pointed
   by al, return the top of the lattice. Of course, this function
   should be avoided as much as possible. */
entity entity_locations_dereference(entity al __attribute__ ((__unused__)))
{
  entity e = entity_undefined;
  e = entity_all_locations();
  return e;
}

/* For debugging the API */
 void check_abstract_locations()
 {
   entity al = entity_undefined;
   /* top */
   al = entity_all_locations();
   fprintf(stderr, "top: %s is %s\n", entity_name(al),
	   entity_all_locations_p(al)?
	   "the set of all application locations" : "bug!!!");

   /* bottom */
   al = entity_nowhere_locations();
   fprintf(stderr, "bottom: %s is %s\n", entity_name(al),
	   entity_nowhere_locations_p(al)?
	   "the bottom of the abstract location lattice" : "bug!!!");

   /* null pointer */
   al = entity_null_locations();
   fprintf(stderr, "null: %s is %s\n", entity_name(al),
	   entity_null_locations_p(al)?
	   "the null pointer" : "bug!!!");

   /* all locations for a given module */
   al = entity_all_module_locations(get_current_module_entity());
   fprintf(stderr, "all module locations: %s is %s\n", entity_name(al),
	   entity_all_module_locations_p(al)?
	   "the set of all locations of a module" : "bug!!!");

   /* all heap locations for a given module */
   al = entity_all_module_heap_locations(get_current_module_entity());
   fprintf(stderr, "all module heap locations: %s is %s\n", entity_name(al),
	   entity_all_module_heap_locations_p(al)?
	   "the set of all heap locations of a module" : "bug!!!");

   /* all stack locations for a given module */
   al = entity_all_module_stack_locations(get_current_module_entity());
   fprintf(stderr, "all module stack locations: %s is %s\n", entity_name(al),
	   entity_all_module_stack_locations_p(al)?
	   "the set of all stack locations of a module" : "bug!!!");

   /* all static locations for a given module */
   al = entity_all_module_static_locations(get_current_module_entity());
   fprintf(stderr, "all module static locations: %s is %s\n", entity_name(al),
	   entity_all_module_static_locations_p(al)?
	   "the set of all static locations of a module" : "bug!!!");

   /* all dynamic locations for a given module */
   al = entity_all_module_dynamic_locations(get_current_module_entity());
   fprintf(stderr, "all module dynamic locations: %s is %s\n", entity_name(al),
	   entity_all_module_dynamic_locations_p(al)?
	   "the set of all dynamic locations of a module" : "bug!!!");


   /* all heap locations for an application */
   al = entity_all_heap_locations();
   fprintf(stderr, "all application heap locations: %s is %s\n", entity_name(al),
	   entity_all_heap_locations_p(al)?
	   "the set of all heap locations of an application" : "bug!!!");

   /* all stack locations for an application */
   al = entity_all_stack_locations();
   fprintf(stderr, "all application stack locations: %s is %s\n", entity_name(al),
	   entity_all_stack_locations_p(al)?
	   "the set of all stack locations of an application" : "bug!!!");

   /* all static locations for an application */
   al = entity_all_static_locations();
   fprintf(stderr, "all application static locations: %s is %s\n", entity_name(al),
	   entity_all_static_locations_p(al)?
	   "the set of all static locations of an applciation" : "bug!!!");

   /* all dynamic locations for an application */
   al = entity_all_dynamic_locations();
   fprintf(stderr, "all module dynamic locations: %s is %s\n", entity_name(al),
	   entity_all_dynamic_locations_p(al)?
	   "the set of all dynamic locations of an application" : "bug!!!");

   /* Should/could be extended to check the max computation... */
 }
