#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "ri-util.h"
#include "effects-util.h"
#include "misc.h"
#include "properties.h"

/*
The ANYWHERE lattice is shown in the figure:

                            *ANY_MODULE*:*ANYWHERE*
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
		             *ANY_MODULE*:*NOWHERE*

It is used to modelize the anywhere abstract locations, but some
elements are missing such as *ANYWHERE*:*HEAP*, *ANYWHERE*:*STACK*,
etc.

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

// FI: redundance between all_locations and anywhere

/*return *ANY_MODULE*:*ANYWHERE* (the top of the lattice)
 *
 * FI->aM: it was first decided to make this entity an area, but areas
 * cannot be typed. So the internal representation must be changed to
 * carry a type usable according to ALIAS_ACROSS_TYPES. The top of the
 * location lattice should use overloaded as type, that is the top of
 * the type lattice.
 */
entity entity_all_locations()
{
  entity anywhere = entity_undefined;
  static const char any_name[] =
    ANY_MODULE_NAME MODULE_SEP_STRING  ANYWHERE_LOCATION;
  anywhere = gen_find_tabulated(any_name, entity_domain);
  if(entity_undefined_p(anywhere)) {
    area a = make_area(0,NIL); /* Size and layout are unknown */
    type t = make_type_area(a);
    anywhere = make_entity(strdup(any_name),
			   t,
			   make_storage_rom(),
			   make_value_unknown());
    entity_kind(anywhere)=ABSTRACT_LOCATION;
  }

  return anywhere;
}

entity entity_anywhere_locations()
{
  return entity_all_locations();
}

/* test if an entity is the top of the lattice 
 *
 * This test does not take typed anywhere into account. This is
 * consistent with the function name, but may be not with its uses.
 *
 * It is not consistent with hiding the impact of
 * ALIASING_ACROSS_TYPES from callers.
 */
bool entity_all_locations_p(entity e)
{

  bool anywhere_p;
  anywhere_p = same_string_p(entity_local_name(e), ANYWHERE_LOCATION);
  anywhere_p = anywhere_p
    && same_string_p(entity_module_name(e), ANY_MODULE_NAME);

  return anywhere_p;
}

entity entity_typed_anywhere_locations(type t)
{
  return entity_all_xxx_locations_typed(ANYWHERE_LOCATION, t);
}

/* test if an entity is the bottom of the lattice*/
bool entity_anywhere_locations_p(entity e)
{
    return same_entity_p(e, entity_all_locations());
}

/* test if a cell is the bottom of the lattice*/
bool cell_typed_anywhere_locations_p(cell c)
{
  reference r = cell_any_reference(c);
  bool anywhere_p = reference_typed_anywhere_locations_p(r);
  return anywhere_p;
}

/* test if a reference is the bottom of the lattice*/
bool reference_typed_anywhere_locations_p(reference r)
{
  entity e = reference_variable(r);
  bool anywhere_p = entity_typed_anywhere_locations_p(e);
  return anywhere_p;
}


/* test if an entity is the bottom of the lattice*/
bool entity_typed_anywhere_locations_p(entity e)
{
  string ln = (string) entity_local_name(e);
  string p = strstr(ln, ANYWHERE_LOCATION);
  return p!=NULL;
}

/* return *ANY_MODULE*:*NOWHERE* */

entity entity_nowhere_locations()
{
  static entity nowhere = entity_undefined;
  static const char any_name [] = ANY_MODULE_NAME MODULE_SEP_STRING NOWHERE_LOCATION ;
  nowhere = gen_find_tabulated(any_name, entity_domain);
  if(entity_undefined_p(nowhere)) {
      area a = make_area(0,NIL); /* Size and layout are unknown */
      type t = make_type_area(a);
      nowhere = make_entity(strdup(any_name),
              t, make_storage_rom(), make_value_unknown());
      entity_kind(nowhere)=ABSTRACT_LOCATION;
      register_static_entity(&nowhere);
  }

   return nowhere;
}

entity entity_typed_nowhere_locations(type t)
{
  return entity_all_xxx_locations_typed(NOWHERE_LOCATION, t);
}

/* test if an entity is the bottom of the lattice 
 *
 * Should we care for the typed nowhere too?
*/
bool entity_nowhere_locations_p(entity e)
{
    return same_entity_p(e,entity_nowhere_locations());
}

/* test if an entity is the bottom of the lattice*/
bool entity_typed_nowhere_locations_p(entity e)
{
  string ln = (string) entity_local_name(e);
  string p = strstr(ln, NOWHERE_LOCATION);
  return p!=NULL;
}


/* return TOP-LEVEL:*NULL_POINTER*
 * The NULL pointer should be a global variable, unique for all modules
 * FI: why isn't it called entity_null_location()?
 */
entity entity_null_locations()
{
    static entity null_pointer = entity_undefined;
    if(entity_undefined_p(null_pointer)) {
        const char any_name [] = TOP_LEVEL_MODULE_NAME MODULE_SEP_STRING NULL_POINTER_NAME;
        area a = make_area(0,NIL); /* Size and layout are unknown */
        type t = make_type_area(a);
        null_pointer = make_entity(strdup(any_name),
                t, make_storage_rom(), make_value_unknown());
        entity_kind(null_pointer) = ABSTRACT_LOCATION;
        register_static_entity(&null_pointer);
    }

    return null_pointer;
}

/* test if an entity is the NULL POINTER*/
bool entity_null_locations_p(entity e)
{
    return same_entity_p(e,entity_null_locations());
}

/* return m:*ANYWHERE*
   Set of all memory locations related to one module.

   FI: This may prove useless unless the compilation unit is taken into
   account.
 */
entity entity_all_module_locations(entity m)
{
  entity anywhere = entity_undefined;
  const char* mn = entity_local_name(m);

  anywhere = FindEntity(mn,ANYWHERE_LOCATION);
  if(entity_undefined_p(anywhere)) {
    area a = make_area(0,NIL); /* Size and layout are unknown */
    type t = make_type_area(a);
    /* FI: any_name? */
    anywhere = CreateEntity(mn,ANYWHERE_LOCATION);
    entity_type(anywhere)=t;
    entity_storage(anywhere)=make_storage_rom();
    entity_initial(anywhere)=make_value_unknown();
    entity_kind(anywhere)=ABSTRACT_LOCATION;
  }

  return anywhere;
}

/* test if an entity is the set of locations defined in a module */
bool entity_all_module_locations_p(entity e)
{

  bool all_module_p;
  all_module_p = same_string_p(entity_local_name(e), ANYWHERE_LOCATION);

  return all_module_p;
}

/* return m:xxx*ANYWHERE*
 * Generic set of functions for all kinds of areas
*/

entity entity_all_module_xxx_locations(entity m, const char *xxx)
{
  entity dynamic = entity_undefined;
  string any_name;
  asprintf(&any_name, "%s" ANYWHERE_LOCATION, xxx);

  //dynamic = gen_find_tabulated(any_name, entity_domain);
  dynamic = FindOrCreateEntity(entity_local_name(m), any_name);
  if(storage_undefined_p(entity_storage(dynamic))) {
    area a = make_area(0,NIL); /* Size and layout are unknown */
    type t = make_type_area(a);
    /*FI: more work to be done here... */
    entity_type(dynamic) = t;
    entity_storage(dynamic) = make_storage_rom();
    entity_initial(dynamic) = make_value_unknown();
    entity_kind(dynamic)=ABSTRACT_LOCATION;
  }
  free(any_name);

  return dynamic;
}

entity entity_all_module_xxx_locations_typed(const char* mn, const char* xxx, type t)
{
  entity e = entity_undefined;
  int count = 0;
  bool found_p = false; // a break could be used instead

  pips_assert("Type t is defined", !type_undefined_p(t));

  for(count = 0; !found_p; count++) {

    type ot = type_undefined;
    char * local_name;

    asprintf(&local_name, "%s_b%d", xxx,count);
    e = FindOrCreateEntity(mn,local_name);
    free(local_name);
    ot = entity_type(e);
    if(type_undefined_p(ot)) {
      /* A new entity has been created */
      //area a = make_area(0,NIL); /* Size and layout are unknown */
      //type t = make_type_area(a);
      /*FI: more work to be done here... */
      entity_type(e) = copy_type(t); /* no aliasing */
      entity_storage(e) = make_storage_rom();
      entity_initial(e) = make_value_unknown();
      found_p = true;

    }
    else if(type_equal_p(t, ot))
      found_p = true;
  }

  // FI: the debug message should be improved with full information
  // about the type... See get_symbol_table() and isolate the code
  // used to prettyprint the type. Too bad it uses the buffer type...
  pips_debug(8, "New abstract location entity \"%s\" found or created"
	     " with type \"%s\"\n", entity_name(e), type_to_string(t));

  return e;
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

/* return *ANY_MODULE*:xxx */
entity entity_all_xxx_locations(string xxx)
{
  entity dynamic = entity_undefined;
  dynamic = FindOrCreateEntity(ANY_MODULE_NAME,xxx);

  if(type_undefined_p(entity_type(dynamic))) {
    //area a = make_area(0,NIL); /* Size and layout are unknown */
    //type t = make_type_area(a);
    /*FI: more work to be done here... */
    /* FI: I'd like to make the type variable, overloaded,... */
    // entity_type(dynamic) = make_type_unknown();
    basic b = make_basic_overloaded();
    variable v = make_variable(b, NIL, NIL);
    entity_type(dynamic) = make_type_variable(v);
    entity_storage(dynamic) = make_storage_rom();
    entity_initial(dynamic) = make_value_unknown();
    entity_kind(dynamic)=ABSTRACT_LOCATION;
    if(same_string_p(xxx,HEAP_AREA_LOCAL_NAME)) entity_kind(dynamic)|=ENTITY_HEAP_AREA;
    else if(same_string_p(xxx,STACK_AREA_LOCAL_NAME)) entity_kind(dynamic)|=ENTITY_STACK_AREA;
    else if(same_string_p(xxx,STATIC_AREA_LOCAL_NAME)) entity_kind(dynamic)|=ENTITY_STATIC_AREA;
    else if(same_string_p(xxx,DYNAMIC_AREA_LOCAL_NAME)) entity_kind(dynamic)|=ENTITY_DYNAMIC_AREA;
  }

  return dynamic;
}

/* FI->AM: the predicate entity_all_xxx_locations_typed_p() is missing... */
entity entity_all_xxx_locations_typed(string xxx, type t)
{
  entity e = entity_undefined;
  int count = 0;
  bool found_p = false; // a break could be used instead

  pips_assert("Type t is defined", !type_undefined_p(t));
  pips_assert("Type t is not functional", !type_functional_p(t));

  for(count = 0; !found_p; count++) {
    string name = string_undefined;
    type ot = type_undefined;

    asprintf(&name, "%s_b%d",xxx,count);
    e = FindOrCreateEntity(ANY_MODULE_NAME, name);
    free(name);
    ot = entity_type(e);
    if(type_undefined_p(ot)) {
      /* A new entity has been created */
      //area a = make_area(0,NIL); /* Size and layout are unknown */
      //type t = make_type_area(a);
      /*FI: more work to be done here... */
      entity_type(e) = copy_type(t);
      entity_storage(e) = make_storage_rom();
      entity_initial(e) = make_value_unknown();
      entity_kind(e) = ABSTRACT_LOCATION;
      found_p = true;
    }
    else if(type_equal_p(t, ot))
      found_p = true;
  }

  // FI: the debug message should be improved with full information
  // about the type... See get_symbol_table() and isolate the code
  // used to prettyprint the type. Too bad it uses the buffer type...
  pips_debug(8, "New abstract location entity \"%s\" found or created"
	     " with type \"%s\"", entity_name(e), type_to_string(t));

  return e;
}

/* test if an entity is the set of all memory locations in the xxx
 * area of any module.
 *
 * FI->AM: how come w generate *HEAP* and we check *HEAP**ANYWHERE*?
 */
bool entity_all_xxx_locations_p(entity e, string xxx)
{
  bool dynamic_p;
  string s = concatenate(xxx, ANYWHERE_LOCATION, NULL);

  dynamic_p = same_string_p(entity_local_name(e), s);
  dynamic_p = dynamic_p
    && same_string_p(entity_module_name(e), ANY_MODULE_NAME);

  return dynamic_p;
}



/* return m:*HEAP**ANYWHERE */
entity entity_all_module_heap_locations(entity m)
{
  return entity_all_module_xxx_locations(m, HEAP_AREA_LOCAL_NAME);
}

entity entity_all_module_heap_locations_typed(entity m, type t)
{
  return entity_all_module_xxx_locations_typed(entity_local_name(m),
					       HEAP_AREA_LOCAL_NAME, t);
}

/* test if an entity is the a heap area*/
bool entity_all_module_heap_locations_p(entity e)
{
  return entity_all_module_xxx_locations_p(e, HEAP_AREA_LOCAL_NAME);
}

/* return ANY_MODULE:*HEAP*
 *
 * FI->AM: I move it to ANY_MODULE:*HEAP**ANYWHERE*
 *
 * Not compatible with entity_all_locations_p()...
 */
entity entity_all_heap_locations()
{
  return entity_all_xxx_locations(HEAP_AREA_LOCAL_NAME ANYWHERE_LOCATION);
}

/* test if an entity is the set of all heap locations
 *
 * We look for "*ANY_MODULE*:*HEAP**ANYWHERE*"... unless it is "foo:*HEAP**ANYWHERE*"
 *
 * FI->AM: this is not compatible with the entity name *ANY-MODULE*:*HEAP*
 */
bool entity_all_heap_locations_p(entity e)
{
  return entity_all_xxx_locations_p(e, HEAP_AREA_LOCAL_NAME);
}

entity entity_all_heap_locations_typed(type t)
{
  return entity_all_xxx_locations_typed(HEAP_AREA_LOCAL_NAME, t);
}

// FI->AM: missing predicate...
//bool entity_all_heap_locations_typed_p(entity e)
//{
//  return entity_all_xxx_locations_typed_p(e, HEAP_AREA_LOCAL_NAME);
//}


/* return m:*STACK**ANYWHERE */
entity entity_all_module_stack_locations(entity m)
{
  return entity_all_module_xxx_locations(m, STACK_AREA_LOCAL_NAME);
}

/* test if an entity is the a stack area*/
bool entity_all_module_stack_locations_p(entity e)
{
  return entity_all_module_xxx_locations_p(e, STACK_AREA_LOCAL_NAME);
}

/* return ANY_MODULE:*STACK* */
entity entity_all_stack_locations()
{
  return entity_all_xxx_locations(STACK_AREA_LOCAL_NAME);
}

/* test if an entity is the set of all stack locations */
bool entity_all_stack_locations_p(entity e)
{
  return entity_all_xxx_locations_p(e, STACK_AREA_LOCAL_NAME);
}


/* return m:*DYNAMIC**ANYWHERE */
entity entity_all_module_static_locations(entity m)
{
  return entity_all_module_xxx_locations(m, STATIC_AREA_LOCAL_NAME);
}

/* test if an entity is the a static area*/
bool entity_all_module_static_locations_p(entity e)
{
  return entity_all_module_xxx_locations_p(e, STATIC_AREA_LOCAL_NAME);
}

/* return  ANY_MODULE:*STATIC* */
entity entity_all_static_locations()
{
  return entity_all_xxx_locations(STATIC_AREA_LOCAL_NAME);
}

/* test if an entity is the set of all static locations*/
bool entity_all_static_locations_p(entity e)
{
  return entity_all_xxx_locations_p(e, STATIC_AREA_LOCAL_NAME);
}


/* return m:*DYNAMIC**ANYWHERE */
entity entity_all_module_dynamic_locations(entity m)
{
  return entity_all_module_xxx_locations(m, DYNAMIC_AREA_LOCAL_NAME);
}

/* test if an entity is the a dynamic area*/
bool entity_all_module_dynamic_locations_p(entity e)
{
  return entity_all_module_xxx_locations_p(e, DYNAMIC_AREA_LOCAL_NAME);
}

/* return ANY_MODULE:*DYNAMIC* */
entity entity_all_dynamic_locations()
{
  return entity_all_xxx_locations(DYNAMIC_AREA_LOCAL_NAME);
}

/* test if an entity is the set of all dynamic locations */
bool entity_all_dynamic_locations_p(entity e)
{
  return entity_all_xxx_locations_p(e, DYNAMIC_AREA_LOCAL_NAME);
}

/* test if an entity is a stub sink for a formal pramater
   e.g. f->_f_1, EXACT
*/
bool entity_stub_sink_p(entity e)
{
  bool stub_sink_p = false;
  bool dummy_target_p = false;
  const char * en = entity_local_name(e);
  storage s = entity_storage(e);
  if(storage_ram_p(s))
    dummy_target_p = pointer_dummy_targets_area_p(ram_section(storage_ram(s)));
  char first = en[0];
  char penultimate = en[strlen(en) - 2];
  if(dummy_target_p && first == '_' && penultimate == '_')
    stub_sink_p = true;

  return stub_sink_p;
}

bool entity_abstract_location_p(entity al)
{
#ifndef NDEBUG
        const char * en = entity_name(al);
        const char * module_sep = strchr(en,MODULE_SEP_CHAR);
        bool abstract_locations_p = (   0 == strncmp(en,ANY_MODULE_NAME,module_sep++ - en) // << FI: this may change in the future and may not be a strong enough condition
                ||   0 == strncmp(module_sep, ANYWHERE_LOCATION, sizeof(ANYWHERE_LOCATION)-1)
                ||   0 == strncmp(module_sep, STATIC_AREA_LOCAL_NAME, sizeof(STATIC_AREA_LOCAL_NAME)-1)
                ||   0 == strncmp(module_sep, DYNAMIC_AREA_LOCAL_NAME, sizeof(DYNAMIC_AREA_LOCAL_NAME)-1)
                ||   0 == strncmp(module_sep, STACK_AREA_LOCAL_NAME, sizeof(STACK_AREA_LOCAL_NAME)-1)
                ||   0 == strncmp(module_sep, HEAP_AREA_LOCAL_NAME, sizeof(HEAP_AREA_LOCAL_NAME)-1)
                ||   0 == strncmp(module_sep, NULL_POINTER_NAME, sizeof(NULL_POINTER_NAME)-1)
                )
            ;
        pips_assert("entity_kind is consistent",abstract_locations_p == ((entity_kind(al)&ABSTRACT_LOCATION)==ABSTRACT_LOCATION));
#endif
    return entity_kind(al) & ABSTRACT_LOCATION;
}


/* returns the smallest abstract locations containing the location of
   variable v.

   This does not work for formal parameters or, if it works, the
   caller module is not known and the resulting abstract location is
   very large. A large abstract location is returned.

   No idea to model return values... even though they are located in
   the stack in real world.

   If v cannot be converted into an abstract location, either the
   function aborts or an undefined entity is returned.
*/
entity variable_to_abstract_location(entity v)
{
  entity al = entity_undefined;

  if(entity_abstract_location_p(v))
    al = v;
  /* NULL is an abstract location  */
  else if(entity_null_locations_p(v))
    al = v;
  else if(entity_variable_p(v)
     && !dummy_parameter_entity_p(v)
     && !variable_return_p(v)) {
    bool typed_p = !get_bool_property("ALIASING_ACROSS_TYPES");

    // Too simplistic
    //al = FindOrCreateEntity(mn, ln);

    if(formal_parameter_p(v))
      if(typed_p)
	//  FI: still to be developped
	//al = entity_all_locations_typed(uvt);
	al = entity_all_locations();
      else
	al = entity_all_locations();



    else { // must be a standard variable
      storage s = entity_storage(v);
      ram r = storage_ram(s);
      entity f = ram_function(r);
      entity a = ram_section(r);
      //string mn = entity_local_name(f);
      const char *ln = string_undefined;
      type uvt = ultimate_type(entity_type(v));

      if(static_area_p(a))
	ln = STATIC_AREA_LOCAL_NAME;
      else if(dynamic_area_p(a))
	ln = DYNAMIC_AREA_LOCAL_NAME;
      else if(stack_area_p(a))
	ln = STACK_AREA_LOCAL_NAME;
      else if(heap_area_p(a))
	ln = HEAP_AREA_LOCAL_NAME;
      else
	pips_internal_error("unexpected area");

      if(typed_p) {
	const char* fn = entity_local_name(f);
	al = entity_all_module_xxx_locations_typed(fn, ln, uvt);
      }
      else
	al = entity_all_module_xxx_locations(f, ln);
      entity_kind(al)=ABSTRACT_LOCATION; // should it be a static/dynamic/stack/heap area too ? not according to static_area_p
    }
  }
  else
    pips_internal_error("arg. not in definition domain");

  pips_assert("al is an abstract location entity",
	      entity_abstract_location_p(al));

  return al;
}


/*returns the smallest abstract location set greater than or equalt to
  al1 and al2.

  If al1 or al2 is nowhere, then return al2 or al1.

  If al1 and al2 are related to the same module, the module can be
  preserved. Else the anywhere module must be used.

  If al1 and al2 are related to the same area, then the area is
  preserved. Else, the *anywhere* area is used.

  FI: The type part of the abstract location lattice is not
  implemented... Since the abstract locations are somewhere defined as
  area, they cannot carry a type. Types are taken care of for heap
  modelization but not more the abstract locations.

  FI: we are in trouble with the NULL pointer...
*/
/* here al1 and al2 must be abstract locations */
entity abstract_locations_max(entity al1, entity al2)
{
  entity e = entity_undefined;

  if (same_entity_p(al1, al2)) /* avoid costly string operations in trivial case */
    e = al1;
  else
    {
      const char* ln1 = entity_local_name(al1);
      const char* ln2 = entity_local_name(al2);
      char* mn1 = strdup(entity_module_name(al1));
      char* mn2 = strdup(entity_module_name(al2));
      const char* ln;
      const char* mn;

      if(!get_bool_property("ALIASING_ACROSS_TYPES")) {
	//pips_internal_error("Option not implemented yet.");
	pips_user_warning("property \"ALIASING_ACROSS_TYPES\" is assumed true"
			  " for abstract locations.\n");
      }

      if(strcmp(ln1, ln2)==0)
	ln = ln1;
      else
	ln = ANYWHERE_LOCATION;

      if(strcmp(mn1, mn2)==0)
	mn = mn1;
      else 
	mn = ANY_MODULE_NAME;
      e = FindEntity(mn, ln);
      free(mn1);free(mn2);
    }
  return e;
}

/* Here, entity al1 and entity al2 can be program variables
 */
entity entity_locations_max(entity al1, entity al2)
{
  entity e = entity_undefined;
  //string ln1 = entity_local_name(al1);
  //string ln2 = entity_local_name(al2);

  if(al1==al2) {
    e = al1;
  }
  else {
    bool al1_abstract_location_p = entity_abstract_location_p(al1);
    bool al2_abstract_location_p = entity_abstract_location_p(al2);
    if(al1_abstract_location_p )
      if(al2_abstract_location_p ) {
	/* Both al1 and al2 are abstract locations and they are
	   different */
	e = abstract_locations_max(al1, al2);
      }
      else {
	entity al = variable_to_abstract_location(al2);
	e = abstract_locations_max(al1, al);
      }
    else
      if(al2_abstract_location_p) {
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
	  const char* mn ;
	  const char* ln ;

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
	  e = FindEntity(mn, ln);
	}
	else
	  pips_internal_error("not implemented");
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

 /**
  * @brief Do these two abstract locations MAY share some real memory
  * locations ?
  */
 bool abstract_locations_may_conflict_p(entity al1, entity al2)
 {
   entity mal = abstract_locations_max(al1, al2); // maximal abstraction location
   bool conflict_p = (mal==al1) || (mal==al2);

   return conflict_p;
 }

 /**
  * @brief Do these two abstract locations MUST share some real memory
  * locations ? Never ! DO NOT USE THIS FUNCTION UNLESS...
  */
bool abstract_locations_must_conflict_p(entity al1 __attribute__ ((__unused__)),
					entity al2 __attribute__ ((__unused__)))
 {

   /* The function is useful in functional drivers to avoid dealing
      with specific cases*/

   //pips_internal_error("abstract_locations_must_conflict_p is a non sense : "
   //    "it's always false ! avoid use it.");

   return false;
 }
