/*
 * These are the functions defined in the Newgen mapping library.
 *
 * This is a temporary implementation used for the Pips Project. The
 * general notion of mapping (i.e., functions) is going to be implemented
 * shortly inside Newgen.
 *
 * This version uses pointers to statements as keys in hash_tables which
 * causes bugs when code in unloaded on and then reloaded from disk.
 * Francois Irigoin, 1 February 1993
 *
 * a useful macro which generates the declaration of a static variable of type
 * statement_mapping, and related functions :
 *       set_**_map
 *       load_statement_**_map
 *       store_statement_**_map
 *       reset_**_map
 *
 * BA, august 26, 1993
 *
 * this macro is redefined here as GENERIC_CURRENT_MAPPING, which uses
 * that type as a parameter, to allow other mappings than statement's ones.
 * I need entity_mapping, control_mapping
 * there is no way to define macros inside the macro, so I cannot
 * generate the usual macros:-(, The code defining them will have to be
 * replicated (done in mapping.h).
 *
 * FC, Feb 21, 1994
 *
 * $RCSfile: newgen_generic_mapping.h,v $ ($Date: 1994/12/30 13:58:50 $, )
 * version $Revision$
 * got on %D%, %T%
 */

#ifndef GENERIC_MAPPING_INCLUDED
#define GENERIC_MAPPING_INCLUDED

/*
 * PIPS level:
 *
 * GENERIC_MAPPING(PREFIX, name, result, type)
 *
 * name: name of the mapping
 * result: type of the result
 * type: type of the mapping key
 * // CTYPE: type of the mapping key in capital letters
 *
 * a static variable 'name'_map of type 'type' is declared
 * and can be accessed thru the definefd functions.
 *
 * the mapping is   'name' = 'type' -> 'result'
 */
#define GENERIC_MAPPING(PREFIX, name, result, type)\
static type##_mapping name##_map = hash_table_undefined;\
\
PREFIX void set_##name##_map(m) \
type##_mapping m;\
{\
    assert(name##_map == hash_table_undefined);\
    name##_map = m;\
}\
\
PREFIX type##_mapping get_##name##_map() \
{\
    return name##_map;\
}\
\
PREFIX void reset_##name##_map()\
{\
     name##_map = hash_table_undefined;\
}\
PREFIX void free_##name##_map() \
{\
     hash_table_free(name##_map);\
     name##_map = hash_table_undefined;\
}\
PREFIX void make_##name##_map() \
{\
     name##_map = hash_table_make(hash_pointer, HASH_DEFAULT_SIZE);\
}\
PREFIX result load_##type##_##name(s)\
type s;\
{\
     result t;\
     assert(s != type##_undefined);\
     t = (result) hash_get((hash_table) (name##_map), (char*) (s));\
     if (t ==(result) HASH_UNDEFINED_VALUE) t = result##_undefined;\
     return t;\
}\
PREFIX bool type##_##name##_undefined_p(s)\
type s;\
{\
    return(load_##type##_##name(s)==result##_undefined);\
}\
\
PREFIX void store_##type##_##name(s,t)\
type s;\
result t;\
{\
    assert(s != type##_undefined && t != result##_undefined);\
    hash_put((hash_table) (name##_map), (char *)(s), (char *)(t));\
}\
\
static void check_##name()\
{\
  type item = (type) check_##name;\
  result r = (result)check_##name;\
  type##_mapping saved = get_##name##_map();\
\
  reset_##name##_map();\
  make_##name##_map(); \
  assert(type##_##name##_undefined_p(item));\
  store_##type##_##name(item,r);\
  assert(load_##type##_##name(item)==r);\
  free_##name##_map();\
  reset_##name##_map();\
  set_##name##_map(saved);\
}

#define GENERIC_CURRENT_MAPPING(name, result, type) \
        GENERIC_MAPPING(/**/, name, result, type)

/*  to allow mappings local to a file.
 *  it seems not to make sense, but I like the interface.
 *  FC 27/12/94
 */
#define GENERIC_LOCAL_MAPPING(name, result, type) \
        GENERIC_MAPPING(static, name, result, type)

/* end GENERIC_MAPPING_INCLUDED */
#endif
