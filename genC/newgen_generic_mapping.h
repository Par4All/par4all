/* -- generic-mapping.h
 *
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
 * PLEASE COPY THIS FILE IN $NEWGENDIR AFTER MODIFICATIONS
 *
 * generic-mapping.h (94/03/10) version 1.6, got on 94/03/10, 13:31:20
 * @(#) generic-mapping.h 1.6@(#)
 */

#ifndef GENERIC_MAPPING_INCLUDED
#define GENERIC_MAPPING_INCLUDED

/*
 * PIPS level:
 *
 * GENERIC_CURRENT_MAPPING(name, result, type, CTYPE)
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
#define GENERIC_CURRENT_MAPPING(name, result, type)\
static type##_mapping name##_map = hash_table_undefined;\
\
void set_##name##_map(m) \
type##_mapping m;\
{\
    assert(name##_map == hash_table_undefined);\
    name##_map = m;\
}\
\
type##_mapping get_##name##_map() \
{\
    return name##_map;\
}\
\
void reset_##name##_map()\
{\
     name##_map = hash_table_undefined;\
}\
void free_##name##_map() \
{\
     hash_table_free(name##_map);\
     name##_map = hash_table_undefined;\
}\
void make_##name##_map() \
{\
     name##_map = hash_table_make(hash_pointer, HASH_DEFAULT_SIZE);\
}\
result load_##type##_##name(s)\
type s;\
{\
     result t;\
     assert(s != type##_undefined);\
     t = (result) hash_get((hash_table) (name##_map), (char*) (s));\
     if (t ==(result) HASH_UNDEFINED_VALUE) t = result##_undefined;\
     return t;\
}\
bool type##_##name##_undefined_p(s)\
type s;\
{\
    return(load_##type##_##name(s)==result##_undefined);\
}\
\
void store_##type##_##name(s,t)\
type s;\
result t;\
{\
    assert(s != type##_undefined && t != result##_undefined);\
    hash_put((hash_table) (name##_map), (char *)(s), (char *)(t));\
}

/* end GENERIC_MAPPING_INCLUDED */
#endif
