/* $RCSfile: newgen_generic_function.h,v $ ($Date: 1995/03/20 13:38:53 $, )
 * version $Revision$
 * got on %D%, %T%
 */

#ifndef GENERIC_FUNCTION_INCLUDED
#define GENERIC_FUNCTION_INCLUDED

#define GENERIC_STATIC_OBJECT(PREFIX, name, type)\
static type name = type##_undefined;\
PREFIX bool name##_undefined_p() { return(name==type##_undefined);}\
PREFIX void reset_##name() { name=type##_undefined;}\
PREFIX void set_##name(o) type o; { name=o;}\
PREFIX type get_##name() { return(name);}

#define GENERIC_STATIC_STATUS(PREFIX, name, type, init, close)\
GENERIC_STATIC_OBJECT(PREFIX, name, type)\
PREFIX void init_##name() { name = init;}\
PREFIX void close_##name() { close(name);}

/* The idea here is to have a static function the name of which is
 * name, and which is a newgen function (that is a ->).
 * It embeds the status of some function related to the manipulated
 * data, with the {init,set,reset,get,close} functions.
 * Plus the extend, update and apply operators. 
 * This could replace all generic_mappings in PIPS, if the mapping
 * types are declared to newgen. It would also ease the db management.
 */

#define GENERIC_FUNCTION(PREFIX, name, type, ktype, vtype)\
GENERIC_STATIC_STATUS(PREFIX, name, type, make_##type(), free_##type)\
PREFIX void extend_##name(k,v) ktype k; vtype v;\
       { extend_##type(name, k, v);}\
PREFIX void update_##name(k,v) ktype k; vtype v;\
       { update_##type(name, k, v);}\
PREFIX vtype apply_##name(k) ktype k;\
       { return(apply_##type(name, k));}\
PREFIX bool apply_##name##_defined_p(k) ktype k; \
       { return(apply_##type(name, k)!=HASH_UNDEFINED_VALUE);}

#define GENERIC_LOCAL_FUNCTION(name, type, ktype, vtype)\
        GENERIC_FUNCTION(static, name, type, ktype, vtype)

#define GENERIC_GLOBAL_FUNCTION(name, type, ktype, vtype)\
        GENERIC_FUNCTION(/**/, name, type, ktype, vtype)

#endif
