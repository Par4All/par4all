/* $RCSfile: newgen_generic_function.h,v $ ($Date: 1995/05/05 15:30:02 $, )
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
 * data, with the {INIT,SET,RESET,GET,CLOSE} functions.
 * Plus the STORE, UPDATE and LOAD operators. and BOUND_P predicate.
 * This could replace all generic_mappings in PIPS, if the mapping
 * types are declared to newgen. It would also ease the db management.
 */

/* STORE and LOAD are prefered to extend and apply because it sounds
 * like the generic mappings, and it feels as a status/static thing
 */

#define GENERIC_FUNCTION(PREFIX, name, type)\
GENERIC_STATIC_STATUS(PREFIX, name, type, make_##type(), free_##type)\
PREFIX void store_##name(k,v) type##_key_type k; type##_value_type v;\
       { extend_##type(name, k, v);}\
PREFIX void update_##name(k,v) type##_key_type k; type##_value_type v;\
       { update_##type(name, k, v);}\
PREFIX type##_value_type load_##name(k) type##_key_type k;\
       { return(apply_##type(name, k));}\
PREFIX type##_value_type delete_##name(k) type##_key_type k;\
       { return(delete_##type(name, k));}\
PREFIX bool bound_##name##_p(k) type##_key_type k; \
       { return(bound_##type##_p(name, k));}

/* plus a non good looking hack to avoid gcc warnings about undefined statics.
 * init, close and store are NOT put in the hack because they MUST be used.
 */
#define GENERIC_LOCAL_FUNCTION(name, type)\
        GENERIC_FUNCTION(static, name, type)\
static int name##_hack()\
{ return((int) name##_undefined_p & (int) reset_##name & \
	 (int) set_##name & (int) get_##name & \
	 (int) update_##name & (int) load_##name & \
	 (int) bound_##name##_p & (int) delete_##type & (int) name##_hack);}

#define GENERIC_GLOBAL_FUNCTION(name, type)\
        GENERIC_FUNCTION(/**/, name, type)

#endif
