/* $RCSfile: newgen_generic_function.h,v $ ($Date: 1995/03/14 18:12:23 $, )
 * version $Revision$
 * got on %D%, %T%
 */

/* The idea here is to have a static function the name of which is
 * name, and which is a newgen function (that is a ->).
 * It embeds the status of some function related to the manipulated
 * data, with the {init,set,reset,get,close} functions.
 * Plus the extend, update and apply operators. 
 * This could replace all generic_mappings in PIPS, if the mapping
 * types are declared to newgen. It would also ease the db management.
 */

#ifndef GENERIC_FUNCTION_INCLUDED
#define GENERIC_FUNCTION_INCLUDED

#define GENERIC_FUNCTION(PREFIX, name, type)\
static type name;\
PREFIX bool name##_undefined_p() { return(name==type##_undefined_p);}\
PREFIX void reset_##name(){ name = type##_undefined;}\
PREFIX void set_##name(v) type v; { assert(name##_undefined_p()); name = v;}\
PREFIX type get_##name() { return(name);}\
PREFIX void init_##name() { name = make_##type();}\
PREFIX void close_##name() { free_##type(name);}\
PREFIX void extend_##name(k,v) char *k,*v;{ extend_##type(name, k, v);}\
PREFIX void update_##name(k,v) char *k,*v;{ update_##type(name, k, v);}\
PREFIX char *apply_##name(k) char *k; { apply_##type(name, k);}

#define GENERIC_LOCAL_FUNCTION(name, type)\
        GENERIC_FUNCTION(static, name, type)

#define GENERIC_GLOBAL_FUNCTION(name, type)\
        GENERIC_FUNCTION(/**/, name, type)

#endif
