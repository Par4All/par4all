/* $Id$ */

#ifndef NEWGEN_GENERIC_FUNCTION_INCLUDED
#define NEWGEN_GENERIC_FUNCTION_INCLUDED

/* Add the __attribute__ ((unused)) for gcc to avoid warning if the
   functions are not used. */
#define GENERIC_STATIC_OBJECT(PREFIX, name, type)			\
  static type name##_object = type##_undefined;				\
  PREFIX bool __attribute__ ((unused)) name##_undefined_p(void) {	\
    return name##_object==type##_undefined;				\
  }									\
  PREFIX void __attribute__ ((unused)) reset_##name(void) {		\
    message_assert("must reset sg defined", !name##_undefined_p());	\
    name##_object=type##_undefined;					\
  }									\
  PREFIX void __attribute__ ((unused)) error_reset_##name(void) {	\
    name##_object=type##_undefined;					\
  }									\
  PREFIX void __attribute__ ((unused)) set_##name(type o) {		\
    message_assert("must set sg undefined", name##_undefined_p());	\
    name##_object=o;							\
  }									\
  PREFIX type __attribute__ ((unused)) get_##name(void) {		\
    message_assert("must get sg defined", !name##_undefined_p());	\
    return name##_object;						\
  }

#define GENERIC_STATIC_STATUS(PREFIX, name, type, init, cloze)		\
  GENERIC_STATIC_OBJECT(PREFIX, name, type)				\
  PREFIX void __attribute__ ((unused)) init_##name(void) {		\
    message_assert("must initialize sg undefined", name##_undefined_p()); \
    name##_object = init;						\
  }									\
  PREFIX void __attribute__ ((unused)) close_##name(void) {		\
    message_assert("must close sg defined", !name##_undefined_p());	\
    cloze(name##_object); name##_object = type##_undefined;		\
}

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
#define GENERIC_FUNCTION(PREFIX, name, type)				\
  GENERIC_STATIC_STATUS(PREFIX, name, type, make_##type(), free_##type)	\
  PREFIX void __attribute__ ((unused)) store_##name(type##_key_type k, type##_value_type v) { \
    extend_##type(name##_object, k, v);					\
  }									\
  PREFIX void __attribute__ ((unused)) update_##name(type##_key_type k, type##_value_type v) { \
    update_##type(name##_object, k, v);					\
  }									\
  PREFIX __attribute__ ((unused)) type##_value_type load_##name(type##_key_type k) { \
    return(apply_##type(name##_object, k));				\
  }									\
  PREFIX __attribute__ ((unused)) type##_value_type delete_##name(type##_key_type k) { \
    return(delete_##type(name##_object, k));				\
  }									\
  PREFIX bool __attribute__ ((unused)) bound_##name##_p(type##_key_type k) { \
    return(bound_##type##_p(name##_object, k));				\
  }									\
  PREFIX void __attribute__ ((unused)) store_or_update_##name(type##_key_type k, type##_value_type v) { \
    if (bound_##name##_p(k))						\
      update_##name(k, v);						\
    else								\
      store_##name(k, v);						\
  }

#define GENERIC_LOCAL_FUNCTION(name, type)\
        GENERIC_FUNCTION(static, name, type)

#define GENERIC_GLOBAL_FUNCTION(name, type)\
        GENERIC_FUNCTION(extern, name, type)

#endif /* NEWGEN_GENERIC_FUNCTION_INCLUDED */
