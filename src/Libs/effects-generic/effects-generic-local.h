/* $Id$
 */


/* some useful SHORTHANDS for EFFECT:
 */
#define effect_entity(e) reference_variable(effect_reference(e))
#define effect_action_tag(eff) action_tag(effect_action(eff))
#define effect_approximation_tag(eff) \
	approximation_tag(effect_approximation(eff))

#define effect_scalar_p(eff) entity_scalar_p(effect_entity(eff))
#define effect_read_p(eff) (action_tag(effect_action(eff))==is_action_read)
#define effect_write_p(eff) (action_tag(effect_action(eff))==is_action_write)
#define effect_may_p(eff) \
        (approximation_tag(effect_approximation(eff)) == is_approximation_may)
#define effect_must_p(eff) \
        (approximation_tag(effect_approximation(eff)) == is_approximation_must)
#define effect_exact_p(eff) \
        (approximation_tag(effect_approximation(eff)) ==is_approximation_exact)


/* some string constants for prettyprints...
 */
#define ACTION_UNDEFINED 	string_undefined
#define ACTION_READ 		"R"
#define ACTION_WRITE 		"W"
#define ACTION_IN    		"IN"
#define ACTION_OUT		"OUT"
#define ACTION_COPYIN		"COPYIN"
#define ACTION_COPYOUT		"COPYOUT"
#define ACTION_PRIVATE		"PRIVATE"


/* prettyprint function types:
 */
#include "text.h" /* hum... */
typedef text (*generic_text_function)(list /* of effect */);
typedef void (*generic_prettyprint_function)(list /* of effect */);
typedef void (*generic_attachment_function)(text);

 
/* for db_* functions 
 */
#define DB_GET_SE(name, NAME)				\
static statement_effects db_get_##name(char * modname)	\
{ return (statement_effects)				\
  db_get_memory_resource(DBR_##NAME, modname, TRUE);}

#define DB_GET_LS(name, NAME)				\
static list db_get_##name(char * modname)		\
{ return effects_to_list((effects)			\
  db_get_memory_resource(DBR_##NAME, modname, TRUE));}

#define DB_PUT_SE(name, NAME)						\
static void db_put_##name(char * modname, statement_effects se)		\
{ DB_PUT_MEMORY_RESOURCE(DBR_##NAME, modname, (char*) se);}

#define DB_PUT_LS(name, NAME)				\
static void db_put_##name(char * modname, list l)	\
{DB_PUT_MEMORY_RESOURCE(DBR_##NAME,modname,(char*)list_to_effects(l));}

#define DB_NOPUT_SE(name)\
static void db_put_##name(char *m, statement_effects se) \
{ free_statement_effects(se); return; }

#define DB_NOPUT_LS(name)\
static void db_put_##name(char *m, list l) \
{ gen_full_free_list(l); return;}

#define DB_GETPUT_SE(name, NAME) DB_GET_SE(name, NAME) DB_PUT_SE(name, NAME)
#define DB_GETNOPUT_SE(name, NAME) DB_GET_SE(name, NAME) DB_NOPUT_SE(name)
#define DB_GETPUT_LS(name, NAME) DB_GET_LS(name, NAME) DB_PUT_LS(name, NAME)
#define DB_GETNOPUT_LS(name, NAME) DB_GET_LS(name, NAME)DB_NOPUT_LS(name)


/* For COMPATIBILITY purpose only - DO NOT USE anymore
 */
#define effect_variable(e) reference_variable(effect_reference(e))

/* end of effects-generic-local.h
 */
