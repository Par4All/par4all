#define STATIC_CONTROLIZE_MODULE_NAME 	"STATCON"
#define NLC_PREFIX 			"NLC"
#define NSP_PREFIX 			"NSP"
#define NUB_PREFIX			"NUB"

#define give_entity(n)	gen_find_tabulated(make_entity_fullname( \
				TOP_LEVEL_MODULE_NAME, n), entity_domain)
#define ENTITY_GE   give_entity(GREATER_OR_EQUAL_OPERATOR_NAME)
#define ENTITY_GT   give_entity(GREATER_THAN_OPERATOR_NAME)
#define ENTITY_LE   give_entity(LESS_OR_EQUAL_OPERATOR_NAME)
#define ENTITY_LT   give_entity(LESS_THAN_OPERATOR_NAME)
#define ENTITY_NOT  give_entity(NOT_OPERATOR_NAME)
#define ENTITY_EQ   give_entity(EQUAL_OPERATOR_NAME)
#define ENTITY_NE   give_entity(NOT_EQUAL_OPERATOR_NAME)
#define ENTITY_OR   give_entity(OR_OPERATOR_NAME)
#define ENTITY_AND  give_entity(AND_OPERATOR_NAME)

#define ENTITY_NLC_P(e) (strncmp(entity_local_name(e), NLC_PREFIX, 3) == 0)
#define ENTITY_NSP_P(e) (strncmp(entity_local_name(e), NSP_PREFIX, 3) == 0)
#define ENTITY_NUB_P(e) (strcmp(entity_local_name(e), NUB_PREFIX, 3) == 0)
#define ENTITY_SP_P(ent) \
   (gen_find_eq(ent, Gstructure_parameters) != chunk_undefined)
#define ENTITY_STRICT_LOGICAL_OPERATOR_P(e) \
		( ENTITY_AND_P(e) || \
		  ENTITY_OR_P(e) || \
		  ENTITY_EQUIV_P(e) || \
		  ENTITY_NON_EQUIV_P(e) || \
		  ENTITY_NOT_P(e) )

extern int Gcount_nlc;
extern int Gcount_nsp;
extern int Gcount_nub;
