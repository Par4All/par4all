 /* value management package for transformers
  *
  * Values could be a new data structure or a special kind of entities.
  * To avoid an increase in data structure declarations and an increased
  * size for the implicit array of entities, values are often entities with
  * special names. Their meaning is defined with respect to an environment
  * defined at the module level and so, constant during the analysis of
  * a module.
  *
  * Three different kind of values are used: new, old and intermediate.
  * New values are used in post-condition predicates. Old Values are
  * used in pre-condition predicates. New and Old values are also used
  * in predicate transformers. Intermediate values are used to combine
  * transformers.
  *
  * New values are directly referenced by the corresponding scalar integer
  * variable (a regular entity). However they are printed with a different
  * name to avoid confusion between variables and values. The value name
  * is the variable name, suffixed by "#new".
  *
  * Old values and intermediate values are referenced via special entities,
  * distinguished by their names.
  *
  * Pre-values for local dynamic entities are named "o#XX", where "XX"
  * is a number. A pre-value is associated to an entity via a hash table
  * used by the function entity_to_old_value. Static aliasing (i.e.
  * EQUIVALENCEs) is dealed with by associating the same value entities
  * to the two aliased variable entities.
  * 
  * Pre-values for integer scalar formal parameters and integer scalar
  * global variables are represented by special entities named from
  * the relevant variable entity, by suffixing its name by "#init". These
  * values have to be represented by specific entities to let us consider
  * parameter values at entry point and to let us perform interprocedural
  * semantics analysis.
  *
  * Intermediate values are referenced by special entities named "i#XX",
  * which are reused from procedure to procedure. Within a procedure,
  * each variable is linked to one intermediate value entity.
  *
  * These conventions are used by value_to_name(e) to generate EXTERNAL
  * names I#new, I#old and I#tmp and to print readable debug information.
  *
  * To accelerate conversion between new, old and intermediate value
  * entities, three hash tables are set up for each module before
  * semantics analysis is started:
  *  - hash_entity_to_new_value associates an entity representing a new
  *    value to each scalar integer variable; this value entity is most
  *    of the time the variable entity itself and thus this hash table
  *    almost represents an identity function; however, perfectly equivalenced
  *    variables are represented by one of them only and some eligible
  *    integer scalar variables are not mapped to anything because they
  *    are equivalenced to an array
  *  - hash_entity_to_old_value associates an entity representing
  *    an old value to each scalar integer variable
  *  - hash_entity_to_intermediate_value
  *  - hash_value_to_name: associates readable names to value entity;
  *    used for debugging and display purposes.
  *
  * Francois Irigoin, December 1989
  *
  * Modifications:
  *  - the three mappings between variable entities and value entities was done
  *    differently for each transformer and based on the arguments field;
  *    efficiency was limited because the mappings were based on lists and
  *    because transformers had to be renamed before they could be combined;
  *    aliasis information had to be computed at each step instead of being
  *    factored out at hashing time (Francois Irigoin, April 90)
  *  - *** assigments to array or to non integer scalar can now affect 
  *    transformer on integer scalar variables (Francois Irigoin, April 90)***
  *    no longer true - such integer scalar variables are just ignored;
  *    however, it would be easy to do slightly better (21 April 1990)
  *  - old value suffix is now "#init" because it's nicer in prettyprinting
  *    predicates; it's also worse for transformer but they are less
  *    interesting for the users (Francois Irigoin, 18 April 1990)
  *  - fourth hash_table entity->new_value added to handle EQUIVALENCEs
  *    (Francois Irigoin, 18 April 1990)
  *  - only integer scalar variables that are not aliased or that are
  *    aliased to another integer scalar variable are analyzed; dubious
  *    interest but April 24th is close (Francois Irigoin, 21 April 1990)
  *
  * Bugs:
  *  - once the hash tables are lost, intraprocedural transformers are
  *    useless while they were still interpretable when mappings were
  *    based on arguments; they could be translated into the previous
  *    representation when their computation is completed (Francois Irigoin,
  *    April 1990)
  *  - hash tables are static; recursion is not possible for interprocedural
  *    analysis; topographical order will have to be used for it
  *    (Francois Irigoin, April 1990)
  *
  * Assumptions:
  *  - toutes les valeurs et seulement les valeurs utilisees dans un module ont
  *    des noms dans value_to_name()
  *  - toutes les variables dont les valeurs peuvent etre importees ont des 
  *    entrees dans les trois tables entity_to_xxx_value; l'information est 
  *    apportee par les effets qui ne prennent pas en compte tous les alias; 
  *    l'aliasing est traite par le biais de ces tables; le representant d'un 
  *    ensemble de variables alias est la variable visible dans le module 
  *    courant (apparaissant la premire?) ou a defaut, si aucune des variables
  *    de l'ensemble n'est visible dans le module courant, n'importe laquelle
  *    d'entre elles (en fait la premiere qui apparait dans les effets
  *    interproceduraux); les variables qui sont seulement
  *    lues n'apparaissent que dans entity_to_new_value;
  *  - un summary_transformer importe peut etre difficile a imprimer
  *  - le probleme des summary_preconditions n'est pas resolue par ces 
  *    conventions pour le moment (faute de repere absolu)
  *
  * Francois Irigoin, 13 January 1994
  */

#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>
#include <string.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "constants.h"

#include "misc.h"

#include "transformer.h"

/* STATIC VARIABLES */

/* Four global hash tables used to map scalar integer variable entities
 * onto the different kind of value entities and to map value entities
 * onto their external names.
 *
 * They prevent recursive calls.
 */
static hash_table hash_entity_to_new_value = hash_table_undefined;
static hash_table hash_entity_to_old_value = hash_table_undefined;
static hash_table hash_entity_to_intermediate_value = hash_table_undefined;
static hash_table hash_value_to_name = hash_table_undefined;

/* Two counters used to assign meaningless value entities to local variables.
 * A special global prefix, SEMANTICS_MODULE_NAME, and two special local
 * prefixes, OLD_VALUE_PREFIX and INTERMEDIATE_VALUE_PREFIX, are also used
 */

static int local_intermediate_value_counter = 0;
static int local_old_value_counter = 0;

void 
reset_value_counters()
{
    local_old_value_counter = 0;
    local_intermediate_value_counter = 0;
}

/* LOCAL VALUE ENTITY */

/* static entity make_local_value_entity(int n, bool old): find or generate
 * an entity representing an old or an intermediate value of number n
 */
static 
entity make_local_value_entity(n, old)
int n;
bool old;
{
    entity v;
    /* 13 is a magic number that can accomodate 2 prefix characters,
       10 digits and a null character */
    char value_name[13];
    char * s;

    if(old==TRUE)
	(void) strcpy(value_name, OLD_VALUE_PREFIX);
    else
	(void) strcpy(value_name, INTERMEDIATE_VALUE_PREFIX);
    (void) sprintf(value_name+2,"%d",n);
    debug(8,"make_local_value_entity","value name: %s\n",value_name);
    s = strdup(concatenate(SEMANTICS_MODULE_NAME,
			   MODULE_SEP_STRING, value_name, (char *) NULL));

    /* find entity or define it */
    v = gen_find_tabulated(s, entity_domain);
    if(v==entity_undefined)
	v = make_entity(s,
			make_scalar_integer_type(4),
			make_storage(is_storage_rom, UU),
			value_undefined);
    else
	free(s);

    return v;
}

static entity 
make_local_old_value_entity()
{
    return make_local_value_entity(local_old_value_counter++, TRUE);
}

static entity 
make_local_intermediate_value_entity()
{
    return make_local_value_entity(local_intermediate_value_counter++, FALSE);
}

bool 
local_old_value_entity_p(e)
entity e;
{
    /* this is not a general test; it will only work for LOCAL values */
    return strncmp(entity_local_name(e), OLD_VALUE_PREFIX, 2) == 0;
}

bool 
local_intermediate_value_entity_p(e)
entity e;
{
    /* this is not a general test; it will only work for LOCAL values */
    return strncmp(entity_local_name(e), INTERMEDIATE_VALUE_PREFIX, 2) == 0;
}

bool 
global_new_value_p(e)
entity e;
{
    bool new = FALSE;
    /* this is not a general test; it will only work for GLOBAL values */

    /* this function should always return FALSE because new value = variable (FI) */

    /* => suf == NULL
    string suf = strchr(entity_local_name(e), SEMANTICS_SEPARATOR);

    pips_assert("global_new_value", suf != NULL);

    new = strcmp(entity_module_name(e), SEMANTICS_MODULE_NAME) != 0 &&
	strcmp(suf, NEW_VALUE_SUFFIX) == 0;
	*/

    pips_assert("global_new_value", new == FALSE);

    return new;
}

bool 
global_old_value_p(e)
entity e;
{
    /* this is not a general test; it will only work for GLOBAL values */
    string suf = strchr(entity_local_name(e), SEMANTICS_SEPARATOR);
    bool old = FALSE;

    if(suf!=NULL)
	old = strcmp(entity_module_name(e), SEMANTICS_MODULE_NAME) != 0 &&
	    strcmp(suf, OLD_VALUE_SUFFIX) == 0;

    return old;
}

bool 
global_intermediate_value_p(e)
entity e;
{
    /* this is not a general test; it will only work for GLOBAL values */
    string suf = strchr(entity_local_name(e), SEMANTICS_SEPARATOR);
    bool intermediate = FALSE;

    if(suf!=NULL)
	intermediate = strcmp(entity_module_name(e), SEMANTICS_MODULE_NAME) != 0 &&
	    strcmp(suf, INTERMEDIATE_VALUE_SUFFIX) == 0;

    return intermediate;
}

entity 
global_new_value_to_global_old_value(entity v_new)
{
    entity v_old = entity_undefined;

    /* There is no real test for global new values */

    pips_assert("global_new_value_to_global_old_value",
		strcmp(entity_module_name(v_new), SEMANTICS_MODULE_NAME) != 0);

    v_old = (entity) gen_find_tabulated(concatenate(entity_name(v_new),
						    OLD_VALUE_SUFFIX,
						    NULL),
					entity_domain);
    return v_old;
}

/* HASH TABLE USE */

string 
external_value_name(e)
entity e;
{
    extern entity get_current_module_entity();
    entity m = get_current_module_entity();
    string s = hash_get(hash_value_to_name, (char *) e);

    if(s==HASH_UNDEFINED_VALUE && !variable_in_module_p(e,m))
    {
	if(global_new_value_p(e)) {
	    entity a = entity_to_new_value(e);
	    s = hash_get(hash_value_to_name, (char *) a);
	}
	else if(global_old_value_p(e)) {
	    entity a = entity_to_old_value(e);
	    s = hash_get(hash_value_to_name, (char *) a);
	}
	else if(global_intermediate_value_p(e)) {
	    entity a = entity_to_intermediate_value(e);
	    s = hash_get(hash_value_to_name, (char *) a);
	}
	else {
	    /* This should never occur. Please core dump! */
	    pips_error("external_value_name", "Unexpected value %s for current module %s\n", 
		       entity_name(e), module_local_name(get_current_module_entity()));
	}
    }
	    
    pips_assert("var must be bounded", s != HASH_UNDEFINED_VALUE);

    s = (strcmp(module_local_name(m), module_name(s)) == 0) ? 
	local_name(s) : s ;

    return s;
}

entity 
entity_to_new_value(e)
entity e;
{
    entity n;
    if((n = (entity) hash_get(hash_entity_to_new_value, (char *) e))
       == entity_undefined)
	pips_error("entity_to_new_value","unbounded entity %s\n",
		   entity_name(e));
    return n;
}

entity 
entity_to_old_value(e)
entity e;
{
    entity o;
    if((o = (entity) hash_get(hash_entity_to_old_value, (char *) e))
       == entity_undefined)
	pips_error("entity_to_old_value","unbounded entity %s\n",
		   entity_name(e));
    return o;
}

entity 
entity_to_intermediate_value(e)
entity e;
{
    entity i;
    if((i = (entity) hash_get(hash_entity_to_intermediate_value, (char *) e))
       == entity_undefined)
	pips_error("entity_to_intermediate_value","unbounded entity %s\n",
		   entity_name(e));
    return i;
}

bool 
entity_has_values_p(e)
entity e;
{
    /* is e variable whose value(s) (already) are analyzed?
     */
    pips_assert("value hash table is defined", 
		!hash_table_undefined_p(hash_entity_to_new_value));

    return hash_defined_p(hash_entity_to_new_value, (char *) e);
}

/* the following three functions are directly or indirectly relative
 * to the current module and its value hash tables.
 */

bool 
new_value_entity_p(e)
entity e;
{
    /* since new values are always variable entities, hash_entity_to_new_value
       can be used for this test */

    pips_assert("new_value_entity_p",e != entity_undefined);

    return (entity) hash_get(hash_entity_to_new_value, (char *) e)
       == e;
}

bool 
old_value_entity_p(e)
entity e;
{
    string s = strstr(external_value_name(e), OLD_VALUE_SUFFIX);

    return s!=NULL;
}

bool 
intermediate_value_entity_p(e)
entity e;
{
    string s = strstr(external_value_name(e), INTERMEDIATE_VALUE_SUFFIX);

    return s!=NULL;
}

bool 
value_entity_p(e)
entity e;
{
    /* tells if e is seen as a value in the current module */
    string s = hash_get(hash_value_to_name, (char *) e);

    if(s == (char*) HASH_UNDEFINED_VALUE) {
	return FALSE;
    }
    else {
	return TRUE;
    }
}

/* used with hash_table_fprintf */
static string 
string_identity(s)
string s;
{ return s;}

void 
print_value_mappings()
{
    (void) fprintf(stderr,"hash table value to name:\n");
    hash_table_fprintf(stderr, dump_value_name, string_identity,
		       hash_value_to_name);

    (void) fprintf(stderr,"hash table entity to new value:\n");
    /*
    hash_table_fprintf(stderr, entity_local_name, external_value_name,
		       hash_entity_to_new_value);
		       */
    hash_table_fprintf(stderr, entity_minimal_name, entity_minimal_name,
		       hash_entity_to_new_value);

    (void) fprintf(stderr,"hash table entity to old value:\n");
    hash_table_fprintf(stderr, entity_minimal_name, external_value_name,
		       hash_entity_to_old_value);

    (void) fprintf(stderr, "hash table entity to intermediate value:\n");
    hash_table_fprintf(stderr, entity_minimal_name, external_value_name,
		       hash_entity_to_intermediate_value);
}

void 
test_mapping_entry_consistency()
{
    int nbo,nbi;
    pips_assert("consistent mapping entry number", 
		((nbo= hash_table_entry_count(hash_entity_to_old_value)) 
		 == (nbi= hash_table_entry_count(hash_entity_to_intermediate_value))) 
		&& ( hash_table_entry_count(hash_value_to_name) >=
		    hash_table_entry_count(hash_entity_to_new_value) 
		    + nbo +nbi));
}

void 
allocate_value_mappings(int n, int o, int i)
{
    pips_assert("undefined mappings for allocation",
		hash_table_undefined_p(hash_entity_to_new_value) &&
		hash_table_undefined_p(hash_entity_to_old_value) &&
		hash_table_undefined_p(hash_entity_to_intermediate_value) &&
		hash_table_undefined_p(hash_value_to_name));

    /* hash_warn_on_redefinition(); */
    hash_entity_to_new_value = hash_table_make(hash_pointer, n);
    hash_entity_to_old_value = hash_table_make(hash_pointer, o);
    hash_entity_to_intermediate_value = 
	hash_table_make(hash_pointer, i);
    hash_value_to_name = 
	hash_table_make(hash_pointer, n + o + i);
}

static void 
reset_value_mappings(void)
{
    hash_entity_to_new_value = hash_table_undefined;
    hash_entity_to_old_value = hash_table_undefined;
    hash_entity_to_intermediate_value = hash_table_undefined;
    hash_value_to_name = hash_table_undefined;
}

/* Normal call to free the mappings */

void 
free_value_mappings(void)
{
    pips_assert("no free of undefined mappings", 
		!hash_table_undefined_p(hash_entity_to_old_value));
    error_free_value_mappings();
}

/* To be called by an error handler */

void 
error_free_value_mappings(void)
{
    /* free previous hash tables, desallocate names; this implies ALL
       value names were malloced and were not pointer to a ri part */

    /* the three tables are assumed to be allocated all together */

    /* free names in hash_value_to_name; the other two hash tables
       contain pointers to the entity tabulated domain and thus need
    no value freeing */
    /* k is discovered unused by lint; it is syntaxically necessary */
    HASH_MAP(k, v, {free(v);}, hash_value_to_name);
    /* free the three tables themselves */
    hash_table_free(hash_entity_to_new_value);
    hash_table_free(hash_entity_to_old_value);
    hash_table_free(hash_entity_to_intermediate_value);
    hash_table_free(hash_value_to_name);
    
    reset_value_mappings();
}

/* HASH TABLE INITIALIZATION */

/* void add_new_value_name(entity e): add a new value name for entity e */
static void 
add_new_value_name(e)
entity e;
{
    string new_value_name = 
	strdup(concatenate(entity_name(e), NEW_VALUE_SUFFIX, 
			   (char *) NULL));
    debug(8,"add_new_value_name","begin: for %s\n", entity_name(e));
    if(hash_get(hash_value_to_name, (char *) e) == HASH_UNDEFINED_VALUE)
	hash_put(hash_value_to_name, (char *) e, (char *) new_value_name);
    else
	free(new_value_name);
}

void 
add_new_value(e)
entity e;
{
    debug(8,"add_new_value","begin: for %s\n", entity_name(e));
    if(hash_get(hash_entity_to_new_value, (char *) e)
       == HASH_UNDEFINED_VALUE) {
	hash_put(hash_entity_to_new_value, (char *) e, (char *) e);
	add_new_value_name(e);
    }
}

void 
add_new_alias_value(e,a)
entity e;
entity a;
{
    entity v = entity_undefined;
    debug(8,"add_new_alias_value","begin: for %s and %s\n",
      entity_name(e),entity_name(a));
    pips_assert("add_new_alias_valued", 
		(hash_get(hash_entity_to_new_value, (char *) e)
		 == HASH_UNDEFINED_VALUE));

    v = entity_to_new_value(a);

    hash_put(hash_entity_to_new_value, (char *) e, (char *) v);
    /* add_new_value_name(e); */
}

entity 
external_entity_to_new_value(e)
entity e;
{
    /* there is no decisive test here; a necessary condition is used instead */

    entity e_new;

    pips_assert("external_entity_to_new_value", entity_integer_scalar_p(e));
    e_new = e;
    return e_new;
}

entity 
external_entity_to_old_value(e)
entity e;
{
    /* find the old value entity if possible or abort
       I should never have to call this function on an local entity
       (local entities include static local variables) */

    entity old_value;
    string old_value_name;

    old_value_name = concatenate(entity_name(e),OLD_VALUE_SUFFIX, 
				 (char *) NULL);
    old_value = gen_find_tabulated(old_value_name, entity_domain);

    pips_assert("external_entity_to_old_value", old_value!=entity_undefined);

    return old_value;
}

void 
add_old_value(e)
entity e;
{
    entity old_value;
    string old_value_name;

    /* find the old value entity if possible, else, generate it */
    old_value_name = concatenate(entity_name(e),OLD_VALUE_SUFFIX, 
				 (char *) NULL);
    old_value = gen_find_tabulated(old_value_name, entity_domain);
    if(old_value == entity_undefined)
	    old_value = make_entity(strdup(old_value_name),
				    make_scalar_integer_type(4),
				    make_storage(is_storage_rom, UU),
				    value_undefined);
    /* add the couple (e, old_value) */
    if(hash_get(hash_entity_to_old_value, (char *) e) == HASH_UNDEFINED_VALUE) {
	hash_put(hash_entity_to_old_value, (char *) e, (char *) old_value);
	/* add its name */
	hash_put(hash_value_to_name, (char *) old_value, 
		 strdup(entity_name(old_value)));
    }
}

void 
add_old_alias_value(e,a)
entity e;
entity a;
{
    entity v = entity_undefined;

    pips_assert("add_old_alias_valued", (hash_get(hash_entity_to_old_value, (char *) e)
       == HASH_UNDEFINED_VALUE));

    v = entity_to_old_value(a);

    hash_put(hash_entity_to_old_value, (char *) e, (char *) v);
    /* add_new_value_name(e); */
}

void 
add_intermediate_value(e)
entity e;
{
    entity intermediate_value;

    /* get a new intermediate value, if necessary */
    if((intermediate_value = 
       (entity) hash_get(hash_entity_to_intermediate_value, (char *) e))
       == (entity) HASH_UNDEFINED_VALUE) {
	intermediate_value = make_local_intermediate_value_entity();
	hash_put(hash_entity_to_intermediate_value, (char *) e, 
		 (char *) intermediate_value);
	/* add its (external) name */
	hash_put(hash_value_to_name, (char *) intermediate_value,
		 strdup(concatenate(entity_name(e),
				    INTERMEDIATE_VALUE_SUFFIX,
				    (char *) NULL)));
    }
}

void 
add_intermediate_alias_value(e,a)
entity e;
entity a;
{
    entity v = entity_undefined;

    pips_assert("add_intermediate_alias_valued",
		(hash_get(hash_entity_to_intermediate_value, (char *) e)
		 == HASH_UNDEFINED_VALUE));

    v = entity_to_intermediate_value(a);

    hash_put(hash_entity_to_intermediate_value, (char *) e, (char *) v);
    /* add_new_value_name(e); */
}

void 
add_local_old_value(e)
entity e;
{
    entity old_value;

    /* get a new old value, if necessary */
    if((old_value = 
	(entity) hash_get(hash_entity_to_old_value, (char *) e))
       == (entity) HASH_UNDEFINED_VALUE) {
	old_value = make_local_old_value_entity();
	hash_put(hash_entity_to_old_value, (char *) e, (char *) old_value);
	/* add its (external) name */
	hash_put(hash_value_to_name, (char *) old_value,
		 strdup(concatenate(entity_name(e), OLD_VALUE_SUFFIX,
				    (char *) NULL)));
    }
}

void 
add_local_intermediate_value(e)
entity e;
{
    entity intermediate_value;

    /* get a new intermediate value, if necessary */
    if((intermediate_value = 
	(entity) hash_get(hash_entity_to_intermediate_value, (char *) e))
       == (entity) HASH_UNDEFINED_VALUE) {
	intermediate_value = make_local_intermediate_value_entity();
	hash_put(hash_entity_to_intermediate_value, (char *) e, 
		 (char *) intermediate_value);
	/* add its (external) name */
	hash_put(hash_value_to_name, (char *) intermediate_value,
		 strdup(concatenate(entity_name(e),
				    INTERMEDIATE_VALUE_SUFFIX,
				    (char *) NULL)));
    }
}

void 
remove_entity_values(e, readonly)
entity e;
bool readonly;
{
    entity new_value = entity_to_new_value(e);
    string s;

    /* pips_assert("remove_entity_values", e != entity_undefined); */
    pips_assert("remove_entity_values", new_value != entity_undefined);

    s = external_value_name(new_value);
    pips_assert("remove_entity_values", s != (char *) NULL);
    (void) hash_del(hash_value_to_name, (char *) new_value);
    (void) hash_del(hash_entity_to_new_value, (char *) e);

    if(!readonly) {
	entity old_value = entity_to_old_value(e);
	entity intermediate_value = entity_to_intermediate_value(e);

	pips_assert("remove_entity_values", old_value != entity_undefined);
	pips_assert("remove_entity_values",
		    intermediate_value!=entity_undefined);

	s = external_value_name(old_value);
	pips_assert("remove_entity_values", s != (char *) NULL);
	(void) hash_del(hash_value_to_name, (char *) old_value);
	s = external_value_name(intermediate_value);
	pips_assert("remove_entity_values", s != (char *) NULL);
	(void) hash_del(hash_value_to_name, (char *) intermediate_value);

	(void) hash_del(hash_entity_to_old_value, (char *) e);
	(void) hash_del(hash_entity_to_intermediate_value, (char *) e);
    }
}

void 
add_synonym_values(e, eq, readonly)
entity e;
entity eq;
bool readonly;
{
    /* e and eq are entities whose values are always equal because they share
     * the exact same memory location (i.e. they are alias) */
    entity new_value = entity_to_new_value(e);
    entity intermediate_value;

    hash_put(hash_entity_to_new_value, (char *) eq,
	     (char *) new_value);
    if(!readonly) {
	entity old_value = entity_to_old_value(e);
	entity old_value_eq = entity_undefined;
	intermediate_value = entity_to_intermediate_value(e);

	add_old_value(eq);
	old_value_eq = entity_to_old_value(eq);
	hash_put(hash_entity_to_intermediate_value, (char *) eq,
		 (char *) intermediate_value);
	hash_update(hash_entity_to_old_value, (char *) eq, 
		 (char *) old_value);
	/* hash_put(hash_value_to_name, (char *) old_value_eq, entity_name(old_value_eq)); */
    }
    /* the name does not change */
    hash_put(hash_value_to_name, (char *) eq,
	     strdup(concatenate(entity_name(eq),
				NEW_VALUE_SUFFIX, (char *) NULL)));
}

/* This function used to be restricted to values seen by the
 * current module. It was extended to values in general to
 * cope with translation issues.
 */

entity 
value_to_variable(val)
entity val;
{
    entity var = entity_undefined;
    int l_suffix = -1; /* for gcc only! */
    string s = hash_get(hash_value_to_name, (char *) val);
    string var_name;
    extern entity get_current_module_entity();

    /* pips_assert("value_to_variable", s != HASH_UNDEFINED_VALUE); */

    /* pips_assert("value_to_variable",
       strchr(entity_name(val), (int) SEMANTICS_SEPARATOR) != NULL); */

    if(s == HASH_UNDEFINED_VALUE) {
	/* this may be a value, but it is unknown in the current module */
	string val_name = entity_name(val);

	if(strstr(val_name, NEW_VALUE_SUFFIX) != NULL) 
	    l_suffix = strlen(NEW_VALUE_SUFFIX);
	else if(strstr(val_name, OLD_VALUE_SUFFIX) != NULL) 
	    l_suffix = strlen(OLD_VALUE_SUFFIX);
	else if(strstr(val_name, INTERMEDIATE_VALUE_SUFFIX) != NULL)
	    l_suffix = strlen(INTERMEDIATE_VALUE_SUFFIX);
	else if(strchr(val_name, (int) SEMANTICS_SEPARATOR) == NULL) {
	    /* new values in fact have no suffixes... */
	    l_suffix = 0;
	}
	else
	    pips_error("value_to_variable",
		       "%s is not a non-local value\n", entity_name(val));

	s = val_name;
    }
    else {
	if(new_value_entity_p(val))
	    l_suffix = strlen(NEW_VALUE_SUFFIX);
	else if(old_value_entity_p(val))
	    l_suffix = strlen(OLD_VALUE_SUFFIX);
	else if(intermediate_value_entity_p(val))
	    l_suffix = strlen(INTERMEDIATE_VALUE_SUFFIX);
	else
	    pips_error("value_to_variable",
		       "%s is not a locally visible value\n",
		       entity_name(val));
    }

    var_name = strdup(s);
    *(var_name+strlen(var_name)-l_suffix) = '\0';
    var = gen_find_tabulated(var_name, entity_domain);
    free(var_name);

    if( var == entity_undefined )
	pips_error("value_to_variable",
		   "no related variable for val=%s\n", entity_name(val));

    return var;
}

entity 
old_value_to_new_value(o_val)
entity o_val;
{
    entity var = entity_undefined;
    entity n_val = entity_undefined;

    var = value_to_variable(o_val);
    /* o_val = variable_to_old_value(var); */
    n_val = entity_to_new_value(var);

    return n_val;
}

entity 
new_value_to_old_value(n_val)
entity n_val;
{
    entity o_val = entity_undefined;

    /* FI: correct code
    entity var = entity_undefined;

    var = value_to_variable(n_val);
    o_val = entity_to_old_value(var);
    */

    /* FI: faster code */
    o_val = entity_to_old_value(n_val);

    return o_val;
}

entity 
value_alias(e)
entity e;
{
    entity a = entity_undefined;

    debug(8,"value_alias","begin: for %s\n", entity_name(e));

    /* lookup the current value name mapping and return an arbitraty "representant"
     of the interprocedural alias set of e; the equivalence relation is "has
     same location" */
    HASH_MAP(var, val, {
	if(variable_entity_p((entity) var) && entity_conflict_p((entity) var, e)) {
	    a = (entity) var;
	    break;
	}
    }, hash_value_to_name);

    if(a==entity_undefined)
	debug(8,"value_alias","return: %s\n", "entity_undefined");
    else
	debug(8,"value_alias","return: %s\n", entity_name(a));

    return a;
}
