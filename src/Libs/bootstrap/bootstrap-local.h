/* 
 * $Id$
 */

/* context for type checking. */
typedef struct 
{
    hash_table types;
    stack stats;
    int number_of_error;
    int number_of_conversion;
    int number_of_simplication;
} type_context_t, * type_context_p;

typedef basic (*typing_function_t)(call, type_context_p);

typedef void (*switch_name_function)(call, type_context_p);


