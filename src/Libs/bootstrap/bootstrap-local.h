/* 
 * $Id$
 */

/* should be some properties to accomodate cray codes?? */
#define INT_LENGTH 4
#define REAL_LENGTH 4
#define DOUBLE_LENGTH 8
#define COMPLEX_LENGTH 8
#define DCOMPLEX_LENGTH 16

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

typedef void (*switch_name_function)(expression, type_context_p);
