
/*
 * Functions
 */

#include <string.h>

#define PVECTOR(v) ((Pvecteur)CHUNK(v))

#define DELTAV        12345
#define TEMPLATEV     12346
#define TSHIFTV       12347

/*
#define expression_normalized_p(e) (expression_normalized(e) != normalized_undefined)
#define expression_undefined_p(e) ((e) == expression_undefined)
#define normalized_undefined_p(e) ((e) == normalized_undefined)
#define statement_undefined_p(e) ((e) == statement_undefined)
#define alignment_undefined_p(e) ((e) == alignment_undefined)
*/

#define entity_variable_p(e) (type_variable_p(entity_type(e)))
#define statement_block_p(stat) instruction_block_p(statement_instruction(stat))

#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)>(b))?(a):(b))

/* integer ceiling function */
#define iceil(a,b) (((a-1)/b)+1)

#define one_statement_unstructured(u) \
    ((control_predecessors(unstructured_control(u)) == NIL) && \
     (control_successors(unstructured_control(u)) == NIL))

/*
 * Constants
 */

#define HOST_NAME "HOST"
#define NODE_NAME "NODE"

#define HPFINTPREFIX 		"I_"
#define HPFFLOATPREFIX 		"F_"
#define HPFLOGICALPREFIX 	"L_"
#define HPFCOMPLEXPREFIX	"C_"

/*
 * new declaration management
 */
#define NO_NEW_DECLARATION      0
#define ALPHA_NEW_DECLARATION   1
#define BETA_NEW_DECLARATION    2
#define GAMMA_NEW_DECLARATION   3
#define DELTA_NEW_DECLARATION   4

#define no_new_declaration_p(i)  (i==NO_NEW_DECLARATION)
#define declaration_alpha_p(i)   (i==ALPHA_NEW_DECLARATION)
#define declaration_beta_p(i)    (i==BETA_NEW_DECLARATION)
#define declaration_gamma_p(i)   (i==GAMMA_NEW_DECLARATION)
#define declaration_delta_p(i)	 (i==DELTA_NEW_DECLARATION)

/*
 * IO Management
 *
 * {"WRITE", (MAXINT)},
 * {"REWIND", (MAXINT)},
 * {"OPEN", (MAXINT)},
 * {"CLOSE", (MAXINT)},
 * {"READ", (MAXINT)},
 * {"BUFFERIN", (MAXINT)},
 * {"BUFFEROUT", (MAXINT)},
 * {"ENDFILE", (MAXINT)},
 * {"IMPLIED-DO", (MAXINT)},
 * {"FORMAT", 1},
 */

#define WRITE_INTRINSIC_P(call)		\
	(!strcmp(entity_local_name(call_function(call)), "WRITE"))
#define REWIND_INTRINSIC_P(call)  	\
	(!strcmp(entity_local_name(call_function(call)), "REWIND"))
#define OPEN_INTRINSIC_P(call)	  	\
	(!strcmp(entity_local_name(call_function(call)), "OPEN"))
#define CLOSE_INTRINSIC_P(call)	  	\
	(!strcmp(entity_local_name(call_function(call)), "CLOSE"))
#define READ_INTRINSIC_P(call)	  	\
	(!strcmp(entity_local_name(call_function(call)), "READ"))
#define BUFFERIN_INTRINSIC_P(call) 	\
	(!strcmp(entity_local_name(call_function(call)), "BUFFERIN"))
#define BUFFEROUT_INTRINSIC_P(call)	\
	(!strcmp(entity_local_name(call_function(call)), "BUFFEROUT"))
#define ENDFILE_INTRINSIC_P(call)  	\
	(!strcmp(entity_local_name(call_function(call)), "ENDFILE"))
#define IMPLIEDDO_INTRINSIC_P(call)	\
	(!strcmp(entity_local_name(call_function(call)), "IMPLIED-DO"))
#define FORMAT_INTRINSIC_P(call)  	\
	(!strcmp(entity_local_name(call_function(call)), "FORMAT"))

#define IO_CALL_P(call) 		\
    (WRITE_INTRINSIC_P(call) 	||	\
     REWIND_INTRINSIC_P(call) 	||	\
     OPEN_INTRINSIC_P(call) 	||	\
     CLOSE_INTRINSIC_P(call) 	||	\
     READ_INTRINSIC_P(call) 	||	\
     BUFFERIN_INTRINSIC_P(call) ||	\
     BUFFEROUT_INTRINSIC_P(call)||	\
     ENDFILE_INTRINSIC_P(call) 	||	\
     IMPLIEDDO_INTRINSIC_P(call)||	\
     FORMAT_INTRINSIC_P(call))


/*
 * Overlap
 */

#define UPPER 1
#define LOWER 0

#define SEND	1
#define RECEIVE 0
