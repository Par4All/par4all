/*
 * HPFC module by Fabien COELHO
 *
 * SCCS stuff:
 * $RCSfile: defines-local.h,v $ ($Date: 1994/11/17 14:19:20 $, ) version $Revision$, got on %D%, %T%
 * $Id$
 */

/*
 * Functions
 */

#include <string.h>

#define PVECTOR(v) ((Pvecteur)CHUNK(v))

/*
 * ??? very beurk!
 */
#define DELTAV        ((Variable) 12)
#define TEMPLATEV     ((Variable) 13)
#define TSHIFTV       ((Variable) 14)

/*
 * Newgen short sentences
 */

#define one_statement_unstructured(u) \
    ((control_predecessors(unstructured_control(u)) == NIL) && \
     (control_successors(unstructured_control(u)) == NIL))

#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)>(b))?(a):(b))

/* integer ceiling function */
#define iceil(a,b) ((((a)-1)/(b))+1)

/*  update a subroutine to a function
 */

#define entity_result_type(e)\
    functional_result(type_functional(entity_type(e)))

#define update_result_type_as_model(e, model) \
    free_type(entity_result_type(e)), \
    entity_result_type(e) = copy_type(entity_result_type(model));

#ifndef bool_undefined
#define bool_undefined ((bool) (-15))
#define bool_undefined_p(b) ((b)==bool_undefined)
#endif

#ifndef int_undefined
#define int_undefined ((int) (-15))
#define int_undefined_p(i) ((i)==int_undefined)
#endif

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
#define NEW_DECLARATION_UNDEFINED -1
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

#define UPPER "UP"
#define LOWER "LO"

#define SEND	1
#define RECEIVE 0

/*
 * debug macro
 */
#define IFDBPRINT(n, func, module, stat)                            \
    ifdebug(n)                                                      \
    {                                                               \
       fprintf(stderr,                                              \
	       "[%s] %s statement:\n",                              \
	       func,entity_name(module));                           \
       print_text(stderr,text_statement(module,0,stat));            \
    }

/*
 * Efficient I/O tags
 */

#define is_movement_collect 0
#define is_movement_update 1
#define movement_collect_p(t) ((t)==(is_movement_collect))
#define movement_update_p(t) ((t)==(is_movement_update))

/*
 * Run-time support functions and subroutine names
 */

#define SND_TO_C        "HPFC_SNDTO_C"
#define SND_TO_H        "HPFC_SNDTO_H"
#define SND_TO_A        "HPFC_SNDTO_A"
#define SND_TO_A_BY_H   "HPFC_HSNDTO_A"
#define SND_TO_O        "HPFC_SNDTO_O"
#define SND_TO_OS       "HPFC_SNDTO_OS"
#define SND_TO_OOS      "HPFC_SNDTO_OOS"
#define SND_TO_HA       "HPFC_SNDTO_HA"
#define SND_TO_NO       "HPFC_SNDTO_NO"
#define SND_TO_HNO      "HPFC_SNDTO_HNO"

#define RCV_FR_S        "HPFC_RCVFR_S"
#define RCV_FR_H        "HPFC_RCVFR_H"
#define RCV_FR_C        "HPFC_RCVFR_C"
#define RCV_FR_mCS      "HPFC_RCVFR_mCS"
#define RCV_FR_mCH      "HPFC_RCVFR_mCH"

#define CMP_COMPUTER    "HPFC_CMPCOMPUTER"
#define CMP_OWNERS      "HPFC_CMPOWNERS"
#define CMP_NEIGHBOUR   "HPFC_CMPNEIGHBOUR"
#define CMP_LID		"HPFC_CMPLID"

#define CND_SENDERP     "HPFC_SENDERP"
#define CND_OWNERP      "HPFC_OWNERP"
#define CND_COMPUTERP   "HPFC_COMPUTERP"
#define CND_COMPINOWNP  "HPFC_COMPUTERINOWNERSP"

#define LOCAL_IND       "HPFC_LOCALIND"
#define LOCAL_IND_GAMMA "HPFC_LOCALINDGAMMA"
#define LOCAL_IND_DELTA "HPFC_LOCALINDDELTA"

#define INIT_NODE       "HPFC_INIT_NODE"
#define INIT_HOST       "HPFC_INIT_HOST"
#define NODE_END        "HPFC_NODE_END"
#define HOST_END        "HPFC_HOST_END"

#define LOOP_BOUNDS     "HPFC_LOOP_BOUNDS"
#define SYNCHRO		"HPFC_SYNCHRO"
#define IDIVIDE		"HPFC_DIVIDE"

#define SND_TO_N        "HPFC_SNDTO_N"
#define RCV_FR_N        "HPFC_RCVFR_N"

/*
 * PVM
 */
#define PVM_INITSEND	"pvmfinitsend"
#define PVM_SEND	"pvmfsend"
#define PVM_CAST	"pvmfmcast"
#define PVM_RECV	"pvmfrecv"
#define PVM_PACK	"pvmfpack"
#define PVM_UNPACK	"pvmfunpack"

/*
 * Variables
 */
#define T_LID		"T_LID"
#define INFO		"HPFC_INFO"
#define BUFID		"HPFC_BUFID"

#define MYPOS		"MYPOS"
#define NODETIDS	"NODETIDS"
#define HOST_TID	"HOSTTID"
#define NBTASKS		"MAXSIZEOFPROCS"
#define MCASTHOST	"MCASTHOST"
#define NODE_CHANNELS	"NODECHANNELS"
#define HOST_CHANNEL	"HOSTCHANNEL"

/*
 *  Very Short and very local functions
 *    moved to macros, FC 17/05/94
 *
 */

#define local_index_is_different_p(array, dim) \
  (new_declaration(array, dim)!=NO_NEW_DECLARATION)

#define FindArrayDimAlignmentOfArray(array, dim) \
  (FindAlignmentOfDim(align_alignment(load_entity_align(array)), dim))

#define FindTemplateDimAlignmentOfArray(array, dim) \
  (FindAlignmentOfTemplateDim(align_alignment(load_entity_align(array)), dim))

#define array_to_template(array) \
  (align_template(load_entity_align(array)))

#define template_to_processors(template) \
  (distribute_processors(load_entity_distribute(template)))

#define array_to_processors(array) \
  (template_to_processors(array_to_template(array)))

#define hpfc_name_to_expression(s) \
  (MakeNullaryCall(hpfc_name_to_entity(s)))

#define condition_computerp() \
  hpfc_name_to_expression(CND_COMPUTERP)

#define condition_senderp() \
  hpfc_name_to_expression(CND_SENDERP)

#define condition_ownerp() \
  hpfc_name_to_expression(CND_OWNERP)

#define condition_computer_in_owners() \
  hpfc_name_to_expression(CND_COMPINOWNP)

#define condition_not_computer_in_owners()\
  (MakeUnaryCall(CreateIntrinsic(NOT_OPERATOR_NAME), \
		 condition_computer_in_owners()))

#define hpfc_name_to_stmt(s) \
  (hpfc_make_call_statement(hpfc_name_to_entity(s), NIL))

#define st_init_host() hpfc_name_to_stmt(INIT_HOST)
#define st_init_node() hpfc_name_to_stmt(INIT_NODE)
#define st_host_end() hpfc_name_to_stmt(HOST_END)
#define st_node_end() hpfc_name_to_stmt(NODE_END)

#define hpfc_name_and_ref_to_stmt(s, val) \
  st_call_send_or_receive(hpfc_name_to_entity(s), val)

/* SND */

#define st_send_to_computer(val) \
  hpfc_name_and_ref_to_stmt(SND_TO_C, val)

#define st_send_to_host(val) \
  hpfc_name_and_ref_to_stmt(SND_TO_H, val)

#define st_send_to_all_nodes(val) \
  hpfc_name_and_ref_to_stmt(SND_TO_A, val)

#define st_host_send_to_all_nodes(val) \
  hpfc_name_and_ref_to_stmt(SND_TO_A_BY_H, val)

#define st_send_to_neighbour() \
  hpfc_make_call_statement(hpfc_name_to_entity(SND_TO_N), NIL)

#define st_send_to_owner(val) \
  hpfc_name_and_ref_to_stmt(SND_TO_O, val)

#define st_send_to_owners(val) \
  hpfc_name_and_ref_to_stmt(SND_TO_OS, val)

#define st_send_to_other_owners(val) \
  hpfc_name_and_ref_to_stmt(SND_TO_OOS, val)

#define st_send_to_host_and_all_nodes(val) \
  hpfc_name_and_ref_to_stmt(SND_TO_HA, val)

#define st_send_to_not_owners(val) \
  hpfc_name_and_ref_to_stmt(SND_TO_NO, val)

#define st_send_to_host_and_not_owners(val) \
  hpfc_name_and_ref_to_stmt(SND_TO_HNO, val)


/* RCV */

#define st_receive_from_neighbour() \
  hpfc_make_call_statement(hpfc_name_to_entity(RCV_FR_N), NIL)

#define st_receive_from_sender(goal) \
  hpfc_name_and_ref_to_stmt(RCV_FR_S, goal)

#define st_receive_from_host(goal) \
  hpfc_name_and_ref_to_stmt(RCV_FR_H, goal)

#define st_receive_from_computer(goal) \
  hpfc_name_and_ref_to_stmt(RCV_FR_C, goal)

#define st_receive_mcast_from_sender(goal) \
  hpfc_name_and_ref_to_stmt(RCV_FR_mCS, goal)

#define st_receive_mcast_from_host(goal) \
  hpfc_name_and_ref_to_stmt(RCV_FR_mCH, goal)

/*
 * that is all
 */

