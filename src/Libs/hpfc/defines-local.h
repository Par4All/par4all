/*
 * HPFC module by Fabien COELHO
 *
 * $RCSfile: defines-local.h,v $ version $Revision$
 * ($Date: 1996/07/23 11:59:17 $, ) 
 */

/* Most includes are centralized here.
 */

/*  standard C includes
 */
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <values.h>
#include <ctype.h>
#include <string.h>

/*  C3/LINEAR
 */
#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"
#include "ray_dte.h"
#include "sommet.h"
#include "polyedre.h"
#include "sg.h"

/*  NEWGEN
 */
#include "genC.h"
#include "text.h"
#include "ri.h"
#include "database.h"
#include "hpf.h"
#include "hpf_private.h"
#include "reductions_private.h"
#include "message.h"
#include "properties.h"

/*  PIPS
 */
#include "misc.h"
#include "text-util.h"
#include "ri-util.h"
#include "hpfc.h"

/* in paf-util.h:
 */
list base_to_list(Pbase base);
void fprint_entity_list(FILE *fp, list l);

#define PVECTOR(v) CHUNK(v)
#define VECTOR gen_chunk*

/* ??? very beurk!
 */
#define DELTAV        ((Variable) 12)
#define TEMPLATEV     ((Variable) 13)
#define TSHIFTV       ((Variable) 14)

/* Newgen short sentences
 */
#define function_mapping(f) (((f)+1)->h)

/* list of variables...
 */
#define add_to_list_of_vars(l, fun, n)\
  l = gen_nconc(make_list_of_dummy_variables(fun, n), l);

#define one_statement_unstructured(u) \
    ((control_predecessors(unstructured_control(u)) == NIL) && \
     (control_successors(unstructured_control(u)) == NIL))

#define entity_functional(e) (type_functional(entity_type(e)))
	    
#define update_functional_as_model(e, model) \
    free_functional(entity_functional(e)), \
    entity_functional(e) = copy_functional(entity_functional(model));

#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)>(b))?(a):(b))

/* integer ceiling function */
#define iceil(a,b) ((((a)-1)/(b))+1)

#ifndef bool_undefined
#define bool_undefined ((bool) (-15))
#define bool_undefined_p(b) ((b)==bool_undefined)
#endif

#ifndef int_undefined
#define int_undefined ((int) (-15))
#define int_undefined_p(i) ((i)==int_undefined)
#endif

/* Constants
 */
#define HOST_NAME "HOST"
#define NODE_NAME "NODE"

/* Overlap
 */
#define UPPER "UP"
#define LOWER "LO"

#define SEND	1
#define RECEIVE 0

/* debug macro
 */
#define GEN_DEBUG(D, W, P) ifdebug(D) { pips_debug(D, "%s:\n", W); P;}

#define DEBUG_STAT(D, W, S) GEN_DEBUG(D, W, print_statement(S))
#define DEBUG_CODE(D, W, M, S) GEN_DEBUG(D, W, hpfc_print_code(stderr, M, S))
#define DEBUG_SYST(D, W, S) GEN_DEBUG(D, W, syst_debug(S))
#define DEBUG_ELST(D, W, L)\
   GEN_DEBUG(D, W, fprint_entity_list(stderr, L); fprintf(stderr, "\n"))
#define DEBUG_BASE(D, W, B)\
   GEN_DEBUG(D, W, base_fprint(stderr, B, entity_local_name);\
	           fprintf(stderr, "\n"))

#define what_stat_debug(level, stat)\
 ifdebug(level) { int so_ = statement_ordering(stat);\
		  pips_debug(level, "statement (%d,%d)\n",\
			  ORDERING_NUMBER(so_), ORDERING_STATEMENT(so_));}

/* Efficient I/O tags
 */
#define is_movement_collect 0
#define is_movement_update 1
#define movement_collect_p(t) ((t)==(is_movement_collect))
#define movement_update_p(t) ((t)==(is_movement_update))

/* Run-time support functions and subroutine names
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

#define TWIN_P		"HPFC_TWIN_P"

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

/* hpfc packing and unpacking
 */
#define BUFPCK		"_HPFC_BUFPCK"
#define BUFUPK		"_HPFC_BUFUPK"

#define BROADCAST	"HPFC_BROADCAST_"
#define GUARDED_BRD	"HPFC_REMAPBRD_"

/* special FCD calls.
 */
#define HOST_TIMEON	"HPFC_HTIMEON"
#define NODE_TIMEON	"HPFC_NTIMEON"
#define HOST_TIMEOFF	"HPFC_HTIMEOFF"
#define NODE_TIMEOFF	"HPFC_NTIMEOFF"

#define RENAME_SUFFIX	"_rename"

/********************************************************************** PVM */

/* PVM functions that may be called by the generated code
 */
#define PVM_INITSEND	"pvmfinitsend"
#define PVM_SEND	"pvmfsend"
#define PVM_CAST	"pvmfmcast"
#define PVM_RECV	"pvmfrecv"
#define PVM_PACK	"pvmfpack"
#define PVM_UNPACK	"pvmfunpack"

/* PVM types encoding for packing and unpacking
 */
#define PVM_BYTE1	"BYTE1"
#define PVM_INTEGER2	"INTEGER2"
#define PVM_INTEGER4	"INTEGER4"
#define PVM_REAL4	"REAL4"
#define PVM_REAL8	"REAL8"
#define PVM_COMPLEX8	"COMPLEX8"
#define PVM_COMPLEX16	"COMPLEX16"
#define PVM_STRING	"STRING"

/********************************************************* COMMON VARIABLES */

#define MYPOS		"MYPOS"
#define MYLID		"MYLID"
#define MSTATUS		"MSTATUS"  		/* remapping status */
#define LIVEMAPPING	"LIVEMAPPING"
#define NODETIDS	"NODETIDS"
#define HOST_TID	"HOSTTID"
#define NBTASKS		"MAXSIZEOFPROCS"
#define MCASTHOST	"MCASTHOST"
#define SEND_CHANNELS	"SENDCHANNELS"
#define RECV_CHANNELS	"RECVCHANNELS"
#define HOST_CHANNEL	"HOSTCHANNEL"

/* common /hpfc_buffers/
 */
#define BUFFER		"_BUFF"
#define BUFSZ		"_BUFFSIZE"
#define BUFFER_INDEX	"BUFINDEX"
#define BUFFER_SIZE	"SIZEOFBUFFER"
#define BUFFER_RCV_SIZE	"SIZEOFRECEIVEDBUFFER"
#define LAZY_SEND	"LAZY_SEND"
#define LAZY_RECV	"LAZY_RECV"
#define SND_NOT_INIT	"SEND_NOT_INITIALIZED"
#define RCV_NOT_PRF	"RECEIVED_NOT_PERFORMED"
#define BUFFER_ENCODING	"BUFFER_ENCODING"

/* Variables
 */
#define T_LID		"T_LID"
#define T_LIDp		"T_LIDp"
#define INFO		"HPFC_INFO"
#define BUFID		"HPFC_BUFID"

/*************************************************************** PROPERTY */

#define LAZY_MESSAGES		"HPFC_LAZY_MESSAGES"
#define USE_BUFFERS		"HPFC_USE_BUFFERS"

/***************************************************************** MACROS */

/*  Very Short and very local functions
 *    moved to macros, FC 17/05/94
 */
#define set_integer(var, i) \
    make_assign_statement(entity_to_expression(var), int_to_expression(i))

#define set_logical(var, b) \
    make_assign_statement(entity_to_expression(var),\
	 make_call_expression(MakeConstant \
	      (b ? ".TRUE." : ".FALSE.", is_basic_logical), NIL))

#define set_expression(var, e) \
    make_assign_statement(entity_to_expression(var), e)

#define local_index_is_different_p(array, dim) \
  (new_declaration_tag(array, dim)!=is_hpf_newdecl_none)

#define FindArrayDimAlignmentOfArray(array, dim) \
  (FindAlignmentOfDim(align_alignment(load_hpf_alignment(array)), dim))

#define FindTemplateDimAlignmentOfArray(array, dim) \
  (FindAlignmentOfTemplateDim(align_alignment(load_hpf_alignment(array)), dim))

#define array_to_template(array) \
  (align_template(load_hpf_alignment(array)))

#define template_to_processors(template) \
  (distribute_processors(load_hpf_distribution(template)))

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
  (MakeUnaryCall(entity_intrinsic(NOT_OPERATOR_NAME), \
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

/*   WARNING
 */
#define hpfc_warning \
    if (!get_bool_property("HPFC_NO_WARNING")) pips_user_warning

/* fake resources...
 */
#define NO_FILE (strdup("fake file name"))

/* that is all
 */
