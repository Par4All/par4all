
#define MAPPING_MODULE_NAME "MAPPING"

#define CONST_COEFF "CC"
#define INDEX_COEFF "XC"
#define PARAM_COEFF "PC"
#define AUXIL_COEFF "AC"
#define LAMBD_COEFF "LC"
#define MU_COEFF    "ZM"
#define INDEX_VARIA "XV"
#define DIFFU_COEFF "DC"


#define VERTEX_DOMAIN(v) \
	dfg_vertex_label_exec_domain((dfg_vertex_label) vertex_vertex_label(v))

#define SUCC_DATAFLOWS(s) \
	dfg_arc_label_dataflows((dfg_arc_label) successor_arc_label(s))

