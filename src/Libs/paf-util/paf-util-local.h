#include "matrix.h"

#define PIP_SOLVE_MIN 0
#define PIP_SOLVE_MAX 1

#define IS_MIN 0
#define IS_MAX 1

#define PAF_UTIL_MODULE_NAME "PAFUTIL"
#define DFG_MODULE_NAME "DFG"
#define MAPPING_MODULE_NAME "MAPPING"
#define DIFF_PREFIX "DIFF"
#define COEFF_PREFIX "COEFF"

#define POSITIVE 1
#define NEGATIVE 0

#define ADD_ELEMENT_TO_LIST( _list, _type, _element) \
    (_list = gen_nconc( _list, CONS( _type, _element, NIL)))

#define BASE_NODE_NUMBER 1000
