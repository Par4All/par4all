
#define PIP_SOLVE_MIN 0
#define PIP_SOLVE_MAX 1
#define PIP_SOLVE_INTEGER 1
#define PIP_SOLVE_RATIONAL 0

#define DFG_MODULE_NAME "DFG"
#define MAPPING_MODULE_NAME "MAPPING"
#define DIFF_PREFIX "DIFF"
#define COEFF_PREFIX "COEFF"

#define POSITIVE 1
#define NEGATIVE 0

#define ADD_ELEMENT_TO_LIST( _list, _type, _element) \
    (_list = gen_nconc( _list, CONS( _type, _element, NIL)))

/* PIP includes */
#include "pip__type.h"
#include "pip__sol.h"
#include "pip__tab.h"

