 /* include file for semantic analysis */

#define SEMANTICS_OPTIONS "?Otcfieod-D:"

#define SEQUENTIAL_TRANSFORMER_SUFFIX ".tran"
#define USER_TRANSFORMER_SUFFIX ".utran"
#define SEQUENTIAL_PRECONDITION_SUFFIX ".prec"
#define USER_PRECONDITION_SUFFIX ".uprec"

/* Maximum number of nodes in a CFG/unstructured for an accurate analysis */
#define SEMANTICS_MAX_CFG_SIZE1 (20)
/* Too big, even for a simple convex union: let's use effects */
#define SEMANTICS_MAX_CFG_SIZE2 (30)
