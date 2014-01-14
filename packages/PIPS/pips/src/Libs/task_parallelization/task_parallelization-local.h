/* parameters of BDSC, to be recovered using pips properties */
int  NBCLUSTERS;
int  MEMORY_SIZE;
string INSTRUMENTED_FILE;
/* Global variables */
gen_array_t annotations;
gen_array_t clusters;


typedef struct {
  double tlevel;
  double blevel;
  double prio;
  double task_time;
  gen_array_t edge_cost;
  list data;
  bool scheduled;
  int order_sched;
  int cluster;
  int nbclusters;
}annotation;

typedef struct {
  double time;
  list data;
}cluster;
