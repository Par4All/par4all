// a minimal list data structure

struct cons_t;
typedef struct cons_t * list;

// empty list
const list nil = ((list) 0);

// alloc/free
list list_cons(double, list);
void list_free(list);
void list_clean(list*);

// getter
list list_next(list);
double list_value(list);

// observer
int list_len(list);

// setter
list list_set_next(list, list);
list list_set_value(list, double);
