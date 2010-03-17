/* Identify properly structure pointed to by expressions */

typedef struct s {
  int foo;
} a_t /* , *p_t*/;
typedef a_t * p_t;

typedef p_t f_t();

typedef struct si {
  int i;
  struct f {
    f_t * init;
    f_t * incr;
    f_t * decr;
  } functions;
} si_t;

f_t ini;
f_t inc;
f_t dec;

si_t tab[] = {{0, {ini, inc, dec}},
	      {1, {ini, inc, dec}},
	      {2, {ini, inc, dec}}};

void initialization04()
{
  /* The top-down approach used to compute the number of elements
     fails here because two elements are seen here: tab2 is analyzed
     as a two element array wile tab3 is analyzed as a one element
     array */
  si_t tab2[] = {0, {ini, inc, dec}};
  si_t tab3[] = {{0, {ini, inc, dec}}};
}
