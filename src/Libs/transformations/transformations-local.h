#define SIGN_EQ(a,b) ((((a)>0 && (b)>0) || ((a)<0 && (b)<0)) ? TRUE : FALSE)
#define FORTRAN_DIV(n,d) (SIGN_EQ((n),(d)) ? ABS(n)/ABS(d) : -(ABS(n)/ABS(d)))
#define FORTRAN_MOD(n,m) (SIGN_EQ((n),(m)) ? ABS(n)%ABS(m) : -(ABS(n)%ABS(m)))


/*
 * EFORMAT: the expression format used in recursiv evaluations.
 * = ((ICOEF * EXPR) + ISHIFT)
 * it is SIMPLER when it is interesting to replace initial expression by the
 * one generated from eformat.
 */
struct eformat {
    expression expr;
    int icoef;
    int ishift;
    boolean simpler;
};


/* What is returned by dead_test_filter : */
enum dead_test { nothing_about_test, then_is_dead, else_is_dead };
typedef enum dead_test dead_test;
