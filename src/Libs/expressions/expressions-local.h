/*
 * $Id$
 */

/*
 * for partial_eval.c:
 *
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

