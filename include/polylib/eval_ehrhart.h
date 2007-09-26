#ifndef _eval_ehrhart_H_
#define _eval_ehrhart_H_

#if defined(__cplusplus)
extern "C" {
#endif

extern double compute_evalue ( evalue *e, Value *list_args );
extern Value *compute_poly (Enumeration *en, Value *list_args);
extern int in_domain(Polyhedron *P, Value *list_args);

#if defined(__cplusplus)
}
#endif

#endif /* _eval_ehrhart_H_ */



