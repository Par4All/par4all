
#ifndef _ehrhart_H_
#define _ehrhart_H_

/*********************** User defines ******************************/

/* Print all overflow warnings, or just one per domain             */
/* #define ALL_OVERFLOW_WARNINGS */

/******************* End of user defines ***************************/


#ifndef ALL_OVERFLOW_WARNINGS
extern int overflow_warning_flag;
#endif


#if __STDC__

extern int count_points ( int pos, Polyhedron *P, Value *context );
extern void eadd ( evalue *e1, evalue *res );
extern enode *ecopy ( enode *e );
extern void edot ( enode *v1, enode *v2, evalue *res );
extern enode *new_enode( enode_type type,int size, int pos );
extern void free_evalue_refs ( evalue *e );
extern Enumeration *Polyhedron_Enumerate ( Polyhedron *P, Polyhedron *C,
                                           unsigned MAXRAYS, char **pname );
extern void print_enode ( FILE *DST, enode *p, char **pname );
extern void print_evalue ( FILE *DST, evalue *e, char **pname );

#else /* __STDC__ */

extern int count_points (/* int pos, Polyhedron *P, Value *context */);
extern void eadd (/* evalue *e1, evalue *res */);
extern enode *ecopy (/* enode *e */);
extern void edot (/* enode *v1, enode *v2, evalue *res */);
extern enode *new_enode(/* enode_type type,int size, int pos*/ );
extern void free_evalue_refs (/* evalue *e */);
extern Enumeration *Polyhedron_Enumerate (/* Polyhedron *P, Polyhedron
                                      *C, unsigned MAXRAYS, char **pname */);
extern void print_enode (/* FILE *DST, enode *p, char **pname */);
extern void print_evalue (/* FILE *DST, evalue *e, char **pname */);

#endif /* __STDC__ */
#endif /* _ehrhart_H_ */
