/* macros */
#define VERTEX_ENCLOSING_SCC(v) \
    sccflags_enclosing_scc(dg_vertex_label_sccflags((dg_vertex_label) \
	vertex_vertex_label(v)))


/* a macro to insert an element at the end of a list. c is the element
to insert.  bl and el are pointers to the begining and the end of the
list. */

#define INSERT_AT_END(bl, el, c) \
    { cons *_insert_ = c; if (bl == NIL) bl = _insert_; else CDR(el) = _insert_; el = _insert_; CDR(el) = NIL; }


/* external variables. see declarations in kennedy.c */
extern graph dg;
extern bool rice_distribute_only;
extern int Nbrdoall;
