struct __ {
    int a;
    int (*fun)(struct __ *);
};


int pain(struct __ * bp) {
    /* no read effect on pain because pain is a (painful) constant */
    bp->fun = pain;
    return 0;
}
