/* local definitions of this library */
#define RETURN( _debug, _function, _ret_obj ) \
   { debug((_debug), (_function), "returning \n"); return( (_ret_obj) ); }
#define ADFG_MODULE_NAME        "ADFG"
#define EXPRESSION_PVECTEUR(e) \
    newgen_Pvecteur(normalized_linear(NORMALIZE_EXPRESSION( e )))
#define ENTRY_ORDER     300000  /* Hard to have a non illegal number for hash_put !!! */
#define EXIT_ORDER      100000
#define TAKE_LAST	TRUE
#define TAKE_FIRST	FALSE

/* Structure for return of a possible source */
typedef struct Sposs_source {
        quast   *qua;
        Ppath   pat;
} Sposs_source, *Pposs_source;


/* Structure to list wich node read or write an effect */
typedef struct Sentity_vertices {
        entity ent;
        list lis;    /* list of vertices */
} Sentity_vertices, *Pentity_vertices;

