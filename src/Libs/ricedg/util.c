
#include "local.h" 

extern bool 
ignore_this_conflict(vertex v1, vertex v2, conflict c, int l);


/* Vertex_to_statement looks for the statement that is pointed to by
vertex v. This information is kept in a static hash_table named
OrderingToStatement. See ri-util/ordering.c for more information.
*/

statement 
vertex_to_statement(v)
vertex v;
{
    dg_vertex_label dvl = (dg_vertex_label) vertex_vertex_label(v);
    return(ordering_to_statement(dg_vertex_label_statement(dvl)));
}

int 
vertex_to_ordering(v)
vertex v;
{
    dg_vertex_label dvl = (dg_vertex_label) vertex_vertex_label(v);
    return dg_vertex_label_statement(dvl);
}


/* Define a mapping from the statement ordering to the dependence
   graph vertices: */
hash_table
compute_ordering_to_dg_mapping(graph dependance_graph)
{
   hash_table ordering_to_dg_mapping = hash_table_make(hash_int, 0);
   
   MAP(VERTEX,
       a_vertex,
       {
          debug(7, "compute_ordering_to_dg_mapping",
                "\tSuccessor list: %p for statement ordering %p\n", 
                vertex_successors(a_vertex),
                dg_vertex_label_statement(vertex_vertex_label(a_vertex)));

          hash_put(ordering_to_dg_mapping,
                   (char *) dg_vertex_label_statement(vertex_vertex_label(a_vertex)),
                   (char *) a_vertex);
       },
       graph_vertices(dependance_graph));
   
   return ordering_to_dg_mapping;
}


static string dependence_graph_banner[8] = {
	"\n *********************** Use-Def Chains *********************\n",
	"\n **************** Effective Dependence Graph ****************\n",
	"\n ********* Dependence Graph (ill. option combination) *******\n",
	"\n ********* Dependence Graph (ill. option combination) *******\n",
	"\n ******** Whole Dependence Graph with Dependence Cones ******\n",
	"\n ********* Dependence Graph (ill. option combination) *******\n",
	"\n ********* Dependence Graph (ill. option combination) *******\n",
	"\n **** Loop Carried Dependence Graph with Dependence Cones ***\n"
    };

/* Print all edges and arcs */
void 
prettyprint_dependence_graph(FILE * fd,
			     statement mod_stat,
			     graph mod_graph)
{
    cons *pv1, *ps, *pc;
    Ptsg gs;
    int banner_number = 0;
    bool sru_format_p = get_bool_property("PRINT_DEPENDENCE_GRAPH_USING_SRU_FORMAT");
    persistant_statement_to_int s_to_l = persistant_statement_to_int_undefined;
    int dl = -1;
    fprintf(fd,"I DONT KNOW WHY\n");

    debug_on("RICEDG_DEBUG_LEVEL");

    if(sru_format_p && !statement_undefined_p(mod_stat)) {
	/* compute line numbers for statements */
	s_to_l = statement_to_line_number(mod_stat);
	dl = module_to_declaration_length(get_current_module_entity());
     }
    else {
	banner_number =
	    get_bool_property("PRINT_DEPENDENCE_GRAPH_WITHOUT_PRIVATIZED_DEPS") +
	    2*get_bool_property
	    ("PRINT_DEPENDENCE_GRAPH_WITHOUT_NOLOOPCARRIED_DEPS") +
	    4*get_bool_property
	    ("PRINT_DEPENDENCE_GRAPH_WITH_DEPENDENCE_CONES");
	fprintf(fd, "%s\n", dependence_graph_banner[banner_number]);
    }

    for (pv1 = graph_vertices(mod_graph); !ENDP(pv1); pv1 = CDR(pv1)) {
	vertex v1 = VERTEX(CAR(pv1));
	statement s1 = vertex_to_statement(v1);

	for (ps = vertex_successors(v1); !ENDP(ps); ps = CDR(ps)) {
	    successor su = SUCCESSOR(CAR(ps));
	    vertex v2 = successor_vertex(su);
	    statement s2 = vertex_to_statement(v2);
	    dg_arc_label dal = (dg_arc_label) successor_arc_label(su);

	    if(!sru_format_p || statement_undefined_p(mod_stat)) {
		/* factorize line numbers */
		fprintf(fd, "\t%02d --> %02d with conflicts\n", 
			statement_number(s1), statement_number(s2));
	    }

	    for (pc = dg_arc_label_conflicts(dal); !ENDP(pc); pc = CDR(pc)) {
		conflict c = CONFLICT(CAR(pc));
		     
		/* if (!entity_scalar_p(reference_variable
		   (effect_reference(conflict_source(c))))) {
		   */
		if(sru_format_p && !statement_undefined_p(mod_stat)) {
		    int l1 = dl + apply_persistant_statement_to_int(s_to_l, s1);
		    int l2 = dl + apply_persistant_statement_to_int(s_to_l, s2);

		    fprintf(fd, "%d %d ", l1, l2);
		    /*
		    fprintf(fd, "%d %d ", 
		    statement_number(s1), statement_number(s2));
		    */
		    fprintf(fd, "%c %c ", 
			    action_read_p(effect_action(conflict_source(c)))? 'R' : 'W',
			    action_read_p(effect_action(conflict_sink(c)))? 'R' : 'W');
		    fprintf(fd, "<");
		    print_words(fd, effect_words_reference(effect_reference(conflict_source(c))));
		    fprintf(fd, "> - <");
		    print_words(fd, effect_words_reference(effect_reference(conflict_sink(c))));
		    fprintf(fd, ">");
		    
		    /* Additional information for EDF prettyprint. 
		       Instruction calls are given with  statement numbers
		       */
		    if (get_bool_property("PRETTYPRINT_WITH_COMMON_NAMES")) {
			if (instruction_call_p(statement_instruction(s1)))
			    fprintf(fd, " %d-%s",statement_number(s1),
				    entity_local_name(call_function(instruction_call(statement_instruction(s1)))));
			else  fprintf(fd, " %d",statement_number(s1));
			if (instruction_call_p(statement_instruction(s2)))
			    fprintf(fd, " %d-%s",statement_number(s2),
				    entity_local_name(call_function(instruction_call(statement_instruction(s2)))));
			else  fprintf(fd, " %d",statement_number(s2));
			     }
		    
		}	
		
		else {
		    fprintf(fd, "\t\tfrom ");
		    print_words(fd, words_effect(conflict_source(c)));

		    fprintf(fd, " to ");
		    print_words(fd, words_effect(conflict_sink(c)));
		}

		if(conflict_cone(c) != cone_undefined){
		    if(sru_format_p && !statement_undefined_p(mod_stat)) {
			fprintf(fd, " HERE levels(");
			MAPL(pl, {
			    fprintf(fd, pl==cone_levels(conflict_cone(c))? "%d" : ",%d",
				    INT(CAR(pl)));
			}, cone_levels(conflict_cone(c)));
			fprintf(fd, ") ");
		    }
		    else {
			fprintf(fd, " at levels ");
			MAPL(pl, {
			    fprintf(fd, " %d", INT(CAR(pl)));
			}, cone_levels(conflict_cone(c)));
			fprintf(fd, "\n");
		    }

		    if(get_bool_property
		       ("PRINT_DEPENDENCE_GRAPH_WITH_DEPENDENCE_CONES")) {
			gs = (Ptsg)cone_generating_system(conflict_cone(c));
			if (!SG_UNDEFINED_P(gs)) {
			    if(sru_format_p && !statement_undefined_p(mod_stat)) {
				if(sg_nbre_sommets(gs)==1 && sg_nbre_rayons(gs)==0
				   && sg_nbre_droites(gs)==0) {
				    /* uniform dependence */
				    fprintf(fd, "uniform");
				    fprint_lsom_as_dense(fd, sg_sommets(gs), gs->base);
				}
				else {
				    sg_fprint_as_ddv(fd, gs);
				}
			    }
			    else {
				/* sg_fprint(fd,gs,entity_local_name); */
				/* FI: almost print_dependence_cone:-( */
				sg_fprint_as_dense(fd, gs, gs->base);
				ifdebug(2) {
				    Psysteme sc1 = sc_new();
				    sc1 = sg_to_sc_chernikova(gs);
				    (void) fprintf(fd,"syst. lin. correspondant au syst. gen.:\n");
				    sc_fprint(fd,sc1,entity_local_name);
				}
			    }
			} 
		    }
		}
		else {
		    if(sru_format_p && !statement_undefined_p(mod_stat)) 
			fprintf(fd, " levels()");
		}
		fprintf(fd, "\n");
	    }
	}
    } 

    if(sru_format_p && !statement_undefined_p(mod_stat)) {
	free_persistant_statement_to_int(s_to_l);
    }
    else {
	fprintf(fd, "\n****************** End of Dependence Graph ******************\n");
    }

    debug_off();
}


/* Do not print vertices and arcs ignored by the parallelization algorithms.
 * At least, hopefully...
 */
void 
prettyprint_dependence_graph_view(FILE * fd,
				  statement mod_stat,
				  graph mod_graph)
{
    cons *pv1, *ps, *pc;
    Ptsg gs;
    int banner_number = 0;

    banner_number =
	get_bool_property("PRINT_DEPENDENCE_GRAPH_WITHOUT_PRIVATIZED_DEPS") +
	    2*get_bool_property
		("PRINT_DEPENDENCE_GRAPH_WITHOUT_NOLOOPCARRIED_DEPS") +
		    4*get_bool_property
			("PRINT_DEPENDENCE_GRAPH_WITH_DEPENDENCE_CONES");

    fprintf(fd, "%s\n", dependence_graph_banner[banner_number]);

    set_enclosing_loops_map( loops_mapping_of_statement(mod_stat) );

    debug_on("RICEDG_DEBUG_LEVEL");

    pv1 = graph_vertices(mod_graph);
    pv1 = CDR(pv1);
    /*    for (pv1 = graph_vertices(mod_graph); !ENDP(pv1); pv1 = CDR(pv1)) {*/
    {
	vertex v1 = VERTEX(CAR(pv1));
	statement s1 = vertex_to_statement(v1);
	list loops1 = load_statement_enclosing_loops(s1);

	for (ps = vertex_successors(v1); !ENDP(ps); ps = CDR(ps)) {
	    successor su = SUCCESSOR(CAR(ps));
	    vertex v2 = successor_vertex(su);
	    statement s2 = vertex_to_statement(v2);
	    list loops2 = load_statement_enclosing_loops(s2);
	    dg_arc_label dal = (dg_arc_label) successor_arc_label(su);

	    int nbrcomloops = FindMaximumCommonLevel(loops1, loops2);
	    for (pc = dg_arc_label_conflicts(dal); !ENDP(pc); pc = CDR(pc)) {
		conflict c = CONFLICT(CAR(pc));

		if(conflict_cone(c) == cone_undefined) continue;
		else {
		    cons * lls = cone_levels(conflict_cone(c));
		    cons *llsred =NIL;
		    if (get_bool_property("PRINT_DEPENDENCE_GRAPH_WITHOUT_PRIVATIZED_DEPS")){
			MAPL(pl,{
			    int level = INT(CAR(pl));
			    if (level <= nbrcomloops) {
				if (! ignore_this_conflict(v1,v2,c,level)) {
				    llsred = gen_nconc(llsred, CONS(INT, level, NIL));
				}
			    }
			    else {
				if (get_bool_property
				    ("PRINT_DEPENDENCE_GRAPH_WITHOUT_NOLOOPCARRIED_DEPS")) {
				    continue;
				}
				else llsred = gen_nconc(llsred, CONS(INT, level, NIL));
			    }
			}, lls);
		    }
		    if (llsred == NIL) continue;
		    else {
			/*if (!	entity_scalar_p(reference_variable
			  (effect_reference(conflict_source(c))))) { */

			fprintf(fd, "\t%02d --> %02d with conflicts\n", 
				statement_number(s1), statement_number(s2));
			
			fprintf(fd, "\t\tfrom ");
			print_words(fd, words_effect(conflict_source(c)));

			fprintf(fd, " to ");
			print_words(fd, words_effect(conflict_sink(c)));
			
			fprintf(fd, " at levels ");
			MAPL(pl, {
			    fprintf(fd, " %d", INT(CAR(pl)));
			}, llsred);

			fprintf(fd, "\n");
			if(get_bool_property
			   ("PRINT_DEPENDENCE_GRAPH_WITH_DEPENDENCE_CONES")) {
			    gs = (Ptsg)cone_generating_system(conflict_cone(c));
			    if (!SG_UNDEFINED_P(gs)) {
				/* sg_fprint(fd,gs,entity_local_name); */
				sg_fprint_as_dense(fd, gs, gs->base);
				ifdebug(2) {
				    Psysteme sc1 = sc_new();
				    sc1 = sg_to_sc_chernikova(gs);
				    (void) fprintf(fd,"syst. lin. correspondant au  syst. gen.:\n");
				    sc_fprint(fd,sc1,entity_local_name);
				}
			    }
			    fprintf(fd, "\n");
			}
		    }
		}
	    }
	}
    }
    clean_enclosing_loops();
    debug_off();
    fprintf(fd, "\n****************** End of Dependence Graph "
	    "******************\n");
}

void 
print_vect_in_vertice_val(fd,v,b)
FILE *fd;
Pvecteur v;
Pbase b;
{
    fprintf(fd,"(");
    for(; b!=NULL; b=b->succ) {
	fprint_string_Value(fd, " ",vect_coeff(b->var,v));
	if(b->succ !=NULL)
	    fprintf(fd,",");
    }
    fprintf(fd," )");
}

void 
print_dependence_cone(fd,dc,basis)
FILE *fd;
Ptsg dc;
Pbase basis;
{
    Psommet v;
    Pray_dte r;
    fprintf(fd,"\nDependence cone :\n");
    if (SG_UNDEFINED_P(dc)) fprintf(fd, "NULL \n");
    else {
	fprintf(fd,"basis :");
	base_fprint(fd,basis,entity_local_name);
	fprintf(fd,"%d vertice(s) :",sg_nbre_sommets(dc));
	v = sg_sommets(dc);
	for(; v!=NULL; v= v->succ) {
	    fprint_string_Value(fd, " \n\t denominator = ", v->denominateur);
	    fprintf(fd, "\t");
	    print_vect_in_vertice_val(fd,v->vecteur,basis);	
	}

	fprintf(fd,"\n%d ray(s) : ",sg_nbre_rayons(dc));
   
	for(r = sg_rayons(dc); r!=NULL; r=r->succ) 
	    print_vect_in_vertice_val(fd,r->vecteur,basis);
    
	fprintf(fd,"\n%d line(s) : ",sg_nbre_droites(dc));
   
	for(r = sg_droites(dc); r!=NULL; r=r->succ) 
	    print_vect_in_vertice_val(fd,r->vecteur,basis);
	fprintf(fd,"\n");
    }
}


/* for an improved dependence test (Beatrice Creusillet)
 *
 * The routine name says it all. Only constraints transitively connected
 * to a constraint referencing a variable of interest with a non-zero 
 * coefficient are copied from sc to sc_res.
 *
 * Input:
 *  sc: unchanged
 *  variables: list of variables of interest (e.g. phi variables of regions)
 * Output:
 *  sc_res: a newly allocated restricted version of sc
 * Temporary:
 *  sc: the pointer is modified to make debugging more interesting:-(
 *      (no impact on the value pointed to by sc on procedure entry)
 *
 * FI: I'm sceptical... OK for speed, quid of accuracy? Nonfeasibility
 * due to existencial variables is lost if these variables are not
 * transitively related to the so-called variables of interest, isn'it?
 * Well, I do not manage to build a counter example because existential
 * problems are caught by the precondition normalization. Although it is
 * not as strong as one could wish, it gets lots of stuff...
 */

Psysteme 
sc_restricted_to_variables_transitive_closure(sc, variables)
Psysteme sc;
Pbase variables;
{
    Psysteme sc_res;
    
    Pcontrainte c, c_pred, c_suiv;
    Pvecteur v;    
    
    /* cas particuliers */ 
    if(sc_rn_p(sc)){ 
	/* we return a sc_rn predicate,on the space of the input variables */ 
	sc_res = sc_rn(variables);
	return sc_res;
    }
    if (sc_empty_p(sc)) {
	/* we return an empty predicate,on the space of the input variables */ 
	sc_res = sc_empty(variables);
	return sc_res;
    } 

    /* sc has no particularity. We just scan its equalities and inequalities 
     * to find which variables are related to the PHI variables */ 
    sc = sc_dup(sc);
    base_rm(sc->base);
    sc->base = BASE_NULLE;
    sc_creer_base(sc);

    sc_res = sc_new();
    sc_res->base = variables;
    sc_res->dimension = vect_size(variables);

    while (vect_common_variables_p(sc->base, sc_res->base)) {
	
	/* equalities first */ 
	c = sc_egalites(sc);
	c_pred = (Pcontrainte) NULL; 
	while (c != (Pcontrainte) NULL) { 
	    c_suiv = c->succ; 

	    /* if a constraint is found in sc, that contains a variable 
	     * that already belongs to the base of sc_res, then this 
	     * constraint is removed from sc, added to sc_res  and its 
	     * variables are added to the base of sc_res */ 
	    if (vect_common_variables_p(c->vecteur,sc_res->base)){ 

		/* the constraint is removed from sc */ 		
		if (c_pred != (Pcontrainte) NULL) 
		    c_pred->succ = c_suiv; 
		else 
		    sc_egalites(sc) = c_suiv; 

		/* and added to sc_res */
		c->succ = (Pcontrainte) NULL;
		sc_add_egalite(sc_res, c);

		/* sc_res base is updated with the variables occuring in c */
		for(v = c->vecteur; !VECTEUR_NUL_P(v); v = v->succ) { 
		    if (vecteur_var(v) != TCST) 
			sc_base_add_variable(sc_res, vecteur_var(v));
		} 
	    } 
	    else 
		c_pred = c; 
	    c = c_suiv;
	} 
	
	/* inequalities then */
	c = sc_inegalites(sc);
	c_pred = (Pcontrainte) NULL; 
	while (c != (Pcontrainte) NULL) { 
	    c_suiv = c->succ; 

	    /* if a constraint is found in sc, that contains a variable 
	     * that already belongs to the base of sc_res, then this 
	     * constraint is removed from sc, added to sc_res  and its 
	     * variables are added to the base of sc_res */ 

	    if (vect_common_variables_p(c->vecteur,sc_res->base)){
 
		/* the constraint is removed from sc */ 		
		if (c_pred != (Pcontrainte) NULL) 
		    c_pred->succ = c_suiv; 
		else 
		    sc_inegalites(sc) = c_suiv; 

		/* and added to sc_res */
		c->succ = (Pcontrainte) NULL;
		sc_add_inegalite(sc_res, c);

		/* sc_res base is updated with the variables occuring in c */
		for(v = c->vecteur; !VECTEUR_NUL_P(v); v = v->succ) { 
		    if (vecteur_var(v) != TCST) 
			sc_base_add_variable(sc_res, vecteur_var(v));
		} 
	    } 
	    else 
		c_pred = c; 
	    c = c_suiv;
	} 
		
	/* update sc base */
	base_rm(sc->base);
	sc_base(sc) = (Pbase) NULL; 
	(void) sc_creer_base(sc); 
    } /* while */ 
    
    
    sc_rm(sc);
    sc = NULL;
        

    return(sc_res);
}

/* That's all */
