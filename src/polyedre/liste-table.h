/*
 * Le fichier a inclure pour qui veut utiliser les fonctions et macros
 * predefinies sur les listes.
 *
 * Ce fichier contient a la fois:
 *    - definitions de fonctions externes,
 *    - definitions de nouveaux types et nouvelles structures,
 *    - macros,
 *    - constantes symboliques,
 *    - etc ...
 * afin d'eviter la proliferation des .h.
 *
 * J'inclus dans le fichier d'autres fichiers .h, tels que types.h.
 * Je ne pense pas cependant qu'il soit necessaire de tenir compte
 * de ces inclusions sur plus d'un niveau dans les makefile, La raison
 * etant que ces fichiers ne seront modifies que pour des ajouts, ne necessitant
 * pas la recompilation des programmes. Le jour ou on decidera de redefinir
 * boolean comme int plutot que comme char, il faudra cependant tout
 * refaire. De toute facon je pense qu'on se plantera a plus ou moins
 * long terme, si on essaie de faire des make trop precis.
 */

/*
 * definitions de nouvelles structures et de nouveaux types
 */

typedef union {
		int	i;
		char	c;
		char *	s;
		float	f;
		char *	p; 
} Ucar;

typedef struct linearcons {
	int typecons;
	Ucar car;
	struct linearcons * cdr;
} Scons, * Pcons;

typedef struct liste {
	int typeliste;
	int nbcons;
	struct linearcons * tete;
	struct linearcons * queue;
} Sliste, * Pliste;



/*
 * definitions de fonctions
 */

/*
 * extern void l_init();
 * extern void t_init();
 * extern void l_addtete();
 * extern void l_addqueue();
 * extern Pliste l_creer();
 * 
 * extern Pcons mkcons();
 * extern anyp rmcons();
 */

/*
 * definitions de constantes symboliques
 */

#include "GENPGM_TAGS.h"

/*
 * definitions de macros
 * =====================
 *									    
 * Ces macros sont a utiliser en association avec l'outil defini par Remi   
 * TRIOLET pour la definition et l'utilisation de domaines construits a     
 * partir de sous-domaines liste et table.				    
 *									    
 * Les macros disponibles sont les suivantes :				    
 *									    
 *  I) Pour les listes :						    
 *     """""""""""""""							    
 *									    
 *	L_GOOVER(pcons_deb,pred_cons,curr_cons,succ_cons)		    
 *		parcours de la liste debutant au Pcons pcons_deb; curr_cons 
 *		est le pointeur Pcons courant, pred_cons le pointeur Pcons  
 *		precedent et succ_cons le pointeur Pcons suivant. Ces trois 
 *		pointeurs doivent etre declares par l'utilisateur.          
 *		Cette macro ouvre un bloc qui devra etre complete par la    
 *		macro L_END_GOOVER.					    
 *		Au premier tour dans la boucle, pred_cons n'a pas de valeur 
 *		definie.						    
 *	L_FOREACH(pcons_deb, curr_cons)					    
 *		idem L_GOOVER mais sans acces au precedent ni au suivant    
 *									    
 *	L_END_GOOVER							    
 *		fin de la boucle L_GOOVER				    
 *									    
 *	L_END_FOREACH							    
 *		fin de la boucle L_FOREACH				    
 *									    
 *	CONSVAL(pcons,cast)						    
 *		valeur de l'enregistrement courant pointe par pcons et de   
 *		type cast.						    
 *									    
 *	L_INS_AFTER(sliste,pcons,q,cast)				    
 *		insertion apres le pointeur pcons d'un enregistrement de    
 *		valeur q. Le type de q est suppose etre celui de cast.	    
 *		ATTENTION : Les pointeurs de parcours de boucle ne sont     
 *		            pas mis a jour - cf remarque 1 - 		    
 *									    
 *	L_INS_HEAD(sliste,q,cast)					    
 *		insertion en tete de liste d'un enregistrement de valeur q. 
 *		Le type de q est suppose etre celui de cast.	            
 *		ATTENTION : Les pointeurs de parcours de boucle ne sont     
 *		            pas mis a jour - cf remarque 1 - 		    
 *									    
 *	L_CHAIN_AFTER(sliste,pcons,pc)					    
 *		insertion apres le pointeur pcons de l'enregistrement 	    
 *		pointe par le pointeur (Pcons) pc.			    
 *		ATTENTION : Les pointeurs de parcours de boucle ne sont     
 *		            pas mis a jour - cf remarque 1 -		    
 *									    
 *	L_CHAIN_HEAD(sliste,pc)						    
 *		insertion en tete de liste de l'enregistrement pointe par   
 *		le pointeur (Pcons) pc.					    
 *		ATTENTION : Les pointeurs de parcours de boucle ne sont     
 *		            pas mis a jour - cf remarque 1 -		    
 *									    
 *	L_REMOVE_CURR(sliste,pcons,ccons,q,cast)			    
 *		destruction de l'enregistrement pointe par ccons, successeur
 *		de pcons. Sa valeur est sauvegardee dans q. 		    
 *		Le type de q est cast. Si ccons ou pcons sont NULL, le 	    
 *		programme s'arrete avec un message d'erreur.		    
 *		ATTENTION : Les pointeurs de parcours de boucle ne sont     
 *		            pas mis a jour - cf remarque 2 -		    
 *									    
 *	L_REMOVE_HEAD(sliste,q,cast)				    
 *		destruction de la tete de liste. Sa valeur est sauvegardee  
 *		dans q dont le type est cast.
 *		ATTENTION : Les pointeurs de parcours de boucle ne sont     
 *		            pas mis a jour - cf remarque 3 -		    
 *									    
 *	L_EXTRACT_CURR(sliste,pcons,currcons,pc)			    
 *		extraction de l'enregistrement suivant pcons. Le pointeur   
 *		pc sauvegarde cet enregistrement. La valeur de currcons,    
 *		pointeur courant dans un parcours de boucle, devient pcons  
 *		Le pointeur predecesseur du pointeur courant de parcours    
 *		de boucle reste inchange.				    
 *		ATTENTION : Les pointeurs de parcours de boucle ne sont     
 *		            pas mis a jour - cf remarque 2 -		    
 *									    
 *	L_EXTRACT_HEAD(sliste,pc)					    
 *		extraction de l'enregistrement tete de liste. Le pointeur   
 *		pc sauvegarde cet enregistrement.			    
 *		ATTENTION : Les pointeurs de parcours de boucle ne sont     
 *		            pas mis a jour - cf remarque 3 -		    
 *									    
 *	L_EMPTY(sliste)							    
 *		vrai si la liste d'entete sliste est vide, faux sinon       
 *									    
 *	L_IS_HEAD(sliste,pcons)						    
 *		vrai si pcons pointe sur la tete de liste, faux sinon	    
 *									    
 *	L_IS_TAIL(sliste,pcons)						    
 *		vrai si pcons pointe sur la fin de la liste, faux sinon     
 *									    
 *	L_HEAD(sliste)							    
 *		renvoie le pointeur de tete de liste			    
 *									    
 *	L_TAIL(sliste)							    
 *		renvoie le pointeur de queue de liste			    
 *									    
 *	L_NEXT(pcons)							    
 *		renvoie le pointeur suivant (Pcons) pcons.		    
 *									    
 *	L_APPEND(sliste_deb,sliste_fin)					    
 *		concatene deux listes de meme type. Le resultat est la liste
 *		sliste_deb. La liste sliste_fin n'est pas modifiee et est   
 *		donc partagee avec la liste concatenee sliste_deb.	    
 *									    
 *	Remarques :							    
 *	~~~~~~~~~							    
 *	   1. Apres une insertion, le pointeur succ du L_GOOVER n'a pas la  
 *	      valeur curr -> cdr. Si l'utilisateur veut utiliser cette      
 *	      valeur, il doit, apres la macro d'insertion, mettre la macro  
 *	      suivante : succ = L_NEXT(curr). Voyons l'exemple suivant :    
 *									    
 *		- avant insertion					    
 *			+-----+	   +-----+				    
 *		... --->|e1| -+--->|e2| -+---> ...			    
 *			+-----+	   +-----+				    
 *			   ^	      ^					    
 *			   |	      |					    
 *			 curr	     succ				    
 *									    
 *		- apres insertion avec L_CHAIN_AFTER (..., curr, e0,...)    
 *			+-----+	   +-----+    +-----+			    
 *		... --->|e1| -+--->|e0| -+--->|e2| -+---> ...		    
 *			+-----+	   +-----+    +-----+			    
 *			   ^	      ^		 ^			    
 *			   |	      |  	 |			    
 *			 curr	  L_NEXT(curr)  succ			    
 *									    
 *	      Si cette insertion se fait dans plusieurs boucles imbriquees  
 *	      et si l'utilisateur veut utiliser les differents succ apres   
 *	      l'insertion, il faut egalement les remettre a jour.    	     
 *									    
 *	   2. Dans le cas d'un L_REMOVE_CURR ou d'un L_EXTRACT_CURR c'est le
 *	      pointeur pred qui n'est plus tel que pred -> cdr = curr. En   
 *	      effet on a pred = curr.					     
 *									    
 *		- avant suppression					    
 *			+-----+	   +-----+				    
 *		... --->|e1| -+--->|e2| -+---> ...			    
 *			+-----+	   +-----+				    
 *			   ^	      ^					    
 *			   |	      |					    
 *			 pred	     curr				    
 *									    
 *		- apres suppression avec L_REMOVE_CURR (..., pred, curr,...)
 *			+-----+	   					    
 *		... --->|e1| -+---> ...		    			    
 *			+-----+	   					    
 *			   ^ ---------+					    
 *			   |	      |   				    
 *			 pred	     curr				    
 *									    
 *	   3. Apres un L_REMOVE_HEAD ou un L_EXTRACT_HEAD si le pointeur    
 *	      etait la tete de liste, il n'est plus positionne sur la    
 *	      liste et sa valeur est indefinie.
 *									    
 *		- avant suppression					    
 *			+-----+	   +-----+    +-----+			    
 *			|TETE-+--->|e1| -+--->|e2| -+---> ...		    
 *			+-----+	   +-----+    +-----+			    
 *			   	      ^		 ^			    
 *			   	      |		 |			    
 *			 	     curr      succ			    
 *									    
 *		- apres suppression avec L_REMOVE_HEAD (....)	    
 *			+-----+	   +-----+				    
 *			|TETE-+--->|e2| -+---> ...    			    
 *			+-----+	   +-----+				    
 *			              ^					    
 *			    	      |   				    
 *			 curr(?)     succ				    
 *									    
 *	   4. Dans le cas d'un if(...) MACRO else ..., il est dangereux de  
 *	      mettre un point virgule apres la macro MACRO. En effet, si la 
 *	      macro MACRO est un bloc {...}, le code deviendra, apres la    
 *	      passe prepocessing : if(...) {...}; else ... ce qui est une   
 *	      erreur de syntaxe C. Voici la liste des macros qui sont un    
 *	      bloc {...} :						    
 *		L_GOOVER  ... L_END_GOOVER				    
 *		L_FOREACH  ... L_END_FOREACH				    
 *		L_INS_AFTER						    
 *		L_INS_HEAD						    
 *		L_CHAIN_AFTER						    
 *		L_CHAIN_HEAD						    
 *		L_REMOVE_CURR						    
 *		L_REMOVE_HEAD						    
 *		L_EXTRACT_CURR						    
 *		L_EXTRACT_HEAD						    
 *		L_APPEND						    
 *									    
 *									    
 *  II) Pour les tables :						    
 *      """""""""""""""							    
 *									    
 *	T_GOOVER( index, table, max_ind, curr_cons)			    
 *		parcours de la table de Pcons "table" avec l'indice "index" 
 *		jusqu'a index = max_ind. Le Pcons curr_cons contient le     
 *		Pcons courant i.e.table[index].Seuls les elements differents
 *		a (Pcons )NULL sont accessibles dans le corps de la boucle. 
 *									    
 *	T_END_GOOVER							    
 *		fin de la boucle T_GOOVER				    
 *									    
 *	T_FOREACH( index, table, max_ind, curr_cons)			    
 *		meme chose que T_GOOVER mais tous les elements, meme = NULL,
 *		sont accessibles dans le corps de la boucle.		    
 *									    
 *	T_END_FOREACH							    
 *		fin de la boucle T_FOREACH				    
 *									    
 *	T_WHILE(index, table, max_ind, curr_cons,test)			    
 *		meme chose que T_GOOVER mais arret si !test		    
 *									    
 *	T_END_WHILE							    
 *		fin de la boucle T_WHILE				    
 *									    
 *	T_INS_END(table,max, q, typcons,place)				    
 *		rajout d'un element de valeur q dans la table (de taille max
 *		a la premiere place libre. Le type du cons cree est typcons.
 *		Le numero de la place est renvoye dans place.		    
 *									    
 */



#define L_EMPTY(sliste)	((sliste).nbcons == 0)
 

#define L_IS_HEAD(sliste,pcons)	((sliste).tete == (pcons))

#define L_IS_TAIL(sliste,pcons)	((sliste).queue == (pcons))

#define L_TAIL(sliste) (sliste).queue

#define L_HEAD(sliste) (sliste).tete

#define L_NEXT(pcons) (pcons) -> cdr

#define L_GOOVER(pcons_deb,pred_cons,curr_cons,succ_cons)		\
for(curr_cons = pcons_deb; curr_cons != (Pcons )NULL; 			\
    pred_cons = curr_cons, curr_cons = succ_cons) {			\
	succ_cons = curr_cons -> cdr;

#define L_FOREACH(pcons_deb, curr_cons)					\
for(curr_cons = pcons_deb; curr_cons != (Pcons )NULL; 			\
    curr_cons = curr_cons -> cdr) {

#define L_END_GOOVER }

#define L_END_FOREACH }

#define CONSVAL(pcons,cast)						\
	(* ((cast *)&((pcons) -> car)))			
		
#define L_INS_AFTER(sliste,pcons,q,cast)				\
	{								\
		Pcons p_tmp ;						\
									\
		p_tmp = (Pcons )mkcons((anyp )NULL,(sliste).typeliste);	\
		CONSVAL(p_tmp,cast) = q;				\
		p_tmp -> cdr = (pcons) -> cdr;				\
		(pcons) -> cdr = p_tmp;					\
		(sliste).nbcons ++ ;					\
		if(L_IS_TAIL(sliste,pcons))				\
			(sliste).queue = p_tmp;				\
	}

#define L_INS_HEAD(sliste,q,cast)					\
	{								\
		Pcons p_tmp ;						\
									\
		p_tmp = (Pcons) mkcons((anyp )NULL, (sliste).typeliste);\
		CONSVAL(p_tmp,cast) = q;				\
		p_tmp -> cdr = (sliste).tete;				\
		(sliste).tete = p_tmp;					\
		(sliste).nbcons ++ ;					\
		if((sliste).nbcons <= 1)				\
			(sliste).queue = p_tmp;				\
	}

#define L_CHAIN_AFTER(sliste,pcons,pc)					\
	{								\
		if(pc == (Pcons ) NULL)					\
		errexit("macro : L_CHAIN_AFTER\n param. 3 = NULL\n");	\
		if(pcons == (Pcons ) NULL)				\
		errexit("macro : L_CHAIN_AFTER\n param. 2 = NULL\n");	\
		pc -> cdr = (pcons) -> cdr;				\
		(pcons) -> cdr = pc;					\
		(sliste).nbcons ++ ;					\
		if(L_IS_TAIL(sliste,pcons))				\
			(sliste).queue = pc;				\
	}

#define L_CHAIN_HEAD(sliste,pc)						\
	{								\
		if(pc == (Pcons ) NULL)					\
		errexit("macro : L_CHAIN_HEAD\n param. 2 = NULL\n");	\
		pc -> cdr = (sliste).tete;				\
		(sliste).tete = pc;					\
		(sliste).nbcons ++ ;					\
		if((sliste).nbcons <= 1)				\
			(sliste).queue = pc;				\
	}
	
#define L_REMOVE_CURR(sliste,pcons,currcons,q,cast)			\
	{								\
		Pcons p_tmp;						\
									\
		if((pcons) == (Pcons ) NULL)				\
 		errexit("macro : L_REMOVE_CURR\n param. 3 = NULL\n");	\
		p_tmp = (pcons) -> cdr;					\
		if(p_tmp == (Pcons ) NULL)				\
		errexit("macro : L_REMOVE_CURR\n  fin de liste\n") ;	\
		q = CONSVAL(p_tmp,cast);				\
		(pcons) -> cdr = p_tmp -> cdr;				\
		currcons = pcons;					\
		FREE(p_tmp, BLK_CONS,"macro : L_REMOVE_CURR");		\
		(sliste).nbcons -- ;					\
		if(pcons -> cdr == (Pcons)NULL)				\
			(sliste).queue = pcons;				\
	}


#define L_REMOVE_HEAD(sliste,q,cast)				\
	{								\
		Pcons p_tmp = (sliste).tete;				\
									\
		if(p_tmp == (Pcons ) NULL)				\
		errexit("macro : L_REMOVE_HEAD\n  liste vide");		\
	/*	if(L_IS_HEAD(sliste,curr)) curr = L_NEXT(curr);	*/	\
		q = CONSVAL(p_tmp,cast);				\
		(sliste).tete = p_tmp -> cdr;				\
		FREE(p_tmp, BLK_CONS, "macro : L_REMOVE_HEAD");		\
		(sliste).nbcons -- ;					\
		if((sliste).nbcons == 0)				\
			(sliste).queue = (Pcons )NULL;			\
	}

#define L_EXTRACT_CURR(sliste,pcons,currcons,pc)			\
	{								\
		Pcons p_tmp;						\
									\
		if((pcons) == (Pcons ) NULL)				\
		errexit("macro : L_EXTRACT_CURR\n param. 3 = NULL\n");	\
		p_tmp = (pcons) -> cdr;					\
		if(p_tmp == (Pcons ) NULL)				\
		errexit("macro : L_EXTRACT_CURR\n  fin de liste\n") ;	\
		pc = p_tmp;						\
		(pcons) -> cdr = p_tmp -> cdr;				\
		currcons = pcons;					\
		(sliste).nbcons -- ;					\
		if(pcons -> cdr == (Pcons)NULL)				\
			(sliste).queue = p_cons;			\
	}

#define L_EXTRACT_HEAD(sliste,pc)					\
	{								\
		Pcons p_tmp = (sliste).tete;				\
									\
		if(p_tmp == (Pcons ) NULL)				\
		errexit("macro : L_EXTRACT_HEAD\n  liste vide");	\
		(sliste).tete = p_tmp -> cdr;				\
		pc = p_tmp;						\
		(sliste).nbcons -- ;					\
		if((sliste).nbcons == 0)				\
			(sliste).queue = (Pcons )NULL;			\
	}
		

#define L_APPEND(sliste_deb,sliste_fin)					\
	{								\
		/*							\
		if((sliste_deb).typeliste != (sliste_fin).typeliste)	\
		errexit("macro : L_APPEND\n types des listes differents");\
		*/							\
		if((sliste_deb).nbcons == 0) 				\
			(sliste_deb) = (sliste_fin);			\
		else {							\
			(sliste_deb).queue -> cdr = (sliste_fin).tete;  \
			(sliste_deb).queue = (sliste_fin).queue;	\
			(sliste_deb).nbcons += (sliste_fin).nbcons;	\
		}							\
	}
			
#define T_GOOVER( ind, table, max_ind,curr_cons)			\
	for( ind = 0; ind < max_ind; ind ++)				\
		if((curr_cons=table[ind]) != (Pcons )NULL) {

#define T_END_GOOVER }

#define T_FOREACH( ind, table, max_ind,curr_cons)			\
	for( ind = 0; ind < max_ind; ind ++ ) {				\
		curr_cons = table[ind];
	
#define T_END_FOREACH }

#define T_WHILE(ind, table, max_ind, curr_cons, test)			\
	for (ind=0, curr_cons=table[0]; (test) && (ind<max_ind); ind ++)\
		if((curr_cons=table[ind]) != (Pcons )NULL) {

#define T_END_WHILE }

#define T_INS_END(table, max, q, typcons,pl)				\
	{								\
		Pcons p = mkcons((anyp )q,typcons);			\
		int tinsend;						\
		for(tinsend=0; tinsend<max; tinsend++)			\
		  if(table[tinsend] == (Pcons)NULL) {			\
		    table[tinsend] = p;					\
		    pl = tinsend;					\
		    break;						\
		  }							\
		if(tinsend == max) {					\
		fprintf(stderr,"macro T_INS_END : table deja pleine\n");\
		exit(1);						\
		}							\
	}
