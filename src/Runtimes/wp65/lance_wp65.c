#include <stdio.h>
#include <setjmp.h>

#include "pvm3.h"

#include "genC.h"
#include "misc.h"

#define MAXSIZEOFPROCS 8

#define IF_DEBUG(niveau, code) if (get_debug_level() >= niveau) { code; }

int numero, banc, mytid, tids[MAXSIZEOFPROCS], limitecalcul;

int nb_taches = MAXSIZEOFPROCS;

/* Mis là simplement pour éviter une erreur au link. S'attendre donc à
   des surprises... */
jmp_buf pips_top_level;

/*
   Routine qui affiche un message d'erreur si valeur de retour < 0 :
*/
void testerreur(char *chaine, int retour)
{
  if (retour < 0) {
    pvm_perror(chaine);
    exit(1);
  }
}


main(int argc, char *argv[])
{      
  int i, nb_t;
  char *argv_fils[3];
  char chaine[100];

  if (argc == 2)
    /* Suppose que c'est le debug_level passé aux fils */
    set_debug_level(atoi(argv[1]));
  else {
    /* Sinon, on le récupère dans l'environnement : */
    debug_on("LANCEWP65_DEBUG_LEVEL");
    sprintf(chaine, "%d", get_debug_level());
    argv_fils[0] = argv[0];
    argv_fils[1] = chaine;
    argv_fils[2] = (char *) NULL;
    argv = argv_fils;
  }
    
      
  mytid = pvm_mytid();
  tids[0] = pvm_parent();

  if (tids[0] < 0) {
    /* Je suis le contro^leur de de'part... */
    nb_t = pvm_spawn("lancewp65", argv,
		     PvmTaskDebug, 
		     "*", 
		     nb_taches - 1, 
		     &tids[1]);
         
    if (nb_t != nb_taches - 1)
      testerreur("lancewp65 : Incapable de lancer les ta^ches",
		 nb_t - (nb_taches - 1));

    tids[0] = mytid;
    testerreur("pvm_initsend",
	       pvm_initsend(PvmDataDefault));
    testerreur("pvm_pkint",
	       pvm_pkint(&tids[1], nb_taches - 1, 1));
      /*    Envoie le vecteur de tids a` tout le monde : */
      testerreur("pvm_mcast",
		 pvm_mcast(&tids[1], nb_taches - 1, 0));
    /*     le contro^leur a le nume'ro 0 : */
    numero = 0;
  }
  else {
    /* Je suis un processeur de banc ou de calcul lance' par le
       contro^leur... */
    testerreur("pvm_recv",
	       pvm_recv(tids[0], 0));
    testerreur("pvm_unpack",
	       pvm_upkint(&tids[1], nb_taches - 1, 1));
         
  }
      
  if (get_debug_level() >= 2)
    for(i = 0; i <= nb_taches - 1; i++)
      fprintf(stderr,"PE %d a un tid %d\n", i, tids[i]);
      
  /*     On est maintenant tous e'quivalents :
	 Cherche le nume'ro d'ordre : */
  for(i = 1; i <= nb_taches; i++)
    if (mytid == tids[i])
      /*     J'ai trouve' mon nume'ro d'ordre : */
      numero = i;

  /*     Reste maintenant a` se partager en noeud de calcul ou de me'moire : */
  limitecalcul = nb_taches/2;

  if (numero <= limitecalcul) {
    debug(2, "lancewp65",
	  "Je suis le PE %d\n", numero);
    WP65_(&numero);
  }
  else {
    banc = numero - limitecalcul;
    debug(2, "lancewp65",
	  "Je suis le banc %d\n", banc);
    BANK_(&banc);
  }

  debug_off();
  exit(0);
}

/*     Fait des conversions : */

void send_4(int tid, float *donnee, int taille)
{
  testerreur("pvm_initsend",
	     pvm_initsend(PvmDataDefault));
  testerreur("pvm_pack",
	     pvm_pkfloat(donnee, taille, 1));
  debug(5, "send_4",
	"pvm_send vers tid 0%o\n", tid);
  testerreur("pvm_send",
	     pvm_send(tid, 0));
}


void receive_4(int tid, float *donnee, int taille)
{
  testerreur("pvm_recv",
	     (int) pvm_recv(tid, 0));
  testerreur("pvm_unpack",
	     pvm_upkfloat(donnee, taille, 1)); 
}


void BANK_SEND_4_(int *proc_id, float *donnee, int *taille)
{
  debug(4, "BANK_SEND_4",
	"Envoi de banc %d -> PE %d, taille = %d\n", banc, proc_id, taille);
  send_4(tids[*proc_id], donnee, *taille);
}

      
void BANK_RECEIVE_4_(int *proc_id, float *donnee, int *taille)
{
  debug(4, "BANK_RECEIVE_4",
	"Re'ception de banc %d -> PE %d, taille = %d\n",
	banc, *proc_id, *taille);
  receive_4(tids[*proc_id], donnee, *taille);
}


void WP65_SEND_4_(int *bank_id, float *donnee, int *taille)
{
  debug(4, "WP65_SEND_4",
	"Envoi de PE %d -> banc %d, taille = %d\n",
	numero, *bank_id, *taille);
  send_4(tids[*bank_id + limitecalcul], donnee, *taille);
}


void WP65_RECEIVE_4_(int *bank_id, float *donnee, int *taille)
{
  debug(4, "WP65_RECEIVE_4",
	"Re'ception de PE %d -> banc %d, taille = %d\n",
	numero, *bank_id, *taille);
  receive_4(tids[*bank_id + limitecalcul], donnee, *taille);
}


/* Horreur pour que la bibliothèque F77 soit contente : */
void MAIN_()
{
  fprintf("Entrée dans MAIN_ !\n");
}
