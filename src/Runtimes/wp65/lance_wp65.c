#include <stdio.h>
#include <stdlib.h>
#include <setjmp.h>

#include "pvm3.h"

#include "genC.h"
#include "misc.h"

#define MAXSIZEOFPROCS 8

#define ENV_DEBUG "LANCEWP65_DEBUG_LEVEL"

#define IF_DEBUG(niveau, code) if (get_debug_level() >= niveau) { code; }

int numero, banc, mytid, tids[MAXSIZEOFPROCS], limitecalcul;

int nb_taches = MAXSIZEOFPROCS;

char chaine_nulle = '\0';
char chaine[1000];
char var[100];
char valeur[900];

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
  char *argv_fils[3], *debut_basename, *p, *chaine_env;
  int niveau_debug_pvm;

  if (argc == 2)
    /* Suppose que c'est le debug_level passé aux fils */
    set_debug_level(atoi(argv[1]));
  else {
    /* Sinon, on le récupère dans l'environnement : */
    debug_on(ENV_DEBUG);
    sprintf(chaine, "%d", get_debug_level());
    argv_fils[0] = argv[0];
    argv_fils[1] = chaine;
    argv_fils[2] = (char *) NULL;
    argv = argv_fils;
  }
    
/* set_debug_level(2); */
      
  mytid = pvm_mytid();
  tids[0] = pvm_parent();

  if (tids[0] < 0) {
    /* Je suis le contro^leur de de'part... */
    debug_on(ENV_DEBUG);
    /* On va lancer l'exécutable de me^me nom comme fils : */
    debut_basename = p = argv[0];
    while(*p != '\0') {
      if (*p == '/') debut_basename = p + 1;
	/* Dans UNIX, un nom d'exécutable ne peut terminer par un "/". */
      p++;
    }

    if (get_debug_level() >= 3)
      niveau_debug_pvm = PvmTaskDebug;
    else
      niveau_debug_pvm = PvmTaskDefault;

    nb_t = pvm_spawn(debut_basename, argv,
		     niveau_debug_pvm, 
		     "*", 
		     nb_taches - 1, 
		     &tids[1]);
         
    if (nb_t != nb_taches - 1)
      testerreur("main : Incapable de lancer les ta^ches",
		 nb_t - (nb_taches - 1));

    tids[0] = mytid;


    /* Envoie des variables d'environnement a` tout les fils : */
    chaine_env = getenv(ENV_DEBUG);
    testerreur("pvm_initsend",
	       pvm_initsend(PvmDataDefault));

    if (chaine_env != NULL) {
    testerreur("pvm_pkstr",
	       pvm_pkstr(ENV_DEBUG));
    testerreur("pvm_pkstr",
	       pvm_pkstr(chaine_env));
    }
    testerreur("pvm_pkstr",
	       pvm_pkstr(&chaine_nulle));
    testerreur("pvm_mcast",
	       pvm_mcast(&tids[1], nb_taches - 1, 0));


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

    /* Récupère des variables d'environnement passées par le père : */
    testerreur("pvm_recv",
	       pvm_recv(tids[0], 0));
    for(;;) {
      testerreur("pvm_upkstr",
		 pvm_upkstr(var));
      if (var[0] == '\0')
	/* Pas ou plus de variable. */
	break;
      testerreur("pvm_upkstr",
		 pvm_upkstr(valeur));
      (void) sprintf(chaine, "%s=%s", var, valeur);
      testerreur("putenv",
		 -putenv(chaine));
    }

    debug_on(ENV_DEBUG);

    testerreur("pvm_recv",
	       pvm_recv(tids[0], 0));
    testerreur("pvm_upkint",
	       pvm_upkint(&tids[1], nb_taches - 1, 1));
  }
      
  if (get_debug_level() >= 2)
    for(i = 0; i <= nb_taches - 1; i++)
      fprintf(stderr,"PE %d a un tid 0x%x\n", i, tids[i]);
      
  /*     On est maintenant tous e'quivalents :
	 Cherche le nume'ro d'ordre : */
  for(i = 1; i <= nb_taches; i++)
    if (mytid == tids[i])
      /*     J'ai trouve' mon nume'ro d'ordre : */
      numero = i;

  /*     Reste maintenant a` se partager en noeud de calcul ou de me'moire : */
  limitecalcul = nb_taches/2;

  if (numero < limitecalcul) {
    debug(2, "main",
	  "Je suis le PE %d de tid 0x%x\n", numero, mytid);
    WP65_(&numero);
  }
  else {
    banc = numero - limitecalcul;
    debug(2, "main",
	  "Je suis le banc %d de tid 0x%x\n", banc, mytid);
    BANK_(&banc);
  }

  debug_off();
  pvm_exit();
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
	"pvm_send vers tid 0x%x\n", tid);
  testerreur("pvm_send",
	     pvm_send(tid, 0));
}


void receive_4(int tid, float *donnee, int taille)
{
  debug(5, "receive_4",
	"pvm_recv depuis le tid 0x%x\n", tid);
  testerreur("pvm_recv",
	     (int) pvm_recv(tid, 0));
  testerreur("pvm_unpack",
	     pvm_upkfloat(donnee, taille, 1)); 
}


void BANK_SEND_4_(int *proc_id, float *donnee, int *taille)
{
  debug(4, "BANK_SEND_4",
	"Envoi de banc %d -> PE %d, taille = %d\n", banc, *proc_id, *taille);
  send_4(tids[*proc_id], donnee, *taille);
}

      
void BANK_RECEIVE_4_(int *proc_id, float *donnee, int *taille)
{
  debug(4, "BANK_RECEIVE_4",
	"Réception de banc %d <- PE %d, taille = %d\n",
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
	"Réception de PE %d <- banc %d, taille = %d\n",
	numero, *bank_id, *taille);
  receive_4(tids[*bank_id + limitecalcul], donnee, *taille);
}


/* Horreur pour que la bibliothèque F77 soit contente : */
void MAIN_()
{
  fprintf("Entrée dans MAIN_ !\n");
}
