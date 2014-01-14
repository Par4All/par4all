/*

  $Id$

  Copyright 1989-2014 MINES ParisTech

  This file is part of PIPS.

  PIPS is free software: you can redistribute it and/or modify it
  under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  any later version.

  PIPS is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.

  See the GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with PIPS.  If not, see <http://www.gnu.org/licenses/>.

*/
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <setjmp.h>
#include <stdarg.h>

#include "pvm3.h"

#include "genC.h"
#include "misc.h"

/* G77 adds 2 underscores to function name which has 1 _ soemewhere
 * and only one to functions that have no underscores at all.
 * I just cannot see the rationale, but this is not the point:-)
 * FC, 30/11/95
 */
#ifdef COMPILE_FOR_G77
#define FUNCTION_(name) name##__
#define FUNCTION(name) name##_
#else
#define FUNCTION_(name) name##_
#define FUNCTION(name) name##_
#endif

extern void FUNCTION(wp65)(int*);
extern void FUNCTION(bank)(int*);

/* not found in any header file */
extern int gethostname(char*, int);

/* Dans #include "wp65.h" : */
extern void get_model(int *ppn, int *pbn, int *pls);

/*
   #define MAXSIZEOFPROCS 8
   #define MAXSIZEOFCONNECTED (MAXSIZEOFPROCS/2)
*/

#define ENV_DEBUG "LANCEWP65_DEBUG_LEVEL"

#define IF_DEBUG(niveau, code) if (get_debug_level() >= niveau) { code; }

#define WP65_PVM_DATA PvmDataRaw

int numero, banc, mytid, pere;

/* Retient la « topologie » de la machine : */
int *tids;

/* Pour permettre la réception d'un paquet en plusieurs morceaux,
   retient des tampons pour chaque PE (en fait si on considère que
   dans WP65 un PE ne cause jamais à un autre PE, idem pour les bancs
   mémoire). */
/* On assume le fait que des données empaquetées en masse peuvent être
   dépaquetées en plusieurs fois dans un tampon PVM. */
#define TAMPON_VIDE -1
int *bufid, *taille_restante;

/* Rajoute une estampille sur les paquets : */
int *estampille;

int nb_taches;
int nb_procs;
int nb_bancs;
int max_bancs_procs;
/* Inutilisée : */
int largeur_banc;

char chaine_nulle = '\0';
char chaine[1000];
char var[100];
char valeur[900];
char machine[100];

char *groupe_final = "Rendez-vous au point d'orgue";

/* Mis là simplement pour éviter une erreur au link. S'attendre donc à
   des surprises... :-) */
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


void sort_erreur(char * format, ...)
{
  va_list args;
  
  va_start(args, format);
  (void) vfprintf(stderr, format, args);
  abort();
  va_end(args);
}


/* Affiche quelque chose dans l'entête d'un xterm et son icône :*/
void affiche_entete_X(char * format, ...)
{
  va_list args;
  char chaine[1000];

  va_start(args, format);
  (void) vsprintf(chaine, format, args);
  (void) fprintf(stderr, "]0;%s", chaine);
  va_end(args);
}


char *basename(char *chaine)
{
  char *debut_basename, *p;
  
  debut_basename = p = chaine;
  while(*p != '\0') {
    if (*p == '/') debut_basename = p + 1;
    /* Dans UNIX, un nom d'exécutable ne peut terminer par un "/". */
    p++;
  }
  return debut_basename;
}

void init_params()
{
  get_model(&nb_procs, &nb_bancs, &largeur_banc);
}


void envoie_params()
{
  testerreur("pvm_pkint",
	     pvm_pkint(&nb_procs, 1, 1));
  testerreur("pvm_pkint",
	     pvm_pkint(&nb_bancs, 1, 1));
  testerreur("pvm_pkint",
	     pvm_pkint(&largeur_banc, 1, 1));
}


void recoit_params()
{
  testerreur("pvm_upkint",
	     pvm_upkint(&nb_procs, 1, 1));
  testerreur("pvm_upkint",
	     pvm_upkint(&nb_bancs, 1, 1));
  testerreur("pvm_upkint",
	     pvm_upkint(&largeur_banc, 1, 1));
}


void init_variables()
{
  int i;

  nb_taches = nb_procs + nb_bancs;
  max_bancs_procs = (nb_procs > nb_bancs) ? nb_procs : nb_bancs;
  tids = (int *) calloc(nb_taches, sizeof(int));
  bufid = (int *) calloc(max_bancs_procs, sizeof(int));
  taille_restante = (int *) calloc(max_bancs_procs, sizeof(int));
  /* calloc met à 0 : */
  estampille = (int *) calloc(2*nb_procs*nb_bancs, sizeof(int));
  
  for(i = 0; i < max_bancs_procs; i++) {
    bufid[i] = TAMPON_VIDE;
    taille_restante[i] = 0;
  }

  testerreur("gethostname",
	     gethostname(machine, sizeof(machine) - 1));
}


/* Rajoute une sorte d'horloge pour chaque liaison : */
/* direction = 0 en réception, 1 en émission. */
int estampiller(int PE, int banc, int taille, int direction)
{
  return estampille[(direction*nb_procs + PE)*nb_bancs + banc] += taille;
}


int main(int argc, char *argv[])
{      
  int i, nb_t;
  char *argv_fils[3], *chaine_env;
  int niveau_debug_pvm;
  
  mytid = pvm_mytid();
  pere = pvm_parent();

  /* Semble planter si on fait un pvm_exit avant que les paquets
     soient réellement partis. Rajout d'une barrière de synchro à la
     fin pour cela... */
  /* Augmente l'efficacité des communications : */
  testerreur("pvm_setopt",
	     pvm_setopt(PvmRoute, PvmRouteDirect));
  
  if (pere < 0) {
    /* Je suis le contrôleur de départ... */
    debug_on(ENV_DEBUG);
    init_params();
    init_variables();
    tids[0] = mytid;
    
    /* On va lancer l'exécutable de même nom comme fils : */
    argv_fils[0] = basename(argv[0]);
    argv_fils[1] = (char *) NULL;
    argv_fils[2] = (char *) NULL;
    /* Si on est sous X11 on passe le DISPLAY comme premier argument.
       Magouille sordide pour faire marcher le debugger légèrement
       modifié... */
    chaine_env = getenv("DISPLAY");
    if (chaine_env != NULL)
      argv_fils[1] = chaine_env;

    if (get_debug_level() >= 3)
      niveau_debug_pvm = PvmTaskDebug;
    else
      niveau_debug_pvm = PvmTaskDefault;

    nb_t = pvm_spawn(argv_fils[0], argv_fils,
		     niveau_debug_pvm, 
		     "*", 
		     nb_taches - 1, 
		     &tids[1]);
         
    if (nb_t != nb_taches - 1)
      testerreur("main : Incapable de lancer les tâches",
		 nb_t - (nb_taches - 1));

    testerreur("pvm_initsend",
	       pvm_initsend(WP65_PVM_DATA));
  
    /* Envoi du model.rc de la machine : */
    envoie_params();
    
    /* Envoi de certaines variables d'environnement à tous les fils : */
    chaine_env = getenv(ENV_DEBUG);
    if (chaine_env != NULL) {
      testerreur("pvm_pkstr",
		 pvm_pkstr(ENV_DEBUG));
      testerreur("pvm_pkstr",
		 pvm_pkstr(chaine_env));
    }
    testerreur("pvm_pkstr",
	       pvm_pkstr(&chaine_nulle));

    testerreur("pvm_pkint",
	       pvm_pkint(&tids[1], nb_taches - 1, 1));

    /* Envoie le vecteur de tids à tout le monde : */
    testerreur("pvm_mcast",
	       pvm_mcast(&tids[1], nb_taches - 1, 0));
    /*     le contrôleur a le numéro 0 : */
    numero = 0;
  }
  else {
    /* Je suis un processeur de banc ou de calcul lancé par le
       contrôleur... */

    /* Récupère du père... */
    testerreur("pvm_recv",
	       pvm_recv(pere, 0));

    /* le model.rc : */
    recoit_params();
    init_variables();
    
    /* les variables d'environnement : */
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

    testerreur("pvm_upkint",
	       pvm_upkint(&tids[1], nb_taches - 1, 1));
    tids[0] = pere;
  }
  
  /* Pour une fin synchrone : */
  /*testerreur("pvm_joingroup",
	     pvm_joingroup(groupe_final));*/
  
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
  if (numero < nb_procs) {
    debug(2, "main",
	  "Je suis le PE %d de tid 0x%x\n", numero, mytid);
    if (get_debug_level() >= 1)
      affiche_entete_X("%s:%s PE %d", machine, basename(argv[0]), numero);
    FUNCTION(wp65)(&numero);
  }
  else {
    banc = numero - nb_procs;
    debug(2, "main",
	  "Je suis le banc %d de tid 0x%x\n", banc, mytid);
    if (get_debug_level() >= 1)
      affiche_entete_X("%s:%s Banc %d", machine, basename(argv[0]), banc);
    FUNCTION(bank)(&banc);
  }

  /* Rajout d'une barrière car il semble que PVM puisse perdre des émissions
     finales si on fait un pvm_exit trop tôt. */
  debug(2, "main", "Attente de la barrière finale pour terminer.\n");
  /* testerreur("pvm_barrier",
	     pvm_barrier(groupe_final, nb_taches));*/

  debug_off();
  pvm_exit();
  exit(0);
}

/*     Fait des conversions : */

void send_4(int tid, float *donnee, int taille)
{
  testerreur("pvm_initsend",
	     pvm_initsend(WP65_PVM_DATA));
  /* Envoie aussi la taille du message pour être capable de le lire
     par morceaux : */
  testerreur("pvm_pack",
	     pvm_pkint(&taille, 1, 1));
  testerreur("pvm_pack",
	     pvm_pkfloat(donnee, taille, 1));

  debug(5, "send_4",
	"pvm_send de tid 0x%x vers tid 0x%x\n", mytid, tid);

  testerreur("pvm_send",
	     pvm_send(tid, 0));
}

void send_8(int tid, double *donnee, int taille)
{
  testerreur("pvm_initsend", pvm_initsend(WP65_PVM_DATA));
  /* Envoie aussi la taille du message pour être capable de le lire
     par morceaux : */
  testerreur("pvm_pack", pvm_pkint(&taille, 1, 1));
  testerreur("pvm_pack", pvm_pkdouble(donnee, taille, 1));

  debug(5, "send_8", "pvm_send de tid 0x%x vers tid 0x%x\n", mytid, tid);

  testerreur("pvm_send", pvm_send(tid, 0));
}


void receive_4(int tid, int proc_or_bank_id, float *donnee, int taille)
{
  int taille_recue;
  int buf_id, old_buf;

  debug(5, "receive_4",
	"pvm_recv de tid 0x%x depuis le tid 0x%x de %d données.\n",
	mytid, tid, taille);

  if(bufid[proc_or_bank_id] != TAMPON_VIDE) {
    if (taille_restante[proc_or_bank_id] != 0) {
      /* Il nous reste des choses à lire dans un ancien tampon. */
      testerreur("pvm_setrbuf",
		 old_buf = pvm_setrbuf(bufid[proc_or_bank_id]));
      if (taille > taille_restante[proc_or_bank_id])
	sort_erreur("Demande de lire %d données alors qu'il n'en reste que %d !\n",
		    taille, taille_restante[proc_or_bank_id]);
      debug(5, "receive_4",
	    "Lecture ancienne de %d données.\n", taille);
      testerreur("pvm_unpack",
		 pvm_upkfloat(donnee, taille, 1)); 
      if ((taille_restante[proc_or_bank_id] -= taille) == 0) {
	/* Le tampon est vide, on le libère : */
	testerreur("pvm_freebuf",
		   pvm_freebuf(bufid[proc_or_bank_id]));
	bufid[proc_or_bank_id] = TAMPON_VIDE;
	debug(5, "receive_4",
	      "Libération du tampon %d...", proc_or_bank_id);
       }
      debug(5, "receive_4",
	    "Reste %d données dans le tampon %d.\n",
	    taille_restante[proc_or_bank_id], proc_or_bank_id);      
    }
    else {
      sort_erreur("Inconsistence entre bufid[%s] et taille_restante[%s] !\n",
		  proc_or_bank_id, proc_or_bank_id);
    }
  }
  else {
    /* Il nous reste rien à lire, il faut donc faire une « vraie »
       réception : */
    testerreur("pvm_recv",
	       buf_id = pvm_recv(tid, 0));
    testerreur("pvm_unpack",
	       pvm_upkint(&taille_recue, 1, 1));
    debug(5, "receive_4",
	  "Nouvelle réception : arrivée de %d données.\n", taille_recue);

    if (taille_recue < taille)
      sort_erreur("Demande de recevoir %d données alors qu'on n'en n'a reçu que %d !\n",
		  taille, taille_recue);
    if (taille_recue != taille) {
      /* On veut recevoir en plus petits morceaux. */
      bufid[proc_or_bank_id] = buf_id;
      taille_restante[proc_or_bank_id] = taille_recue - taille;
      debug(5, "receive_4",
	    "Découpe en petit morceaux : reste %d données à lire pour une prochaine fois.\n",
	    taille_restante[proc_or_bank_id]);
    }
    /* Lit déjà ce qu'on veut : */
    testerreur("pvm_unpack",
	       pvm_upkfloat(donnee, taille, 1));
  }
}

void receive_8(int tid, int proc_or_bank_id, double *donnee, int taille)
{
  int taille_recue;
  int buf_id, old_buf;

  debug(5, "receive_8",
	"pvm_recv de tid 0x%x depuis le tid 0x%x de %d données.\n",
	mytid, tid, taille);

  if(bufid[proc_or_bank_id] != TAMPON_VIDE) {
    if (taille_restante[proc_or_bank_id] != 0) {
      /* Il nous reste des choses à lire dans un ancien tampon. */
      testerreur("pvm_setrbuf",
		 old_buf = pvm_setrbuf(bufid[proc_or_bank_id]));
      if (taille > taille_restante[proc_or_bank_id])
	sort_erreur("Demande de lire %d données alors qu'il n'en reste que %d !\n",
		    taille, taille_restante[proc_or_bank_id]);
      debug(5, "receive_8",
	    "Lecture ancienne de %d données.\n", taille);
      testerreur("pvm_unpack", pvm_upkdouble(donnee, taille, 1)); 
      if ((taille_restante[proc_or_bank_id] -= taille) == 0) {
	/* Le tampon est vide, on le libère : */
	testerreur("pvm_freebuf", pvm_freebuf(bufid[proc_or_bank_id]));
	bufid[proc_or_bank_id] = TAMPON_VIDE;

	debug(5, "receive_8", "Libération du tampon %d...", proc_or_bank_id);
    }

      debug(5, "receive_8", "Reste %d données dans le tampon %d.\n",
	    taille_restante[proc_or_bank_id], proc_or_bank_id);      
    }
    else {
      sort_erreur("Inconsistence entre bufid[%s] et taille_restante[%s] !\n",
		  proc_or_bank_id, proc_or_bank_id);
    }
  }
  else {
    /* Il nous reste rien à lire, il faut donc faire une « vraie »
       réception : */
    testerreur("pvm_recv", buf_id = pvm_recv(tid, 0));
    testerreur("pvm_unpack", pvm_upkint(&taille_recue, 1, 1));
    debug(5, "receive_8",
	  "Nouvelle réception : arrivée de %d données.\n", taille_recue);

    if (taille_recue < taille)
      sort_erreur("Demande de recevoir %d données alors qu'on n'en n'a reçu que %d !\n",
		  taille, taille_recue);
    if (taille_recue != taille) {
      /* On veut recevoir en plus petits morceaux. */
      bufid[proc_or_bank_id] = buf_id;
      taille_restante[proc_or_bank_id] = taille_recue - taille;
      debug(5, "receive_8",
	    "Découpe en petit morceaux : reste %d données à lire pour une prochaine fois.\n",
	    taille_restante[proc_or_bank_id]);
    }
    /* Lit déjà ce qu'on veut : */
    testerreur("pvm_unpack", pvm_upkdouble(donnee, taille, 1));
  }
}

void FUNCTION_(bank_send_4)(int *proc_id, float *donnee, int *taille)
{
  debug(4, "BANK_SEND_4",
	"Envoi de banc %d -> PE %d, taille = %d (estampille %d)\n",
	banc, *proc_id, *taille,
	estampiller(*proc_id, banc, *taille, 1));
  send_4(tids[*proc_id], donnee, *taille);
}

void FUNCTION_(bank_send_8)(int *proc_id, double *donnee, int *taille)
{
  debug(4, "BANK_SEND_8",
	"Envoi de banc %d -> PE %d, taille = %d (estampille %d)\n",
	banc, *proc_id, *taille,
	estampiller(*proc_id, banc, *taille, 1));
  send_8(tids[*proc_id], donnee, *taille);
}

void FUNCTION_(bank_receive_4)(int *proc_id, float *donnee, int *taille)
{
  debug(4, "BANK_RECEIVE_4",
	"Réception de banc %d <- PE %d, taille = %d (estampille %d)\n",
	banc, *proc_id, *taille,
	estampiller(*proc_id, banc, *taille, 0));
  receive_4(tids[*proc_id], *proc_id, donnee, *taille);
}

void FUNCTION_(bank_receive_8)(int *proc_id, double *donnee, int *taille)
{
  debug(4, "BANK_RECEIVE_8",
	"Réception de banc %d <- PE %d, taille = %d (estampille %d)\n",
	banc, *proc_id, *taille,
	estampiller(*proc_id, banc, *taille, 0));
  receive_8(tids[*proc_id], *proc_id, donnee, *taille);
}

void FUNCTION_(wp65_send_4)(int *bank_id, float *donnee, int *taille)
{
  debug(4, "WP65_SEND_4",
	"Envoi de PE %d -> banc %d, taille = %d (estampille %d)\n",
	numero, *bank_id, *taille,
	estampiller(numero, *bank_id, *taille, 1));
  send_4(tids[*bank_id + nb_procs], donnee, *taille);
}

void FUNCTION_(wp65_send_8)(int *bank_id, double *donnee, int *taille)
{
  debug(4, "WP65_SEND_8",
	"Envoi de PE %d -> banc %d, taille = %d (estampille %d)\n",
	numero, *bank_id, *taille,
	estampiller(numero, *bank_id, *taille, 1));
  send_8(tids[*bank_id + nb_procs], donnee, *taille);
}


void FUNCTION_(wp65_receive_4)(int *bank_id, float *donnee, int *taille)
{
  debug(4, "WP65_RECEIVE_4",
	"Réception de PE %d <- banc %d, taille = %d (estampille %d)\n",
	numero, *bank_id, *taille,
	estampiller(numero, *bank_id, *taille, 0));
  receive_4(tids[*bank_id + nb_procs], *bank_id, donnee, *taille);
}

void FUNCTION_(wp65_receive_8)(int *bank_id, double *donnee, int *taille)
{
  debug(4, "WP65_RECEIVE_8",
	"Réception de PE %d <- banc %d, taille = %d (estampille %d)\n",
	numero, *bank_id, *taille,
	estampiller(numero, *bank_id, *taille, 0));
  receive_8(tids[*bank_id + nb_procs], *bank_id, donnee, *taille);
}

int FUNCTION(idiv)(int * i, int * j)
{
    return (*i)>=0 ? (*i)/(*j) : (-(-(*i)+(*j)-1)/(*j));
}

/* Horreur pour que la bibliothèque F77 soit contente : */
#ifndef COMPILE_FOR_G77
void FUNCTION(MAIN)()
{
  /* Jamais exécuté, je pense... */
  fprintf(stderr, "Entrée dans MAIN_ !\n");
}
#endif
