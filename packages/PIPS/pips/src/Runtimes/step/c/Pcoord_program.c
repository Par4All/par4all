#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <error.h>
#include "trace.h"

#define true 1
#define false 0
typedef int bool;

/* FSC taille de tableau fixe. Pourquoi?

2 choix:

1) tailles fixes
- les mettre dans des macros (#define MAX_NB_CRITICALS 10...) c fait

- ajouter des tests dans le code permettant une sortie propre du programme avec un message d'erreur explicite

2) remplacer des tableaux dynamiques

*/

#define MAX_NB_CRITICALS 10
#define MAX_NB_PROCESSES 100
#define MAX_NB_GLOBALUPDATES 1000

#define STOPPCOORD_CRITICAL_TAG 3
#define REQUEST_CRITICAL_TAG 0
#define RELEASE_CRITICAL_TAG 1
#define LASTUPDATE_CRITICAL_TAG 2

struct critical_infos_s 
{
  int next_process_rank; 
  int nb_global_update; 
  int first_process_p;
}critical_infos;

int main(int argc, char **argv)
{
/* FSC renommer glob_maj en global_update_array et decrire glob_maj c fait*/
/* revoir pourquoi un tableau par processus */
/* global_update_array contains the ranks of the process that handled global updates */
	
        int   i,p, ierr,num_current_critical,rank, size,newrank,length;
	int global_update_array[MAX_NB_CRITICALS][MAX_NB_PROCESSES][MAX_NB_GLOBALUPDATES]; //ranks of processes which have sent data
	int nb_global_update_array[MAX_NB_CRITICALS][MAX_NB_PROCESSES];// number of global_updates
	/* FSC renommer first en first_p */
	bool first_p[MAX_NB_CRITICALS];//pour chaque section critique
  	MPI_Comm spawn_intercomm;
	MPI_Status status;
	int msg_process;
/* FSC remplacer election par la structure critical_infos */
  	/* contient le nombre de processus qu'on a dans la file d'attente */
        /* FSC decrire critical_queue_array */
        /* FSC: faire un tableau de structures de taille "le nombre de sections critiques"
                chaque element contenant la taille de la file et la file */
        /* FSC: renommer file_attente en critical_queue_array */
	int critical_queue_array[ MAX_NB_CRITICALS][MAX_NB_PROCESSES];//queue of processes for each critical section
	char name[MPI_MAX_PORT_NAME];	
	ierr = MPI_Init(&argc, &argv);
	ierr = MPI_Comm_rank(MPI_COMM_WORLD, &newrank);
        /* FSC renommer parent en spawn_intercomm c fait*/
  	ierr = MPI_Comm_get_parent (&spawn_intercomm);
	ierr = MPI_Comm_size(spawn_intercomm, &size);
        /* FSC renommer rank en intercomm_rank */
	ierr = MPI_Comm_rank(spawn_intercomm, &rank);

	//IN_TRACE("rank = %d", rank);
        /* FSC verifier le comportement de error: affiche puis sort ou juste affiche --> oui affiche puis sort */
  	if(spawn_intercomm == MPI_COMM_NULL) error (EXIT_FAILURE, 0, "Pcoord has no parent: exit...");
	MPI_Get_processor_name(name,&length);

        /* FSC Mettre les commentaires a conserver en anglais sinon les supprimer */
	
        /* FSC voir si on utilise les nb_sections,non, on l'utilise pas si oui renommer nb_sections et supprimer les valeurs en dur */
	printf("I m the additional process (rank) %d, size= %d, newrank=%d, my name is %s\n",rank,size, newrank,name);
	bool stop_p=false;
	for(i=0;i<MAX_NB_CRITICALS;i++)
	{	
		first_p[i]=true;
	}
	while(stop_p==false)
	{	
	  ierr = MPI_Recv(&msg_process, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, spawn_intercomm, &status); 
	  /* FSC renommer nm_section en num_current_critical */
	  num_current_critical=msg_process;
	  switch (status.MPI_TAG)
	    {	
/* FSC: a remplacer par une macro: REQUEST_CRITICAL_TAG c fait */
	    case REQUEST_CRITICAL_TAG:
	      { 
/* FSC: relire et decrire la structure file d'attente */
        /* Case where queue is empty */
		if (critical_queue_array[num_current_critical][0]==0)
		{    	
                  /* FSC remplacer l'index complique par une variable temporaire pour lisibilité
                     FSC revoir le nom de la variable*/
		  int nb_process_queue;
		  nb_process_queue = critical_queue_array[num_current_critical][0];
		  critical_queue_array[num_current_critical][nb_process_queue+1]=status.MPI_SOURCE;    
		  critical_queue_array[num_current_critical][0]++;
		  critical_infos.next_process_rank=-1;
                   /* FSC ajouter une variable supplementaire pour stocker le nombre de MAJ globale */
		  critical_infos.nb_global_update=nb_global_update_array[num_current_critical][status.MPI_SOURCE];
		  nb_global_update_array[num_current_critical][status.MPI_SOURCE]=0;
		  if(first_p[num_current_critical])
		    {
		      critical_infos.first_process_p=1;   //you are the first process (don t recv)
		      first_p[num_current_critical]=false;
		    }
		  else
		    critical_infos.first_process_p=0;  //you are n t the first process
		  ierr = MPI_Send(&critical_infos, sizeof(struct critical_infos_s), MPI_INT, status.MPI_SOURCE, 0, spawn_intercomm);
		  if (critical_infos.nb_global_update>0)	
		     ierr = MPI_Send(global_update_array[num_current_critical][status.MPI_SOURCE],critical_infos.nb_global_update,MPI_INT,status.MPI_SOURCE, 4, spawn_intercomm);//rank of processes that have updated data
		  ierr = MPI_Recv(&msg_process, 1, MPI_INT, status.MPI_SOURCE, 1, spawn_intercomm, &status); //après request de pi   
		}
		else 
		{
		  critical_queue_array[num_current_critical][critical_queue_array[num_current_critical][0]+1]=status.MPI_SOURCE;    	
		  critical_queue_array[num_current_critical][0]++;
		}    
		break;
	      }
	      /* FSC: a remplacer par une macro: RELEASE_CRITICAL_TAG c fait*/
	    case  RELEASE_CRITICAL_TAG :
	      {
		/* Case where another process is waiting for the critical section */
		if (critical_queue_array[num_current_critical][0]>1)
		{
		  critical_infos.next_process_rank=critical_queue_array[num_current_critical][2];
		   ierr = MPI_Send(&critical_infos, sizeof(struct critical_infos_s), MPI_INT, status.MPI_SOURCE, 10, spawn_intercomm); 
		   ierr = MPI_Recv(&msg_process,1, MPI_INT, MPI_ANY_SOURCE, 9,spawn_intercomm,&status);
		   critical_infos.nb_global_update=nb_global_update_array[num_current_critical][critical_infos.next_process_rank]+1;//this is an another update(partial update)
		   critical_infos.first_process_p=0;
		   global_update_array[num_current_critical][critical_infos.next_process_rank][critical_infos.nb_global_update-1]= status.MPI_SOURCE;
		   /* FSC mettre des affichages de controle pendant l'execution 
		      IN_TRACE("Pcoord to %d: ", ) 
		      next_process_p, nombre de mises a jour globale a faire...
		   */
		    ierr = MPI_Send(&critical_infos, sizeof(struct critical_infos_s), MPI_INT,critical_infos.next_process_rank, 0, spawn_intercomm);
		    ierr = MPI_Send(global_update_array[num_current_critical][critical_infos.next_process_rank],critical_infos.nb_global_update,MPI_INT,critical_infos.next_process_rank, 4, spawn_intercomm);
		    /* Reset the number of global updates for the current process */
/* FSC revoir 999??????????????????????????????? c fait*/ 
		  nb_global_update_array[num_current_critical][critical_infos.next_process_rank]=0;
		  /* FSC verifier l'utilité de cette communication : communication utile sauf si on remplace SEND par SSEND */
		  ierr = MPI_Recv(&msg_process,1,MPI_INT,critical_infos.next_process_rank,1,spawn_intercomm,&status); //bloqué jusqu'à terminaison de la réception(après request aussi) 		
		  /* Update the critical_queue to remove current process and go to next process */
		  for( i=0;i<critical_queue_array[num_current_critical][0];i++)
		    critical_queue_array[num_current_critical][i+1]=critical_queue_array[num_current_critical][i+2];	   
		}
		else
		{
		  /* Case where no process is waiting for the critical section, launches a global updates for all processes */
		  critical_infos.next_process_rank=-1;
		   ierr = MPI_Send(&critical_infos, sizeof(struct critical_infos_s), MPI_INT, status.MPI_SOURCE, 10, spawn_intercomm); 
		   ierr = MPI_Recv(&msg_process,1,MPI_INT,status.MPI_SOURCE,9,spawn_intercomm,&status);
		   /* Update the number of global updates for all the processes */
		   /* FSC ne pas mettre 100 en dur : soit mettre le vrai nombre soit macro c fait*/
		   for (p=0;p<MAX_NB_PROCESSES;p++)
		     {
		       /* FSC a revoir apres renommage avec des structures, si necessaire introduire une variable temporaire */
		      global_update_array[num_current_critical][p][(nb_global_update_array[num_current_critical][p])]= status.MPI_SOURCE;
		      nb_global_update_array[num_current_critical][p] ++;
		     }
		}
              /* Decrease the number of processes waiting */
		critical_queue_array[num_current_critical][0]--;	
		break;
	      }
	      /* FSC: a remplacer par une macro: LASTUPDATE_CRITICAL_TAG c fait */
	    case LASTUPDATE_CRITICAL_TAG:
	      {
		critical_infos.nb_global_update=nb_global_update_array[num_current_critical][status.MPI_SOURCE];
		critical_infos.first_process_p=0;
		ierr = MPI_Send(&critical_infos, sizeof(struct critical_infos_s), MPI_INT,status.MPI_SOURCE,2, spawn_intercomm);
		if (critical_infos.nb_global_update>0)	
		   ierr = MPI_Send(global_update_array[num_current_critical][status.MPI_SOURCE],critical_infos.nb_global_update,MPI_INT,status.MPI_SOURCE,2, spawn_intercomm);
		ierr = MPI_Recv(&msg_process,1,MPI_INT,status.MPI_SOURCE,1, spawn_intercomm, &status); 
		break;
	      }
/* FSC: a remplacer par une macro: STOPPCOORD_CRITICAL_TAG c fait */
	// la fin du programme
	    case  STOPPCOORD_CRITICAL_TAG: 
	      stop_p=true;
	      break;
	    }
	}	
	ierr=MPI_Finalize();
	return 0;
}
