#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <error.h>
#include "trace.h"
#include "critical.h"

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

/* FSC renommer MAX_NB_GLOBALUPDATES  en CRITICAL_MAX_NB_GLOBALUPDATES  */

#define MAX_NB_PROCESSES 100


int main(int argc, char **argv)
{

/* FSC revoir pourquoi un tableau par processus */
	
  int   i,p, ierr,num_current_critical, commsize,newrank,length, intercomm_rank;


	/*
	  pending_updates_ranklist

	  For each critical section 
             For each process

               Storage of a list of ranks of processes that performed
               global updates 

               This list is reset for a given process when entering a
               critical section and performing global update
	*/
	int pending_updates_ranklist[MAX_NB_CRITICALS][MAX_NB_PROCESSES][CRITICAL_MAX_NB_GLOBALUPDATES];


	/* Number of global updates 
	   In one case, contains also the current partial update */
	int nb_pending_updates[MAX_NB_CRITICALS][MAX_NB_PROCESSES];

	/* first_p

	   For each critical section
             Storage of a boolean indicating whether a process alredy
             entered the critical section
	*/
	bool first_p[MAX_NB_CRITICALS];


	/* 
	   critical_queue_array

	   For each critical section
                Storage of the ranks of the processes waiting for the critical section
	*/
  	/* contient le nombre de processus qu'on a dans la file d'attente */
	typedef struct  
	{
	  int size; 
	  int array[MAX_NB_PROCESSES];
	} critical_queue_s;
	critical_queue_s critical_queue_array [MAX_NB_CRITICALS];
	
	char name[MPI_MAX_PORT_NAME];	
	bool stop_p;
       	
	ierr = MPI_Init(&argc, &argv);
	ierr = MPI_Comm_rank(MPI_COMM_WORLD, &newrank);
  	ierr = MPI_Comm_get_parent (&spawn_intercomm);
	ierr = MPI_Comm_size(spawn_intercomm, &commsize);
	ierr = MPI_Comm_rank(spawn_intercomm, &intercomm_rank);

	//IN_TRACE("intercomm_rank = %d", intercomm_rank);
  	if(spawn_intercomm == MPI_COMM_NULL) error (EXIT_FAILURE, 0, "Pcoord has no parent: exit...");
	MPI_Get_processor_name(name,&length);

	printf("I m the additional process rank = %d, commsize = %d, newrank = %d running on %s\n", intercomm_rank, commsize, newrank, name);
	for(i=0;i<MAX_NB_CRITICALS;i++)
	  {	
	    first_p[i]=true;
	    critical_queue_array[i].size=0;
	  }
	
	stop_p = false;
	while(stop_p == false)
	{	
	  
	  critical_queue_s  *current_queue;
	  

	  ierr = MPI_Recv(&num_current_critical, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, spawn_intercomm, &status); 
	  /* il connait pas cette fontion : TRACE_P*/
	  //TRACE_P("Pcoord have received a msg from %d, tag=%d, num_current_critical=%d\n", status.MPI_SOURCE, status.MPI_TAG, num_current_critical);  


	  current_queue = &(critical_queue_array[num_current_critical]);
	  
	  switch (status.MPI_TAG)
	    {	
	    case CRITICAL_REQUEST_TAG:
	      { 

		/* Case where queue is empty */
		if (current_queue->size == 0)
		{    	
		  current_queue->array[0] = status.MPI_SOURCE;    
		  current_queue->size ++;
		  critical_infos.next_process_rank = -1;

		  critical_infos.nb_pending_updates = nb_pending_updates[num_current_critical][status.MPI_SOURCE];
		  /* reset the number of global updates for this process */
		  nb_pending_updates[num_current_critical][status.MPI_SOURCE] = 0;

		  if(first_p[num_current_critical])
		    {
		      /* Case when this is the first process */
		      critical_infos.first_process_p = 1;   
		      first_p[num_current_critical] = false;
		    }
		  else
		    critical_infos.first_process_p = 0;

		  ierr = MPI_Send(&critical_infos, sizeof(struct critical_infos_s), MPI_INT, status.MPI_SOURCE, CRITICAL_INFOS_TAG, spawn_intercomm);

		  if (critical_infos.nb_pending_updates > 0)	
		    ierr = MPI_Send(pending_updates_ranklist[num_current_critical][status.MPI_SOURCE], critical_infos.nb_pending_updates, MPI_INT, status.MPI_SOURCE,  CRITICAL_PENDING_UPTODATE_TAG, spawn_intercomm);

		  ierr = MPI_Recv(&msg_process, 1, MPI_INT, status.MPI_SOURCE, CRITICAL_AKNOWLEGMENT_TAG, spawn_intercomm, &status); 
		}
		else 
		{
		  /* Case where queue is not empty */

		  /* Add process to queue */
		  /* FSC ajouter des tests pour vérifier que tu ne dépasses pas la taille de la file ou faire dynamique */

		  current_queue->array[current_queue->size]=status.MPI_SOURCE;    	
		  current_queue->size ++;
		}    
		break;
	      }

	    case  CRITICAL_RELEASE_TAG :
	      {
		/* Case where another process is waiting for the critical section */
		if (current_queue->size > 1)
		{
		  /* next process always located at index 1 */
		  critical_infos.next_process_rank = current_queue->array[1];

		  ierr = MPI_Send(&critical_infos, sizeof(struct critical_infos_s), MPI_INT, status.MPI_SOURCE, CRITICAL_NEXTPROCESS_TAG, spawn_intercomm); 
		  ierr = MPI_Recv(&msg_process, 1, MPI_INT, MPI_ANY_SOURCE, CRITICAL_AKNOWLEGMENT_TAG, spawn_intercomm,&status);

		  /* Increment updates to perform previous global updates + the current partial update */
		  critical_infos.nb_pending_updates = nb_pending_updates[num_current_critical][critical_infos.next_process_rank] + 1; 

		   critical_infos.first_process_p = 0;
		   pending_updates_ranklist[num_current_critical][critical_infos.next_process_rank][critical_infos.nb_pending_updates - 1] = status.MPI_SOURCE;

		   /* FSC mettre des affichages de controle pendant l'execution 
		      IN_TRACE("Pcoord to %d: ", ) 
		      next_process_p, nombre de mises a jour globale a faire...
		   */

		    ierr = MPI_Send(&critical_infos, sizeof(struct critical_infos_s), MPI_INT, critical_infos.next_process_rank, CRITICAL_INFOS_TAG, spawn_intercomm);

		    if (critical_infos.nb_pending_updates > 0)
		      ierr = MPI_Send(pending_updates_ranklist[num_current_critical][critical_infos.next_process_rank], critical_infos.nb_pending_updates, MPI_INT, critical_infos.next_process_rank, CRITICAL_PENDING_UPTODATE_TAG, spawn_intercomm);

		  /* reset the number of global updates for this process */
		    nb_pending_updates[num_current_critical][critical_infos.next_process_rank] = 0;
		    
		    ierr = MPI_Recv(&msg_process, 1, MPI_INT, critical_infos.next_process_rank,  CRITICAL_AKNOWLEGMENT_TAG, spawn_intercomm, &status); 
		    
		    /* Update the critical_queue to remove current process and go to next process */
		    for( i=0; i <  current_queue->size; i++)
		      current_queue->array[i]=current_queue->array[i+1];	   
		}
		else
		{
		  
		  /* Case where release critical section and no process is waiting for the critical section, 
		     current process will perform a global update */

		  critical_infos.next_process_rank = -1;
		  ierr = MPI_Send(&critical_infos, sizeof(struct critical_infos_s), MPI_INT, status.MPI_SOURCE, CRITICAL_NEXTPROCESS_TAG, spawn_intercomm); 
		  ierr = MPI_Recv(&msg_process, 1, MPI_INT, status.MPI_SOURCE, CRITICAL_AKNOWLEGMENT_TAG, spawn_intercomm, &status);
		  
		  /* Update the number of global updates for all the processes */
		  for (p = 0; p < MAX_NB_PROCESSES; p++)
		    {
		      int nb_glob;
		      
		      /* process MPI_SOURCE also updates the list */
		      nb_glob = nb_pending_updates[num_current_critical][p];
		      pending_updates_ranklist[num_current_critical][p][nb_glob]= status.MPI_SOURCE;
		      nb_pending_updates[num_current_critical][p] ++;
		    }
		}
		
		/* Decrease the number of processes waiting */
		current_queue->size--;	
		break;
	      }

	    case CRITICAL_FINALUPDATE_TAG:
	      {
		critical_infos.nb_pending_updates = nb_pending_updates[num_current_critical][status.MPI_SOURCE];
		critical_infos.first_process_p = 0;


		ierr = MPI_Send(&critical_infos, sizeof(struct critical_infos_s), MPI_INT, status.MPI_SOURCE, CRITICAL_INFOS_TAG, spawn_intercomm);
		

		if (critical_infos.nb_pending_updates > 0)	
		  ierr = MPI_Send(pending_updates_ranklist[num_current_critical][status.MPI_SOURCE], critical_infos.nb_pending_updates, MPI_INT, status.MPI_SOURCE, CRITICAL_PENDING_UPTODATE_TAG, spawn_intercomm);

		/* reset the number of global updates for this process */
		nb_pending_updates[num_current_critical][status.MPI_SOURCE] = 0;

		ierr = MPI_Recv(&msg_process, 1, MPI_INT, status.MPI_SOURCE,  CRITICAL_AKNOWLEGMENT_TAG, spawn_intercomm, &status); 
		break;
	      }

	    case  CRITICAL_STOPPCOORD_TAG: 
	      stop_p=true;
	      break;
	    }
	}	
	ierr = MPI_Finalize();
	return EXIT_SUCCESS;
}
