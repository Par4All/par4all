#define MAX_NB_CRITICALS 10
#define CRITICAL_MAX_NB_GLOBALUPDATES 1000

#define CRITICAL_REQUEST_TAG 0
#define CRITICAL_RELEASE_TAG 1
#define CRITICAL_FINALUPDATE_TAG 2
#define CRITICAL_STOPPCOORD_TAG 3
/* FSC prefixer avec CRITICAL */
#define CRITICAL_PENDING_UPTODATE_TAG 4
#define CRITICAL_INFOS_TAG 5
#define CRITICAL_AKNOWLEGMENT_TAG 9
#define CRITICAL_NEXTPROCESS_TAG 10

MPI_Comm spawn_intercomm;             /* intercommunicator */
MPI_Status status;
int msg_process;

struct critical_infos_s 
{
  int next_process_rank; 
  int nb_pending_updates; 
  int first_process_p;
} critical_infos;
