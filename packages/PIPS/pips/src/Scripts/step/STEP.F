* Copyright 2007, 2008 Alain Muller, Frederique Silber-Chaussumier
*
*This file is part of STEP.
*
*The program is distributed under the terms of the GNU General Public
*License.




      subroutine STEP_get_myloopslice(i)
      INTEGER STEP_I_LOW, STEP_I_UP, STEP_MYLOOPSLICE
      COMMON /MYLOOPSLICE/ STEP_I_LOW, STEP_I_UP, STEP_MYLOOPSLICE

      i = STEP_MYLOOPSLICE
      end

      subroutine STEP_get_i_low(i)
      INTEGER STEP_I_LOW, STEP_I_UP, STEP_MYLOOPSLICE
      COMMON /MYLOOPSLICE/ STEP_I_LOW, STEP_I_UP, STEP_MYLOOPSLICE

      i = STEP_I_LOW
      END

      subroutine STEP_get_i_up(i)
      INTEGER STEP_I_LOW, STEP_I_UP, STEP_MYLOOPSLICE
      COMMON /MYLOOPSLICE/ STEP_I_LOW, STEP_I_UP, STEP_MYLOOPSLICE

      i = STEP_I_UP
      end

      subroutine STEP_Init()
      implicit none
      include 'STEP.h'
      integer*4 ierr,iBidon
      logical Initialized
      call MPI_Initialized(Initialized,ierr )
      if (.not. Initialized) then
         call MPI_Init ( ierr )
!     print *, STEP_size, STEP_rank
      end if
      call MPI_Comm_size ( MPI_COMM_WORLD, STEP_size, ierr )
      call MPI_Comm_rank ( MPI_COMM_WORLD, STEP_rank, ierr )      

      if (STEP_size .gt. STEP_MAX_NBNODE) then
         print *,"STEP_Init() : STEP_MAX_NBNODE
     &trop petit (update STEP.h) STEP_size = ",STEP_size
         call MPI_ABORT(MPI_COMM_WORLD, iBidon, ierr)
      end if
      end
      
      subroutine STEP_Finalize()
      implicit none
      include 'mpif.h'
      integer*4 ierr
      call MPI_Finalize(ierr)
      end

      subroutine STEP_Barrier()
      implicit none
      include 'mpif.h'
      integer*4 ierr
      call MPI_BARRIER(MPI_COMM_WORLD,ierr)
      end

      subroutine STEP_Get_size(size)
      implicit none
      include 'mpif.h'
      integer*4 size,ierr
      call MPI_Comm_size (  MPI_COMM_WORLD, size, ierr )
      end

      subroutine STEP_Get_rank(rank)
      implicit none
      include 'mpif.h'
      integer*4 rank,ierr
      call MPI_Comm_rank ( MPI_COMM_WORLD, rank, ierr )
      end
	
      function STEP_Get_thread_num()
      implicit none
      integer STEP_Get_thread_num
      call STEP_Get_rank(STEP_Get_thread_num)
      end

      subroutine STEP_ComputeLoopSlices(from,to,step,nb,size,bounds)
      implicit none
      integer F,T,from,to,step,nb,size
      parameter (F=1,T=2)
      integer bounds(F:T,size)
      integer nb_indices,nb_i,nb_e,i
      
c      print *,'from',from,'to',to,'step',step,'nb',nb,'size',size,
c     * 'bounds'

      do 5 i=1,size
         bounds(F,i)=-1
         bounds(T,i)=-1
 5    continue
      if (((step.GT.0) .AND. ((to-from).LT.step)) .OR.
     *     ((step.LT.0) .AND. ((from-to).LT.-step))) then ! pas d'iteration
c         print *,'pas d''iteration'
         do 10 i=1,nb
            bounds(F,i)=to
 10         bounds(T,i)=from
      else
         nb_indices=(to-from)/step+1
c         print *,nb_indices,' iterations'
         if (nb_indices.LE.nb) then ! une iteration par noeud
c            print *,'au plus une iteration par noeud'
            bounds(F,1)=from
            bounds(T,1)=from
            do 20 i=2,nb_indices
               bounds(F,i)= bounds(F,i-1)+step
 20            bounds(T,i)= bounds(F,i)
            do 25 i=nb_indices+1,nb ! plus d'iteration pour les autres noeuds
               bounds(F,i)= to
 25            bounds(T,i)= from
         else                   !au moins un noeud avec deux iterations
            nb_i=MOD(nb_indices,nb)
            nb_e=(nb_indices-nb_i)/nb
            if (nb_i.EQ.0) then
               nb_i=nb
               nb_e=nb_e-1
            endif
            bounds(F,1)=from
            bounds(T,1)=from+nb_e*step
            do 30 i=2,nb_i
c               print *, 'do30',i
               bounds(F,i)=bounds(T,i-1)+step
 30            bounds(T,i)=bounds(F,i)+nb_e*step
            do 35 i=nb_i+1,nb
c               print *,'do35',i
               bounds(F,i)=bounds(T,i-1)+step
 35            bounds(T,i)=bounds(F,i)+(nb_e-1)*step
         endif
      endif
    
c      print *,bounds
      end
      
******************************************
c
c     procédure produisant type_sub_region (handler de  type MPI) 
c     représentant la sous-region subregion définit dans l'espace d'indice region
c     d'élément de type type_region (handler de type MPI)
c
      subroutine type_subRegion(dim,region,sub_region,type_region,
     &     type_sub_region)
      implicit none
      include 'STEP.h'
      integer dim               !< nombre de dimension de l'espace
      integer L,U
      parameter (L=1, U=2)
      integer region(L:U,dim)      !< borne min/max des indices selon chaque dimension
      integer*4 sub_region(L:U,dim)      !< borne min/max des indices selon chaque dimension
      integer type_region       !< type MPI décrivant un element de la region
      integer type_sub_region   !> type MPI décrivant la sous-region
      
      integer*4 array_type(0:STEP_MAX_DIM),data_size(0:STEP_MAX_DIM),i
      integer*4 iBidon,IERROR
      logical valide

      if (dim .gt. STEP_MAX_DIM) then
         print *,"type_subRegion() : STEP_MAX_DIM 
     &trop petit (update STEP.h) dim = ",dim
         call MPI_ABORT(MPI_COMM_WORLD, iBidon, IERROR)
      end if
!      print *,'dim=',dim,'region =',region,
!     &     'sub_region =',sub_region

      array_type(0)=type_region
      call MPI_TYPE_EXTENT(array_type(0),data_size(0),IERROR)
      i=1
      valide=.TRUE.
      do while (valide .AND. (i.LE.dim))
c     verification des définitions des régions/sous-regions
         if (.NOT.((region(L,i) .LE. sub_region(L,i)) .AND.
     &        (sub_region(L,i) .LE. sub_region(U,i)) .AND.
     &        (sub_region(U,i) .LE. region(U,i)))) then
            valide=.FALSE.
            array_type(dim)=MPI_DATATYPE_NULL
         else
            data_size(i)=data_size(i-1)*
     &           (region(U,i)-region(L,i)+1)
            call MPI_TYPE_HVECTOR(
     &           sub_region(U,i)-sub_region(L,i)+1, 1,
     &           data_size(i-1), array_type(i-1), array_type(i),IERROR)
         endif
         i=i+1
      enddo
      type_sub_region=array_type(dim)
      end
      
******************************************
c     
c     retourne la taille d'un tableau donc l'espace des indices 
c     est définit par region
      
      function STEP_SizeRegion(dim,region)
      implicit none
      integer dim,d,STEP_sizeRegion
      integer L,U
      parameter (L=1, U=2)
      integer region(L:U,dim)
      STEP_sizeRegion=1
      do 10 d=1,dim
         STEP_sizeRegion=STEP_sizeRegion*(region(U,d)-region(L,d)+1)
 10   continue
      end

c     
c     retourne l'indice dans un tableau linéaire dont l'indice commence à 1,
c     d'une case de cordonnées coords définit pour l'espace d'indice  region 
c     
      function indice(dim,coords,region)
      implicit none
      integer dim,d,indice
      integer coords(dim)
      integer L,U
      parameter (L=1, U=2)
      integer region(L:U,dim)
      indice = 0
      do 10 d=dim,1,-1
         indice=indice*(region(U,d)-region(L,d)+1)+
     &        coords(d)-region(L,d)
 10   continue
      indice=indice+1
      end
      
c     
c     retourne l'indice dans un tableau linéaire dont l'indice commence à 1,
c     de l'origine d'une sous-region subregion, définit dans l'espace d'indice
c     definit par region
c     
      function origine(dim,region,subregion)
      implicit none
      include 'STEP.h'
      integer dim,d,origine,indice
      integer L,U
      parameter (L=1,U=2)
      integer region(L:U,dim)
      integer subregion(L:U,dim)
      integer coords(STEP_MAX_DIM)
      integer*4 iBidon, IERROR

      if (dim .gt. STEP_MAX_DIM) then
         print *,"origine() : STEP_MAX_DIM 
     &trop petit (update STEP.h) dim = ",dim
         call MPI_ABORT(MPI_COMM_WORLD, iBidon, IERROR)
      end if
      do 10 d=1,dim
         coords(d)=subregion(L,d)
 10   continue
      origine=indice(dim,coords,region)
!      print *,'\nregion=',region,'\nsubregion=',subregion,
!     &     '\ncoords=',coords,'\norigine=',origine
      end

******************************************
       subroutine STEP_WaitAll(NbReq,Request)
      implicit none
      include 'STEP.h'
      integer*4 NbReq,Request(NbReq)
      integer*4 Status(MPI_STATUS_SIZE,STEP_MAX_NBREQ)
      integer*4 iErr,iBidon

      if (NbReq .gt. STEP_MAX_NBREQ) then
         print *,"STEP_WaitAll() : STEP_MAX_NBREQ
     &trop petit (update STEP.h) NbReq = ",NbReq
         call MPI_ABORT(MPI_COMM_WORLD, iBidon, iErr)
      end if
      call MPI_BARRIER(MPI_COMM_WORLD,iErr)
      if (iErr .ne. MPI_SUCCESS) then
         print *,'STEP_WaitAll() : MPI_BARRIER() Pb'
         call MPI_ABORT(MPI_COMM_WORLD, iBidon, iErr)
      end if
      if (NbReq .ne. 0) then
         call mpi_waitall(NbReq,Request,Status,iErr)
         if (iErr .ne. MPI_SUCCESS) then
            print *,'STEP_WaitAll() : mpi_wait_all() Pb'
            call MPI_ABORT(MPI_COMM_WORLD, iBidon, iErr)
         end if
      end if
      end

#include "STEP_COM_SUBROUTINES_BODY.f"
#include "STEP_COM_SUBROUTINES_BODY_INTERLACED.f"

