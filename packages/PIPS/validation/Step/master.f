!
!
! Example of a master directive
!
! 2008
! Creation: A. Muller, 2008
!
      program master
      implicit none
      integer N
      parameter (N=10)
      integer i,j,a(N)

      i=-3 
!$omp PARALLEL
!$omp DO
      do 10 j=1,N
         a(j)=j
         print *,"do2",j
 10   continue
!$omp END DO
      print *,"parallel1",a,i
      i=-1
!$omp MASTER
      print *,"master",a,i

      do 20 i=1,N
         a(i)=2*i
 20   continue
!$omp END MASTER
!$omp BARRIER   ! no implict barrier for "END MASTER" directive

      print *,"parallel2",a,i
!$omp END PARALLEL

      end
