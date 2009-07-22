!
! sum program
!
! Example of a do directive
!
! 2007,2008
! Creation: A. Muller, 2007
! Modification: F. Silber-Chaussumier

      program sum
      implicit none
      integer N
      parameter (N=10)
      integer i,a(N,2),b(N),c(N)
      
      do 5 i=1,N
         a(i,1)=0
         a(i,2)=0
         b(i)=0
         c(i)=-1
 5    continue
!$omp PARALLEL DO
      do 10 i=1,N
         a(i,1)=i
         a(i,2)=2*i
         b(i)=i*10
         c(i)=0
 10   continue
      
!$omp PARALLEL DO
      do 20 i=2,N
         c(i)=a(i-1,1)+a(i,1)+b(i)
 20   continue

!$omp PARALLEL DO
      do 30 i=1,N/2
         b(i)=i
         b(N+1-i)=i
 30   continue
      

      
      print *,a
      print *,b
      print *,c
      end
