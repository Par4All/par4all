!
! do_reduction.f
! Example of reduction clauses on a parallel do directive
!
! 2008
! Creation: A. Muller, 2008

      program do_reduction
      implicit none
      integer N
      parameter (N=10)
      integer i,a,t
      integer d(N)
      real*8 b
      real c

      a=0
      b=1
      c=-1

!$OMP PARALLEL DO reduction ( +:A ,c) reduction(*:b)
      do 10 i=1,N
         d(i)=i
         a= a+d(i)
         b= b*i
         c= c+i
 10   continue
!$OMP END PARALLEL DO

      print *,a
      print *,b
      print *,c
      print *,d

      end
