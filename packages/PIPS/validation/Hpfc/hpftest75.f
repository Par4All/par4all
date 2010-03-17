! dynamic liveness analysis
      program dla
      real A(10,10)
      integer i, j, answer
!hpf$ dynamic A, T
!hpf$ processors P(2,2)
!hpf$ processors Q(4)
!hpf$ template T(10,10)
!hpf$ align A with T
!hpf$ distribute T(block,block) onto P
!hpf$ independent(j,i)
      do j=1, 10
         do i=1, 10
            A(i,j) = i+j
         enddo
      enddo
      print *, '1 or 0'
      read *, answer
      if (answer.eq.1) then
!
! initial A is not live here, because A is reset.
!
!hpf$    redistribute T(block,*) onto Q
!hpf$    independent(j,i)
         do j=1, 9
            do i=1, 10
               A(i,j) = A(i,j)+A(i,j+1)
            enddo
         enddo
      else
!
! initial A remains live from here
!
!hpf$    redistribute T(*,block) onto Q
         print *, A(5,5)
      endif
!
! bask to initial distribution
!
!hpf$ redistribute T(block,block) onto P
!hpf$ independent(j,i)
      do j=1, 10
         do i=1, 10
            A(i,j) = A(i,j)+1
         enddo
      enddo
      end
