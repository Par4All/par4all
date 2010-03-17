! cellular automata (from adaptor/simple/ca.hpf)
      program automata

! n = size of automata
! m = number of iterations
      integer n, m, i, j
      parameter (n=70,m=40)

      logical field(0:n+1,3)
!hpf$ processors p(2)
!hpf$ distribute field(block,*) onto p

      character s(n)
!hpf$ align with field(i,*):: s(i)

! initialize
!hpf$ independent
      do i=1, n
         field(i,1)=.false.
      enddo
!fcd$ local
      field(n/2, 1)=.true.
!fcd$ end local
      
      do j=1, m
         
! I must model the Circular SHIFTs...
!fcd$    local
            field(0,1)=field(n,1)
!fcd$    end local
!fcd$    local
            field(n+1,1)=field(1,1)
!fcd$    end local
!hpf$    independent
         do i=1, n
            field(i,2)=field(i-1,1)
            field(i,3)=field(i+1,1)
         enddo

! now compute new value 
!hpf$    independent
         do i=1, n
            field(i,1)=field(i,2).neqv.field(i,3)
         enddo

! compose and print out the result
!hpf$    independent
         do i=1, n
            if (field(i,1)) then
               s(i)= ' '
            else
               s(i)= '*'
            endif
         enddo

         print *, s

      enddo

      end
