! kill fc directive with several arguments
      program kills
      integer a(10), b(10), c(10), i

!hpf$ dynamic a,b,c
!hpf$ processors p(2)
!hpf$ align with a:: b, c
!hpf$ distribute a(block) onto p

!hpf$ independent
      do i=1, 10
         a(i) = 100+i
         b(i) = 200+i
         c(i) = 300+i
      enddo

!fcd$ kill a,b,c
!hpf$ redistribute a(cyclic) onto p      

!hpf$ independent
      do i=1, 10
         a(i) = 100+i
         b(i) = 200+i
         c(i) = 300+i
      enddo

      print *, a(3), b(5), c(8)

      end
