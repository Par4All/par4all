C Check optimization for code generated in Guuillaume Capron's PhD dissertation

      program capron02

      logical a(10)
      integer c

      read *, a

      c = 0
      do i = 1, n
         if(a(i)) then
            c = c + 1
         endif
      enddo

      print *, c
      end
