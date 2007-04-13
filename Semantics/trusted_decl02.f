      subroutine trusted_decl02

      real a(100)

      read *, n

      call init(a, n)

      print *, n

      end

      subroutine init(a, n)
      real a(n)
      real b(1, 4)

      do i = 1, n
         a(i) = 0.
      enddo

      n = 0

      end

