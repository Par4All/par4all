      subroutine trusted_decl01

C     Use advanced mode!
C     It requires SEMANTICS_TRUST_ARRAY_DECLARATIONS property to be set.

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

      end

