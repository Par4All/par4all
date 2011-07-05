      subroutine trusted_ref01(X, N, M)

C     Use advanced mode!
C     It requires SEMANTICS_TRUST_ARRAY_REFERENCES property to be set.

      real X(N)

      do I = 1, M
         X(i) = 0.
      enddo

      print *, N, M

      end
