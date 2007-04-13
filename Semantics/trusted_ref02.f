      subroutine trusted_ref02(X, N, M)

      real X(N)

      if(M.lt.1) stop 'illegal parameter'

      do I = 1, M
         X(i) = 0.
      enddo

      print *, N, M

      end
