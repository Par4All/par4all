      subroutine trusted_ref03(X, N1, N2, M1, M2)

      real X(N1, N2)

      if(M1.lt.1.or.M2.lt.1) stop 'illegal parameter'

      do I = 1, M1
         do J = 1, M2
            X(i,j) = 0.
         enddo
      enddo

      print *, N1, N2, M1, M2

      end
