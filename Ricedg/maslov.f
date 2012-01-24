C     example by Vadim Maslov, SIGPLAN PLDI 92
      program maslov
      real c(0:99)

      do i = 0, 4
         do j = 0, 9
            c(i+10*j) = c(i+10*j+5)
         enddo
      enddo

      end
