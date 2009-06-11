      program interpol
      common /sum/ s(10)
      s = 2.0
      do j=1, 10
         do i=1, 10
            s(i) = s(i) + i
            call addto(s(i), j)
            call addtos(i, j)
         enddo
      enddo
      end
      subroutine addto(x, j)
      x = x + j
      end
      subroutine addtos(i, j)
      common /sum/ s(10)
      s(i)=s(i)+j
      end
