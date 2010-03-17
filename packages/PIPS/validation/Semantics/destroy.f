      subroutine destroy

c     Check transformer for called user routine. It's not possible
c     to neglect effect and to use only the callee's transformer

      common /foo/i,j,k,l

      i = i + 1
      j = j + 2
      k = k + 3
      l = l + 4

      call bar

      print *, i, j, k, l

      end

      subroutine bar
      common /foo/ii,ndummy(2),l
      real l

      do i = 1, 2
cfirst         ndummy(i) = 0
      enddo

cfirst      l = 0.

      end
