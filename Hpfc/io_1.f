c test io directive
      program io_1
      real A(100)
      integer i, n
      external print_result
cfcd$ io print_result
chpf$ processors P(4)
chpf$ distribute A(block) onto P
chpf$ independent
      do i=1, 100
         A(i) = real(100-i)
      enddo
      read *, n
      call print_result(A, n, n+5)
      end
      subroutine print_result(X, l, u)
cfcd$ io
      real X(100)
      integer l, u
      integer i
      print *, 'result is: ', (X(i), i=l, u)
      end
