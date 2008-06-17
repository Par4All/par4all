      program descriptive_2
      parameter(n=20)
      integer n, A(n,2), i
chpf$ processors P(4)
chpf$ distribute A(block,*) onto P
chpf$ independent
      do i=1, n
         A(i,1)=i
      enddo
      call iter(A)
      print *, (A(i,2), i=1, n)
      end
      subroutine iter(A)
      parameter(n=20)
      integer n, A(n,2), i
chpf$ processors P(4)
chpf$ distribute A * (block,*) onto P
chpf$ independent
      do i=2, n-1
         A(i,2) = A(i-1,1)+A(i,1)+A(i+1,1)
      enddo
      end
