c descriptive and overlaps
      program descriptive_3
      integer n
      parameter(n=20)
      integer A(n,2)
chpf$ processors P(4)
chpf$ distribute A(block,*) onto P
      call init(A)
      call iter(A)
      call prnt(A, 2)
      end
c
      subroutine iter(A)
      integer n
      parameter(n=20)
      integer A(n,2), i
chpf$ processors P(4)
chpf$ distribute A * (block,*) onto P ! descriptive
chpf$ independent
      do i=2, n-1
         A(i,2) = A(i-1,1)+A(i,1)+A(i+1,1)
      enddo
      end
c
      subroutine init(A)
      integer n
      parameter(n=20)
      integer A(n,2), i
chpf$ processors P(4)
chpf$ distribute A * (block,*) onto P
chpf$ independent
      do i=1, n
         A(i,1)=i
         A(i,2)=0
      enddo
      end
c
      subroutine prnt(A, side)
      integer n
      parameter(n=20)
      integer A(n,2), i, side
chpf$ processors P(4)
chpf$ distribute A * (block,*) onto P
      print *, (A(i,side), i=1, n)
      end
