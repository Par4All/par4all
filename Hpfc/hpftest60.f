c scratched dimension and non ordered independent
      program hpftest60
      integer i, j, k, n
      parameter (n=10)
      integer a(n,n,n)
chpf$ template t(n,n)
chpf$ processors p(2,2)
chpf$ distribute t(block,block) onto p
chpf$ align a(*,i,j) with t(i,j)
      print *, 'hpftest60 running'
chpf$ independent(i,j,k)
      do k=1, n
         do j=1, n
            do i=1, n
               a(i,j,k) = (i-1)*100 + (j-1)*10 + k - 1
            enddo
         enddo
      enddo
      print *, (a(i,i,i), i=1, n)
      end
