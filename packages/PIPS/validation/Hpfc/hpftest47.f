      program hpftest47
      parameter (n=100)
      real a(n), amin
chpf$ template t(n)
chpf$ processors p(2)
chpf$ align a(i) with t(i)
chpf$ distribute t(block) onto p
      print *, 'hpftest47 running, reduction example'
chpf$ independent(i)
      do i=1, n
         a(i) = real(mod(i+n+30, n+50)+30)
      enddo
      amin = redmin1(a(1), 1, n)
      print *, 'min of a is ', amin
      end
c
      real function redmin1(a,l,u)
      integer l, u
      real a(l:u), amin
      amin = a(l)
      do i=l+1, u
         if (a(i).LT.amin) amin = a(i)
      enddo
      redmin1 = amin
      return
      end
