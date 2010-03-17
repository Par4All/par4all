      program hpftest22
      integer j(10),k(10)
chpf$ template t(10)
chpf$ processors p(5)
chpf$ align j(i), k(i) with t(i)
chpf$ distribute t(block) onto p
      print *, 'hpftest22 running'
chpf$ independent(i)
      do i=1, 10
         k(i) = mod(i+6,10)+1
         j(i) = 11-i
      enddo
      do i=1, 10
         print *,'j(k(i)) = ',j(k(i))
      enddo
      if (j(k(3)).LT.j(k(7))) then
         print *, 'quite interesting a property...'
      endif
      print *, 'hpftest22 end'
      end
