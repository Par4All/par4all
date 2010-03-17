      program hpftest24
      integer j(10), i, k, d
chpf$ template t(20)
chpf$ align j(i) with t(2*i-1)
chpf$ processors p(3)
chpf$ distribute t(block) onto p
      print *, 'hpftest24 running'
chpf$ INDEPENDENT(i)
      do i=1, 10
         j(i) = i
      enddo
chpf$ independent(i)
      do i=1,5
         k = 11-i
         d = j(i)
         j(i) = j(k)
         j(k) = d
      enddo
      do i=1,10
         print *, 'j(',i,') = ',j(i)
      enddo
      print *, 'hpftest24 ended'
      end
