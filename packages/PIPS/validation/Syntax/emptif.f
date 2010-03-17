      program emptif

c     Comments in the empty true branch of a test are printed out
c     in the false branch

c     this is the comment about the test
      if(i.gt.1) then
         i = 1
c     this is part of the true branch
      else
         i = 2
c     this is part of the false branch
      endif
c     this is the end of the test
      end
