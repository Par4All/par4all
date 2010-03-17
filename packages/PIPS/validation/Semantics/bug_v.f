      program bug_v

C     Bug found in CEA's benchv

C     Double occurence of the same variable in a 
C     precondition argument list

      if(x.gt.0.) then
         y = gama(xx, ier)
      else
         n = 3
      endif

      print *, y, n

      end

      function gama(xx,ier)
      common /cg_rgr/xv, n
      XV=XX
      N=2 
      gama = xv
      end
