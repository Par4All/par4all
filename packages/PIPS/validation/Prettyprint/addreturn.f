      subroutine addreturn(n)
C     Bug: an explicit RETURN is added by the prettyprinter; PIPS output
C     are not PIPS compatible as input in some cases (WP65)
      n = 1
      if(n.ne.1) then
         return
      else
         n = 3
      endif
      end
