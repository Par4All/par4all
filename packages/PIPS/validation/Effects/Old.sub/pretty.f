      program pretty

c     Check prettyprinting of effects

      call foo

      call bar

      end

      subroutine foo
      common /foo_common/ alpha, beta, gamma, delta

      alpha = 0.
      beta = 0.
      gamma = 0.
      delta = 0.

      end

      subroutine bar
      common /bar_common/ alpha, beta, gamma, delta

      alpha = 0.
      beta = 0.
      gamma = 0.
      delta = 0.

      end
