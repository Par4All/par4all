C     bug: information maltraitee quand elle porte sur une variable non visible
C     sur un sous-programme intermediaire, ici datast; cet exemple est tire de ocean.f

C     C'est un choix explicite de semantics, implemente dans translate_global_value()
C     Ca conduit a des erreurs puisqu'une modif de variable globale passe inapercue.
C     J'hesite quand meme a modifier quelque chose d'aussi fondamental tout de suite,
C     dans la mesure ou des ajouts de declaration de COMMONs permettent de contourner
C     le probleme. Le seul cas de coincement vient de la presence de deux variables
C     dans deux COMMONs differents, portant le meme nom.

      program ind
      integer count
      common /foo/ count
      data count /0/

      call datast
      i = 1

      end

      subroutine datast

      call out
      j = 2

      end

      subroutine out
      integer count
      common /foo/ count

      count = count + 1

      end
