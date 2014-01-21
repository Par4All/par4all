Génération de programmes optimisés lors de l’exécution avec support par analyse statique et dynamique
=====================================================================================================

*Optimized program generation with static and dynamic analysis support*

Sujet de stage d’ingénieur ou M2

**Mots-clés** : compilation, parallélisation, accélérateurs matériels,
machines virtuelles, transformation de programmes, analyse statique,
runtime, optimisation

Afin de fournir aux utilisateurs toujours plus de puissance de calcul, on
assiste actuellement au déploiement de nombreuses architectures parallèles
bon marché basées sur différentes technologies : processeurs multi-cœurs,
multiprocesseurs hétérogènes sur puce (Cell, Larrabee…), des coprocesseurs
graphiques (GPGPU nVidia, AMD/ATI…) ou des accélérateurs à base de
circuits logiques reconfigurables.

De nombreux outils basés sur des analyses statiques du code
(parallélisation automatique…) ont été développés par le passé pour cibler
ce genre d’architectures. Malheureusement de nombreuses questions sont
indécidables à la compilation (en fonction de données…) et les
performances ne sont souvent pas à la hauteur.

À l’opposé des approches statiques, des approches purement dynamiques
basées éventuellement sur des principes de machines virtuelles permettent
d’exploiter des connaissances à l’exécution. Malheureusement, ces
approches sont souvent inefficaces dans le cas où on aurait pu optimiser
le code de manière statique.

Le but de ce stage est d’étudier les approches mixtes et en particulier la
génération de manière statique de différents programmes optimisés qui
seront sélectionnés selon des choix faits à l’exécution au moment où on a
les réponses à certaines questions (« ce bout de programme est parallèle
dans le contexte de ces données… », etc).

Le travail consistera à réaliser des phases de transformation en utilisant
le système d’analyse sémantique (basé sur de l’algèbre linéaire) de
l’outil de transformation PIPS http://pips4u.org intégré dans
l’environnement de parallélisation automatique Par4All
(http://par4all.org)

Profil recherché : passionné d’informatique voulant faire de la vraie
informatique et n’ayant pas peur de réfléchir (formalisme mathématique et
sémantique) ni de programmer. Bonnes connaissances utiles : C, Python,
Unix, Emacs, programmation parallèle…

Ce stage peut enchaîner avec une embauche et/ou une thèse.

Encadrant : Ronan Keryell (rk at hpc-project dot com) et collaboration
avec Mines ParisTech/CRI et membres du projet OpenGPU

http://par4all.org

Lieu : HPC Project http://www.hpc-project.com, start-up de 35 personnes en
pleine croissance.

..
  # Some Emacs stuff:
  ### Local Variables:
  ### mode: rst,flyspell
  ### ispell-local-dictionary: "french"
  ### End:
