;;;; Initialization file for reduction.cl

(provide :init)

;; Load default path search rules.

(require :pipsinit (pathname "~pips/Pips/pipsinit.cl"))

;; Load NewGen stuff.

(require :newgen (pathname "genLisplib"))

;; Load PIPS internal representation

(require :ri)

;; Use NewGen and Ri packages

(use-package '(:newgen :ri))

;; Initialize NewGen library

(gen-read-spec)
