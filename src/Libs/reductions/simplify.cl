;;;; Simplification of expressions

(require :init)
(require :ri)
(require :util)

(use-package '(:newgen :ri :reduction))

(defun simplify-expression (e)
    (gen-recurse
     e
     ((expression syntax) syntax)
     ((syntax tag) tag)
     ((reference indices)
      (make-expression-reference (reference-variable reference)
				 indices))
     ((call arguments)
      (simplify-call call arguments))))

(defun simplify-call (call arguments)
    (let ((f (call-function call)))
	(cond ((entity-and-p f)
	       (simplify-and arguments))
	      (t (make-expression-call f arguments)))))

(defun simplify-and (arguments)
    (let ((opd1 (first arguments))
	  (opd2 (second arguments)))
	(cond ((true-expression-p opd1) opd2)
	      ((true-expression-p opd2) opd1)
	      (t (make-expression-call f arguments)))))

;;;

(provide :simplify)
