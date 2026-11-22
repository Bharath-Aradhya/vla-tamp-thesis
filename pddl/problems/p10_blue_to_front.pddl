;; Three bars in a row; red goes onto green and blue moves to the front slot t1.
(define (problem p10-blue-to-front)
  (:domain tabletop-single-arm)
  (:objects red blue green - item  t1 t2 t3 bin1 bin2 - location)
  (:init (hand-empty) (table-spot t1) (table-spot t2) (table-spot t3)
         (reachable t1) (reachable t2) (reachable t3) (reachable bin1) (reachable bin2)
         (at red t1) (clear red) (at blue t2) (clear blue) (at green t3) (clear green)
         (free bin1) (free bin2))
  (:goal (and (at blue t1) (on red green))))
