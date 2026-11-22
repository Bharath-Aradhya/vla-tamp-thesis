;; Three bars in a row; build a tower with green at the bottom, red in the middle and blue on top.
(define (problem p08-tower-blue-red-green)
  (:domain tabletop-single-arm)
  (:objects red blue green - item  t1 t2 t3 bin1 bin2 - location)
  (:init (hand-empty) (table-spot t1) (table-spot t2) (table-spot t3)
         (reachable t1) (reachable t2) (reachable t3) (reachable bin1) (reachable bin2)
         (at red t1) (clear red) (at blue t2) (clear blue) (at green t3) (clear green)
         (free bin1) (free bin2))
  (:goal (and (on red green) (on blue red))))
