from ishamil import check_hc
from numpy.random import choice

def voisin1(sol):
        index1, index2 = choice(len(sol), size=2, replace=False)
        sol[index1],sol[index2]=sol[index2],sol[index1]
        return sol