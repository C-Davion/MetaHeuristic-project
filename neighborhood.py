from numpy import exp,where
from numpy.random import choice, permutation

def voisin1(sol):
        index1, index2 = choice(len(sol), size=2, replace=False)
        sol[index1],sol[index2]=sol[index2],sol[index1]
        return sol

def sigmoid(lower,upper,coeff,x): #makes sigmoid go from lower to upper
        num=upper-lower
        return num/(1+exp(coeff*x))+lower
        

def voisin2(sol,temp):
        ratio=0.5 #ratio for the start of the first selection.
        sigmoid_coeff=0.5 #coefficient for the sigmoid. the higher it goes the sharper it drops
        chosen=int(sigmoid(lower=2,upper=len(sol)*ratio,coeff=sigmoid_coeff,x=temp))
        index_to_permute=choice(len(sol),size=chosen,replace=False)
        print(index_to_permute)
        permuted=permutation(index_to_permute)
        sol = [sol[i] if i not in index_to_permute else sol[permuted[where(index_to_permute == i)[0][0]]] for i in range(len(sol))]
        return sol

