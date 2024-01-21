from ishamil import check_hc
from score import load_instance, compute_score_with_mat
import json
from numpy import exp,array
from numpy.random import rand, shuffle
import neighborhood
from matplotlib import pyplot as plt
import numpy as np
from copy import deepcopy

def to_Str(list):
    return [str(elem) for elem in list]


def first_sol(instance):
    sol_init=[i+1 for i in range(len(instance))]
    shuffle(sol_init)
    if check_hc(sol_init):
        return to_Str(sol_init)
    else:
        raise ValueError("It is not a hamiltonian cycle")

def exp_temp(debut,fin,tries,temp,x=0.01):
    A=debut-fin

    C=np.log(A/x/debut)/tries
    k=np.log(A/(temp-fin))/C
    return A*np.exp(-k*C)

def temperature_heuristic(instance,sol_init,dist_mat,tries=100000,temp_init=500,temp_end=10,update_t=None,voisin=None):
    temp=temp_init
    curr_sol=sol_init
    score=compute_score_with_mat(instance,curr_sol,dist_mat)
    nb_violation=score[1]
    curr_score=score[0]
    score_list=[curr_score]
    temp_list=[temp_init]
    proba=1
    num=0

    while tries>num:
        num+=1
        if voisin is None:
            copydesol=deepcopy(curr_sol)
            next_sol=neighborhood.voisin1(copydesol)
        else:
            next_sol=voisin(curr_sol)
        A=compute_score_with_mat(instance,next_sol,dist_mat)
        next_score=A[0]
        
        
        
        if next_score<score_list[-1]:
            proba=1
        else:
            proba=exp(-(next_score-score_list[-1])/temp)
        if rand()<proba:
            curr_sol=next_sol
            score_list.append(next_score)
            nb_violation=A[1]
        else:
            score_list.append(score_list[-1]) #else doesn't show that we retain de same solution
        if update_t is None:
            temp=temp-(temp_init-temp_end)/tries
        else:
            temp=update_t(temp_init,temp_end,tries,temp)
        temp_list.append(temp)
    return curr_sol, score_list,temp_list,nb_violation

def main():
    inst ="data/inst1"

    with open("distmat.json","r") as file:
        dict_data=json.load(file)

    mat1=dict_data["inst1"]
    instance=load_instance(inst)
    keys=instance.keys()
    keys=list(keys)
    #print(type(keys[0]))

    mat1=array(mat1) #convert to numpy array so that score doens't f up
    sol_init=first_sol(instance)
    #print(sol_init)
    print(compute_score_with_mat(instance,sol_init,mat1))
    min=-1
    for i in range(1):
        sol,score_list,temp_list,nb_violation=temperature_heuristic(instance,sol_init,mat1,update_t=None) 
        print(nb_violation)
        if min<0 or score_list[-1]<min:
            min=score_list[-1] 
    
    plt.plot([i for i in range(len(score_list))],score_list)
    # plt.gca().invert_xaxis(); probleme de proba trop elevel; augmente le nombre d iteration. bien baisser les probas. fonction de score: afficher le score
    # pour une meme heuristiaue;courbe+ Intervalle de confiance: avec plusieur fonction de voisinage; et plusieur jeux de parametre.
    
    plt.ylabel("score")
    plt.xlabel("iteration") 
    plt.title(f'Score as a function of temp, with {sol} as the solution for {inst}')
    plt.show()

    



    
if __name__=="__main__":
    main()    



    

