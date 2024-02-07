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

def temperature_heuristic(instance,sol_init,dist_mat,tries=10000,temp_init=5000,temp_end=10,update_t=None,voisin=None):
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
            temp=temp*(temp_end/temp_init)**(1/tries)
        temp_list.append(temp)
    return curr_sol, score_list,temp_list,nb_violation

def average_and_error(mat):
    avg=[]
    error=[]
    for i in range(len(mat[0])):
        temp=np.zeros(len(mat))
        for j in range(len(mat)):
            temp[j]=mat[j][i]
        average=np.average(temp)
        avg.append(average)
        #low=np.percentile(temp,2.5)
        #up=np.percentile(temp,97.5)
        #error.append(temp[(temp >= low) & (temp <= up)])
        error.append(max(temp)-min(temp)*0.5)
    return avg, error
def main():
    inst ="data/inst1"

    with open("distmat.json","r") as file:
        dict_data=json.load(file)

    mat1=dict_data["inst1"]
    instance=load_instance(inst)
    keys=instance.keys()
    keys=list(keys)
    #print(type(keys[0]))
    hold=[]

    mat1=array(mat1) #convert to numpy array so that score doens't f up
   
    #print(sol_init)
    #print(compute_score_with_mat(instance,sol_init,mat1))
    #min=-1
    for i in range(10):
        sol_init=first_sol(instance)
        _,score_list,_,nb_violation=temperature_heuristic(instance,sol_init,mat1,update_t=exp_temp) 
        hold.append(score_list)
        print(nb_violation)
        #if min<0 or score_list[-1]<min:
        #    min=score_list[-1] 
    avg,error=average_and_error(hold)
    error=np.array(error)
    error_indices = np.arange(len(avg)) % 1000 == 0
    
    plt.errorbar([i for i in range(len(score_list))],avg,yerr=[error[i] if error_indices[i] else 0 for i in range(len(error))])
    # plt.gca().invert_xaxis(); probleme de proba trop elevel; augmente le nombre d iteration. bien baisser les probas. fonction de score: afficher le score
    # pour une meme heuristiaue;courbe+ Intervalle de confiance: avec plusieur fonction de voisinage; et plusieur jeux de parametre.
    
    plt.ylabel("score")
    plt.xlabel("iteration") 
    #plt.xscale('log')
    plt.title(f'Average function of score as a function of iterations, with 95% errors as the solution for {inst}')
    plt.show()

    



    
if __name__=="__main__":
    main()    



    

