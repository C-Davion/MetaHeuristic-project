from ishamil import check_hc
from score import load_instance, compute_score_with_mat
import json
from numpy import exp,array
from numpy.random import rand, shuffle
import neighborhood
from matplotlib import pyplot as plt


def to_Str(list):
    return [str(elem) for elem in list]


def first_sol(instance):
    sol_init=[i+1 for i in range(len(instance))]
    shuffle(sol_init)
    if check_hc(sol_init):
        return to_Str(sol_init)
    else:
        raise ValueError("It is not a hamiltonian cycle")


def temperature_heuristic(instance,sol_init,dist_mat,tries=100,temp_init=80,temp_end=5,update_t=None,voisin=None):
    temp=temp_init
    curr_sol=sol_init
    curr_score=compute_score_with_mat(instance,curr_sol,dist_mat)
    score_list=[curr_score]
    temp_list=[temp_init]
    proba=1
    while temp>temp_end:
        if voisin is None:
            next_sol=neighborhood.voisin1(curr_sol)
        else:
            next_sol=voisin(curr_sol)
        next_score= compute_score_with_mat(instance,next_sol,dist_mat)
        
        if next_score<score_list[-1]:
            proba=1
        else:
            proba=exp(-(next_score-curr_score)/temp)
        if rand()<proba:
            curr_sol=next_sol
            score_list.append(next_score)
        if update_t is None:
            temp=temp-(temp_init-temp_end)/tries
        else:
            temp=update_t(temp)
        temp_list.append(temp)
    return curr_sol, score_list,temp_list

def main():
    inst1 ="data/inst1"

    with open("distmat.json","r") as file:
        dict_data=json.load(file)

    mat1=dict_data["inst1"]
    instance=load_instance(inst1)
    keys=instance.keys()
    keys=list(keys)
    print(type(keys[0]))

    mat1=array(mat1) #convert to numpy array so that score doens't f up
    sol_init=first_sol(instance)
    print(sol_init)
    print(compute_score_with_mat(instance,sol_init,mat1))
'''
sol,score_list,temp_list=temperature_heuristic(instance,sol_init,mat1)   
    plt.plot(temp_list,score_list)
    plt.title(f'Score as a function of temp, with{sol} as the solution for {inst1}')
    plt.show()

'''
    

    
if __name__=="__main__":
    main()    



    

