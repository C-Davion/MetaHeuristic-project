from evaluation_st7 import dummyJ
import numpy as np
from configparticule import Walls,W,cognitive_coef,social_coef,Vitesse_scale,Position_scale,iteration_numbers,number_of_particle
from numpy.random import rand

"""
Research on PSO were mostly on how to determine the hyperparameters 
 or varying their values as the algorithm progressed. For example, there are proposals making the inertia weight linear decreasing. There are also proposals trying to make the cognitive coefficient 
 decreasing while the social coefficient 
 increasing to bring more exploration at the beginning and more exploitation at the end. See, for example, Shi and Eberhart (1998) and Eberhart and Shi (2000).
"""


class Particule: # if doesn't work, add the optimize function to the class
    """
        Vitesse and Position can be 3D (4D) arrays to track location and intensity of the explosion
        
        inertie in [0,1]

        param1 cognitive coeff
        param2 social coeff
    """
    def __init__(self,Vitesse_init:np.ndarray,Pos_init:np.ndarray,inertie:np.float64,param1:np.float64,param2:np.float64):
        self.vitesse=Vitesse_init
        self.position=Pos_init
        self.inertie=inertie
        self.c1=param1
        self.c2=param2
        self.pbest=Pos_init
        self.pbest_score=dummyJ(self.position)
        #self.pbest_score=J(self.position) # doit etre modifier
    
    def update_pos_and_speed(self,r1:np.float64,r2:np.float64,gbest:np.ndarray):
        self.vitesse=self.inertie*self.vitesse+self.c1*r1*(self.pbest-self.position)+self.c2*r2*(gbest-self.position)
        self.position=self.position+self.vitesse
        #Check if the particle is beyond the wall, if so, put it to the closest wall and invert the speed
        for i in range(self.position.shape[0]):
            if self.position[i]<Walls[i][0] or self.position[i]>Walls[i][1]:
                self.position[i]=Walls[i][0] if abs(self.position[i]-Walls[i][0])<abs(self.position[i]-Walls[i][1]) else Walls[i][1]
                self.vitesse[i]=-self.vitesse[i]
    def update_pbest(self):
        #score=dummy_J(self.position)
        score=dummyJ(self.position)
        if score<self.pbest_score: 
            self.pbest=self.position



   

def find_gbest(particles):
    
    list_of_pbest=np.array([particle.pbest for particle in particles])
    list_of_pbest_score=np.array([particle.pbest_score for particle in particles])
    gbest_score=min(list_of_pbest_score)
    gbest=list_of_pbest[list_of_pbest_score.argmin()]
    return gbest_score,gbest[0]
        
 

if __name__=="__main__":
    particules=[Particule(Vitesse_init=rand(len(Walls),1).flatten()*Vitesse_scale,Pos_init=rand(len(Walls),1).flatten()*Position_scale,inertie=W,param1=cognitive_coef,param2=social_coef) for _ in range(number_of_particle)]
   
    for k in range(iteration_numbers): #can't do much about this loop
        print(k)
        gbest_score,gbest=find_gbest(particules)
        for particule in particules: #this can be parallalized
            [r1,r2]=rand(2)
            particule.update_pos_and_speed(r1,r2,gbest) #particle is at new postion
            particule.update_pbest() #check if new position is better than previous one
    print(gbest_score,gbest)





    
        
    





    

    
    

