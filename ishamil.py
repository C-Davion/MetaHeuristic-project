import json

with open("distmat.json", 'r') as file:
    dico=json.load(file)
def check_hc(candidate,graph=None): # it's a click so only check if every node is in it once and once only.
    hash=[0]*(len(candidate)+1) #candidate goes from 1 to ...
    for x in candidate:
        if hash[x]!=0:
            return False
        else:
            hash[x]+=1
    return True


    