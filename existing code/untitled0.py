import random 
 
def twoRandomNumbers(a,b): 
    test = random.random() # random float 0.0 <= x < 1.0 
    
    if test < 0.5: 
        return a 
    else: 
        return b

N1=20
for x in range(N1):
    a=twoRandomNumbers(0, 1)
    print(a)