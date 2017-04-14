
from random import seed,random,randrange,randint,choice,shuffle
from math import exp,sqrt,ceil
from copy import deepcopy
from datetime import datetime,date
from os import mkdir
from random import choice
from operator import itemgetter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import scatter
import statsmodels.api as sm
from scipy.stats import truncnorm
from scipy.stats import beta
#This program simulates a poplation of males and females who have binomially distributed levels of European admixture
# (represented as a percentage of their genome).

#-------------------------------------------------------------------------
#-----------------------    Paramaters    --------------------------------
#-------------------------------------------------------------------------
number_pairs=10 #choose even number
p=.9
mu=1/(79+79+31)*(79*.172+79*.256+31*.472)
sigma=1/(79+79+31)*(79*(.134**2+.172)+79*(.142**2+.256)+31*(.158**2+.472))-mu**2
alpha=((1-mu)/sigma**2-1/mu)*mu**2
beta=alpha*(1/mu-1)
encounters=number_pairs  #must be less than popsize
Males_choose=True  #If false, then females propose to males
#-----------------------------------------------------------------------------------
#-----------------------    Init Population Functions    ---------------------------
#-----------------------------------------------------------------------------------

#def init_pop(popsize,mean,var):
#    males=[]
#    females=[]
#    for i in range(ceil(popsize/2)):
#        a=np.random.normal(mean, var, 1)
#        males.append(float("{0:.2f}".format(a[0]))) #restrict admixture fractions to 2 decimal places
#        a=np.random.normal(mean, var, 1)
#        females.append(float("{0:.2f}".format(a[0])))
#    return males, females
 
def init_pop_normal(mu,sigma,number_pairs):
    male_variates=np.random.normal(mu, sigma, number_pairs)
    female_variates=np.random.normal(mu, sigma, number_pairs)
    males=[ [i,male_variates[i]] for i in range(number_pairs)]
    females=[ [i,female_variates[i]] for i in range(number_pairs)]
    return males, females

def init_pop_beta(mu,sigma,number_pairs):
    alpha=((1-mu)/sigma**2-1/mu)*mu**2
    beta=alpha*(1/mu-1)
    male_variates=np.random.beta(alpha, beta, number_pairs)
    female_variates=np.random.beta(alpha, beta, number_pairs)
    males=[ [i,male_variates[i]] for i in range(number_pairs)]
    females=[ [i,female_variates[i]] for i in range(number_pairs)]
    return males, females

def init_pop_binom(popsize,mean):  #This function produces a group of males and females who 
    males=[]
    females=[]
    n=100
    for i in range(ceil(popsize/2)):
        a=np.random.binomial(n, p, 1)
        males.append([i,a[0]]) #restrict admixture fractions to 2 decimal places
        a=np.random.binomial(n, p, 1)
        females.append([i,a[0]])
    return males, females

#-------------------------------------------------------------------------------
#-----------------------    Interaction Functions    ---------------------------
#-------------------------------------------------------------------------------     
     
def remove_multiples(pref_list,tag):
    temp_list=[]
    for mate in pref_list[tag]:
        if mate not in temp_list: temp_list.append(mate)
    pref_list[tag]=temp_list
    return pref_list[tag]

    
def show_multiples(pref_list):
    items_to_remove=[]
    for i in range(len(pref_list)):
        for j in range(len(pref_list)):
            if pref_list[i]==pref_list[j] and i!=j: items_to_remove.append(pref_list[j])
    print('items to remove = ',items_to_remove)
#    for index in range(len(items_to_remove)): pref_list.remove(items_to_remove[index])

def simple_build_preferences(males,females):
    male_pref_list=[]
    female_pref_list=[]
#    males.sort(key=itemgetter(1))
#    females.sort(key=itemgetter(1))
    for f in range(len(females)): 
        female_pref_list.append(sorted(males,key=itemgetter(1)))
        male_pref_list.append(sorted(females,key=itemgetter(1)))
    #for i in range(len(males)):
     #   males.sort(key=itemgetter(1))
      #  females.sort(key=itemgetter(1))
       # male_pref_list.append(females)
        #female_pref_list.append(males)
    return male_pref_list,female_pref_list

def build_preferences(males,females,encounters):               #build preference list from N encounters  We have to fix this to get the rpeferences ordering right
    male_pref_list=[[] for i in range(len(males))]
    female_pref_list=[[] for i in range(len(females))]
    for i in range(encounters):                             #Here we assign partners in the random encounters
        male_perm=np.random.permutation(males).tolist()
        for f in range(len(females)): 
            female_pref_list[f].append(male_perm[f])
            ind=int(male_perm[f][0])
            male_pref_list[ind].append(females[f])
    for m in range(len(males)):
        extra_females=[]
        male_pref_list[m]=remove_multiples(male_pref_list,m)
        male_pref_list[m].sort(key=itemgetter(1)) 
#        show_multiples(male_pref_list[m])
#        print('male ',m,' length is ',len(male_pref_list[m]))
        for female in females:
            if female not in male_pref_list[m]: extra_females.append(female)
        unencountered_females=np.random.permutation(extra_females).tolist()
        for mate in unencountered_females: male_pref_list[m].append(mate)
#        print('length is ',str(len(male_pref_list[m])))
    for f in range(len(females)):
        extra_males=[]
        female_pref_list[f]=remove_multiples(female_pref_list,f)
        female_pref_list[f].sort(key=itemgetter(1))
#        show_multiples(female_pref_list[f])
#       print('female ',f,' length is ',len(female_pref_list[f]))
        for male in males:
            if male not in female_pref_list[f]: extra_males.append(male)
        unencountered_males=np.random.permutation(extra_males).tolist()
        #if len(unencountered_males)+len(female_pref_list[f])!=ceil(popsize/2):
            #print('size problem at ',f)
            #print('unenc + enc = ',len(unencountered_males),' + ',len(female_pref_list[f]))
        for mate in unencountered_males: female_pref_list[f].append(mate)
    return male_pref_list,female_pref_list
#This function produces a specified number of random encounters and then
#determines each individuals preferences across those encounters.  All
# the individuals not encountered will be ranked equally at the end of the 
#preference list.

#Also need to deal with the fact that a male may run out of the prefered females and must choose at random from females that have not been encountered.
           
def Top_choices(suitors_list,female_pref_list):
    global best_suitors
    best_suitors=[[] for i in range(len(female_pref_list))]
    for s,suitor in enumerate(suitors_list):
        if suitor:
            for mate in suitor: 
                if mate not in female_pref_list[s]: print('item error for ',mate)
            index_list=[female_pref_list[s].index(mate) for mate in suitor]  #list of all indices for suitors of a given female
            best_suitors[s].append(female_pref_list[s][max(index_list)])  #find suitor with the highest index
    #print(best_suitors)
    return best_suitors

     
def Gale_Shapley(males,females,Males_choose,male_pref_list,female_pref_list,suitors_list):
    for m,male in enumerate(males):
        proposal=male_pref_list[m].pop()
        suitors_list[int(proposal[0])].append(male)
        #print(male)
    suitors_list=Top_choices(suitors_list,female_pref_list)
    unproposed=[suitor for suitor in suitors_list if not suitor]
    print(unproposed)
    if unproposed:
        leftover_males=[male for male in males if male not in suitors_list]
        return Gale_Shapley(leftover_males,females,Males_choose,male_pref_list,female_pref_list,suitors_list)
    else:
        return suitors_list
                    

#------------------------------------------------------------------------
#-----------------------    Simulation    ---------------------------
#-------------------------------------------------------------------------    

#males,females=init_pop_normal(mu,sigma,number_pairs)
#males,females=init_pop_beta(mu,sigma,number_pairs)
male_pref_list,female_pref_list=build_preferences(males,females,encounters)
male_pref_list,female_pref_list=simple_build_preferences(males,females)
suitors_list=[[] for i in range(len(female_pref_list))]
male_final_choices=Gale_Shapley(males,females,Males_choose,male_pref_list,female_pref_list,suitors_list)
male_final_choices=[item for place in male_final_choices for item in place]
male_admix=[male[1] for male in male_final_choices]
female_admix=[female[1] for female in females]
pairs =[[male[0],females[male_final_choices.index(male)]] for male in male_final_choices]

#define plot size in inches (width, height) & resolution(DPI)
plt.figure(figsize=(5, 5), dpi=100)
plt.rc("font", size=16)
results = sm.OLS(male_admix,sm.add_constant(female_admix)).fit()
scatter(female_admix,male_admix)
X_plot = np.linspace(0,.1,1.0)
line=plt.plot(X_plot, X_plot*results.params[0],'-',linewidth=2.0)
fit = np.polyfit(female_admix,male_admix,1)
fit_fn = np.poly1d(fit) 
plt.plot(female_admix,male_admix, 'yo', female_admix, fit_fn(female_admix), '--k')
plt.ylabel('male admixture')
plt.xlabel('female admixture')
plt.axis([0, 1, 0, 1])
plt.show()
figsize=(1, 1)


#def Gale_shapley(males,females,Males_choose,male_pref_list,female_pref_list,suitors_list):     #Use Gale Shapley algorithm to pair individuals
#    for m in range(len(males)): 
#        female_ind=male_pref_list[m].pop()
#        ind=int(female_ind[0])
#        suitors_list[ind].append(m)        
#    for f in range(len(female_pref_list)):
#        if not suitors_list[f]: suitors_list[f]=suitors_list[f]
#        else: suitors_list[f]=check_suitors(suitors_list,female_pref_list,f)
#    leftover_males=[]
#    for m in range(len(males)):
#        if m not in suitors_list: leftover_males.append(m)
#    unproposed=0
#    for female in suitors_list: 
#        if not female: unproposed+=1
#    if unproposed>0: Gale_shapley(leftover_males,females,Males_choose,male_pref_list,female_pref_list,suitors_list)
#    else: 
#        return male_final_choices
 

#def check_suitors(suitors_list,female_pref_list,f):
#    index_list=[]
#    for suitor in suitors_list[f]:
#        #print('the suitors are ',suitor)
#        #print('the pref list is ',female_pref_list[f])
#        #if mate not in female_pref_list[f]: print('suitors are ',suitors_list[f])
#        if suitor not in female_pref_list[f]: print(suitor,'not in our list for female ',f)
#        index_list.append(female_pref_list[f].index(suitor))
#    best=female_pref_list[f][int(max(index_list))]
#    #print('best is ',best)
#    return best       
        
       

        
#def Gale_shapley(males,females,Males_choose,male_pref_list,female_pref_list,suitors_list,recursion_counter):     #Use Gale Shapley algorithm to pair individuals
#    recursion_counter+=1
#    print('recursion counter= ',recursion_counter)    
#    male_final_choices=[]
#    #print('length of suitors_list is ',len(suitors_list))
#    for male in males:
#        male_final_choices.append([]) #building the list this way assumes equal number of men and women
#        female_preferred=male_pref_list[male[0]].pop()
#        ind=int(female_preferred[0])
# #       if len(males[m])<2: print('here is issue',males[m])
#        suitors_list[ind].append(male) 
#    for suitor in suitors_list: print('suitor is ',suitor)
#    for f in range(len(female_pref_list)):
#        if not suitors_list[f]: suitors_list[f]=suitors_list[f]
#        else: 
#            #print('calling check_suitors')
#            #print('index is ',f)
#            best=check_suitors(suitors_list,female_pref_list,f)
#            for i in suitors_list[f]: suitors_list[f].remove(i)
#            suitors_list[f].append(best)
#    leftover_males=[]
#    for m in range(len(males)):
#        counter=0
#        for suitor in suitors_list:
#            if males[m] in suitor: counter+=1
#        if counter==0: leftover_males.append(males[m])
#    unproposed=0
#    for female in suitors_list: 
#        if not female: unproposed+=1
#    if recursion_counter>10: sys.exit()
#    #print('length of suitors_list is ',len(suitors_list))
#    if unproposed>0:
#        print('left overs are ',leftover_males)
#        Gale_shapley(leftover_males,females,Males_choose,male_pref_list,female_pref_list,suitors_list,recursion_counter)
#    else:
#        male_final_choices=suitors_list
#        return male_final_choices           