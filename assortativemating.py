
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
from scipy.stats import truncnorm, pearsonr
from scipy.stats import beta
from itertools import permutations
import pandas as pd
#This program simulates a poplation of males and females who have binomially distributed levels of European admixture
# (represented as a percentage of their genome).

#-------------------------------------------------------------------------
#-----------------------    Parameters    --------------------------------
#-------------------------------------------------------------------------
number_pairs=10 #choose even number
p=.9
mu=1/(79+79+31)*(79*.172+79*.256+31*.472)
sigma=1/(79+79+31)*(79*(.134**2+.172)+79*(.142**2+.256)+31*(.158**2+.472))-mu**2
alpha=((1-mu)/sigma**2-1/mu)*mu**2
beta=alpha*(1/mu-1)
#encounters=10 #must be less than/equal to popsize
Males_choose=True  #If false, then females propose to males
Assort=True
Uniform=False
Random=False
num_runs=2
show_plot3=False
mult_gen=True #use input to control this
num_generations=100
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

def init_mult_gen(males,females,final_choice):
    new_males=[]
    new_females=[]
    for m in range(len(males)):
        mom=np.random.randint(0,len(females)-1)
        dad=final_choice[mom]
        new_males.append([m,(males[dad][1]+females[mom][1])/2])
        mom=np.random.randint(0,len(females)-1)
        dad=final_choice[mom]
        new_females.append([m,(males[dad][1]+females[mom][1])/2])
    return new_males,new_females

#-------------------------------------------------------------------------------
#-----------------------    Interaction Functions    ---------------------------
#-------------------------------------------------------------------------------     
     
def build_preferences_assortative(males,females,encounters):
    male_pref_list=[[] for i in range(len(males))]
    female_pref_list=[[] for i in range(len(females))]
    for ind, male in enumerate(males):
        meetings=[[] for i in range(encounters)]
        sample=np.random.choice(number_pairs,encounters,replace=False).tolist()
        for index, element in enumerate(meetings):
            element.append(sample[index])
            element.append(abs(male[1]-females[sample[index]][1]))
        meetings.sort(key=itemgetter(1),reverse=False)
        for item in meetings: male_pref_list[ind].append(females[item[0]])
        female_index=[n for n in range(number_pairs) if n not in sample]
        female_index=np.random.permutation(female_index).tolist()
        for index in female_index: male_pref_list[ind].append(females[index])
    for ind, female in enumerate(females):
        meetings=[[] for i in range(encounters)]
        sample=np.random.choice(number_pairs,encounters,replace=False).tolist()
        for index, element in enumerate(meetings):
            element.append(sample[index])
            element.append(abs(female[1]-males[sample[index]][1]))
        meetings.sort(key=itemgetter(1),reverse=False)
        for item in meetings: female_pref_list[ind].append(males[item[0]])
        male_index=[n for n in range(number_pairs) if n not in sample]
        male_index=np.random.permutation(male_index).tolist()
        for index in male_index: female_pref_list[ind].append(males[index])
    return male_pref_list,female_pref_list

def build_preferences_random(males,females,encounters):
    male_pref_list=[[] for i in range(len(males))]
    female_pref_list=[[] for i in range(len(females))]
    for ind, male in enumerate(males):
        meetings=np.random.choice(number_pairs,encounters,replace=False).tolist()
        for item in meetings: male_pref_list[ind].append(females[item])
        female_index=[n for n in range(number_pairs) if n not in meetings]
        female_index=np.random.permutation(female_index).tolist()
        for index in female_index: male_pref_list[ind].append(females[index])
    for ind, female in enumerate(females):
        meetings=np.random.choice(number_pairs,encounters,replace=False).tolist()
        for item in meetings: female_pref_list[ind].append(males[item])
        male_index=[n for n in range(number_pairs) if n not in meetings]
        male_index=np.random.permutation(male_index).tolist()
        for index in male_index: female_pref_list[ind].append(males[index])
    return male_pref_list,female_pref_list

def build_preferences_uniform(males,females,encounters):
    male_pref_list=[[] for i in range(len(males))]
    female_pref_list=[[] for i in range(len(females))]
    for ind, male in enumerate(males):
        meetings=np.random.choice(number_pairs,encounters,replace=False).tolist()
        for item in meetings: male_pref_list[ind].append(females[item])
        male_pref_list[ind].sort(key=itemgetter(1),reverse=True)                   
        female_index=[n for n in range(number_pairs) if n not in meetings]
        female_index=np.random.permutation(female_index).tolist()
        for index in female_index: male_pref_list[ind].append(females[index])
    for ind, female in enumerate(females):
        meetings=np.random.choice(number_pairs,encounters,replace=False).tolist()
        for item in meetings: female_pref_list[ind].append(males[item])
        female_pref_list[ind].sort(key=itemgetter(1),reverse=True)
        male_index=[n for n in range(number_pairs) if n not in meetings]
        male_index=np.random.permutation(male_index).tolist()
        for index in male_index: female_pref_list[ind].append(males[index])
    return male_pref_list,female_pref_list
#Important: This function sorts highest preference from right to left so that the preferences can be popped in the Gale Shapley function.        

def gale_shapley(leftover_males,males,females,Males_choose,male_pref_list,female_pref_list,suitors_list):
    for m, male in enumerate(males):
        if male in leftover_males:
            proposal=male_pref_list[m].pop()
            suitors_list[proposal[0]].append(m)
    for female,proposals in enumerate(suitors_list):
        if proposals:
            suitors_indices=[female_pref_list[female].index(males[m]) for m in proposals]
            #print('suitors indices = ',suitors_indices)
            suitors_list[female]=[female_pref_list[female][max(suitors_indices)][0]]
    check_list=[y for x in suitors_list for y in x]
    leftover_males=[male for male in males if male[0] not in check_list]
    if leftover_males:
        return gale_shapley(leftover_males,males,females,Males_choose,male_pref_list,female_pref_list,suitors_list)
    else:
        return suitors_list


def iterative_gale_shapley(leftover_males,males,females,Males_choose,male_pref_list,female_pref_list,suitors_list):
    max_iterations=number_pairs**2-2*number_pairs+2    
    for i in range(max_iterations):
        for m, male in enumerate(males):
            if male in leftover_males:
                proposal=male_pref_list[m].pop()
                suitors_list[proposal[0]].append(m)
        for female,proposals in enumerate(suitors_list):
            if proposals:
                suitors_indices=[female_pref_list[female].index(males[m]) for m in proposals]
                suitors_list[female]=[female_pref_list[female][max(suitors_indices)][0]]
        check_list=[y for x in suitors_list for y in x]
        leftover_males=[male for male in males if male[0] not in check_list]
        if not leftover_males: break
    return suitors_list
#This function produces a specified number of random encounters and then
#determines each individuals preferences across those encounters.  All
# the individuals not encountered will be ranked equally at the end of the 
#preference list.

#------------------------------------------------------------------------
#-----------------------    Simulation    ---------------------------
#-------------------------------------------------------------------------    



def simulation(Assort,Uniform,Random,number_pairs,encounters,mu,num_runs,show_plots):
    corrs=0
    for t in range(num_runs):    
        males,females=init_pop_beta(mu,sigma,number_pairs)
        #males,females=init_pop_normal(mu,sigma,number_pairs)
        if Uniform==True: male_pref_list,female_pref_list=build_preferences_uniform(males,females,encounters)
        if Assort==True: male_pref_list,female_pref_list=build_preferences_assortative(males,females,encounters)
        if Random==True: male_pref_list,female_pref_list=build_preferences_random(males,females,encounters)
        for ind in range(len(male_pref_list)): male_pref_list[ind]=list(reversed(male_pref_list[ind]))
        for ind in range(len(female_pref_list)): female_pref_list[ind]=list(reversed(female_pref_list[ind]))
        suitors_list=[[] for i in range(number_pairs)]
        counter=0
        leftover_males=males
        final_choice=iterative_gale_shapley(leftover_males,males,females,Males_choose,male_pref_list,female_pref_list,suitors_list)
        #final_choice=gale_shapley(leftover_males,males,females,Males_choose,male_pref_list,female_pref_list,suitors_list)
        final_choice=[y for x in final_choice for y in x]
        male_admix=[males[i][1] for i in final_choice]
        female_admix=[female[1] for female in females]
        #define plot size in inches (width, height) & resolution(DPI)
        if show_plots==True:
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
            print(results.summary())
        corr_mat=np.corrcoef(male_admix,female_admix)
        corrs+=corr_mat[0,1]/num_runs
    #print('Correlation for Assort= ',str(Assort),', Uniform= ',str(Uniform),', Random= ',str(Random),' = ',str(corrs))
    return corrs
    
def simulation_mult_generations(Assort,Uniform,Random,number_pairs,encounters,mu,num_runs,show_plots,num_generations):
    data1=[] 
    data2=[]      
    for t in range(num_runs):
        corrs_vector=[]
        mean_admix=[]
        males,females=init_pop_beta(mu,sigma,number_pairs)
        for gen in range(num_generations):
            if gen>0: males,females=init_mult_gen(males,females,final_choice)
            if Uniform==True: male_pref_list,female_pref_list=build_preferences_uniform(males,females,encounters)
            if Assort==True: male_pref_list,female_pref_list=build_preferences_assortative(males,females,encounters)
            if Random==True: male_pref_list,female_pref_list=build_preferences_random(males,females,encounters)
            for ind in range(len(male_pref_list)): male_pref_list[ind]=list(reversed(male_pref_list[ind]))
            for ind in range(len(female_pref_list)): female_pref_list[ind]=list(reversed(female_pref_list[ind]))
            suitors_list=[[] for i in range(number_pairs)]
            counter=0
            leftover_males=males
            final_choice=iterative_gale_shapley(leftover_males,males,females,Males_choose,male_pref_list,female_pref_list,suitors_list)
            #final_choice=gale_shapley(leftover_males,males,females,Males_choose,male_pref_list,female_pref_list,suitors_list)
            final_choice=[y for x in final_choice for y in x]
            male_admix=[males[i][1] for i in final_choice]
            female_admix=[female[1] for female in females]
            corr_mat=np.corrcoef(male_admix,female_admix)
            corrs_vector.append(corr_mat[0,1])
            mean_admix.append(np.mean(male_admix+female_admix))
        data1.append(corrs_vector)
        data2.append(mean_admix)
        #corrs+=corr_mat[0,1]/num_runs
    #print('Correlation for Assort= ',str(Assort),', Uniform= ',str(Uniform),', Random= ',str(Random),' = ',str(corrs))
    return data1,data2    
  
  
#-------------------------------------------------------------------------
#----------------------           Main          --------------------------
#-------------------------------------------------------------------------
#mult_gen=bool(input('Multipe generations results? (True/False) \n'))
#single_gen=bool(input('Single generation results? (True/False) \n))

#f single_gen==True:
assort_corrs=[]
for i in range(number_pairs):
    Assort=True
    Uniform=False
    Random=False
    encounters=i+1  
    corrs=simulation(Assort,Uniform,Random,number_pairs,encounters,mu,num_runs,show_plots)    
    assort_corrs.append(corrs) 
uniform_corrs=[]
for i in range(number_pairs):
    Assort=False
    Uniform=True
    Random=False 
    encounters=i+1
    corrs=simulation(Assort,Uniform,Random,number_pairs,encounters,mu,num_runs,show_plots)    
    uniform_corrs.append(corrs)
d={'assort':assort_corrs,'uniform':uniform_corrs}
df=pd.DataFrame(data=d)
df.plot(kind='line',title='mean corr. vs. sample size (single generation)')


encounters=number_pairs
if mult_gen==True:
    #num_generations=int(input('number of generations \n'))
    Assort=False
    Uniform=True
    Random=False 
    cdata,mdata=simulation_mult_generations(Assort,Uniform,Random,number_pairs,encounters,mu,num_runs,show_plots,num_generations)
    cdata=np.transpose(cdata)
    mdata=np.transpose(mdata)
    meancorr=[np.mean(cdata[i]) for i in range(len(cdata))]
    meanadmix=[np.mean(mdata[i]) for i in range(len(mdata))]
    column_names=['run'+str(i+1) for i in range(len(cdata[0]))]
    unicdf=pd.DataFrame(data=cdata,columns=column_names)
    unimdf=pd.DataFrame(data=mdata,columns=column_names)
    unimeancdf=pd.DataFrame(data=meancorr)
    unimeanmdf=pd.DataFrame(data=meanadmix)    
    Assort=True
    Uniform=False
    Random=False 
    cdata,mdata=simulation_mult_generations(Assort,Uniform,Random,number_pairs,encounters,mu,num_runs,show_plots,num_generations)
    cdata=np.transpose(cdata)
    mdata=np.transpose(mdata)
    meancorr=[np.mean(cdata[i]) for i in range(len(cdata))]
    meanadmix=[np.mean(mdata[i]) for i in range(len(mdata))]
    column_names=['run'+str(i+1) for i in range(len(cdata[0]))]
    asscdf=pd.DataFrame(data=cdata,columns=column_names)
    assmdf=pd.DataFrame(data=mdata,columns=column_names)
    assmeancdf=pd.DataFrame(data=meancorr)
    assmeanmdf=pd.DataFrame(data=meanadmix) 
    
unicdf.plot(title='corr vs gen (uniform)',ylim=[0,1])
unimdf.plot(title='mean admix (uniform)',ylim=[0,1])
asscdf.plot(title='corr vs gen (assort)',ylim=[0,1])
assmdf.plot(title='mean admix (assort)',ylim=[0,1])
unimeancdf.plot(title='mean corr vs. gen (uniform)',ylim=[0,1])
unimeanmdf.plot(title='mean admix vs. gen (uniform)',ylim=[0,1])
assmeancdf.plot(title='mean corr vs. gen (assort)',ylim=[0,1])
assmeanmdf.plot(title='mean admix vs. gen (assort)',ylim=[0,1])

 



    



 
