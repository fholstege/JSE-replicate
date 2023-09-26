


import numpy as np
import pandas as pd
import torch


from JSE.data import *
from JSE.settings import data_info, optimizer_info
from JSE.models import *
from JSE.training import *
import math

from scipy.special import binom as binom_coef
from scipy.stats import binom

import argparse
import os
import sys
from scipy.stats import multinomial

# combine the weighted distributions - for each observation, there is a probaility p_1, ..., p_4 that it comes from d_1, ..., d_4
def draw_mixture(p_1, p_2, p_3, p_4, d_1, d_2, d_3, d_4, n):

            elements = [1, 2, 3, 4]
            probabilities = [p_1, p_2, p_3, p_4]
            print('probabilities: ', probabilities)
            g = np.random.choice(elements, n, p=probabilities)

            mixture = []
            for i in range(n):
                

                group = g[i]
                if group ==1:
                    mixture.append(d_1[i])
                elif group ==2:
                    mixture.append(d_2[i])
                elif group ==3:
                    mixture.append(d_3[i])
                elif group ==4:
                    mixture.append(d_4[i])


            return mixture, g


def analytical_mean( mu_1, mu_2, mu_3, mu_4):

        mean_mixture = (1/4)*(mu_1  + mu_2 + mu_3 + mu_4)

        return mean_mixture


def calc_variance(mu_wd_est, p_1_est, p_2_est, p_3_est, p_4_est, second_moment_1_est, second_moment_2_est, second_moment_3_est, second_moment_4_est):

        p_1_est_inv = 1/p_1_est
        p_2_est_inv = 1/p_2_est
        p_3_est_inv = 1/p_3_est
        p_4_est_inv = 1/p_4_est
        first_part = (1/16)*((p_1_est_inv*second_moment_1_est) + (p_2_est_inv*second_moment_2_est) + (p_3_est_inv*second_moment_3_est) + (p_4_est_inv*second_moment_4_est))
    
        second_part = mu_wd_est**2
    
        var_est = first_part - second_part
    
        return var_est

def mean_estimator(d_1, d_2, d_3, d_4):
      
      
        mean_d_1 = np.mean(d_1)
        mean_d_2 = np.mean(d_2)
        mean_d_3 = np.mean(d_3)
        mean_d_4 = np.mean(d_4)

        mean_est = (1/4)*(mean_d_1 + mean_d_2 +mean_d_3 + mean_d_4)

        return mean_est


def mean_estimator_uncertainty(d_1, d_2, d_3, d_4, n, p_1_est, p_2_est, p_3_est, p_4_est):
     
    expectation_adjustment_1 = calc_expectation_adjustment(n, p_1_est)
    expectation_adjustment_2 = calc_expectation_adjustment(n, p_2_est)
    expectation_adjustment_3 = calc_expectation_adjustment(n, p_3_est)
    expectation_adjustment_4 = calc_expectation_adjustment(n, p_4_est)

    mean_1_est = np.mean(d_1)
    mean_2_est = np.mean(d_2)
    mean_3_est = np.mean(d_3)
    mean_4_est = np.mean(d_4)

    mean_est = (1/4)*((expectation_adjustment_1*mean_1_est) + (expectation_adjustment_2*mean_2_est) + (expectation_adjustment_3*mean_3_est) + (expectation_adjustment_4*mean_4_est))

    return mean_est

def mean_estimator_analytical(mu_1, mu_2, mu_3, mu_4, n, p_1, p_2, p_3, p_4):
     
    expectation_adjustment_1 = calc_expectation_adjustment(n, p_1)
    expectation_adjustment_2 = calc_expectation_adjustment(n, p_2)
    expectation_adjustment_3 = calc_expectation_adjustment(n, p_3)
    expectation_adjustment_4 = calc_expectation_adjustment(n, p_4)
    print('expectation_adjustment_1: ', expectation_adjustment_1)
    print('expectation_adjustment_2: ', expectation_adjustment_2)
    print('expectation_adjustment_3: ', expectation_adjustment_3)
    print('expectation_adjustment_4: ', expectation_adjustment_4)

    mean_est = (1/4)*((expectation_adjustment_1*mu_1) + (expectation_adjustment_2*mu_2) + (expectation_adjustment_3*mu_3) + (expectation_adjustment_4*mu_4))

    return mean_est

     



def calc_variance_estimator(n, sigma_1, sigma_2, sigma_3, sigma_4, n_1, n_2, n_3, n_4):
     
     constant = (1/16)

     var_est = constant * ((sigma_1/n_1+  sigma_2/n_2 + sigma_3/n_3 +  sigma_4/n_4))

     return var_est

def calc_variance_for_group_estimator(n, p_g_est, var_g_est):



    expected_k = 1/(n*p_g_est )
    variance_group = expected_k *var_g_est
    

    return variance_group



def calc_variance_estimator_alt(n, p_1_est, p_2_est, p_3_est, p_4_est,  var_d_1_est, var_d_2_est, var_d_3_est, var_d_4_est, sum_of_weights=1):

        term_1 = (var_d_1_est + var_d_2_est + var_d_3_est + var_d_4_est)/n
        term_2 = (((var_d_1_est/(n*p_1_est))) + ((var_d_2_est/(n*p_2_est))) + ((var_d_3_est/(n*p_3_est))) + ((var_d_4_est/(n*p_4_est))))

        p_1_correction = (1/p_1_est) - 1
        p_2_correction = (1/p_2_est) - 1
        p_3_correction = (1/p_3_est) - 1
        p_4_correction = (1/p_4_est) - 1

        correction_for_term_2 = (var_d_1_est*p_1_correction + var_d_2_est*p_2_correction + var_d_3_est*p_3_correction + var_d_4_est*p_4_correction)/n


        var_est = (1/16 )  * (term_2 - correction_for_term_2)
    
        return var_est


def calc_variance_estimator_alt_2( p_1_est, p_2_est, p_3_est, p_4_est,  mu_wd_est, second_moment_1_est, second_moment_2_est, second_moment_3_est, second_moment_4_est):
     
     second_part = mu_wd_est**2

     first_part = (1/16)*((p_1_est*second_moment_1_est) + (p_2_est*second_moment_2_est) + (p_3_est*second_moment_3_est) + (p_4_est*second_moment_4_est))

     var_est = first_part - second_part

     return var_est


     
     

     

def calc_t_stat_wd(mean_est, var_est, x, n):
        std_est = np.sqrt(var_est)
         
        t_stat = (mean_est - x)/(std_est/np.sqrt(n))
    
        return t_stat

def calc_t_stat(mean_est, var_est, x):
        std_est = np.sqrt(var_est)
         
        t_stat = (mean_est - x)/(std_est)
    
        return t_stat
    
def calc_fourth_moment_binom(n, p):
        first_term = n*p
        second_term = n*(n-1)*(n-2)*(n-3)*(p**4)
        third_term = n*(n-1)*(n-2)*(p**3)
        fourth_term =n*(n-1)*(p**2)


        fourth_moment = first_term + second_term + 6*third_term + 7*fourth_term

        return fourth_moment

def calc_second_moment_binom(n, p):
        first_term = n*p
        second_term = n*(n-1)*(p**2)
        second_moment = first_term + second_term

        return second_moment

def calc_psi(n, p):
      
    fourth_moment_binom = calc_fourth_moment_binom(n, p)
    second_moment_binom = calc_second_moment_binom(n, p)
    var_binom_squared = fourth_moment_binom - (second_moment_binom**2)

    first_term = (1/second_moment_binom) + (1/(second_moment_binom**3))*var_binom_squared
    
    psi = n**2*p**2*first_term
    return psi

def calc_expectation_adjustment(n, p):
     return 1 + ((1/p) - 1)/n

def calc_var_group(psi, expectation_adjustment, second_moment, mu):
     return (psi*second_moment) - ((expectation_adjustment**2) * (mu**2))

def calc_variance_uncertainty(n, p_est_1, p_est_2, p_est_3, p_est_4, second_moment_est_1, second_moment_est_2, second_moment_est_3, second_moment_est_4, mu_est_1, mu_est_2, mu_est_3, mu_est_4):
     

    psi_1 = calc_psi(n, p_est_1)
    psi_2 = calc_psi(n, p_est_2)
    psi_3 = calc_psi(n, p_est_3)
    psi_4 = calc_psi(n, p_est_4)

    expectation_adjustment_1 = calc_expectation_adjustment(n, p_est_1)
    expectation_adjustment_2 = calc_expectation_adjustment(n, p_est_2)
    expectation_adjustment_3 = calc_expectation_adjustment(n, p_est_3)
    expectation_adjustment_4 = calc_expectation_adjustment(n, p_est_4)

    var_1 = calc_var_group(psi_1, expectation_adjustment_1, second_moment_est_1, mu_est_1)
    var_2 = calc_var_group(psi_2, expectation_adjustment_2, second_moment_est_2, mu_est_2)
    var_3 = calc_var_group(psi_3, expectation_adjustment_3, second_moment_est_3, mu_est_3)
    var_4 = calc_var_group(psi_4, expectation_adjustment_4, second_moment_est_4, mu_est_4)

    var_est = (1/16)*(var_1 + var_2 + var_3 + var_4)
    return var_est

def calc_variance_uncertainty_2(n, n_1, n_2, n_3, n_4, sigma_1_est, sigma_2_est, sigma_3_est, sigma_4_est, p_1_est, p_2_est, p_3_est, p_4_est):
     
    first_term_1 = (sigma_1_est)/n_1
    first_term_2 = (sigma_2_est)/n_2
    first_term_3 = (sigma_3_est)/n_3
    first_term_4 = (sigma_4_est)/n_4

    second_term_adjustment_1 = (1/(n**2))* (((n/n_1) - 1)**2)
    second_term_adjustment_2 = (1/(n**2))* (((n/n_2) - 1)**2)
    second_term_adjustment_3 = (1/(n**2))* (((n/n_3) - 1)**2)
    second_term_adjustment_4 = (1/(n**2))* (((n/n_4) - 1)**2)

    var_1 = (1 + second_term_adjustment_1)* first_term_1
    var_2 = (1 + second_term_adjustment_2)* first_term_2
    var_3 = (1 + second_term_adjustment_3)* first_term_3
    var_4 = (1 + second_term_adjustment_4)* first_term_4


    constant = (1/16)
    var_est = constant*(var_1 + var_2 + var_3 + var_4)

    return var_est
      

def calc_variance_uncertainty_analytical(n, p_1, p_2, p_3, p_4, second_moment_1, second_moment_2, second_moment_3, second_moment_4, mu_1, mu_2, mu_3, mu_4):
     
    psi_1 = calc_psi(n, p_1)
    psi_2 = calc_psi(n, p_2)
    psi_3 = calc_psi(n, p_3)
    psi_4 = calc_psi(n, p_4)

    expectation_adjustment_1 = calc_expectation_adjustment(n, p_1)
    expectation_adjustment_2 = calc_expectation_adjustment(n, p_2)
    expectation_adjustment_3 = calc_expectation_adjustment(n, p_3)
    expectation_adjustment_4 = calc_expectation_adjustment(n, p_4)

    var_1 = calc_var_group(psi_1, expectation_adjustment_1, second_moment_1, mu_1)
    var_2 = calc_var_group(psi_2, expectation_adjustment_2, second_moment_2, mu_2)
    var_3 = calc_var_group(psi_3, expectation_adjustment_3, second_moment_3, mu_3)
    var_4 = calc_var_group(psi_4, expectation_adjustment_4, second_moment_4, mu_4)

    var_est = (1/16)*(var_1 + var_2 + var_3 + var_4)

    return var_est
##################################

from math import comb
from decimal import *
from scipy.special import factorial, comb
from scipy.stats import norm

def approx_binom_pmf(k_values, n, p):
     
    # Calculate the mean (μ) and standard deviation (σ) for the binomial distribution: μ = np, σ = √(np(1-p))

    mean = n*p
    std = np.sqrt(n*p*(1-p))

    pdf_func = stats.norm(mean, std)

    # Calculate the probability density function (PDF) for the normal distribution using SciPy: norm.pdf(x, loc, scale)
    pdf_for_k = pdf_func.pdf(k_values)
    return pdf_for_k
    


def binom_pmf(k, n, p):
    binom_coef =comb(n, k, exact=True)
    prob = (p**k) * ((1-p)**(n-k))

    
    return binom_coef* prob
     

def calc_psi_hat(p_g, var_g, n):
     
    total = 0
    k_values = list(range(1, n+1))
    pdf_for_k = approx_binom_pmf(k_values, n, p_g)

    for k in range(1, n+1):
        probability_of_k = pdf_for_k[k-1]
   
        var_est_for_k = (1/k)* var_g

        total += (probability_of_k * var_est_for_k)

    psi = total 
 
    return psi



def var_estimator_n_g_uncertain(n, p_1_est, p_2_est, p_3_est, p_4_est, var_1_est, var_2_est, var_3_est, var_4_est):

    psi_1_est = calc_psi_hat(p_1_est, var_1_est, n)
    psi_2_est = calc_psi_hat(p_2_est, var_2_est, n)
    psi_3_est = calc_psi_hat(p_3_est, var_3_est, n)
    psi_4_est = calc_psi_hat(p_4_est, var_4_est, n)

    constant = (1/16)
   

    var = constant*(psi_1_est + psi_2_est + psi_3_est + psi_4_est)

    return var

def calc_xi_hat( p_g, var_g_est, n):
    total = 0
    total_probability = 0

    k_values = list(range(1, n+1))
    pdf_for_k = approx_binom_pmf(k_values, n, p_g)
    for k in range(1, n+1):
        
        
        probability_of_k = pdf_for_k[k-1]
        total_probability += probability_of_k


        term_at_k = (((n/k) - 1)**2) * (1/k)* var_g_est

        total += (probability_of_k * term_at_k)

    xi = total 
 
    return xi

     


def var_estimator_n_g_uncertain_and_uncertain_p(n, p_1_est, p_2_est, p_3_est, p_4_est, var_1_est, var_2_est, var_3_est, var_4_est):
    psi_1 = calc_psi_hat(p_1_est, var_1_est, n)
    psi_2 = calc_psi_hat(p_2_est, var_2_est, n)
    psi_3 = calc_psi_hat(p_3_est, var_3_est, n)
    psi_4 = calc_psi_hat(p_4_est, var_4_est, n)


    xi_1 = calc_xi_hat( p_1_est, var_1_est, n)
    xi_2 = calc_xi_hat( p_2_est, var_2_est, n)
    xi_3 = calc_xi_hat( p_3_est, var_3_est, n)
    xi_4 = calc_xi_hat( p_4_est, var_4_est, n)

    var_1 = (psi_1 + ((1/(n**2)) * xi_1))
    var_2 = (psi_2 + ((1/(n**2)) * xi_2))
    var_3 = (psi_3 + ((1/(n**2)) * xi_3))
    var_4 = (psi_4 + ((1/(n**2)) * xi_4))

    constant = (1/16)
    var = constant*(var_1 + var_2 + var_3 + var_4)

    return var



#####################################

def main():



    # n_per_sim = 5
    # n_sims = 10000
    # our_t_stat = [None]*n_sims

    # for i in range(n_sims):
         

    #     # simulate a standard normal with n_per_sim observations
    #     d_1 = np.random.normal(0, 1, n_per_sim)

    #     mean_est =d_1.mean()
    #     var_est = d_1.var()
    #     std_est = np.sqrt(var_est)

    #     t_stat = mean_est/(std_est/np.sqrt(n_per_sim))
    #     our_t_stat[i] = t_stat



    #  # values of alpha
    # alpha_values = np.linspace(0, 1, 1000)

    # def get_t_stat_for_alpha(alpha, df_t_test):
    #         t = stats.t.ppf(alpha, df_t_test)
    #         return t
        
    #     # for each alpha, calculate the t-statistic
    # critical_value_per_alpha = [get_t_stat_for_alpha(1-alpha, n_per_sim-1) for alpha in alpha_values]

    #     # calculate the rejection rateƒ
    # rejection_rate_per_alpha = [np.mean(our_t_stat > critical_value) for critical_value in critical_value_per_alpha]

    #     # plot the rejection rate as a function of alpha
    # plt.plot(alpha_values, rejection_rate_per_alpha, color='orange', label =  'original t-statistic')
    # plt.plot(alpha_values, alpha_values, linestyle='--', color='black')
    # plt.xlabel(r'Significance level: $\alpha$')
    # plt.ylabel(r'Rejection rate of test')
    # plt.legend(loc='upper left')
    # plt.savefig('illustration_consistency_n_per_sim{}.png'.format(n_per_sim))
    # plt.show()
    
    # sys.exit()

    n_per_sim = 10000
    n_sims = 10000
    cases = [2, 3, 4]
    # define the case

    for case_i in cases:

        if case_i == 1:
        
            # define four normal distributions; d_1, d_2, d_3, d_4, each with mean mu_1, ..., and variance sigma_1, ...
            mu_1, sigma_1 = 0, 1
            mu_2, sigma_2 = 0, 1
            mu_3, sigma_3 = 0, 1
            mu_4, sigma_4 = 0, 1

            # now, define the probabilities of each distribution
            p_1 = 0.25
            p_2 = 0.25
            p_3 = 0.25
            p_4 = 0.25

        elif case_i == 2:
            
            # define four normal distributions; d_1, d_2, d_3, d_4, each with mean mu_1, ..., and variance sigma_1, ...
            mu_1, sigma_1 = 0, 1
            mu_2, sigma_2 = 0, 1
            mu_3, sigma_3 = 0, 1
            mu_4, sigma_4 = 0, 1

            # now, define the probabilities of each distribution
            p_1 = 0.4
            p_2 = 0.1
            p_3 = 0.1
            p_4 = 0.4

        elif case_i == 3:
            
            # define four normal distributions; d_1, d_2, d_3, d_4, each with mean mu_1, ..., and variance sigma_1, ...
            mu_1, sigma_1 = -1, 1
            mu_2, sigma_2 = 0, 3
            mu_3, sigma_3 = 0, 5
            mu_4, sigma_4 = 1.5, 1

            # now, define the probabilities of each distribution
            p_1 = 0.25
            p_2 = 0.25
            p_3 = 0.25
            p_4 = 0.25

        elif case_i == 4:
            
            # define four normal distributions; d_1, d_2, d_3, d_4, each with mean mu_1, ..., and variance sigma_1, ...
            mu_1, sigma_1 = -1, 1
            mu_2, sigma_2 = 0, 3
            mu_3, sigma_3 = 0, 5
            mu_4, sigma_4 = 1.5, 1

            # now, define the probabilities of each distribution
            p_1 = 0.4
            p_2 = 0.1
            p_3 = 0.1
            p_4 = 0.4

            

            
            



        observed_mean = [None]*n_sims
        observed_var = [None]*n_sims
        observed_std = [None]*n_sims

        our_est_mean = [None]*n_sims
        our_est_var = [None]*n_sims

    
        our_t_stat = [None]*n_sims
        orig_t_stat = [None]*n_sims

        determined_mean =mean_estimator_analytical(mu_1, mu_2, mu_3, mu_4, n_per_sim, p_1, p_2, p_3, p_4)


        for i in range(n_sims):
            print(' at simulation ', i)


        

            d_1 = np.random.normal(mu_1, np.sqrt(sigma_1), n_per_sim)
            d_2 = np.random.normal(mu_2, np.sqrt(sigma_2), n_per_sim)
            d_3 = np.random.normal(mu_3, np.sqrt(sigma_3), n_per_sim)
            d_4 = np.random.normal(mu_4, np.sqrt(sigma_4), n_per_sim)

            mixture_before_weighting, g = draw_mixture(p_1, p_2, p_3, p_4, d_1, d_2, d_3, d_4, int(n_per_sim))


            p_1_est = np.mean(g==1)
            p_2_est = np.mean(g==2)
            p_3_est = np.mean(g==3)
            p_4_est = np.mean(g==4)

            print('p_1_est: ', p_1_est)
            print('p_2_est: ', p_2_est)
            print('p_3_est: ', p_3_est)
            print('p_4_est: ', p_4_est)

            w_1_est = 1/(4*p_1_est)
            w_2_est = 1/(4*p_2_est)
            w_3_est = 1/(4*p_3_est)
            w_4_est = 1/(4*p_4_est)
            weights = [w_1_est, w_2_est, w_3_est, w_4_est]


            weighting_variable =  [weights[group-1]  for group in g]
            mixture = [weighting_variable[i]*mixture_before_weighting[i] for i in range(len(mixture_before_weighting))]


            # calculate the mean and variance of the mixture
            mean_mixture = np.mean(mixture)
            var_mixture = np.var(mixture)
            std_mixture = np.sqrt(var_mixture)
            print('Mean of mixture (taken mean of wd): ', mean_mixture)
            print('Variance of mixture (taken var of wd): ', var_mixture)
            observed_mean[i] = mean_mixture
            observed_var[i] = var_mixture
            observed_std[i] = std_mixture

            d_1_from_mixture_before_weighting = np.array(mixture_before_weighting)[g==1]
            d_2_from_mixture_before_weighting = np.array(mixture_before_weighting)[g==2]
            d_3_from_mixture_before_weighting = np.array(mixture_before_weighting)[g==3]
            d_4_from_mixture_before_weighting = np.array(mixture_before_weighting)[g==4]

            n_1 = len(d_1_from_mixture_before_weighting)
            n_2 = len(d_2_from_mixture_before_weighting)
            n_3 = len(d_3_from_mixture_before_weighting)
            n_4 = len(d_4_from_mixture_before_weighting)

            

           
            mean_est = mean_estimator_uncertainty(d_1_from_mixture_before_weighting, 
                                                  d_2_from_mixture_before_weighting, 
                                                  d_3_from_mixture_before_weighting,
                                                  d_4_from_mixture_before_weighting,
                                                    n_per_sim, p_1_est, p_2_est, p_3_est, p_4_est
                                                  )


            print('Estimated mean: ', mean_est)
            print('Estimated probabilities: ', p_1_est, p_2_est, p_3_est, p_4_est)

            
            our_est_mean[i] = mean_est

            # check if mean_est is finite
            if not np.isfinite(mean_est):
                print('mean estimator is not finite')
                print('p_1_est: ', p_1_est)
                print('p_2_est: ', p_2_est)
                print('p_3_est: ', p_3_est)
                print('p_4_est: ', p_4_est)

               
                sys.exit()

            print('mean of d_1 (sample)', np.mean(d_1_from_mixture_before_weighting))
            print('mean of d_2 (sample)', np.mean(d_2_from_mixture_before_weighting))
            print('mean of d_3 (sample)', np.mean(d_3_from_mixture_before_weighting))
            print('mean of d_4 (sample)', np.mean(d_4_from_mixture_before_weighting))


            mu_1_est = np.mean(d_1_from_mixture_before_weighting)
            mu_2_est = np.mean(d_2_from_mixture_before_weighting)
            mu_3_est = np.mean(d_3_from_mixture_before_weighting)
            mu_4_est = np.mean(d_4_from_mixture_before_weighting)

            var_1_est = np.var(d_1_from_mixture_before_weighting)
            var_2_est = np.var(d_2_from_mixture_before_weighting)
            var_3_est = np.var(d_3_from_mixture_before_weighting)
            var_4_est = np.var(d_4_from_mixture_before_weighting)

            second_moment_1_est = np.mean(d_1_from_mixture_before_weighting**2)
            second_moment_2_est = np.mean(d_2_from_mixture_before_weighting**2)
            second_moment_3_est = np.mean(d_3_from_mixture_before_weighting**2)
            second_moment_4_est = np.mean(d_4_from_mixture_before_weighting**2)
            
            #var_est = calc_variance(mean_est, p_1_est, p_2_est, p_3_est, p_4_est, second_moment_1_est, second_moment_2_est, second_moment_3_est, second_moment_4_est)
            #our_est_var[i] = var_est
            #print('our estimated variance: ', var_est)


            #variance_estimator = calc_variance_estimator_alt_2(p_1_est, p_2_est, p_3_est, p_4_est,  mean_est, second_moment_1_est, second_moment_2_est, second_moment_3_est, #second_moment_4_est)
            #variance_estimator = calc_variance_estimator_alt(n_per_sim, p_1_est, p_2_est, p_3_est, p_4_est,  var_1_est, var_2_est, var_3_est, var_4_est, #sum_of_weights=sum_of_weights)
            #variance_estimator = calc_variance_estimator(n_per_sim, var_1_est, var_2_est, var_3_est, var_4_est, n_1, n_3, n_3, n_4)
            #print('Variance of the estimator: ', variance_estimator)
            #variance_estimator = calc_variance_uncertainty_2(n_per_sim, n_1, n_2, n_3, n_4, var_1_est, var_2_est, var_3_est, var_4_est, p_1_est, p_2_est, p_3_est, p_4_est)

            variance_estimator = var_estimator_n_g_uncertain_and_uncertain_p(n_per_sim, p_1_est, p_2_est, p_3_est, p_4_est, var_1_est, var_2_est, var_3_est, var_4_est)
            t_stat = calc_t_stat(mean_est, variance_estimator, 0)
            orig_t_stat_i = mean_mixture/(std_mixture/np.sqrt(n_per_sim))
            our_t_stat[i] = t_stat
            orig_t_stat[i] = orig_t_stat_i
            print('variannce estimator: ', variance_estimator)


        

        # draw from a standard normal distribution
        standard_normal = np.random.normal(0, 1, n_sims)
        bins_standard_normal =   list(np.linspace(-4,4,50))


        # values of alpha
        alpha_values = np.linspace(0, 1, 1000)

        def get_t_stat_for_alpha(alpha, df_t_test):
            t = stats.t.ppf(alpha, df_t_test)
            return t
        
        # for each alpha, calculate the t-statistic
        critical_value_per_alpha = [get_t_stat_for_alpha(1-alpha, n_per_sim-1) for alpha in alpha_values]

        # calculate the rejection rateƒ
        rejection_rate_per_alpha = [np.mean(our_t_stat > critical_value) for critical_value in critical_value_per_alpha]
        rejection_rate_orig_t_stat = [np.mean(orig_t_stat > critical_value) for critical_value in critical_value_per_alpha]

        # plot the rejection rate as a function of alpha
        plt.plot(alpha_values, rejection_rate_per_alpha, color='blue', label = 'Our t-statistic')
        plt.plot(alpha_values, rejection_rate_orig_t_stat, color='orange', label = 'Original t-statistic')
        plt.plot(alpha_values, alpha_values, linestyle='--', color='black')
        plt.xlabel(r'Significance level: $\alpha$')
        plt.ylabel(r'Rejection rate of test')
        plt.legend(loc='upper left')
        plt.savefig('size_plot_n_per_sim_{}_case_{}'.format(n_per_sim, case_i), dpi=300)
       #plt.show()
    

        




        avg_observed_mean = np.mean(np.array(observed_mean))
    
        avg_our_estimator = np.mean(np.array(our_est_mean))
    

    
        # make a plot of the observed mean, and our estimated mean 
        plt.hist(observed_mean, bins=50, alpha=0.5, label='observed mean', color='blue')
        plt.hist(our_est_mean, bins=50, alpha=0.5, label='our est. mean', color='orange')
        plt.vlines(determined_mean, 0, n_sims/20, label='analytical mean', color='red')
        plt.vlines(avg_observed_mean, 0, n_sims/20, label='avg. observed mean', color='blue')
        plt.vlines(avg_our_estimator, 0, n_sims/20, label='avg. our estimator', color='orange')
        plt.legend(loc='upper right')
        plt.savefig('mean.png', dpi=300)
        #plt.show()

        print('Variance of our Estimated mean: ', np.var(np.array(our_est_mean), ddof=1))

        n_1_analytical = int(n_per_sim*p_1)
        n_2_analytical = int(n_per_sim*p_2)
        n_3_analytical = int(n_per_sim*p_3)
        n_4_analytical = int(n_per_sim*p_4)

        analytical_variance_estimated_mean = calc_variance_uncertainty_2(n_per_sim, n_1_analytical, n_2_analytical, n_3_analytical, n_4_analytical, sigma_1, sigma_2, sigma_3,sigma_4, p_1, p_2, p_3, p_4)
        print('Analytical variance of our Estimated mean: ', analytical_variance_estimated_mean)


        
        #print('Analytical variance of our Estimated mean (alt): ', analytical_var_estimator_mean(n_per_sim, p_1, p_2, p_3, p_4, sigma_1, sigma_2, sigma_3,sigma_4))
        #print('Analytical variance of Estimated mean: ', calc_variance_estimator(n_per_sim, sigma_1, sigma_2, sigma_3,sigma_4))

        analytical_mean_uncertainty_adjusted = mean_estimator_analytical(mu_1, mu_2, mu_3, mu_4, n_per_sim, p_1, p_2, p_3, p_4)
        print('Analytical mean of our Estimated mean (uncertainty adjusted): ', analytical_mean_uncertainty_adjusted)
        print('mean of mixture: ', np.mean(np.array(our_est_mean)))


        # draw from a standard normal distribution
        standard_normal = np.random.normal(0, 1, n_sims)
        bins_standard_normal =   list(np.linspace(-4,4,50))

        # plot a histogram of the t-statistic and a standard normal distribution
        plt.hist(our_t_stat, bins=50, alpha=0.5, label='t-statistic')
        plt.hist(standard_normal, bins=50, alpha=0.5, label='standard normal')
        plt.legend(loc='upper right')
        plt.savefig('t_statistic.png', dpi=300)
        #pplt.show()


        

if __name__ == "__main__":


    
    main()

