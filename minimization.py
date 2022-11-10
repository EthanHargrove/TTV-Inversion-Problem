import rebound
import numpy as np
from transit_times import transits
from system_generation import generate_guess

def cost(true_trans, params_guess, target_acc, num_bod):
    nonzero_true_trans = true_trans[np.nonzero(true_trans)]
    sim = rebound.Simulation()
    sim.units = ("day", "AU", "Msun")
    sim.add(m=params_guess[0])
    for q in np.arange(0,7*(num_bod),7):
        sim.add(m=params_guess[q+1],P=params_guess[q+2],e=params_guess[q+3],inc=params_guess[q+4],Omega=params_guess[q+5],omega=params_guess[q+6],f=params_guess[q+7])
    sim.move_to_com()
    guess_trans = transits(sim, method="newtons", bodies=None, N=len(true_trans[0]), accuracy=0.9*target_acc, fast=True)
    nonzero_guess_trans = guess_trans[np.nonzero(guess_trans)]
    if len(nonzero_true_trans) == len(nonzero_guess_trans):
        max_diff = abs(nonzero_true_trans - nonzero_guess_trans).max()
        mean_diff = abs(nonzero_true_trans - nonzero_guess_trans).mean()
        chi_sqr = sum(((nonzero_guess_trans - nonzero_true_trans)**2) / nonzero_true_trans)
    else:
        max_diff = -1
        mean_diff = -1
        chi_sqr = -1
    return max_diff, mean_diff, chi_sqr


def simplex_point(start_point, ind, alpha, transittimes, target_acc, num_bod):
    point_guess = np.copy(start_point)
    point_guess[ind] *= (1 + alpha)
    max_diff, mean_diff, chi_sqr = cost(transittimes, point_guess, target_acc, num_bod)
    if mean_diff > 0:
        return mean_diff, point_guess
    else:
        start_point[ind] /= (1 + alpha)
        start_point[ind] *= (1 + alpha)
        max_diff, mean_diff, chi_sqr = cost(transittimes, point_guess, target_acc, num_bod)
        return mean_diff, point_guess


def nelder_mead(transittimes, star_mass, target_acc):
    """
    Gao, F., & Han, L. 2012
    """
    num_bod = len(transittimes)
    num_dim = num_bod * 7
    max_iter = num_dim ** 3
    α = 1
    β = 1 + 2 / num_dim
    γ = 0.75 - 1 / (2 * num_dim)
    δ = 1 - 1 / num_dim
    for i in range(5*(num_dim+1)):
        guess = generate_guess(transittimes, star_mass)
        max_diff, mean_diff, chi_sqr = cost(transittimes, guess, target_acc, num_bod)
        if i == 0:
            if mean_diff > 0:
                best_mean_diff = mean_diff
                best_guess = guess
            else:
                best_mean_diff = 1e7
        elif (mean_diff < best_mean_diff) and (mean_diff > 0):
            best_mean_diff = mean_diff
            best_guess = guess
    
    print(best_mean_diff)
    print(best_guess)
    costs = np.zeros(num_dim+1)
    points = np.empty(num_dim+1, dtype="object")
    costs[0] = best_mean_diff
    points[0] = best_guess
    
    for p in range(len(best_guess)):
        print(f"p = {p}")
        curr_guess = np.copy(best_guess)
        if p != 0:
            alph = 2
            mean_diff, point_guess = simplex_point(curr_guess, p, alph, transittimes, target_acc, num_bod)
            print(f"mean_diff = {mean_diff}")
            print(f"point_guess = {point_guess}")
            while mean_diff < 0:
                alph /= 4
                print(f"alph = {alph}")
                mean_diff, point_guess = simplex_point(curr_guess, p, alph, transittimes, target_acc, num_bod)
                print(f"mean_diff = {mean_diff}")
                print(f"point_guess = {point_guess}")
            costs[p] = mean_diff
            points[p] = point_guess
    
    order = np.argsort(costs)
    costs = costs[order]
    points = points[order]
    num_iter = 0
    print("initial")
    while num_iter < max_iter:
        param_sum = np.zeros(len(points[0]))
        for params in points[:-1]:
            param_sum += params
        centroid = param_sum / num_dim
        skip_flag = 0
        print(f"num_iter = {num_iter}")
        print(f"costs = {costs}")
        reflected = centroid + α * (centroid - points[-1])
        for j, param_guess in enumerate(reflected):
                if j != 0:
                    if ((j-3) % 7 == 0) and ((param_guess < 0) or (param_guess >= 1)):
                        reflected_mean_diff = -1
                        skip_flag = 1
        if skip_flag == 0:
            reflected_max_diff, reflected_mean_diff, reflected_chi_sqr = cost(transittimes, reflected, target_acc, num_bod)
        if (reflected_mean_diff <= costs[0]) and (reflected_mean_diff > 0):
            expanded = centroid + β * (reflected - centroid)
            for j, param_guess in enumerate(expanded):
                if j != 0:
                    if ((j-3) % 7 == 0) and ((param_guess < 0) or (param_guess >= 1)):
                        reflected_mean_diff = -1
                        expanded_mean_diff = 1000
                        skip_flag = 1
            if skip_flag == 0:
                expanded_max_diff, expanded_mean_diff, expanded_chi_sqr = cost(transittimes, expanded, target_acc, num_bod)
            if expanded_mean_diff <= reflected_mean_diff:
                print("Expanded")
                costs[-1] = expanded_mean_diff
                points[-1] = expanded
                order = np.argsort(costs)
                costs = costs[order]
                points = points[order]
                num_iter += 1
                continue
            else:
                print("Reflected")
                costs[-1] = reflected_mean_diff
                points[-1] = reflected
                order = np.argsort(costs)
                costs = costs[order]
                points = points[order]
                num_iter += 1
                continue
        elif (reflected_mean_diff <= costs[-2]) and (reflected_mean_diff > 0):
            print("Reflected")
            costs[-1] = reflected_mean_diff
            points[-1] = reflected
            order = np.argsort(costs)
            costs = costs[order]
            points = points[order]
            num_iter += 1
            continue
        elif costs[-2] <= reflected_mean_diff < costs[-1]:
            contracted = centroid + γ * (reflected - centroid)
            for j, param_guess in enumerate(contracted):
                if j != 0:
                    if ((j-3) % 7 == 0) and ((param_guess < 0) or (param_guess >= 1)):
                        reflected_mean_diff = -1
                        contracted_mean_diff = 1000
                        skip_flag = 1
            if skip_flag == 0:
                contracted_max_diff, contracted_mean_diff, contracted_chi_sqr = cost(transittimes, contracted, target_acc, num_bod)
            if (contracted_mean_diff <= reflected_mean_diff) and (contracted_mean_diff > 0):
                print("Outside Contracted")
                costs[-1] = contracted_mean_diff
                points[-1] = contracted
                order = np.argsort(costs)
                costs = costs[order]
                points = points[order]
                num_iter += 1
                continue
            else:
                worst_cost = costs[-1]
                for i, point in enumerate(points[1:]):
                    points[i+1] = points[0] + δ * (point - points[0])
                    max_diff, mean_diff, chi_sqr = cost(transittimes, points[i+1], target_acc, num_bod)
                    if mean_diff < 0:
                        while True:
                            guess = generate_guess(transittimes, star_mass)
                            max_diff, mean_diff, chi_sqr = cost(transittimes, guess, target_acc, num_bod)
                            if 0 < mean_diff < worst_cost:
                                break
                        points[i+1] = guess
                    costs[i+1] = mean_diff
                print("Shrunk")
                order = np.argsort(costs)
                costs = costs[order]
                points = points[order]
                num_iter += 1
                continue
        elif (costs[-1] <= reflected_mean_diff) or (reflected_mean_diff < 0):
            inside_contracted = centroid - γ * (reflected - centroid)
            ic_max_diff, ic_mean_diff, ic_chi_sqr = cost(transittimes, inside_contracted, target_acc, num_bod)
            if (ic_mean_diff < costs[-1]) and (ic_mean_diff > 0):
                print("Inside Contracted")
                costs[-1] = ic_mean_diff
                points[-1] = inside_contracted
                order = np.argsort(costs)
                costs = costs[order]
                points = points[order]
                num_iter += 1
                continue
            else:
                worst_cost = costs[-1]
                for i, point in enumerate(points[1:]):
                    points[i+1] = points[0] + δ * (point - points[0])
                    max_diff, mean_diff, chi_sqr = cost(transittimes, points[i+1], target_acc, num_bod)
                    if mean_diff < 0:
                        while True:
                            guess = generate_guess(transittimes, star_mass)
                            max_diff, mean_diff, chi_sqr = cost(transittimes, guess, target_acc, num_bod)
                            if 0 < mean_diff < worst_cost:
                                break
                        points[i+1] = guess
                    costs[i+1] = mean_diff
                print("Shrink")
                order = np.argsort(costs)
                costs = costs[order]
                points = points[order]
                num_iter += 1
                continue
        else:
            print("Error, no cases fulfilled")
            break
    return points[0]