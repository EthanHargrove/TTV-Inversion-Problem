import rebound
import numpy as np
from root_finders import newtons, bisection


def transits(sim, method=None, bodies=None, N=None, tmax=None, accuracy=None, fast=False, max_time=None):
    """
    """
    #If user doesn't specify method, choose newton's
    if method is None:
        method = "newtons"
        
    # If user doesn't specify which bodies to find transits for, find for all
    if bodies is None:
        bodies = np.arange(1, sim.N) 
        
    # Store particles in variable    
    p = sim.particles
    
    # Store initial periods of planets in an array if needed
    t0 = sim.t
    periods = np.empty(len(bodies) + 1)
    periods[-1] = 1e99
    for i, body_i in enumerate(bodies):
        periods[i] = p[int(body_i)].calculate_orbit(primary=p[0]).P
    ttvamp = periods.min() / 4
            
    # Default simulation duration and accuracy          
    if tmax is None and N is None:
        N = 10
    if accuracy is None:
        accuracy = 1e-6 
        
    # Setup transit time guesses for fastmode using a maximum simulation time
    if fast and N is None:
        sim.save("checkpoint.bin")
        # Array to hold analytical Keplerian transittimes (our guesses)
        guesses2D = np.zeros((len(bodies), int(np.ceil((tmax-t0) / periods.min()))))
        orbits = sim.calculate_orbits()
        for j, body in enumerate(bodies):
            # Determine the time of first transit
            # See <my-paper.pdf> for derivation
            o = orbits[body-1]
            M_1 = o.M
            true_anom = -np.arctan2(np.sin(o.Omega), (np.cos(o.inc) * np.cos(o.Omega))) - o.omega
            sin_E = (np.sqrt(1 - (o.e **2)) * np.sin(true_anom)) / (1 + (o.e * np.cos(true_anom)))
            cos_E = (o.e + np.cos(true_anom)) / (1 + (o.e * np.cos(true_anom)))
            M_2 = (np.arctan2(sin_E, cos_E) % (2 * np.pi)) - (o.e * sin_E)
            if M_1 > M_2:
                M_1 -= (2 * np.pi)
            first_trans = ((M_2 - M_1) / o.n) + t0
            # To get each subsequent transit-time add one period
            for k in range(len(guesses2D[0])):
                guess = (k * o.P) + first_trans
                if guess < tmax:
                    guesses2D[j, k] = guess
                else:
                    break
        # Flatten and sort the guesses
        guesses1D = np.sort(np.concatenate(guesses2D))
        guesses1D = guesses1D[np.nonzero(guesses1D)]
        if len(guesses1D) == 0:
            print("no guesses")
            return np.array([])
        
    # Setup transit-time guesses for fastmode using a maximum number of transits
    elif fast and (type(N) is int):
        sim.save("checkpoint.bin")
        # Array to hold analytical Keplerian transittimes (our guesses)
        guesses2D = np.zeros((len(bodies), N))
        orbits = sim.calculate_orbits()
        for j, body in enumerate(bodies):
            # Determine the time of first transit
            # See <my-paper.pdf> for derivation
            o = orbits[body-1]
            M_1 = o.M
            true_anom = -np.arctan2(np.sin(o.Omega), (np.cos(o.inc) * np.cos(o.Omega))) - o.omega
            sin_E = (np.sqrt(1 - (o.e **2)) * np.sin(true_anom)) / (1 + (o.e * np.cos(true_anom)))
            cos_E = (o.e + np.cos(true_anom)) / (1 + (o.e * np.cos(true_anom)))
            M_2 = (np.arctan2(sin_E, cos_E) % (2 * np.pi)) - (o.e * sin_E)
            if M_1 > M_2:
                M_1 -= (2 * np.pi)
            first_trans = ((M_2 - M_1) / o.n) + t0
            # To get each subsequent transit-time add one period
            for k in range(len(guesses2D[0])):
                guess = (k * o.P) + first_trans
                if np.all(guesses2D[:,-1] == 0.) or np.any(guess < guesses2D[:,-1]):
                    guesses2D[j, k] = guess
                else:
                    break
        # Flatten and sort the guesses
        guesses1D = np.sort(np.concatenate(guesses2D))
        guesses1D = guesses1D[np.nonzero(guesses1D)]
        if len(guesses1D) == 0:
            print("no guesses")
            return np.array([])
        
    elif fast and (type(N) is list):
        sim.save("checkpoint.bin")
        # Array to hold analytical Keplerian transittimes (our guesses)
        guesses2D = np.zeros((len(bodies), N[0]))
        orbits = sim.calculate_orbits()
        for j, body in enumerate(bodies):
            # Determine the time of first transit
            # See <my-paper.pdf> for derivation
            o = orbits[body-1]
            M_1 = o.M
            true_anom = -np.arctan2(np.sin(o.Omega), (np.cos(o.inc) * np.cos(o.Omega))) - o.omega
            sin_E = (np.sqrt(1 - (o.e **2)) * np.sin(true_anom)) / (1 + (o.e * np.cos(true_anom)))
            cos_E = (o.e + np.cos(true_anom)) / (1 + (o.e * np.cos(true_anom)))
            M_2 = (np.arctan2(sin_E, cos_E) % (2 * np.pi)) - (o.e * sin_E)
            if M_1 > M_2:
                M_1 -= (2 * np.pi)
            first_trans = ((M_2 - M_1) / o.n) + t0
            # To get each subsequent transit-time add one period
            for k in range(N[j]):
                guess = (k * o.P) + first_trans
                guesses2D[j, k] = guess
        # Flatten and sort the guesses
        guesses1D = np.sort(np.concatenate(guesses2D))
        guesses1D = guesses1D[np.nonzero(guesses1D)]
        if len(guesses1D) == 0:
            print("no guesses")
            return np.array([])
        
    # Find the transit-times using fast-mode using a maximum simulation time
    if N is None and fast:
        # Create an array to store the transit-times and a counter to keep track of the number of transits detected
        transittimes = np.zeros((9, 2088))
        num = 0
        # Determine the transit-times
        while sim.t <= tmax:
            # Obtain current guess from our array of guesses
            curr_guess = guesses1D[num]
            # Find the planet associated with the current guess
            body_i = int(np.where(abs(guesses2D - curr_guess) < 1e-7)[0][0]) + 1
            # Integrate to a time before and after the current guess, keep track of the time and y position
            sim.integrate(curr_guess - ttvamp)
            old_pos = p[body_i].y - p[0].y
            t_old = sim.t
            sim.integrate(curr_guess + ttvamp)
            new_pos = p[body_i].y - p[0].y
            t_new = sim.t
            # Use a root finder to determine the actual transit-time
            if old_pos < 0. and new_pos >= 0.:
                if method == "newtons":
                    transittimes[body_i][np.where(transittimes[body_i]==0)[0][0]] = newtons(sim, p, int(body_i), t_old, t_new, new_pos, accuracy)
                elif method == "bisection":
                    transittimes[body_i][np.where(transittimes[body_i]==0)[0][0]] = bisection(sim, p, int(body_i), t_old, t_new, new_pos, accuracy)
            else:
                sim = rebound.Simulation("checkpoint.bin")
                print("shift to slow mode")
                transittimes = transits(sim, method="newtons", bodies=bodies, tmax=tmax, accuracy=accuracy, fast=False)
                return transittimes
            # Increment our transit counter and break if all have been found
            num += 1
            if num == len(guesses1D):
                break
            sim.integrate(t_new)

        return transittimes
    
    # Find the transit-times using a maximum simulation time
    elif N is None:
        # Create an array to store the transit-times and array to keep track of the y-positions of each planet
        transittimes = np.zeros((8, 2088))
        old_pos = np.zeros(len(bodies))
        new_pos = np.zeros(len(bodies))
        # Flag to indicate start of the simulation
        start = True
        # Determining the transit-times
        while sim.t <= tmax:
            if not start:
                # Set the old time and positions equal to the new counterparts of the previous iteration
                old_pos = np.copy(new_pos)
                t_old = t_new
            else:
                # Determine the old time and y-positions
                for i, body_i in enumerate(bodies):
                    old_pos[i] = p[int(body_i)].y - p[0].y
                t_old = sim.t
                start = False
            # Integrate one time-step
            sim.step()
            # Determine the new time and y-positions
            for i, body_i in enumerate(bodies):
                new_pos[i] = p[int(body_i)].y - p[0].y
            t_new = sim.t
            # Check each planet to see if a transit has occurred
            for i, body_i in enumerate(bodies):
                if old_pos[i] < 0. and new_pos[i] >= 0.:
                    trans_found = True
                    if method == "newtons":
                        transittimes[i][np.where(transittimes[i]==0)[0][0]] = newtons(sim, p, int(body_i), t_old, t_new, new_pos[i], accuracy)
                    elif method == "bisection":
                        transittimes[i][np.where(transittimes[i]==0)[0][0]] = bisection(sim, p, int(body_i), t_old, t_new, new_pos[i], accuracy)
                else:
                    trans_found = False
            # If a transit was found, integrate to t_new to prevent cases of double-counting
            if trans_found:
                sim.integrate(t_new)
        return transittimes
    
    # Find the transit-times using fast-mode using a maximum number of transits      
    elif fast:
        if type(N) is list:
            N = N[0]
        transittimes = np.zeros((sim.N, N))
        num = 0
        while np.all(transittimes[:,-1]) == 0:
            curr_guess = guesses1D[num]
            body_i = int(np.where(abs(guesses2D - curr_guess) < 1e-7)[0][0]) + 1
            sim.integrate(curr_guess - ttvamp)
            old_pos = p[body_i].y - p[0].y
            t_old = sim.t
            sim.integrate(curr_guess + ttvamp)
            new_pos = p[body_i].y - p[0].y
            t_new = sim.t
            print_flag = np.random.rand()
            if old_pos < 0. and new_pos >= 0.:
                if transittimes[body_i,-1] == 0:
                    if method == "newtons":
                        transittimes[body_i][np.where(transittimes[body_i]==0)[0][0]] = newtons(sim, p, int(body_i), t_old, t_new, new_pos, accuracy)
                    elif method == "bisection":
                        transittimes[body_i][np.where(transittimes[body_i]==0)[0][0]] = bisection(sim, p, int(body_i), t_old, t_new, new_pos, accuracy)
            else:
                sim = rebound.Simulation("checkpoint.bin")
                max_time = 2*guesses2D.max()
                print("shift to slow mode")
                transittimes = transits(sim, method="newtons", bodies=bodies, N=N, accuracy=accuracy, fast=False, max_time=max_time)
                return transittimes
            num += 1
            if num == len(guesses1D):
                break
            sim.integrate(t_new)
        return transittimes[1:]

    # Find the transit-times using a maximum number of transits       
    else:
        transittimes = np.zeros((sim.N, N))
        old_pos = np.zeros(len(bodies))
        new_pos = np.zeros(len(bodies))
        start = True
        while np.all(transittimes[:-1,-1]) == 0:
            if not start:
                old_pos = np.copy(new_pos)
                t_old = t_new
            else:
                for i, body_i in enumerate(bodies):
                    old_pos[i] = p[int(body_i)].y - p[0].y
                t_old = sim.t
                start = False
            sim.step()
            for i, body_i in enumerate(bodies):
                new_pos[i] = p[int(body_i)].y - p[0].y
            t_new = sim.t
            if t_new >= max_time:
                print("max time exceed")
                return np.array([])
            print_flag = np.random.rand()
            for i, body_i in enumerate(bodies):
                if old_pos[i] < 0. and new_pos[i] >= 0.:
                    if transittimes[i,-1] == 0:
                        trans_found = True
                        if method == "newtons":
                            transittimes[i][np.where(transittimes[i]==0)[0][0]] = newtons(sim, p, int(body_i), t_old, t_new, new_pos[i], accuracy)
                        elif method == "bisection":
                            transittimes[i][np.where(transittimes[i]==0)[0][0]] = bisection(sim, p, int(body_i), t_old, t_new, new_pos[i], accuracy)
                else:
                    trans_found = False
            if trans_found:
                sim.integrate(t_new)
        return transittimes[:-1]
