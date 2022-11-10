import rebound
import numpy as np


def get_ecc(period):
    if period < 1:
        epsi = (1/3)*(10/12)
        ecc = np.random.f(3,10) * (0.001/epsi)
        ecc = min(0.1,ecc)
    elif 1 <= period < 3:
        epsi = (1/3)*(18/20)
        ecc = np.random.f(3,18) * (0.005/epsi)
        diff = ecc - 0.005
        if diff > 0:
            ecc += 2.8 * np.random.rand() * diff
            ecc = min(0.25,ecc)
        else:
            ecc += 0.5 * np.random.rand() * diff
            ecc = max(0,ecc)
    elif 3 <= period < 11:
        epsi = (1/3)*(16/18)
        ecc = np.random.f(3,16) * (0.01/epsi)
        diff = ecc - 0.01
        if diff > 0:
            ecc += 2.8 * np.random.rand() * diff
            ecc = min(0.575,ecc)
        else:
            ecc += 0.5 * np.random.rand() * diff
            ecc = max(0,ecc)
    elif 11 <= period < 100:
        epsi = (2/4)*(37/39)
        ecc = np.random.f(4,37) * (0.05/epsi)
        diff = ecc - 0.05
        if diff > 0:
            ecc = min(0.85,ecc)
        else:
            ecc += 0.5 * np.random.rand() * diff
            ecc = max(0,ecc)
    elif 100 <= period:
        epsi = (4/6)*(70/72)
        ecc = np.random.f(6,70) * (0.1/epsi)
        diff = ecc - 0.1
        if diff > 0:
            ecc += 0.95 * np.random.rand() * diff
        ecc = min(0.9, ecc)
        
    return ecc


def deg_to_rad(deg_inc):
    deg_sep = 90 - deg_inc
    rad_inc = deg_sep * (np.pi/180)
    
    return rad_inc


def get_inclination(period):
    if 4 <= period <= 9:
        mini = 21.8604 + 23.05143*period - 2.848991*period**2 + 0.1174863*period**3
        mini *= np.random.uniform(0.98,1.02)
    if 7 <= period < 26:
        epsi = (98/100)*(99/101)
        inc = np.random.f(100,99) * (89.9 / epsi)
        diff = inc - 89.9
        plusminus = np.random.rand()
        maxi = 88.5 + plusminus * 1.6
        if (period < 4) or (period > 9):
            mini = 90 - plusminus * 6
        if diff < 0:
            inc -= 1.25 * np.random.rand() * diff
            inc = min(maxi, inc)
            inc = max(mini, inc)
        else:
            inc = max(mini, inc)
            inc = min(maxi, inc)
            
        if plusminus > 0.9975:
            inc += 6 * np.random.rand()
            
        elif plusminus < 0.0002:
            inc = np.random.uniform(80,90.1)
            inc = 2.970297*inc - 187.6238     
    if 2 <= period < 7:
        inc = np.random.uniform(79,91)
        inc = inc + np.random.rand()*np.random.rand() - np.random.rand()*np.random.rand()
        diff = inc - 88
        plusminus = np.random.rand()
        maxi = 88 + plusminus * 2.1
        if (period < 4) or (period > 9):
            mini = 82 - plusminus * 2
        inc = max(mini, inc)
        inc = min(maxi, inc)
        
        if plusminus > 0.9995:
            inc += 11 * np.random.rand()
            
        elif plusminus < 0.002:
            inc = np.random.uniform(80,90.1)
            inc = 2.970297*inc - 187.6238
        
    
    elif period < 2:
        inc = np.random.uniform(77.5,90.1)
        plusminus = np.random.rand()
        maxi = 87 + plusminus * 3.1
        mini = 80 - plusminus * 2.5
        inc = min(maxi,inc)
        inc = max(mini,inc)
            
    elif 26 <= period < 600:
        inc = np.random.uniform(84,90)
        diff = abs(inc - 88)
        plusminus = np.random.rand()
        maxi = 89 + plusminus * 1.1
        mini = 88 - plusminus * 2
        inc += 5 * np.random.rand() * diff
        inc = min(maxi, inc)
        inc = max(mini, inc)
        
        if plusminus > 0.9995:
            inc += 7 * np.random.rand()
        elif plusminus < 0.0075:
            inc = np.random.uniform(86,90)
            inc = 3.5 * inc - 255
            
    elif period >= 600:
        inc = np.random.uniform(88,90)
        plusminus = np.random.rand()
        if plusminus < 0.065:
            inc = 27.61905*inc - 2398.476
        inc = max(inc,46)
        if inc == 46:
            shift = np.random.rand()
            if shift <= 0.25:
                inc = np.random.uniform(46,50)
            elif 0.25 < shift <= 0.5:
                inc = np.random.uniform(50,55)
            elif 0.5 < shift <= 0.75:
                inc = np.random.uniform(55,60)
            else:
                inc = np.random.uniform(46,60)
    rad_inc = deg_to_rad(inc)
    
    return inc, rad_inc


def generate_guess(transittimes, star_mass):
    sim = rebound.Simulation()
    sim.units = ("day", "AU", "Msun")
    sim.add(m=star_mass)
    body_index = 1
    num_bod = len(transittimes)
    params_guess = np.zeros((num_bod+1,7))
    if 0.65 <= star_mass <= 1.5:
        cold_jup_odds = 0.8555 / (num_bod - 0.5)
    else:
        cold_jup_odds = 0.0
    if 0.31 <= star_mass <= 2.22:
        hot_jup_odds = 0.052961 / (num_bod - 0.5)
    else:
        hot_jup_odds = 0.0
    for i, x in enumerate(transittimes):
        while (body_index - 1) == i:
            x = x[np.nonzero(x)[0]]
            if len(x) == 1:
                period = x[0]
            else:
                A = np.vstack([np.ones(len(x)), range(len(x))]).T
                c, period = np.linalg.lstsq(A, x, rcond=-1)[0]
                period *= np.random.uniform(0.999, 1.001)
            if period > 850:
                mass_epsi = (23/25)*(13/15)
                jup_mass = np.random.f(25,13) * (2/mass_epsi)
                mass_diff = jup_mass - 2
                if mass_diff < 0:
                    jup_mass += 0.2 * mass_diff
                    jup_mass = max(jup_mass,0.3)
                else:
                    jup_mass += 0.2 * mass_diff
                    jup_mass = min(jup_mass,20)
                mass = jup_mass / 1047.57
            elif period > 575:
                terr_odds = 1.0 - cold_jup_odds
                plan_type = np.random.choice(2,p=[terr_odds,cold_jup_odds])
                if (0.65 > star_mass) or (star_mass > 1.5):
                    plan_type = 0
                if plan_type == 1:
                    mass_epsi = (23/25)*(13/15)
                    jup_mass = np.random.f(25,13) * (2/mass_epsi)
                    mass_diff = jup_mass - 2
                    if mass_diff < 0:
                        jup_mass += 0.2 * mass_diff
                        jup_mass = max(jup_mass,0.3)
                    else:
                        jup_mass += 0.2 * mass_diff
                        jup_mass = min(jup_mass,20)
                    mass = jup_mass / 1047.57
                else:
                    jup_mass = 0 
                    while jup_mass < 4e-2:
                        mass_epsi = (7/9)*(20/22)
                        mass = np.random.f(9,20)*(1.43e-5/mass_epsi)
                        if 9.5459e-6 <= mass <= 2.8638e-5:
                            diff = mass - 1.43e-5
                            mass += 0.05*diff
                        elif mass > 2.8638e-5:
                            diff = mass - 2.8638e-5
                            move = np.random.random()
                            if move <= 1/7:
                                mass += 4 * diff
                            elif 1/7 < move <= 2/7:
                                mass += 3.25 * diff
                            elif 2/7 < move <= 3/7:
                                mass += 2.5 * diff
                            elif 3/7 < move <= 4/7:
                                mass += 1.25 * diff
                            elif 4/7 < move <= 5/7:
                                mass += 0.5 * diff
                            elif 5/7 < move <= 6/7:
                                mass += 0.125 * diff
                            mass = min(3.8e-4,mass)
                            if mass == 3.8e-4:
                                bump = np.random.rand()
                                if bump <= 0.25:
                                    mass /= 1 + np.random.rand()
                                elif 0.25 < bump <= 0.5:
                                    mass /= 2 + np.random.rand()
                                elif 0.5 < bump <= 0.75:
                                    mass /= 3 + np.random.rand()
                                else:
                                    mass /= 4 + np.random.rand()
                            switch = np.random.rand()
                            if switch <= 0.6:
                                mass = np.random.uniform(9.5459e-6,2.8638e-5)
                        elif mass < 9.5459e-6:
                            diff = mass - 9.5459e-6
                            mass += 0.5 * diff
                            mass = max(4.5e-7,mass)
                            switch = np.random.rand()
                            if switch <= 0.1:
                                mass = np.random.uniform(9.5459e-6,2.8638e-5)
                            if mass == 4.5e-7:
                                bump = np.random.rand()
                                if bump <= 0.25:
                                    mass *= 3 + np.random.rand()
                                elif 0.25 < bump <= 0.5:
                                    mass *= 4 + np.random.rand()
                                elif 0.5 < bump <= 0.75:
                                    mass *= 5 + np.random.rand()
                                else:
                                    mass *= 6 + np.random.rand()
                        jup_mass = mass * 1047.57
            elif period > 150:
                terr_odds = 1.0 - cold_jup_odds
                plan_type = np.random.choice(2,p=[terr_odds,cold_jup_odds])
                if (0.65 > star_mass) or (star_mass > 1.5):
                    plan_type = 0
                if plan_type == 1:
                    mass_epsi = (23/25)*(13/15)
                    jup_mass = np.random.f(25,13) * (2/mass_epsi)
                    mass_diff = jup_mass - 2
                    if mass_diff < 0:
                        jup_mass += 0.2 * mass_diff
                        jup_mass = max(jup_mass,0.3)
                    else:
                        jup_mass += 0.2 * mass_diff
                        jup_mass = min(jup_mass,20)
                    mass = jup_mass / 1047.57
                else:
                    jup_mass = 0 
                    while jup_mass < 1.3e-2:
                        mass_epsi = (7/9)*(20/22)
                        mass = np.random.f(9,20)*(1.43e-5/mass_epsi)
                        if 9.5459e-6 <= mass <= 2.8638e-5:
                            diff = mass - 1.43e-5
                            mass += 0.05*diff
                        elif mass > 2.8638e-5:
                            diff = mass - 2.8638e-5
                            move = np.random.random()
                            if move <= 1/7:
                                mass += 4 * diff
                            elif 1/7 < move <= 2/7:
                                mass += 3.25 * diff
                            elif 2/7 < move <= 3/7:
                                mass += 2.5 * diff
                            elif 3/7 < move <= 4/7:
                                mass += 1.25 * diff
                            elif 4/7 < move <= 5/7:
                                mass += 0.5 * diff
                            elif 5/7 < move <= 6/7:
                                mass += 0.125 * diff
                            mass = min(3.8e-4,mass)
                            if mass == 3.8e-4:
                                bump = np.random.rand()
                                if bump <= 0.25:
                                    mass /= 1 + np.random.rand()
                                elif 0.25 < bump <= 0.5:
                                    mass /= 2 + np.random.rand()
                                elif 0.5 < bump <= 0.75:
                                    mass /= 3 + np.random.rand()
                                else:
                                    mass /= 4 + np.random.rand()
                            switch = np.random.rand()
                            if switch <= 0.6:
                                mass = np.random.uniform(9.5459e-6,2.8638e-5)
                        elif mass < 9.5459e-6:
                            diff = mass - 9.5459e-6
                            mass += 0.5 * diff
                            mass = max(4.5e-7,mass)
                            switch = np.random.rand()
                            if switch <= 0.1:
                                mass = np.random.uniform(9.5459e-6,2.8638e-5)
                            if mass == 4.5e-7:
                                bump = np.random.rand()
                                if bump <= 0.25:
                                    mass *= 3 + np.random.rand()
                                elif 0.25 < bump <= 0.5:
                                    mass *= 4 + np.random.rand()
                                elif 0.5 < bump <= 0.75:
                                    mass *= 5 + np.random.rand()
                                else:
                                    mass *= 6 + np.random.rand()
                        jup_mass = mass * 1047.57
            elif period >= 50:
                terr_odds = 1.0 - cold_jup_odds
                plan_type = np.random.choice(2,p=[terr_odds,cold_jup_odds])
                if (0.65 > star_mass) or (star_mass > 1.5):
                    plan_type = 0
                if plan_type == 1:
                    mass_epsi = (23/25)*(13/15)
                    jup_mass = np.random.f(25,13) * (2/mass_epsi)
                    mass_diff = jup_mass - 2
                    if mass_diff < 0:
                        jup_mass += 0.2 * mass_diff
                        jup_mass = max(jup_mass,0.3)
                    else:
                        jup_mass += 0.2 * mass_diff
                        jup_mass = min(jup_mass,20)
                    mass = jup_mass / 1047.57
                else:
                    jup_mass = 0 
                    while jup_mass < 2e-3:
                        mass_epsi = (7/9)*(20/22)
                        mass = np.random.f(9,20)*(1.43e-5/mass_epsi)
                        if 9.5459e-6 <= mass <= 2.8638e-5:
                            diff = mass - 1.43e-5
                            mass += 0.05*diff
                        elif mass > 2.8638e-5:
                            diff = mass - 2.8638e-5
                            move = np.random.random()
                            if move <= 1/7:
                                mass += 4 * diff
                            elif 1/7 < move <= 2/7:
                                mass += 3.25 * diff
                            elif 2/7 < move <= 3/7:
                                mass += 2.5 * diff
                            elif 3/7 < move <= 4/7:
                                mass += 1.25 * diff
                            elif 4/7 < move <= 5/7:
                                mass += 0.5 * diff
                            elif 5/7 < move <= 6/7:
                                mass += 0.125 * diff
                            mass = min(3.8e-4,mass)
                            if mass == 3.8e-4:
                                bump = np.random.rand()
                                if bump <= 0.25:
                                    mass /= 1 + np.random.rand()
                                elif 0.25 < bump <= 0.5:
                                    mass /= 2 + np.random.rand()
                                elif 0.5 < bump <= 0.75:
                                    mass /= 3 + np.random.rand()
                                else:
                                    mass /= 4 + np.random.rand()
                            switch = np.random.rand()
                            if switch <= 0.6:
                                mass = np.random.uniform(9.5459e-6,2.8638e-5)
                        elif mass < 9.5459e-6:
                            diff = mass - 9.5459e-6
                            mass += 0.5 * diff
                            mass = max(4.5e-7,mass)
                            switch = np.random.rand()
                            if switch <= 0.1:
                                mass = np.random.uniform(9.5459e-6,2.8638e-5)
                            if mass == 4.5e-7:
                                bump = np.random.rand()
                                if bump <= 0.25:
                                    mass *= 3 + np.random.rand()
                                elif 0.25 < bump <= 0.5:
                                    mass *= 4 + np.random.rand()
                                elif 0.5 < bump <= 0.75:
                                    mass *= 5 + np.random.rand()
                                else:
                                    mass *= 6 + np.random.rand()
                        jup_mass = mass * 1047.57
            elif period > 40:
                jup_mass = 0 
                while jup_mass < 2e-3:
                    mass_epsi = (7/9)*(20/22)
                    mass = np.random.f(9,20)*(1.43e-5/mass_epsi)
                    if 9.5459e-6 <= mass <= 2.8638e-5:
                        diff = mass - 1.43e-5
                        mass += 0.05*diff
                    elif mass > 2.8638e-5:
                        diff = mass - 2.8638e-5
                        move = np.random.random()
                        if move <= 1/7:
                            mass += 4 * diff
                        elif 1/7 < move <= 2/7:
                            mass += 3.25 * diff
                        elif 2/7 < move <= 3/7:
                            mass += 2.5 * diff
                        elif 3/7 < move <= 4/7:
                            mass += 1.25 * diff
                        elif 4/7 < move <= 5/7:
                            mass += 0.5 * diff
                        elif 5/7 < move <= 6/7:
                            mass += 0.125 * diff
                        mass = min(3.8e-4,mass)
                        if mass == 3.8e-4:
                            bump = np.random.rand()
                            if bump <= 0.25:
                                mass /= 1 + np.random.rand()
                            elif 0.25 < bump <= 0.5:
                                mass /= 2 + np.random.rand()
                            elif 0.5 < bump <= 0.75:
                                mass /= 3 + np.random.rand()
                            else:
                                mass /= 4 + np.random.rand()
                        switch = np.random.rand()
                        if switch <= 0.6:
                            mass = np.random.uniform(9.5459e-6,2.8638e-5)
                    elif mass < 9.5459e-6:
                        diff = mass - 9.5459e-6
                        mass += 0.5 * diff
                        mass = max(4.5e-7,mass)
                        switch = np.random.rand()
                        if switch <= 0.1:
                            mass = np.random.uniform(9.5459e-6,2.8638e-5)
                        if mass == 4.5e-7:
                            bump = np.random.rand()
                            if bump <= 0.25:
                                mass *= 3 + np.random.rand()
                            elif 0.25 < bump <= 0.5:
                                mass *= 4 + np.random.rand()
                            elif 0.5 < bump <= 0.75:
                                mass *= 5 + np.random.rand()
                            else:
                                mass *= 6 + np.random.rand()
                    jup_mass = mass * 1047.57
            elif period > 35:
                terr_odds = 1.0 - hot_jup_odds
                plan_type = np.random.choice(2,p=[terr_odds,hot_jup_odds])
                if (0.31 > star_mass) or (star_mass > 2.22):
                    plan_type = 0
                if plan_type == 1:
                    mass_epsi = (86/88) * (11/13)
                    jup_mass = np.random.f(88,11) * (1.5/mass_epsi)
                    mass_diff = jup_mass - 1.5
                    if mass_diff < 0:
                        jup_mass += 0.75 * mass_diff
                        jup_mass = max(jup_mass,0.3)
                    else:
                        jup_mass += 1.0 * mass_diff
                        jup_mass = min(jup_mass,15)
                    mass = jup_mass / 1047.57
                else:
                    jup_mass = 0 
                    while jup_mass < 2e-3:
                        mass_epsi = (7/9)*(20/22)
                        mass = np.random.f(9,20)*(1.43e-5/mass_epsi)
                        if 9.5459e-6 <= mass <= 2.8638e-5:
                            diff = mass - 1.43e-5
                            mass += 0.05*diff
                        elif mass > 2.8638e-5:
                            diff = mass - 2.8638e-5
                            move = np.random.random()
                            if move <= 1/7:
                                mass += 4 * diff
                            elif 1/7 < move <= 2/7:
                                mass += 3.25 * diff
                            elif 2/7 < move <= 3/7:
                                mass += 2.5 * diff
                            elif 3/7 < move <= 4/7:
                                mass += 1.25 * diff
                            elif 4/7 < move <= 5/7:
                                mass += 0.5 * diff
                            elif 5/7 < move <= 6/7:
                                mass += 0.125 * diff
                            mass = min(3.8e-4,mass)
                            if mass == 3.8e-4:
                                bump = np.random.rand()
                                if bump <= 0.25:
                                    mass /= 1 + np.random.rand()
                                elif 0.25 < bump <= 0.5:
                                    mass /= 2 + np.random.rand()
                                elif 0.5 < bump <= 0.75:
                                    mass /= 3 + np.random.rand()
                                else:
                                    mass /= 4 + np.random.rand()
                            switch = np.random.rand()
                            if switch <= 0.6:
                                mass = np.random.uniform(9.5459e-6,2.8638e-5)
                        elif mass < 9.5459e-6:
                            diff = mass - 9.5459e-6
                            mass += 0.5 * diff
                            mass = max(4.5e-7,mass)
                            switch = np.random.rand()
                            if switch <= 0.1:
                                mass = np.random.uniform(9.5459e-6,2.8638e-5)
                            if mass == 4.5e-7:
                                bump = np.random.rand()
                                if bump <= 0.25:
                                    mass *= 3 + np.random.rand()
                                elif 0.25 < bump <= 0.5:
                                    mass *= 4 + np.random.rand()
                                elif 0.5 < bump <= 0.75:
                                    mass *= 5 + np.random.rand()
                                else:
                                    mass *= 6 + np.random.rand()
                        jup_mass = mass * 1047.57
            elif period >= 2.5:
                terr_odds = 1.0 - hot_jup_odds
                plan_type = np.random.choice(2,p=[terr_odds,hot_jup_odds])
                if (0.31 > star_mass) or (star_mass > 2.22):
                    plan_type = 0
                if plan_type == 1:
                    mass_epsi = (86/88) * (11/13)
                    jup_mass = np.random.f(88,11) * (1.5/mass_epsi)
                    mass_diff = jup_mass - 1.5
                    if mass_diff < 0:
                        jup_mass += 0.75 * mass_diff
                        jup_mass = max(jup_mass,0.3)
                    else:
                        jup_mass += 1.0 * mass_diff
                        jup_mass = min(jup_mass,15)
                    mass = jup_mass / 1047.57
                else:
                    jup_mass = 0 
                    while True:
                        mass_epsi = (7/9)*(20/22)
                        mass = np.random.f(9,20)*(1.43e-5/mass_epsi)
                        if 9.5459e-6 <= mass <= 2.8638e-5:
                            diff = mass - 1.43e-5
                            mass += 0.05*diff
                        elif mass > 2.8638e-5:
                            diff = mass - 2.8638e-5
                            move = np.random.random()
                            if move <= 1/7:
                                mass += 4 * diff
                            elif 1/7 < move <= 2/7:
                                mass += 3.25 * diff
                            elif 2/7 < move <= 3/7:
                                mass += 2.5 * diff
                            elif 3/7 < move <= 4/7:
                                mass += 1.25 * diff
                            elif 4/7 < move <= 5/7:
                                mass += 0.5 * diff
                            elif 5/7 < move <= 6/7:
                                mass += 0.125 * diff
                            mass = min(3.8e-4,mass)
                            if mass == 3.8e-4:
                                bump = np.random.rand()
                                if bump <= 0.25:
                                    mass /= 1 + np.random.rand()
                                elif 0.25 < bump <= 0.5:
                                    mass /= 2 + np.random.rand()
                                elif 0.5 < bump <= 0.75:
                                    mass /= 3 + np.random.rand()
                                else:
                                    mass /= 4 + np.random.rand()
                            switch = np.random.rand()
                            if switch <= 0.6:
                                mass = np.random.uniform(9.5459e-6,2.8638e-5)
                        elif mass < 9.5459e-6:
                            diff = mass - 9.5459e-6
                            mass += 0.5 * diff
                            mass = max(4.5e-7,mass)
                            switch = np.random.rand()
                            if switch <= 0.1:
                                mass = np.random.uniform(9.5459e-6,2.8638e-5)
                            if mass == 4.5e-7:
                                bump = np.random.rand()
                                if bump <= 0.25:
                                    mass *= 3 + np.random.rand()
                                elif 0.25 < bump <= 0.5:
                                    mass *= 4 + np.random.rand()
                                elif 0.5 < bump <= 0.75:
                                    mass *= 5 + np.random.rand()
                                else:
                                    mass *= 6 + np.random.rand()
                        jup_mass = mass * 1047.57
                        break
            elif period >= 0.9:
                terr_odds = 1 - hot_jup_odds
                plan_type = np.random.choice(2,p=[terr_odds,hot_jup_odds])
                if (0.31 > star_mass) or (star_mass > 2.22):
                    plan_type = 0
                if plan_type == 1:
                    mass_epsi = (86/88) * (11/13)
                    jup_mass = np.random.f(88,11) * (1.5/mass_epsi)
                    mass_diff = jup_mass - 1.5
                    if mass_diff < 0:
                        jup_mass += 0.75 * mass_diff
                        jup_mass = max(jup_mass,0.3)
                    else:
                        jup_mass += 1.0 * mass_diff
                        jup_mass = min(jup_mass,15)
                    mass = jup_mass / 1047.57
                else:
                    jup_mass = 10
                    while jup_mass >= 4e-2:
                        mass_epsi = (7/9)*(20/22)
                        mass = np.random.f(9,20)*(1.43e-5/mass_epsi)
                        if 9.5459e-6 <= mass <= 2.8638e-5:
                            diff = mass - 1.43e-5
                            mass += 0.05*diff
                        elif mass > 2.8638e-5:
                            diff = mass - 2.8638e-5
                            move = np.random.random()
                            if move <= 1/7:
                                mass += 4 * diff
                            elif 1/7 < move <= 2/7:
                                mass += 3.25 * diff
                            elif 2/7 < move <= 3/7:
                                mass += 2.5 * diff
                            elif 3/7 < move <= 4/7:
                                mass += 1.25 * diff
                            elif 4/7 < move <= 5/7:
                                mass += 0.5 * diff
                            elif 5/7 < move <= 6/7:
                                mass += 0.125 * diff
                            mass = min(3.8e-4,mass)
                            if mass == 3.8e-4:
                                bump = np.random.rand()
                                if bump <= 0.25:
                                    mass /= 1 + np.random.rand()
                                elif 0.25 < bump <= 0.5:
                                    mass /= 2 + np.random.rand()
                                elif 0.5 < bump <= 0.75:
                                    mass /= 3 + np.random.rand()
                                else:
                                    mass /= 4 + np.random.rand()
                            switch = np.random.rand()
                            if switch <= 0.6:
                                mass = np.random.uniform(9.5459e-6,2.8638e-5)
                        elif mass < 9.5459e-6:
                            diff = mass - 9.5459e-6
                            mass += 0.5 * diff
                            mass = max(4.5e-7,mass)
                            switch = np.random.rand()
                            if switch <= 0.1:
                                mass = np.random.uniform(9.5459e-6,2.8638e-5)
                            if mass == 4.5e-7:
                                bump = np.random.rand()
                                if bump <= 0.25:
                                    mass *= 3 + np.random.rand()
                                elif 0.25 < bump <= 0.5:
                                    mass *= 4 + np.random.rand()
                                elif 0.5 < bump <= 0.75:
                                    mass *= 5 + np.random.rand()
                                else:
                                    mass *= 6 + np.random.rand()
                        jup_mass = mass * 1047.57
            elif period >= 0.8:
                jup_mass = 10
                while jup_mass >= 4e-2:
                    mass_epsi = (7/9)*(20/22)
                    mass = np.random.f(9,20)*(1.43e-5/mass_epsi)
                    if 9.5459e-6 <= mass <= 2.8638e-5:
                        diff = mass - 1.43e-5
                        mass += 0.05*diff
                    elif mass > 2.8638e-5:
                        diff = mass - 2.8638e-5
                        move = np.random.random()
                        if move <= 1/7:
                            mass += 4 * diff
                        elif 1/7 < move <= 2/7:
                            mass += 3.25 * diff
                        elif 2/7 < move <= 3/7:
                            mass += 2.5 * diff
                        elif 3/7 < move <= 4/7:
                            mass += 1.25 * diff
                        elif 4/7 < move <= 5/7:
                            mass += 0.5 * diff
                        elif 5/7 < move <= 6/7:
                            mass += 0.125 * diff
                        mass = min(3.8e-4,mass)
                        if mass == 3.8e-4:
                            bump = np.random.rand()
                            if bump <= 0.25:
                                mass /= 1 + np.random.rand()
                            elif 0.25 < bump <= 0.5:
                                mass /= 2 + np.random.rand()
                            elif 0.5 < bump <= 0.75:
                                mass /= 3 + np.random.rand()
                            else:
                                mass /= 4 + np.random.rand()
                        switch = np.random.rand()
                        if switch <= 0.6:
                            mass = np.random.uniform(9.5459e-6,2.8638e-5)
                    elif mass < 9.5459e-6:
                        diff = mass - 9.5459e-6
                        mass += 0.5 * diff
                        mass = max(4.5e-7,mass)
                        switch = np.random.rand()
                        if switch <= 0.1:
                            mass = np.random.uniform(9.5459e-6,2.8638e-5)
                        if mass == 4.5e-7:
                            bump = np.random.rand()
                            if bump <= 0.25:
                                mass *= 3 + np.random.rand()
                            elif 0.25 < bump <= 0.5:
                                mass *= 4 + np.random.rand()
                            elif 0.5 < bump <= 0.75:
                                mass *= 5 + np.random.rand()
                            else:
                                mass *= 6 + np.random.rand()
                    jup_mass = mass * 1047.57
            elif period < 0.8:
                jup_mass = 10
                while jup_mass >= 1.3e-2:
                    mass_epsi = (7/9)*(20/22)
                    mass = np.random.f(9,20)*(1.43e-5/mass_epsi)
                    if 9.5459e-6 <= mass <= 2.8638e-5:
                        diff = mass - 1.43e-5
                        mass += 0.05*diff
                    elif mass > 2.8638e-5:
                        diff = mass - 2.8638e-5
                        move = np.random.random()
                        if move <= 1/7:
                            mass += 4 * diff
                        elif 1/7 < move <= 2/7:
                            mass += 3.25 * diff
                        elif 2/7 < move <= 3/7:
                            mass += 2.5 * diff
                        elif 3/7 < move <= 4/7:
                            mass += 1.25 * diff
                        elif 4/7 < move <= 5/7:
                            mass += 0.5 * diff
                        elif 5/7 < move <= 6/7:
                            mass += 0.125 * diff
                        mass = min(3.8e-4,mass)
                        if mass == 3.8e-4:
                            bump = np.random.rand()
                            if bump <= 0.25:
                                mass /= 1 + np.random.rand()
                            elif 0.25 < bump <= 0.5:
                                mass /= 2 + np.random.rand()
                            elif 0.5 < bump <= 0.75:
                                mass /= 3 + np.random.rand()
                            else:
                                mass /= 4 + np.random.rand()
                        switch = np.random.rand()
                        if switch <= 0.6:
                            mass = np.random.uniform(9.5459e-6,2.8638e-5)
                    elif mass < 9.5459e-6:
                        diff = mass - 9.5459e-6
                        mass += 0.5 * diff
                        mass = max(4.5e-7,mass)
                        switch = np.random.rand()
                        if switch <= 0.1:
                            mass = np.random.uniform(9.5459e-6,2.8638e-5)
                        if mass == 4.5e-7:
                            bump = np.random.rand()
                            if bump <= 0.25:
                                mass *= 3 + np.random.rand()
                            elif 0.25 < bump <= 0.5:
                                mass *= 4 + np.random.rand()
                            elif 0.5 < bump <= 0.75:
                                mass *= 5 + np.random.rand()
                            else:
                                mass *= 6 + np.random.rand()
                    jup_mass = mass * 1047.57
                    
            ecc = get_ecc(period)
            deg_inc, rad_inc = get_inclination(period)
            long_asc = np.random.uniform(0,2*np.pi)
            arg_peri = np.random.uniform(0,2*np.pi)
            true_anom = np.random.uniform(0,2*np.pi)

            sim.add(m=mass,P=period,e=ecc,inc=rad_inc,Omega=long_asc,omega=arg_peri,f=true_anom)

            if body_index == 1:
                params_guess[i+1,0] = mass
                params_guess[i+1,1] = period
                params_guess[i+1,2] = ecc
                params_guess[i+1,3] = rad_inc
                params_guess[i+1,4] = long_asc
                params_guess[i+1,5] = arg_peri
                params_guess[i+1,6] = true_anom
                body_index += 1
                
            else:
                pass_flag = 1
                orbits = sim.calculate_orbits()
                true_anoms = np.linspace(0, 2*np.pi,250)
                x_points = np.empty(len(orbits), dtype="object")
                y_points = np.empty(len(orbits), dtype="object")
                z_points = np.empty(len(orbits), dtype="object")
                for j, o in enumerate(orbits):
                    r = o.a * ((1 - o.e**2)/(1 + o.e*np.cos(true_anoms)))
                    x_points[j] = r * (np.cos(o.Omega)*np.cos(o.omega+true_anoms) - np.sin(o.Omega)*np.sin(o.omega+true_anoms)*np.cos(o.inc))
                    y_points[j] = r * (np.sin(o.Omega)*np.cos(o.omega+true_anoms) + np.cos(o.Omega)*np.sin(o.omega+true_anoms)*np.cos(o.inc))
                    z_points[j] = r * (np.sin(o.omega+true_anoms)*np.sin(o.inc))
                for k in range(len(orbits[:-1])):
                    x2 = x_points[k]
                    y2 = y_points[k]
                    z2 = z_points[k]
                    x1 = x_points[-1]
                    y1 = y_points[-1]
                    z1 = z_points[-1]
                    d = np.zeros(len(x2)**2)
                    n = 0
                    for l in range(len(x2)):
                        for m in range(len(x1)):
                            d[n] = np.sqrt((x2[l]-x1[m])**2 + (y2[l]-y1[m])**2 + (z2[l]-z1[m])**2)
                            n += 1
                    if d.min() < 5 * max(orbits[1].rhill, orbits[-1].rhill):
                        pass_flag = 0
                        sim.remove(body_index)
                        break

                if pass_flag == True:
                    params_guess[i+1,0] = mass
                    params_guess[i+1,1] = period
                    params_guess[i+1,2] = ecc
                    params_guess[i+1,3] = rad_inc
                    params_guess[i+1,4] = long_asc
                    params_guess[i+1,5] = arg_peri
                    params_guess[i+1,6] = true_anom
                    body_index += 1

    params_guess[0,0] = star_mass
    params_guess = params_guess.flatten()
    params_guess = np.delete(params_guess, [1,2,3,4,5,6])
    
    return params_guess