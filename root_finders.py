import rebound
import numpy as np

def newtons(sim, p, body_i, t_old, t_new, pos_b, accuracy):
    """
    """
    # keep track of upper (b) and lower (a) bracket
    a = t_old
    b = t_new
    # keep track of old and new transit guesses
    old_guess_t = t_new
    # first guess is the interval's midpoint
    new_guess_t = (a + b) / 2
    # calculate the y position and velocity at the midpoint
    sim.integrate(new_guess_t)
    new_guess_y = p[body_i].y - p[0].y
    new_guess_v = p[body_i].vy - p[0].vy
    # keep track of number of iterations and set a maximum amount
    num_iter = 0 
    max_iter = 20
    # loop until target accuracy
    while (abs(new_guess_t - old_guess_t) > accuracy) and (num_iter < max_iter):
        num_iter += 1
        # Newton's step
        old_guess_t = new_guess_t
        if new_guess_y == 0.0 or new_guess_v == 0.0:
            return new_guess_t
        new_guess_t -= (new_guess_y / new_guess_v)
        # check within bracketed interval
        if a < new_guess_t < b:
            # calculate t, y, v for new guess
            sim.integrate(new_guess_t)
            new_guess_y = p[body_i].y - p[0].y
            new_guess_v = p[body_i].vy - p[0].vy
        else:
            # bisection step
            midpoint_t = (a + b) / 2
            sim.integrate(new_guess_t)
            midpoint_y = p[body_i].y - p[0].y
            # set midpoint as new bracket
            if pos_b * midpoint_y < 0.:
                a = midpoint_t
            else:
                b = midpoint_t
            # use midpoint of new interval as next guess for Newton's
            new_guess_t = (a + b) / 2
            sim.integrate(new_guess_t)
            new_guess_y = p[body_i].y - p[0].y
            new_guess_v = p[body_i].vy - p[0].vy
    return (new_guess_t + old_guess_t) / 2


def bisection(sim, p, body_i, t_old, t_new, pos_b, accuracy):
    """

    """
    a = t_old
    b = t_new
    max_iter = round(np.log((abs(a - b)) / accuracy) / np.log(2))
    steps_taken = 0
    while (abs(a - b) > accuracy) and steps_taken < max_iter:
        guess_t = (a + b) / 2
        sim.integrate(guess_t)
        guess_y = p[body_i].y - p[0].y
        if guess_y == 0.:
            break
        elif pos_b * guess_y < 0.:
            a = guess_t
        else:
            b = guess_t
        steps_taken += 1
    return (a + b) / 2