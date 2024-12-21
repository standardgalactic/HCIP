import numpy as np
import scipy.stats as stats
from os.path import exists
from scipy.stats import norm


# to not always return the idx of the first max element
def fuzzy_argmax(data, axis=0):
    max_mask = data == data.max(axis=axis)[:, None]
    
    # Generate random numbers of the same shape as the input
    random_selection = np.random.random(data.shape)
    
    # Zero out random numbers where values are not maximum
    random_selection = random_selection * max_mask
    
    # Get the indices of the maximum random value for each row
    return np.argmax(random_selection, axis=axis)

def _likelihood_for_arms(arms, wb, ts):
    """
    Calculate the likelihood for each arm given the correct arm

    arms: possible arms

    wb: standard deviation

    ts: the correct length

    """


    # sort arms to compute sequentially the Gaussian intersections
    arms_sortid = np.argsort(arms)
    arms_sorted=arms[arms_sortid]

    # if some lengths have duplicates, treat them as a single Gaussian
    # then the likelihood is divided by # duplicates 
    arms_unique = []
    for arm in arms_sorted:
        if not arms_unique:
            arms_unique.append([arm, 1])
        else:
            if arm == arms_unique[-1][0]:
                arms_unique[-1][1] += 1
            else:
                arms_unique.append([arm, 1])
    
    # if all arms are the same, return uniform distribution
    if len(arms_unique) == 1:
        return np.ones(len(arms))/len(arms)
    
    # compute intersections of Gaussian PDFs from left to right
    intervals = []
    for i in range(len(arms_unique)-1):
        mu1 = arms_unique[i][0]
        mu2 = arms_unique[i+1][0]
        intervals.append(_find_intersection(mu1, mu1*wb, mu2, mu2*wb))

    intervals.insert(0, -np.inf)
    intervals.append(np.inf)

    # calculate the cdf under each interval
    cdf = []
    for i in range(len(intervals)-1):
        cdf.append(norm.cdf(intervals[i+1], loc=ts, scale=ts*wb) - norm.cdf(intervals[i], loc=ts, scale=ts*wb))
    
    cdf_each = []
    for i, (arm, count) in enumerate(arms_unique):
        cdf_each.extend((np.ones(count)*cdf[i]/count).tolist())
    
    cdf_each = np.array(cdf_each)

    # sort the likelihoods back to the original order
    cdf_each = cdf_each[np.argsort(arms_sortid)]
    return cdf_each

def _find_intersection(mu1, sigma1, mu2, sigma2):
    """
    Find the intersection point of two Gaussian PDFs.
    Returns both solutions of the quadratic equation.
    """
    # If distributions are identical, return the mean
    if mu1 == mu2 and sigma1 == sigma2:
        return mu1
    
    # If standard deviations are equal, intersection is halfway between means
    if sigma1 == sigma2:
        return (mu1 + mu2) / 2
    
    # Otherwise, solve the quadratic equation
    # From equating two Gaussian PDFs and taking ln of both sides:
    # ax² + bx + c = 0
    a = -(1/(2*sigma1**2)) + (1/(2*sigma2**2))
    b = (mu1/(sigma1**2)) - (mu2/(sigma2**2))
    c = -(mu1**2/(2*sigma1**2)) + (mu2**2/(2*sigma2**2)) + np.log(sigma2/sigma1)
    
    # Quadratic formula
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        return None  # No intersection
    
    x1 = (-b + np.sqrt(discriminant))/(2*a)
    x2 = (-b - np.sqrt(discriminant))/(2*a)
    
    # Return the intersection point that lies between the means
    if mu1 < mu2:
        return max(x1, x2)
    else:
        return min(x1, x2)


def optimal_emission(trials, wb, n_sim=10000):   
    in_arms = trials['in_arms']
    labels = trials['labels'].astype(int)
    arm_pairs = np.array([[0, 2],
                          [0, 3],
                          [1, 4],
                          [1, 5]])
    n_trials = in_arms.shape[0]

    emission = np.zeros((n_trials, 4))
    for n in range(n_trials):

        ts1 = in_arms[n,arm_pairs[labels[n],0], 0]
        ts2 = in_arms[n,arm_pairs[labels[n],1], 0]

        tm1 = np.random.normal(ts1, ts1*wb, n_sim)
        tm2 = np.random.normal(ts2, ts2*wb, n_sim)

        arm_l = np.repeat(in_arms[n, 0, 0], n_sim)
        arm_r = np.repeat(in_arms[n, 1, 0], n_sim)
        arm_lu = np.repeat(in_arms[n, 2, 0], n_sim)
        arm_ld = np.repeat(in_arms[n, 3, 0], n_sim)
        arm_ru = np.repeat(in_arms[n, 4, 0], n_sim)
        arm_rd = np.repeat(in_arms[n, 5, 0], n_sim)

        p_l = norm.pdf(tm1, arm_l, arm_l*wb)
        p_r = norm.pdf(tm1, arm_r, arm_r*wb)
        p_lu = norm.pdf(tm2, arm_lu, arm_lu*wb)
        p_ld = norm.pdf(tm2, arm_ld, arm_ld*wb)
        p_ru = norm.pdf(tm2, arm_ru, arm_ru*wb)
        p_rd = norm.pdf(tm2, arm_rd, arm_rd*wb)


        p1 = p_l*p_lu
        p2 = p_l*p_ld
        p3 = p_r*p_ru
        p4 = p_r*p_rd

        p = np.stack([p1, p2, p3, p4], axis=1)
        choices = fuzzy_argmax(p, axis=1)
        for i in range(4):
            emission[n, i] = np.sum(choices == i)/n_sim

    return emission


def postdictive_emission(trials, wb, n_sim=10000):   
    in_arms = trials['in_arms']
    labels = trials['labels'].astype(int)
    arm_pairs = np.array([[0, 2],
                          [0, 3],
                          [1, 4],
                          [1, 5]])
    n_trials = in_arms.shape[0]

    emission = np.zeros((n_trials, 4))
    for n in range(n_trials):

        ts1 = in_arms[n,arm_pairs[labels[n],0], 0]
        ts2 = in_arms[n,arm_pairs[labels[n],1], 0]

        tm1 = np.random.normal(ts1, ts1*wb, n_sim)
        tm2 = np.random.normal(ts2, ts2*wb, n_sim)

        arm_l = np.repeat(in_arms[n, 0, 0], n_sim)
        arm_r = np.repeat(in_arms[n, 1, 0], n_sim)
        arm_lu = np.repeat(in_arms[n, 2, 0], n_sim)
        arm_ld = np.repeat(in_arms[n, 3, 0], n_sim)
        arm_ru = np.repeat(in_arms[n, 4, 0], n_sim)
        arm_rd = np.repeat(in_arms[n, 5, 0], n_sim)

        p_l = norm.pdf(tm1, arm_l, arm_l*wb)
        p_r = norm.pdf(tm1, arm_r, arm_r*wb)
        p_lu = norm.pdf(tm2, arm_lu, arm_lu*wb)
        p_ld = norm.pdf(tm2, arm_ld, arm_ld*wb)
        p_ru = norm.pdf(tm2, arm_ru, arm_ru*wb)
        p_rd = norm.pdf(tm2, arm_rd, arm_rd*wb)


        p_h = np.stack([p_l*(p_lu+p_ld), p_r*(p_ru+p_rd)], axis=1)
        choices_h = fuzzy_argmax(p_h, axis=1)
        choices_l = fuzzy_argmax(np.stack([p_lu, p_ld], axis=1), axis=1)
        choices_r = fuzzy_argmax(np.stack([p_ru, p_rd], axis=1), axis=1)

        emission[n, 0] = np.sum(np.logical_and(choices_h==0, choices_l==0))/n_sim
        emission[n, 1] = np.sum(np.logical_and(choices_h==0, choices_l==1))/n_sim
        emission[n, 2] = np.sum(np.logical_and(choices_h==1, choices_r==0))/n_sim
        emission[n, 3] = np.sum(np.logical_and(choices_h==1, choices_r==1))/n_sim

    return emission


def hierarchy_emission(trials, wb, n_sim=10000):   
    in_arms = trials['in_arms']
    labels = trials['labels'].astype(int)
    arm_pairs = np.array([[0, 2],
                          [0, 3],
                          [1, 4],
                          [1, 5]])
    n_trials = in_arms.shape[0]

    emission = np.zeros((n_trials, 4))
    for n in range(n_trials):

        ts1 = in_arms[n,arm_pairs[labels[n],0], 0]
        ts2 = in_arms[n,arm_pairs[labels[n],1], 0]

        tm1 = np.random.normal(ts1, ts1*wb, n_sim)
        tm2 = np.random.normal(ts2, ts2*wb, n_sim)

        arm_l = np.repeat(in_arms[n, 0, 0], n_sim)
        arm_r = np.repeat(in_arms[n, 1, 0], n_sim)
        arm_lu = np.repeat(in_arms[n, 2, 0], n_sim)
        arm_ld = np.repeat(in_arms[n, 3, 0], n_sim)
        arm_ru = np.repeat(in_arms[n, 4, 0], n_sim)
        arm_rd = np.repeat(in_arms[n, 5, 0], n_sim)

        p_l = norm.pdf(tm1, arm_l, arm_l*wb)
        p_r = norm.pdf(tm1, arm_r, arm_r*wb)
        p_lu = norm.pdf(tm2, arm_lu, arm_lu*wb)
        p_ld = norm.pdf(tm2, arm_ld, arm_ld*wb)
        p_ru = norm.pdf(tm2, arm_ru, arm_ru*wb)
        p_rd = norm.pdf(tm2, arm_rd, arm_rd*wb)


        p_h = np.stack([p_l, p_r], axis=1)
        choices_h = fuzzy_argmax(p_h, axis=1)
        choices_l = fuzzy_argmax(np.stack([p_lu, p_ld], axis=1), axis=1)
        choices_r = fuzzy_argmax(np.stack([p_ru, p_rd], axis=1), axis=1)

        emission[n, 0] = np.sum(np.logical_and(choices_h==0, choices_l==0))/n_sim
        emission[n, 1] = np.sum(np.logical_and(choices_h==0, choices_l==1))/n_sim
        emission[n, 2] = np.sum(np.logical_and(choices_h==1, choices_r==0))/n_sim
        emission[n, 3] = np.sum(np.logical_and(choices_h==1, choices_r==1))/n_sim

    return emission


def secondary_emission(trials, wb, n_sim=10000):   
    in_arms = trials['in_arms']
    labels = trials['labels'].astype(int)
    arm_pairs = np.array([[0, 2],
                          [0, 3],
                          [1, 4],
                          [1, 5]])
    n_trials = in_arms.shape[0]

    emission = np.zeros((n_trials, 4))
    for n in range(n_trials):

        ts2 = in_arms[n,arm_pairs[labels[n],1], 0]

        tm2 = np.random.normal(ts2, ts2*wb, n_sim)

        arm_lu = np.repeat(in_arms[n, 2, 0], n_sim)
        arm_ld = np.repeat(in_arms[n, 3, 0], n_sim)
        arm_ru = np.repeat(in_arms[n, 4, 0], n_sim)
        arm_rd = np.repeat(in_arms[n, 5, 0], n_sim)

        p_lu = norm.pdf(tm2, arm_lu, arm_lu*wb)
        p_ld = norm.pdf(tm2, arm_ld, arm_ld*wb)
        p_ru = norm.pdf(tm2, arm_ru, arm_ru*wb)
        p_rd = norm.pdf(tm2, arm_rd, arm_rd*wb)


        p = np.stack([p_lu, p_ld, p_ru, p_rd], axis=1).T
        choices = fuzzy_argmax(p, axis=0)
        for i in range(4):
            emission[n, i] = np.sum(choices == i)/n_sim

    return emission


def counterfactual_emission(trials, wb, wb_inc, threshold=0.05, n_sim=10000):
    in_arms = trials['in_arms']
    labels = trials['labels'].astype(int)
    arm_pairs = np.array([[0, 2],
                          [0, 3],
                          [1, 4],
                          [1, 5]])
    n_trials = in_arms.shape[0]

    emission = np.zeros((n_trials, 4))
    for n in range(n_trials):

        ts1 = in_arms[n,arm_pairs[labels[n],0], 0]
        ts2 = in_arms[n,arm_pairs[labels[n],1], 0]

        tm1 = np.random.normal(ts1, ts1*wb, n_sim)
        tm2 = np.random.normal(ts2, ts2*wb, n_sim)

        arm_l = np.repeat(in_arms[n, 0, 0], n_sim)
        arm_r = np.repeat(in_arms[n, 1, 0], n_sim)
        arm_lu = np.repeat(in_arms[n, 2, 0], n_sim)
        arm_ld = np.repeat(in_arms[n, 3, 0], n_sim)
        arm_ru = np.repeat(in_arms[n, 4, 0], n_sim)
        arm_rd = np.repeat(in_arms[n, 5, 0], n_sim)

        p_l = norm.pdf(tm1, arm_l, arm_l*wb)
        p_r = norm.pdf(tm1, arm_r, arm_r*wb)
        p_lu = norm.pdf(tm2, arm_lu, arm_lu*wb)
        p_ld = norm.pdf(tm2, arm_ld, arm_ld*wb)
        p_ru = norm.pdf(tm2, arm_ru, arm_ru*wb)
        p_rd = norm.pdf(tm2, arm_rd, arm_rd*wb)



        p_h = np.stack([p_l, p_r], axis=1)
        choices_h = fuzzy_argmax(p_h, axis=1)

        p_v = np.stack([np.stack([p_lu, p_ld], axis=1), np.stack([p_ru, p_rd], axis=1)], axis=2)

        likelihood_1st_attempt = np.max(p_h, 1)/np.min(p_h, 1)*np.max(p_v[np.arange(n_sim),:,choices_h],axis=1)

        switch = likelihood_1st_attempt < threshold
        choices_h[switch] = 1 - choices_h[switch]
        tm2[switch] = np.random.normal(tm2[switch], abs(wb_inc*ts2))

        p_lu[switch] = norm.pdf(tm2[switch], arm_lu[switch], arm_lu[switch]*wb)
        p_ld[switch] = norm.pdf(tm2[switch], arm_ld[switch], arm_ld[switch]*wb)
        p_ru[switch] = norm.pdf(tm2[switch], arm_ru[switch], arm_ru[switch]*wb)
        p_rd[switch] = norm.pdf(tm2[switch], arm_rd[switch], arm_rd[switch]*wb)


        choices_l = fuzzy_argmax(np.stack([p_lu, p_ld], axis=1), axis=1)
        choices_r = fuzzy_argmax(np.stack([p_ru, p_rd], axis=1), axis=1)

        emission[n, 0] = np.sum(np.logical_and(choices_h==0, choices_l==0))/n_sim
        emission[n, 1] = np.sum(np.logical_and(choices_h==0, choices_l==1))/n_sim
        emission[n, 2] = np.sum(np.logical_and(choices_h==1, choices_r==0))/n_sim
        emission[n, 3] = np.sum(np.logical_and(choices_h==1, choices_r==1))/n_sim
    return emission

def counterfactual_(trials, wb, wb_inc, threshold=0.05):
    in_arms = trials['in_arms']
    in_time = trials['in_time']
    n_trials = in_arms.shape[0]
    # wb = 0.15
    # wb_inc = 0.175
    # std = 0.8
    # std_inc = 1.2

    probs1 = np.zeros((n_trials, 2))
    tm1 = np.max(in_time[:, 0, :], axis=1)
    tm2 = np.max(in_time[:, 1, :], axis=1)
    for j in range(2):
        ts1 = in_arms[:, j, 0]
        probs1[:, j] = stats.norm(ts1, wb*ts1).sf(ts1 + np.abs(ts1 - tm1))

    choice1 = fuzzy_argmax(probs1, axis=1)

    probs2 = np.zeros((n_trials, 2))
    sec_arms = np.array([[2, 3], [4, 5]])
    for j in range(2):
        ts2 = in_arms[np.arange(n_trials), sec_arms[choice1, j], 0]
        probs2[:, j] = stats.norm(ts2, wb*ts2).sf(ts2 + np.abs(ts2 - tm2))

    prob_commit = np.max(probs1, 1)/np.min(probs1, 1)*np.max(probs2, 1)

    # not to make a choice yet
    switch = prob_commit < threshold

    # flip the choice for primary choice
    choice1[switch] = 1 - choice1[switch]

    # degrade tm2 for the other side
    tm2[switch] = np.random.normal(tm2[switch], abs(wb_inc*ts2[switch]))

    # do secondary choice again
    for j in range(2):
        ts2 = in_arms[np.arange(n_trials), sec_arms[choice1, j], 0]
        probs2[:, j] = stats.norm(ts2, wb*ts2).sf(ts2 + np.abs(ts2 - tm2))

    choice2 = fuzzy_argmax(probs2, axis=1)
    mapping = {
        (0, 0): 0,
        (0, 1): 1,
        (1, 0): 2,
        (1, 1): 3
    }

    # compile choice1 and choice2
    choices = np.zeros(n_trials)
    for i in range(n_trials):
        choices[i] = mapping[(choice1[i], choice2[i])]

    return choices


def counterfactual(trials, wb, wb_inc, threshold=0.05):
    in_arms = trials['in_arms']
    in_time = trials['in_time']
    n_trials = in_arms.shape[0]
    # wb = 0.15
    # wb_inc = 0.175
    # std = 0.8
    # std_inc = 1.2

    probs1 = np.zeros((n_trials, 2))
    tm1 = np.max(in_time[:, 0, :], axis=1)
    tm2 = np.max(in_time[:, 1, :], axis=1)
    for j in range(2):
        ts1 = in_arms[:, j, 0]
        probs1[:, j] = norm.pdf(tm1, ts1, wb*ts1)

    choice1 = fuzzy_argmax(probs1, axis=1)

    probs2 = np.zeros((n_trials, 2))
    sec_arms = np.array([[2, 3], [4, 5]])
    for j in range(2):
        ts2 = in_arms[np.arange(n_trials), sec_arms[choice1, j], 0]
        probs2[:, j] = norm.pdf(tm2, ts2, wb*ts2)

    prob_commit = np.max(probs1, 1)/np.min(probs1, 1)*np.max(probs2, 1)

    # not to make a choice yet
    switch = prob_commit < threshold

    # flip the choice for primary choice
    choice1[switch] = 1 - choice1[switch]

    # degrade tm2 for the other side
    tm2[switch] = np.random.normal(tm2[switch], abs(wb_inc*ts2[switch]))

    # do secondary choice again
    for j in range(2):
        ts2 = in_arms[np.arange(n_trials), sec_arms[choice1, j], 0]
        probs2[:, j] = norm.pdf(tm2, ts2, wb*ts2)

    choice2 = fuzzy_argmax(probs2, axis=1)
    mapping = {
        (0, 0): 0,
        (0, 1): 1,
        (1, 0): 2,
        (1, 1): 3
    }

    # compile choice1 and choice2
    choices = np.zeros(n_trials)
    for i in range(n_trials):
        choices[i] = mapping[(choice1[i], choice2[i])]

    return choices


def optimal_(trials, wb):
    in_arms = trials['in_arms']
    in_time = trials['in_time']
    n_trials = in_arms.shape[0]
    # wb = 0.15
    arm_pairs = np.array([[0, 2],
                          [0, 3],
                          [1, 4],
                          [1, 5]])

    probs = np.zeros((n_trials, len(arm_pairs)))
    tm1 = np.max(in_time[:, 0, :], axis=1)
    tm2 = np.max(in_time[:, 1, :], axis=1)
    for j in range(probs.shape[1]):
        ts1 = in_arms[:, arm_pairs[j, 0], 0]
        ts2 = in_arms[:, arm_pairs[j, 1], 0]

        p_tm1 = stats.norm(ts1, ts1*wb).sf(ts1 + np.abs(ts1 - tm1))
        p_tm2 = stats.norm(ts2, ts2*wb).sf(ts2 + np.abs(ts2 - tm2))
        probs[:, j] = p_tm1 * p_tm2

    choices = fuzzy_argmax(probs, axis=1)

    return choices

def optimal(trials, wb):
    in_arms = trials['in_arms']
    in_time = trials['in_time']
    n_trials = in_arms.shape[0]
    # wb = 0.15
    arm_pairs = np.array([[0, 2],
                          [0, 3],
                          [1, 4],
                          [1, 5]])

    probs = np.zeros((n_trials, len(arm_pairs)))
    tm1 = np.max(in_time[:, 0, :], axis=1)
    tm2 = np.max(in_time[:, 1, :], axis=1)
    for j in range(probs.shape[1]):
        ts1 = in_arms[:, arm_pairs[j, 0], 0]
        ts2 = in_arms[:, arm_pairs[j, 1], 0]

        p_tm1 = norm.pdf(tm1, ts1, ts1*wb)
        p_tm2 = norm.pdf(tm2, ts2, ts2*wb)
        probs[:, j] = p_tm1 * p_tm2

    choices = fuzzy_argmax(probs, axis=1)

    return choices


def hierarchy_(trials, wb):
    in_arms = trials['in_arms']
    in_time = trials['in_time']
    n_trials = in_arms.shape[0]
    # wb = 0.15
    probs1 = np.zeros((n_trials, 2))
    tm1 = np.max(in_time[:, 0, :], axis=1)
    tm2 = np.max(in_time[:, 1, :], axis=1)
    for j in range(2):
        ts1 = in_arms[:, j, 0]
        probs1[:, j] = stats.norm(ts1, ts1*wb).sf(ts1 + np.abs(ts1 - tm1))

    choice1 = fuzzy_argmax(probs1, axis=1)

    probs2 = np.zeros((n_trials, 2))
    sec_arms = np.array([[2, 3], [4, 5]])
    for j in range(2):
        ts2 = in_arms[np.arange(n_trials), sec_arms[choice1, j], 0]
        probs2[:, j] = stats.norm(ts2, ts2*wb).sf(ts2 + np.abs(ts2 - tm2))
    choice2 = fuzzy_argmax(probs2, axis=1)
    mapping = {
        (0, 0): 0,
        (0, 1): 1,
        (1, 0): 2,
        (1, 1): 3
    }

    # compile choice1 and choice2
    choices = np.zeros(n_trials)
    for i in range(n_trials):
        choices[i] = mapping[(choice1[i], choice2[i])]

    return choices

def hierarchy(trials, wb):
    in_arms = trials['in_arms']
    in_time = trials['in_time']
    n_trials = in_arms.shape[0]
    # wb = 0.15
    probs1 = np.zeros((n_trials, 2))
    tm1 = np.max(in_time[:, 0, :], axis=1)
    tm2 = np.max(in_time[:, 1, :], axis=1)
    for j in range(2):
        ts1 = in_arms[:, j, 0]
        probs1[:, j] = norm.pdf(tm1, ts1, ts1*wb)

    choice1 = fuzzy_argmax(probs1, axis=1)

    probs2 = np.zeros((n_trials, 2))
    sec_arms = np.array([[2, 3], [4, 5]])
    for j in range(2):
        ts2 = in_arms[np.arange(n_trials), sec_arms[choice1, j], 0]
        probs2[:, j] = norm.pdf(tm2, ts2, ts2*wb)
    choice2 = fuzzy_argmax(probs2, axis=1)
    mapping = {
        (0, 0): 0,
        (0, 1): 1,
        (1, 0): 2,
        (1, 1): 3
    }

    # compile choice1 and choice2
    choices = np.zeros(n_trials)
    for i in range(n_trials):
        choices[i] = mapping[(choice1[i], choice2[i])]

    return choices


def postdictive_(trials, wb):
    in_arms = trials['in_arms']
    in_time = trials['in_time']
    n_trials = in_arms.shape[0]
    # wb = 0.15
    #std = 0.8
    probs1 = np.zeros((n_trials, 2))
    tm1 = np.max(in_time[:, 0, :], axis=1)
    tm2 = np.max(in_time[:, 1, :], axis=1)
    mapping = {
        0: [2, 3],
        1: [4, 5]
    }
    for j in range(2):
        ts1 = in_arms[:, j, 0]
        ts2u = in_arms[:, mapping[j][0], 0]
        ts2d = in_arms[:, mapping[j][1], 0]
        probs1[:, j] = stats.norm(ts1, ts1*wb).sf(ts1 + np.abs(ts1 - tm1))*(
            stats.norm(ts2u, ts2u*wb).sf(ts2u + np.abs(ts2u - tm2)) + stats.norm(ts2d, ts2d*wb).sf(ts2d + np.abs(ts2d - tm2))
        )

    choice1 = fuzzy_argmax(probs1, axis=1)

    probs2 = np.zeros((n_trials, 2))
    sec_arms = np.array([[2, 3], [4, 5]])
    for j in range(2):
        ts2 = in_arms[np.arange(n_trials), sec_arms[choice1, j], 0]
        probs2[:, j] = stats.norm(ts2, ts2*wb).sf(ts2 + np.abs(ts2 - tm2))
    choice2 = fuzzy_argmax(probs2, axis=1)
    mapping = {
        (0, 0): 0,
        (0, 1): 1,
        (1, 0): 2,
        (1, 1): 3
    }

    # compile choice1 and choice2
    choices = np.zeros(n_trials)
    for i in range(n_trials):
        choices[i] = mapping[(choice1[i], choice2[i])]

    return choices

def postdictive(trials, wb):
    in_arms = trials['in_arms']
    in_time = trials['in_time']
    n_trials = in_arms.shape[0]
    # wb = 0.15
    #std = 0.8
    probs1 = np.zeros((n_trials, 2))
    tm1 = np.max(in_time[:, 0, :], axis=1)
    tm2 = np.max(in_time[:, 1, :], axis=1)
    mapping = {
        0: [2, 3],
        1: [4, 5]
    }
    for j in range(2):
        ts1 = in_arms[:, j, 0]
        ts2u = in_arms[:, mapping[j][0], 0]
        ts2d = in_arms[:, mapping[j][1], 0]
        probs1[:, j] = norm.pdf(tm1,ts1, ts1*wb)*(
            norm.pdf(tm2, ts2u, ts2u*wb)+ norm.pdf(tm2,ts2d, ts2d*wb)
        )

    choice1 = fuzzy_argmax(probs1, axis=1)

    probs2 = np.zeros((n_trials, 2))
    sec_arms = np.array([[2, 3], [4, 5]])
    for j in range(2):
        ts2 = in_arms[np.arange(n_trials), sec_arms[choice1, j], 0]
        probs2[:, j] = norm.pdf(tm2, ts2, ts2*wb)
    choice2 = fuzzy_argmax(probs2, axis=1)
    mapping = {
        (0, 0): 0,
        (0, 1): 1,
        (1, 0): 2,
        (1, 1): 3
    }

    # compile choice1 and choice2
    choices = np.zeros(n_trials)
    for i in range(n_trials):
        choices[i] = mapping[(choice1[i], choice2[i])]

    return choices

def secondary(trials, wb):
    in_arms = trials['in_arms']
    in_time = trials['in_time']
    n_trials = in_arms.shape[0]
    # wb = 0.15
    arm_pairs = np.array([[0, 2],
                          [0, 3],
                          [1, 4],
                          [1, 5]])

    probs = np.zeros((n_trials, len(arm_pairs)))
    tm2 = np.max(in_time[:, 1, :], axis=1)
    for j in range(probs.shape[1]):
        ts2 = in_arms[:, arm_pairs[j, 1], 0]
        p_tm2 = norm.pdf(tm2, ts2, ts2*wb)
        probs[:, j] = p_tm2

    choices = fuzzy_argmax(probs, axis=1)

    return choices


    