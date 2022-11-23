import time
import numpy as np
import matplotlib.pyplot as plt


    
def load_permutations(filename="preferences.csv"):
    """
    Load all permutations from a file

    Parameters
    ----------
    filename: string
        Path to a file
    
    Returns
    -------
    animals: A list of animals in alphabetical order
    raters: dictionary( 
        string (Ranker's name): list (This person's permutation as a list of numbers
                                      corresponding to the indices in animals)
    )
    """
    raters = {}
    fin = open(filename)
    lines = fin.readlines()
    fin.close()
    animals = [s.rstrip().replace("\"", "") for s in lines[0].split(",")[1::]]
    for line in lines[1::]:
        fields = line.split(",")
        rater = fields[0].replace("\"", "")
        fields = [int(f) for f in fields[1::]]
        raters[rater] = [0]*len(fields)
        for i, x in enumerate(fields):
            raters[rater][x-1] = i
    return animals, raters


def mds(D):
    """
    Perform multidimensional scaling

    Parameters
    ----------
    D: ndarray(N, N)
        A matrix of pairwise similarities
    
    Return
    ------
    Y: ndarray(N, N)
        MDS projection, with columns in order of variance
        explained
    """
    from numpy import linalg
    N = D.shape[0]
    H = np.eye(N) - np.ones((N, N))/N
    B = -0.5*(H.dot((D*D).dot(H)))
    U, s, V = linalg.svd(B)
    Y = np.sqrt(s[None, :])*U
    return Y

def plot_mds_distances(raters, random_state=0):
    """
    Compute all pairwise Kendall-Tau distances and plot a dimension 
    reduction from the Kendall-Tau metric space to 2D to visualize how
    similar different raters are

    Parameters
    ----------
    raters: dictionary( 
        string (Ranker's name): list (This person's permutation as a list of numbers
                                      corresponding to the indices in animals)
    random_state: int
        A seed to determine which random isometry to use for MDS
    """
    N = len(raters)
    D = np.zeros((N, N))
    rlist = [r for r in raters]
    for i, rater1 in enumerate(rlist):
        for j in range(i+1, N):
            rater2 = rlist[j]
            D[i, j] = kendall_tau(raters[rater1], raters[rater2])
    D = D+D.T
    X = mds(D)
    plt.scatter(X[:, 0], X[:, 1])
    for i, r in enumerate(rlist):
        plt.text(X[i, 0], X[i, 1], r)
    plt.title("MDS Projected Kendall-Tau Distances")



    
    


def diameter(raters):
    """
    Prints the two permutations that achieve
    the largest kendall tau distance

    Parameters
    ----------
    raters : dictionary(
            string (rater's name): list (rater's permutation)
        )

    Returns
    -------
    None.

    """
    N = len(raters)
    distances = np.zeros((N,N))
    rlist = [r for r in raters]
    for i, rater1 in enumerate(rlist):
        for j in range(i+1, N):
            rater2 = rlist[j]
            
            distances[i, j] = kendall_tau(raters[rater1], raters[rater2])
            
    largestdistance = 0
    largesti = -1
    largestj = -1
    for i in range(0,N):
        for j in range(0,N):
            if j > i:
                if distances[i,j] > largestdistance:
                    largestdistance = distances[i,j]
                    
                    largesti = i
                    largestj = j
    print("\n{} and {} achieve the diameter with a distance of {}.".format(rlist[largesti], rlist[largestj], largestdistance))
    print("{}'s ranking: {}".format(rlist[largesti], raters[rlist[largesti]]))
    print("{}'s ranking: {}".format(rlist[largestj], raters[rlist[largestj]]))


def get_average_rankings(animals, raters):
    """
    Print the average ranking among all given permutations

    Parameters
    ----------
    animals : string list
        list of animal names
    raters : dictionary(
            string (rater's name): list (rater's permutation)
        )

    Returns
    -------
    None.

    """
    print(animals)
    averages = [0]*8
    
    rlist = [r for r in raters]
    for i in range(len(rlist)):
        for j in range(len(animals)):
            averages[raters[rlist[i]][j]] += j + 1
    for i in range(len(averages)):
        averages[i] /= len(rlist)
    indices = np.argsort(averages)
    
    print("\nAverage Ranking:")
    for i in range(len(indices)):
        print(animals[indices[i]])
        
 
def merge(A, B, L, inversions = 0):
    i = 0 # index into A
    j = 0 # index into B
    while i < len(A) and j < len(B):
        if A[i] < B[j]:
            L[i+j] = A[i]
            i = i + 1
        else:
            #count inversions when element from right half is sliding ahead of left half
            inversions += len(A) - i
            L[i+j] = B[j]
            j = j + 1
    # Add any remaining elements once one list is empty
    L[i+j:] = A[i:] + B[j:]
    return inversions


def mergesort(L):
    # Base Case!
    if len(L) > 1:
        # Divide!
        mid = len(L) // 2
        A = L[:mid]
        B = L[mid:]
        # Conquer!
        inversions = 0
        inversions += mergesort(A)
        inversions += mergesort(B)
        
        # Combine!
        return merge(A, B, L) + inversions
    else:
        return 0
    


def kendall_tau(p1, p2):
    """
    Optimized kendall tau using mergesort 
    to count inversions between permutations

    Parameters
    ----------
    p1 : list
        first permutation to compare
    p2 : list
        second permutation to compare

    Returns
    -------
    number of inversions

    """
    
    dictionary = {}
    
    keys = list(np.arange(0,len(p1)))
    for i in range(len(keys)):
        dictionary[keys[i]] = p1.index(keys[i])
    
    
    """
    for each value in p2, that is the key whose value you want to replace it with
    so in loop: counter will be the key, and that key's value becomes the new value in p2 at that index
    """
    p3 = [0]*len(p2)
    for i in range(len(dictionary)):
        p3[i] = dictionary.get(p2[i])
    return mergesort(p3)

def bruteforce_kendall_tau(rank1,rank2):
    """
    a brute force method to find the kendall tau distance

    Parameters
    ----------
    rank1 : list
        first permutation
    rank2 : list
        second permutation

    Returns
    -------
    discordants : int
        number of discordants between the
        two permutations

    """
    table1 = np.zeros((len(rank1), len(rank2)))
    table2 = np.zeros((len(rank1), len(rank2)))
    
    for i in range(0,len(rank1)):
        for j in range(0,len(rank1)):
            if j > i:
                if rank1[i] < rank1[j]:
                    table1[rank1[i]][rank1[j]] = 1
                    table1[rank1[j]][rank1[i]] = 1
                else:
                    table1[rank1[i]][rank1[j]] = -1
                    table1[rank1[j]][rank1[i]] = -1
                
                if rank2[i] < rank2[j]:
                    table2[rank2[i]][rank2[j]] = 1
                    table2[rank2[j]][rank2[i]] = 1
                else:
                    table2[rank2[i]][rank2[j]] = -1
                    table2[rank2[j]][rank2[i]] = -1
    
    
    discordants = 0
    for i in range(0,len(rank1)):
        for j in range(0,len(rank1)):
            if j > i:
                if table1[i][j] != table2[i][j]:
                    discordants += 1
                    
    return discordants


def swap(arr, i, j):
    temp = arr[j]
    arr[j] = arr[i]
    arr[i] = temp
    
def kemeny_optimal_ranking(arr, raters, info, idx = 0):
    """
    finds the permutation that minimizes the sum of 
    kendall tau distances with the given rankings

    Parameters
    ----------
    arr : list
        current permutation to compare
    raters : dictionary(
            string (rater): list (ranking)
        )
        
    info : list
        info[0] -> sum of kendall tau distances
        info[1] -> permutation that achieved the sum
    idx : 
        the current element of arr that
        we're at while building the optimal
        permutation. The default is 0.

    Returns
    -------
    None.

    """
    Sum = 0
    
    rlist= [r for r in raters]
    if idx == len(arr)-1:
        
        for rater in rlist:
            
            Sum += kendall_tau(raters[rater], arr)
        if Sum < info[0]:
            
            info[0] = Sum
            info[1] = arr.copy()
            #print(info)
    else:
        for i in range(idx, len(animals)):
            swap(arr, i, idx) # Swap in arr[i] for arr[idx]
            kemeny_optimal_ranking(arr, raters, info, idx+1) # Keep going recursively to figure out arr[idx+1]
            swap(arr, i, idx) # Swap back, and try something else



animals, raters = load_permutations()
diameter(raters)
get_average_rankings(animals, raters)
smallest = np.inf
info = [smallest,0]
arr = [0,1,2,3,4,5,6,7]
kemeny_optimal_ranking(arr, raters, info)
print("\nThe optimal distance of {} was achieved by {}".format(info[0], info[1]))
