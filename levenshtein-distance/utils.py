import numpy as np

def min_edit_distance(source, target, ins_cost, del_cost, rep_cost):
    '''
    Input: 
        source: a string corresponding to the string you are starting with
        target: a string corresponding to the string you are ending with
        ins_cost: an integer setting the insert cost
        del_cost: an integer setting the delete cost
        rep_cost: an integer setting the replace cost
    Output:
        D: a matrix of len(source)+1 by len(target)+1 containing minimum edit distances
        med: the minimum edit distance (med) required to convert the source string to the target
    '''
    
    m = len(source) 
    n = len(target)
    
    # initializing cost matrix with zeros and dimensions (m+1,n+1) 
    D = np.zeros((m+1, n+1), dtype=int) 
    
    # Filling in column 0, from row 1 to row m, both inclusive
    for row in range(1, m+1):
        D[row,0] = D[row-1,0] + del_cost
        
    # Filling in row 0, for all columns from 1 to n, both inclusive
    for col in range(1, n+1):
        D[0,col] = D[0,col-1] + ins_cost
        
    # Looping through row 1 to row m, both inclusive
    for row in range(1, m+1): 
        
        # Looping through column 1 to column n, both inclusive
        for col in range(1, n+1):
            
            # Intializing r_cost to the 'replace' cost that is passed into this function
            r_cost = rep_cost
            
            # Checking to see if source character at the previous row matches the target character at the previous column
            if source[row-1] == target[col-1]:
                # Updating the replacement cost to 0 if source and target are the same
                r_cost = 0
            
            # Updating the cost at row, col based on previous entries in the cost matrix            
            mins_list = []
            min_del = D[row-1, col] + del_cost
            min_ins = D[row, col-1] + ins_cost
            min_rep = D[row-1, col-1] + r_cost
            mins_list.extend((min_del, min_ins, min_rep))
            
            D[row,col] = min(mins_list)
          
    # Setting the minimum edit distance with the cost found at row m, column n
    med = D[m,n]
    
    return D, med
