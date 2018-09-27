import numpy as np
import copy

from math import sqrt
#def distance(x1, y1, x2, y2):
#    ret = sqrt((y1-y2)**2) 
#    #why distance computed only by y?
#    #Ans:x is time and difference between the two time indexes should not be penalized    
#    return ret

def distance(x1, y1, x2, y2):
    #print x1,x2
    delta = 10
    ret = abs(y1-y2)
    h_dist = abs(x1-x2)
    alpha = 0.2
    penalty = 1
    c = int(np.ceil(h_dist))
    for i in range(c):
        penalty = penalty * alpha
    penalty = 1.0 / penalty
    if(h_dist>delta):
        ret= abs(y1-y2)*penalty

    return ret

# return: dist    distance
#         cost    2-d array of cost
#         path    list of path
#path 0 - time of dtw_y1(traffic), path 1 - time of dtw_y2(occupancy)
def dtw(y1_arr, y2_arr, distFunc=distance):
    if (y1_arr.size != y2_arr.size):
        print "two sequences should have same length"
        #keeping the same length is for simplicity
        #how to modify the two arrary to the same size?
        #Ans: use dataAlign in mlUtils
        return 
    ts_len = y1_arr.size
    cost = computeCost(y1_arr, y2_arr, distFunc)
    #cost is computed by distance, d(i, j)
    pre_arr_1 = np.ndarray((ts_len, ts_len), np.int32) 
    pre_arr_2 = np.ndarray((ts_len, ts_len), np.int32)
    #pre_array is to storage which element this one points to
    #1 for i(or x), 2 for j(or y)    
    min_cost = np.ndarray((ts_len, ts_len), np.float64)
    min_cost.fill(-1)
    #mincost, g(i, j), is computed by previous mincost and d(i,j)
    
    #init column 0 and row 0
    min_cost[0][0] =0
    for i in range(1,ts_len): 
    #start from 1 to avoid IndexOutOfBounds Exception
        min_cost[0][i] = min_cost[0][i-1] + cost[0][i]
        min_cost[i][0] = min_cost[i-1][0] + cost[i][0]
        pre_arr_1[0][i] = 0
        pre_arr_2[0][i] = i-1
        pre_arr_1[i][0] = i-1
        pre_arr_2[i][0] = 0
        
    #compute column 1,: and row 1,:    
    for i in range(1,ts_len):
        for j in range(1,ts_len):
            #Search sub arrays of [i-1 ~ i][j-1~j]
            cost_ij = cost[i][j]
            min_cost0_1 = min_cost[i][j-1]
            min_cost1_0 = min_cost[i-1][j]
            min_cost1_1 = min_cost[i-1][j-1]
            #select the sub arrays with minimum cost
            if (min_cost0_1<=min_cost1_0) and (min_cost0_1<=min_cost1_1):
                pre_arr_1[i][j]=i
                pre_arr_2[i][j]=j-1
                min_cost[i][j] = min_cost0_1 + cost_ij
                continue
            if (min_cost1_0<=min_cost1_1) and (min_cost1_0<=min_cost0_1):
                pre_arr_1[i][j]=i-1
                pre_arr_2[i][j]=j
                min_cost[i][j] = min_cost1_0 + cost_ij
                continue
            if (min_cost1_1<=min_cost1_0) and (min_cost1_1<=min_cost0_1):
                pre_arr_1[i][j]=i-1
                pre_arr_2[i][j]=j-1
                min_cost[i][j] = min_cost1_1 + cost_ij
                continue
            
    #trace back and store pathlist
    cur_i = ts_len-1
    cur_j = ts_len-1
    path_list = []
    while (cur_i != 0) or (cur_j !=0):
        path_list.append((cur_i, cur_j))
        temp_cur_i = pre_arr_1[cur_i][cur_j]
        temp_cur_j = pre_arr_2[cur_i][cur_j]
        cur_i = temp_cur_i
        cur_j = temp_cur_j
        
    path_list.append((0,0))
    
    #the pathlist above is reversed and need rearranged
    path_len = len(path_list)
    path_0 = np.arange(path_len)
    path_1 = np.arange(path_len)
    for i in range(path_len):
        path_0[i] = path_list[path_len -1 - i][0]
        path_1[i] = path_list[path_len -1 - i][1]
        #path 0 for cur_i, path 1 for cur_j
    
    return min_cost[ts_len-1][ts_len-1], cost, (path_0, path_1)
        
    

def computeCost(y1_arr, y2_arr, distFunc):
    ts_len = y1_arr.size 
    cost = np.ndarray((ts_len, ts_len), np.float64)
    for i in range(ts_len):
        for j in range(ts_len):
            cost[i][j]=distFunc(i, y1_arr[i], j, y2_arr[j])
    return cost
    
#############################
## below added by zimu
## 2015.3
############################    
    
#Translate y1 to fit y0 according to DTW path result, note that y0 remain the same,
#Be careful with that which is y0 and y1,
#since the function "dataAlign" could change them unpredictably  
#DTW should be carried out first to get the array "path0 path1"
#path 0 - time of dtw_y1(traffic), path 1 - time of dtw_y2(occupancy)
def transition(y1,path0,path1):   
    y1t = copy.deepcopy(y1) #shallow copy, which changes y0 gradually, will cause error
    pre=0    
    t=1
    while t < path0.size:
        st=pre
        ed=t
        if pre >= path0.size or t >= path0.size:
            break
        if path0[pre]==path0[t]:
            for ed in range(t, path0.size,1):
                if path0[ed]!=path0[t]:
                    break
            if ed != path0.size - 1: 
                ed = ed - 1
            part = y1[path1[st]:path1[ed]]
            if part.size != 0:
                #print pre, t
                #print path0[pre],path0[t]
                #print st, ed
                #print path1[st],path1[ed]
                mean = np.mean(part)
                y1t[path0[st]] = mean
            #print 'path0, map ', path1[st], ' - ', path1[ed], ' to ', path0[st] 
            t = ed + 1
        elif path1[pre]==path1[t]:
            for ed in range(t,path1.size,1):
                if path1[ed]!=path1[t]:
                    break
            if ed != (path1.size-1):
                ed = ed -1
            for i in range(path0[st],path0[ed]+1,1):            
                y1t[i] = y1[path1[st]]
            #print 'path1, map ', path1[st], ' to ', path0[st], ' - ', path0[ed]
            t = ed + 1
        else:
            y1t[path0[st]]=y1[path1[st]] 
            ed = st
            #print 'path2, map ', path1[st], ' to ', path0[st]
            #print 'change the value to ', y1t[path0[st]]
        pre = t
        t = t + 1
    return y1t
    
    
def distanceT(x1, y1, t1, x2, y2, t2):
    ret = abs(y1-y2)
    h_dist = abs(x1-x2)
    coef = 0.2
    if(np.abs(t1-t2)>1):
        for i in range(h_dist):
            coef = coef * 0.2
        ret= abs(y1-y2)/coef
    return ret

def computeCostT(y1_arr, y1_t,y2_arr, y2_t,distFunc):
    ts_len = y1_arr.size 
    ts_len2 = y2_arr.size 
    cost = np.ndarray((ts_len, ts_len2), np.float64)
    for i in range(ts_len):
        for j in range(ts_len2):
            cost[i][j]=distFunc(i, y1_arr[i], y1_t[i], j, y2_arr[j], y2_t[j])
    return cost
    
# return: dist    distance
#         cost    2-d array of cost
#         path    list of path
#path 0 - time of dtw_y1(traffic), path 1 - time of dtw_y2(occupancy)
def dtwT(y1_arr, y1_t, y2_arr, y2_t, distFunc=distanceT):
    ts_len = y1_arr.size
    ts_len2 = y2_arr.size
    cost = computeCostT(y1_arr, y1_t, y2_arr, y2_t, distFunc)
    #cost is computed by distance, d(i, j)
    pre_arr_1 = np.ndarray((ts_len, ts_len2), np.int32) 
    pre_arr_2 = np.ndarray((ts_len, ts_len2), np.int32)
    #pre_array is to storage which element this one points to
    #np.ndarray((max_i,max_j),np.int32)
    #1 for i(or x), 2 for j(or y)    
    min_cost = np.ndarray((ts_len, ts_len2), np.float64)
    min_cost.fill(-1)
    #mincost, g(i, j), is computed by previous mincost and d(i,j)
    
    #init column 0 and row 0
    min_cost[0][0] =0
    for i in range(1,ts_len): 
    #start from 1 to avoid IndexOutOfBounds Exception
        min_cost[i][0] = min_cost[i-1][0] + cost[i][0]
        pre_arr_1[i][0] = i-1
        pre_arr_2[i][0] = 0
    for j in range(1,ts_len2):
        min_cost[0][j] = min_cost[0][j-1] + cost[0][j]
        pre_arr_1[0][j] = 0
        pre_arr_2[0][j] = j-1
        
    #compute column 1,: and row 1,:    
    for i in range(1,ts_len):
        for j in range(1,ts_len2):
            #Search sub arrays of [i-1 ~ i][j-1~j]
            cost_ij = cost[i][j]
            min_cost0_1 = min_cost[i][j-1]
            min_cost1_0 = min_cost[i-1][j]
            min_cost1_1 = min_cost[i-1][j-1]
            #select the sub arrays with minimum cost
            if (min_cost0_1<=min_cost1_0) and (min_cost0_1<=min_cost1_1):
                pre_arr_1[i][j]=i
                pre_arr_2[i][j]=j-1
                min_cost[i][j] = min_cost0_1 + cost_ij
                continue
            if (min_cost1_0<=min_cost1_1) and (min_cost1_0<=min_cost0_1):
                pre_arr_1[i][j]=i-1
                pre_arr_2[i][j]=j
                min_cost[i][j] = min_cost1_0 + cost_ij
                continue
            if (min_cost1_1<=min_cost1_0) and (min_cost1_1<=min_cost0_1):
                pre_arr_1[i][j]=i-1
                pre_arr_2[i][j]=j-1
                min_cost[i][j] = min_cost1_1 + cost_ij
                continue
            
    #trace back and store pathlist
    cur_i = ts_len-1
    cur_j = ts_len2-1
    path_list = []
    while (cur_i != 0) or (cur_j !=0):
        path_list.append((cur_i, cur_j))
        temp_cur_i = pre_arr_1[cur_i][cur_j]
        temp_cur_j = pre_arr_2[cur_i][cur_j]
        cur_i = temp_cur_i
        cur_j = temp_cur_j
        
    path_list.append((0,0))
    
    #the pathlist above is reversed and need rearranged
    path_len = len(path_list)
    path_0 = np.arange(path_len)
    path_1 = np.arange(path_len)
    for i in range(path_len):
        path_0[i] = path_list[path_len -1 - i][0]
        path_1[i] = path_list[path_len -1 - i][1]
        #path 0 for cur_i / x-axis, path 1 for cur_j/ y-axis
        #step[0] = [path[0][0], path[1][0]]
        #step[1] = [path[0][1], path[1][1]]
        #step[2] = [path[0][2], path[1][2]]
    
    return min_cost[ts_len-1][ts_len2-1], cost, (path_0, path_1)
