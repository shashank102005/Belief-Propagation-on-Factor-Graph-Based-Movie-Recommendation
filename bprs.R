
####### constants 
num_movies = 1682
num_features = 19
num_users = 943



######### read the main user-ratings data set
user_ratings = read.csv(file="F:/Projects/PGM/Movie Recommendation/ml-100k/u_data.csv",header=TRUE,
                        sep=",",na.strings=c(""))

######### read the main user-ratings data set
film_info = read.csv(file="F:/Projects/PGM/Movie Recommendation/ml-100k/u_items.csv",header=TRUE,
                     sep=",",na.strings=c(""))



######## separate train and test user ratings data
samp = sample(nrow(user_ratings),0.2*nrow(user_ratings))
user_ratings_test = user_ratings[samp,]
user_ratings_train = user_ratings[-samp,]


######### create adjacency matrix of the bipartite factor graph for training
user_mat = matrix(0, nrow = 943 , ncol = 1682)
for(i in 1:nrow(user_ratings_train))
{
  r = user_ratings_train$user.id[i]
  c = user_ratings_train$item.id[i]
  user_mat[r,c] = user_ratings_train$rating[i]
}

fact_graph = as.data.frame(user_mat,row.names = seq(1,num_users))


####### select the target user
targ_user = 112


######## prune the bipartite factor graph for the target user, get the user and items vectors
### vector of users in the factor graph
user_vec = numeric(0) 
count = 1
for(i in 1:nrow(fact_graph))
{ 
  targ_arr = fact_graph[targ_user,]
  curr_arr = fact_graph[i,]
  if( any( targ_arr *  curr_arr ) > 0) 
  { user_vec[count] = i
  count = count+1
  }
}

###### create item vector
item_vec = numeric(0)
for(i in 1:length(user_vec))
  item_vec = union(item_vec,which(fact_graph[user_vec[i],] != 0))


######################## Belief Propagation Message Passing Algorithm  ##############################
N_f = which(fact_graph[targ_user,] != 0) ### neighbours of the target user
gamma = c(1,2,3,4,5)  ### set of ratings
epsilon = 0.8         ### initial value for the lambda messages
lambda = matrix(epsilon,num_users,num_movies) ### initialize the lambda messages
Mu = array(0,dim=c(num_movies,num_users,5) ) ### initialize the Mu messages
A_f = mean(user_mat[targ_user,N_f])  ### average rating of the target user
rho = 4

den1 = 0  ### required to compute the Mu messages
for(i in 1:5)
{
  den1 = den1 + 1/abs(A_f - i)
}

###### determine the items to predict the ratings for
lone_items = which(colSums(user_mat) == 0)
to_pred = setdiff( (subset(user_ratings_test,user.id == targ_user)[,2]) , lone_items ) ### items for which ratings are to be predicted

###### initialize the predictions
Pred = matrix(0,15,length(to_pred)) ### initialize the predictions
final_lambda = matrix(0,15,943)


################## Do the following on every iteration ##############################################
for(i in 11:15)  
{  
  
#################### First half: Mu messages (from items to users) ##################################

###### for all the items(Movies) in the factor graph
for(j in 1:length(item_vec))
{
  #### Do the following for each item
  N_a = which(user_mat[,item_vec[j]] != 0) ### determine the neighbours of the item
  N_a = intersect(N_a,user_vec)
  
  #### Do the following for each neighbour of the item
  for(k in 1:length(N_a)) ### a single Mu a->k message
  {
    if( length(N_a) == 1)
    {
      for(l in 1:5)
      { 
        if( (user_mat[N_a[k],item_vec[j]] == l) )
        { Mu[item_vec[j],N_a[k],l] = 1 }
        else
        { Mu[item_vec[j],N_a[k],l] = 0 }
      }
    }
      
      
    else if( (item_vec[j] %in% N_f) == FALSE)
    {   
      N_a_k = setdiff(N_a,N_a[k])
      num = matrix(1,5,length(N_a_k)) 
      log_num = numeric(5)
      
      for(l in 1:5)  ### for each value of the ratings
      { 
        for(n in 1:length(N_a_k))
         { 
           if(user_mat[N_a_k[n],item_vec[j]] == l)
           { 
             num[l,n] = (lambda[N_a_k[n],item_vec[j]] + (1-lambda[N_a_k[n],item_vec[j]] )* ( (1/abs(A_f - l))/ den1))
           }
           else
           { 
             tt = abs(A_f - user_mat[N_a_k[n],item_vec[j]])
             num[l,n] = (1-lambda[N_a_k[n],item_vec[j]] )  *  ( (1/tt)/ den1)
           }
         }
       }
       
      for(l in 1:5)  ### for each value of the ratings
      { 
        temp = num[l,]
        if( length(temp[temp > 0]) == 0 )
        { 
          for(n in 1:length(N_a_k))
          {
            log_num[l] = log_num[l] + log((-1)*num[l,n])   
          } 
        }
        else
        {
           mean = mean(temp[temp > 0])
           for(n in 1:length(N_a_k))
           {
            if(num[l,n] > 0) 
              log_num[l] = log_num[l] + log(num[l,n])
            else
              log_num[l] = log_num[l] + log(mean)
           }
         }
       }
       
      
        ##### normalizing the log likelihoods into probabilities
        max_val = max(log_num)
        for( l in 1:5)
        {
          log_num[l] = log_num[l] - max_val 
        }  
        
        norm = numeric(5)
        for(l in 1:5)
        {
          if(log_num[l] >= -37.93997)
             { norm[l] = exp(log_num[l]) }
          else
             { norm[l] = 0 }
        }
        
        for(l in 1:5)
        {
          Mu[item_vec[j],N_a[k],l] = norm[l]/sum(norm)
        }
       
      }
      
      else 
      {
        for(l in 1:5)
         { 
          if( (user_mat[targ_user,item_vec[j]] == l) )
           { Mu[item_vec[j],N_a[k],l] = 1 }
          else
           { Mu[item_vec[j],N_a[k],l] = 0 }
        }
      }  
    
  } ### loop for each neighbour of the item ends
  
} ### first half of the iteration ends
 
  
########################## Second Half: Lambda Messages (from users to items)  #######################

#######do the following for each user in the factor graph
for(j in 1:length(user_vec))
{
  #### Do the following for each user
  N_k = which(user_mat[user_vec[j],] != 0) ### neighbours of the user
  
  
  #### Do the following for each neighbour of the user
  for(k in 1:length(N_k)) ### a single Lambda k-> a message
  {
    sum = 0
    for(l in setdiff(N_k,N_k[k]))
    {
      for(m in 1:5)
      {
        sum = sum + (m - user_mat[user_vec[j],l]) * Mu[l,user_vec[j],m]
        
      }
    }
     
    mod =  length(setdiff(N_k,N_k[k]))
    lambda[user_vec[j],N_k[k]] = 1 -(1 / (rho* mod)) * sum
  }
  
}


######### computation of the predicted ratings for the Movies not rated by the target user ###########

for(j in 1:length(to_pred))
{
  Prob = numeric(5)
  
  N_a = which(user_mat[,to_pred[j]] != 0) ### determine the neighbours of the item
  N_a = intersect(N_a,user_vec)
  
  if( length(N_a) == 1)
  {
    for(l in 1:5)
    { 
      if( (user_mat[N_a[1],to_pred[j]] == l) )
      { Prob[l] = 1 }
      else
      { Prob[l] = 0 }
    }
  }
  
  else if( (to_pred[j] %in% N_f) == FALSE)
  { 
    
   num = matrix(1,5,length(N_a)) 
   log_num = numeric(5)
  
     for(l in 1:5)  ### for each value of the ratings
     { 
       for(n in 1:length(N_a))
       { 
         if(user_mat[N_a[n],to_pred[j]] == l)
         { 
          num[l,n] = (lambda[N_a[n],to_pred[j]] + (1-lambda[N_a[n],to_pred[j]] )* ( (1/abs(A_f - l))/ den1))
         }
         else
         { 
          tt = abs(A_f - user_mat[N_a[n],to_pred[j]])
          num[l,n] = (1-lambda[N_a[n],to_pred[j]] )  *  ( (1/tt)/ den1)
         }
       }
     }
  
     for(l in 1:5)  ### for each value of the ratings
     { 
       temp = num[l,]
       if( length(temp[temp > 0]) == 0 )
       { 
         for(n in 1:length(N_a))
         {
           log_num[l] = log_num[l] + log((-1)*num[l,n])   
         } 
       }
       else
       {
         mean = mean(temp[temp > 0])
         for(n in 1:length(N_a))
         {
           if(num[l,n] > 0) 
             {log_num[l] = log_num[l] + log(num[l,n])}
           else
             {log_num[l] = log_num[l] + log(mean)}
         }
       }
      }
    
    ##### normalizing the log likelihoods into probabilities
    max_val = max(log_num)
    for( l in 1:5)
    {
      log_num[l] = log_num[l] - max_val 
    }  
    
    norm = numeric(5)
    for(l in 1:5)
    {
      if(log_num[l] >= -37.93997)
      { norm[l] = exp(log_num[l]) }
      else
      { norm[l] = 0 }
    }
    
    for(l in 1:5)
    {
      Prob[l] = norm[l]/sum(norm)
    }
  
  }
  
  else 
  {
    for(l in 1:5)
    { 
      if( (user_mat[targ_user,to_pred[j]] == l) )
      { Prob[l] = 1 }
      else
      { Prob[l] = 0 }
    }
  }  
  
  Pred[i,j] = 0
  for(k in 1:5)
  {
    Pred[i,j] = Pred[i,j] + k * Prob[k]
  }
  
} ### prediction computation loop ends

##### computation of the average belief of each of the user , for the target user   
for( j in 1:length(user_vec))
{
  final_lambda[i,j] = mean(lambda[j, which(user_mat[j,] !=0)])  
}


} #### main loop ends



