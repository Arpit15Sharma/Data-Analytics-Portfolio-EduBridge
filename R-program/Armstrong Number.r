isArmstrong <- function(x){
  sum <- 0
  y <- x
  while(y > 0){
    digit <- y %% 10
    sum <- sum + (digit^3)
    y <- floor(y/10)
   
  }

  if(x == sum){
    print(paste(x,' *** is an Armstrong Number ***'))
  } else {
    print(paste(x,'is not an Armstrong Number'))
  }
}

isArmstrong(153)
isArmstrong(23)