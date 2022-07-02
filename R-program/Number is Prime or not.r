isprime <- function(n) {
    lim <- n/2
    prime <- T
    for( i in 2:lim) {
      if(n %% i == 0)
        prime <- FALSE
     
  }

if(n==2) prime <- T
    if(prime) print(paste(n," is a Prime Number"))
    else print(paste(n," is a Composite Number"))
}
