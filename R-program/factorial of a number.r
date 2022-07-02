findfactorial <- function(n){

factorial <- 1

if ((n==0)|(n==1))
    factorial <- 1

else{
    for( i in 1:n)
        factorial <- factorial * i
      }
return (factorial)
}