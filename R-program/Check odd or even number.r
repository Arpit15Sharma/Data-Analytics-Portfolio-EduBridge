i <- readline(prompt="Enter a number:  ")
x <- as.integer(i)
if( (x%%2) == 0) {
print(paste("The number ",i," is Even"))
} else {
print(paste("The number ",i," is Odd"))