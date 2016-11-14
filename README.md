# krr-py
kernel ridge regression utility for simple data files

# example usage:
> krr.py example.dat 
> krr.py -n 2 -s 0.001 -a 0.1 example.dat
> krr.py -r 0:10:11,1 example.dat
> krr.py -f example.dat -n 1 example.dat
> krr.py -r 1:4:10,0:0.3:3 example.dat

# dependencies:
* numpy version 1.8.2
* mlpy version 2.2.0

# TODO:
* automatic determination of optimal sigma and alpha
* normalization of independent variables so that sigma and alpha remain optimal for all input dimensions
* nicer output formatting
