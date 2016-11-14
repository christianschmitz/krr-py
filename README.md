# krr-py
kernel ridge regression utility for simple data files

# example usage:
* Simplest usage: 
> krr.py example.dat 
* 2 indep variables columns, custom sigma and alpha:
> krr.py -n 2 -s 0.001 -a 0.1 example.dat
* predicted value ranges: 
> krr.py -r 1:4:10,0:0.3:3 example.dat
* custom range file (observed variables in this file are ignored)
> krr.py -f example.dat -n 1 example.dat

# dependencies:
* numpy version 1.8.2
* mlpy version 2.2.0

# TODO:
* automatic determination of optimal sigma and alpha
* normalization of independent variables so that sigma and alpha remain optimal for all input dimensions
* nicer output formatting
