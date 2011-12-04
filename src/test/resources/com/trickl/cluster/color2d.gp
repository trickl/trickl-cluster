set  autoscale                        # scale axes automatically
unset log                              # remove any log-scaling
unset label                            # remove any previous labels
set xtic auto                          # set xtics automatically
set ytic auto                          # set ytics automatically
set title "Color 2D Data Plot"
set pm3d map
set pal rgb 7,5,15
set key off
splot "kernel-pnn-cluster-0.dat" using 1:2:3 with points lt pal 
pause -1 "\nPush 'q' and 'return' to exit Gnuplot ...\n"
