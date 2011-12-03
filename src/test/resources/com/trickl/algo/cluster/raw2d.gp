set  autoscale                        # scale axes automatically
unset log                              # remove any log-scaling
unset label                            # remove any previous labels
set xtic auto                          # set xtics automatically
set ytic auto                          # set ytics automatically
set title "Raw 2D Data Plot"
plot "kernel-pnn-cluster-0.dat" 
pause -1 "\nPush 'q' and 'return' to exit Gnuplot ...\n"
