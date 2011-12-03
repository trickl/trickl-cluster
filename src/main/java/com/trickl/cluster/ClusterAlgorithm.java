package com.trickl.cluster;

import cern.colt.matrix.DoubleMatrix2D;

public interface ClusterAlgorithm {

   void cluster(DoubleMatrix2D data, int clusters);

   DoubleMatrix2D getPartition();
}
