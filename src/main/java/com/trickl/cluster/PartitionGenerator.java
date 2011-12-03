package com.trickl.cluster;

import cern.colt.matrix.DoubleMatrix2D;
import cern.jet.random.engine.RandomEngine;

public interface PartitionGenerator {

   void generate(DoubleMatrix2D partition);

   void setRandomEngine(RandomEngine randomEngine);
}
