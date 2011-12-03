package com.trickl.cluster;

import cern.colt.matrix.DoubleMatrix2D;
import cern.jet.random.Uniform;
import cern.jet.random.engine.MersenneTwister;
import cern.jet.random.engine.RandomEngine;

public class HardRandomPartitionGenerator implements PartitionGenerator {

   private RandomEngine randomEngine;

   public HardRandomPartitionGenerator() {
      randomEngine = new MersenneTwister();
   }

   @Override
   public void generate(DoubleMatrix2D partition) {
      // Initialise U randomly
      partition.assign(0);

      Uniform uniform = new Uniform(randomEngine);

      for (int i = 0; i < partition.rows(); ++i)
      {
         // Randomise
         int k = uniform.nextIntFromTo(0, partition.columns() - 1);
         partition.setQuick(i, k, 1);
      }
   }

   public RandomEngine getRandomEngine() {
      return randomEngine;
   }

   @Override
   public void setRandomEngine(RandomEngine random) {
      this.randomEngine = random;
   }
}
