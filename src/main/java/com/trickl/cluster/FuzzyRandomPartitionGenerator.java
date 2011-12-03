package com.trickl.cluster;

import cern.colt.matrix.DoubleMatrix2D;
import cern.jet.random.engine.MersenneTwister;
import cern.jet.random.engine.RandomEngine;

public class FuzzyRandomPartitionGenerator implements PartitionGenerator {

   private RandomEngine randomEngine;

   public FuzzyRandomPartitionGenerator() {
      randomEngine = new MersenneTwister();
   }

   @Override
   public void generate(DoubleMatrix2D partition) {
      for (int i = 0; i < partition.rows(); ++i) {
         // Randomise
         double sum = 0;
         for (int k = 0; k < partition.columns(); ++k) {
            double u = randomEngine.nextDouble();
            partition.setQuick(i, k, u);
            sum += u;
         }

         // Normalise the weights
         for (int k = 0; k < partition.columns(); ++k) {
            partition.setQuick(i, k, partition.getQuick(i, k) / sum);
         }
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
