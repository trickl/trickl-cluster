/*
 * This file is part of the Trickl Open Source Libraries.
 *
 * Trickl Open Source Libraries - http://open.trickl.com/
 *
 * Copyright (C) 2011 Tim Gee.
 *
 * Trickl Open Source Libraries are free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Trickl Open Source Libraries are distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this project.  If not, see <http://www.gnu.org/licenses/>.
 */
package com.trickl.cluster;

import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.doublealgo.Statistic;
import cern.colt.matrix.doublealgo.Statistic.VectorVectorFunction;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.impl.SparseDoubleMatrix2D;
import cern.jet.random.engine.MersenneTwister;
import cern.jet.random.engine.RandomEngine;

public class FuzzyCMeans implements ClusterAlgorithm {

   private DoubleMatrix2D means;
   private DoubleMatrix2D partition;
   private double fuzzification = 2.0;
   private double epsilon = 1e-7;
   private int maxIterations = 1000;
   private RandomEngine randomEngine = new MersenneTwister();
   private PartitionGenerator partitionGenerator = new FuzzyRandomPartitionGenerator();
   private VectorVectorFunction distanceMeasure = Statistic.EUCLID;

   public FuzzyCMeans() {
   }

   @Override
   public void cluster(DoubleMatrix2D data, int clusters) {
      int n = data.rows(); // Number of features
      int p = data.columns(); // Dimensions of features

      partition = new SparseDoubleMatrix2D(n, clusters);
      partitionGenerator.setRandomEngine(randomEngine);
      partitionGenerator.generate(partition);

      means = new DenseDoubleMatrix2D(p, clusters);

      // Begin the main loop of alternating optimization
      double stepSize = epsilon;
      for (int itr = 0; itr < maxIterations && stepSize >= epsilon; ++itr) {
         // Get new prototypes (v) for each cluster using weighted median
         for (int k = 0; k < clusters; k++) {

            for (int j = 0; j < p; j++) {
               double sumWeight = 0;
               double sumValue = 0;

               for (int i = 0; i < n; i++) {
                  double Um = Math.pow(partition.getQuick(i, k), fuzzification);
                  sumWeight += Um;
                  sumValue += data.getQuick(i, j) * Um;
               }

               means.setQuick(j, k, sumValue / sumWeight);
            }
         }

         // Calculate distance measure d:
         DoubleMatrix2D distances = new DenseDoubleMatrix2D(n, clusters);
         for (int k = 0; k < clusters; k++) {
            for (int i = 0; i < n; i++) {
               // Euclidean distance calculation
               double distance = distanceMeasure.apply(means.viewColumn(k), data.viewRow(i));
               distances.setQuick(i, k, distance);
            }
         }

         // Get new partition matrix U:
         stepSize = 0;
         for (int k = 0; k < clusters; k++) {
            for (int i = 0; i < n; i++) {
               double u = 0;

               if (distances.getQuick(i, k) == 0) {
                  // Handle this awkward case
                  u = 1;
               } else {
                  double sum = 0;
                  for (int j = 0; j < clusters; j++) {
                     // Exact analytic solution given by Lagrange multipliers
                     sum += Math.pow(distances.getQuick(i, k) / distances.getQuick(i, j),
                                     1.0 / (fuzzification - 1.0));
                  }
                  u = 1 / sum;
               }

               double u0 = partition.getQuick(i, k);
               partition.setQuick(i, k, u);

               // Stepsize is max(delta(U))
               if (u - u0 > stepSize) {
                  stepSize = u - u0;
               }
            }
         }
      }
   }

   public DoubleMatrix2D getMeans() {
      return means;
   }

   @Override
   public DoubleMatrix2D getPartition() {
      return partition;
   }

   public double getFuzzification() {
      return fuzzification;
   }

   public void setFuzzification(double fuzzification) {
      this.fuzzification = fuzzification;
   }

   public double getEpsilon() {
      return epsilon;
   }

   public void setEpsilon(double epsilon) {
      this.epsilon = epsilon;
   }

   public int getMaxIterations() {
      return maxIterations;
   }

   public void setMaxIterations(int maxIterations) {
      this.maxIterations = maxIterations;
   }

   public RandomEngine getRandomEngine() {
      return randomEngine;
   }

   public void setRandomEngine(RandomEngine random) {
      this.randomEngine = random;
   }
}
