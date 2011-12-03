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

public class KMeans implements ClusterAlgorithm {

   private DoubleMatrix2D means;
   private DoubleMatrix2D partition;
   private int maxIterations = 1000;
   private RandomEngine randomEngine = new MersenneTwister();
   private PartitionGenerator partitionGenerator = new HardRandomPartitionGenerator();
   private VectorVectorFunction distanceMeasure = Statistic.EUCLID;

   public KMeans() {
   }

   @Override
   public void cluster(DoubleMatrix2D data, int clusters) {
      int n = data.rows(); // Number of features
      int p = data.columns(); // Dimensions of features

      partition = new SparseDoubleMatrix2D(n, clusters);
      partitionGenerator.setRandomEngine(randomEngine);
      partitionGenerator.generate(partition);

      means = new DenseDoubleMatrix2D(p, clusters);

      boolean changedPartition = true;

      // Begin the main loop of alternating optimization
      for (int itr = 0; itr < maxIterations && changedPartition; ++itr) {
         // Get new prototypes (v) for each cluster using weighted median
         for (int k = 0; k < clusters; k++) {

            for (int j = 0; j < p; j++) {
               double sumWeight = 0;
               double sumValue = 0;

               for (int i = 0; i < n; i++) {
                  double Um = partition.getQuick(i, k);
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
         changedPartition = false;
         for (int i = 0; i < n; i++) {
            double minDistance = Double.MAX_VALUE;
            int closestCluster = 0;

            for (int k = 0; k < clusters; k++) {
               // U = 1 for the closest prototype
               // U = 0 otherwise

               if (distances.getQuick(i, k) < minDistance) {
                  minDistance = distances.getQuick(i, k);
                  closestCluster = k;
               }
            }

            if (partition.getQuick(i, closestCluster) == 0) {
               changedPartition = true;

               for (int k = 0; k < clusters; k++) {
                  partition.setQuick(i, k, (k == closestCluster) ? 1 : 0);
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

   public VectorVectorFunction getDistanceMeasure() {
      return distanceMeasure;
   }

   public void setDistanceMeasure(VectorVectorFunction distanceMeasure) {
      this.distanceMeasure = distanceMeasure;
   }
}
