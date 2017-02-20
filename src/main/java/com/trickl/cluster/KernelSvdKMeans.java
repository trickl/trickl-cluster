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

import cern.colt.function.IntIntDoubleFunction;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.impl.SparseDoubleMatrix2D;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;

/**
 * See Graph Nodes Clustering based on the Commute-Time Kernel
 * @author tgee
 */
public class KernelSvdKMeans {

   private DoubleMatrix2D prototypeVectors;
   private DoubleMatrix2D partition;
   private int maxIterations = 1000;
   private RandomGenerator randomGenerator = new MersenneTwister();
   private PartitionGenerator partitionGenerator = new HardRandomPartitionGenerator();

   public KernelSvdKMeans() {
   }

   /**
    * @param kernel K = X * Xt
    * @param clusters
    */
   public void cluster(final DoubleMatrix2D U, final double[] singularValues, final DoubleMatrix2D V, int clusters) {
      int n = U.rows(); // Number of features
      final int p = singularValues.length;

      partition = new SparseDoubleMatrix2D(n, clusters);
      partitionGenerator.setRandomGenerator(randomGenerator);
      partitionGenerator.generate(partition);

      prototypeVectors = new SparseDoubleMatrix2D(n, clusters);

      boolean changedPartition = true;

      // Work matrices
      final DoubleMatrix1D clusterMembershipSums = new DenseDoubleMatrix1D(clusters);
      final DoubleMatrix1D clusterSpans = new DenseDoubleMatrix1D(clusters);
      final DoubleMatrix2D Vtg = new DenseDoubleMatrix2D(p, n);

      // Begin the main loop of alternating optimization
      for (int itr = 0; itr < maxIterations && changedPartition; ++itr) {
         // Get new prototypes (v) for each cluster using weighted median
         clusterMembershipSums.assign(0);

         partition.forEachNonZero(new IntIntDoubleFunction() {

            @Override
            public double apply(int i, int k, double value) {
               double sum = clusterMembershipSums.getQuick(k);
               clusterMembershipSums.setQuick(k, sum + value);
               return value;
            }
         });

         // Reset the old prototype vectors
         prototypeVectors.forEachNonZero(new IntIntDoubleFunction() {

            @Override
            public double apply(int i, int i1, double d) {
               return 0;
            }
         });

         partition.forEachNonZero(new IntIntDoubleFunction() {

            @Override
            public double apply(int i, int k, double value) {
               prototypeVectors.setQuick(i, k, value / clusterMembershipSums.getQuick(k));
               return 0;
            }
         });

         // Calculate Vt * gamma, O(n * s)
         Vtg.assign(0);
         prototypeVectors.forEachNonZero(new IntIntDoubleFunction() {
            @Override
            public double apply(int i, int k, double value) {
               for (int s = 0; s < p; ++s) {
                  double productSum = Vtg.getQuick(s, k);
                  productSum += V.getQuick(i, s) * value;
                  Vtg.setQuick(s, k, productSum);
               }
               return value;
            }
         });

         // Calculate cluster spans O(n*s)
         clusterSpans.assign(0);
         prototypeVectors.forEachNonZero(new IntIntDoubleFunction() {

            @Override
            public double apply(int i, int k, double value) {
               double clusterSpan = clusterSpans.getQuick(k);
               double contribution = 0;
               for (int s = 0; s < p; ++s) {
                  double u = U.getQuick(i, s);
                  double vtg = Vtg.getQuick(s, k);
                  contribution += value * u * singularValues[s] * vtg;
               }

               clusterSpan += contribution;
               clusterSpans.setQuick(k, clusterSpan);
               return value;
            }
         });

         // Calculate cluster distance O(n*k*s)
         changedPartition = false;
         for (int i = 0; i < n; i++) {
            double minDistance = Double.MAX_VALUE;
            int closestCluster = 0;

            // TODO: Only consider nearby clusters
            for (int k = 0; k < clusters; k++) {
               // U = 1 for the closest prototype
               // U = 0 otherwise
               double clusterDistance = 0;
               for (int s = 0; s < p; ++s) {
                  clusterDistance += 2 * U.getQuick(i, s) * singularValues[s]
                                       * Vtg.getQuick(s, k);
               }

               double distance = clusterSpans.getQuick(k) - clusterDistance;
               if (distance < minDistance) {
                  minDistance = distance;
                  closestCluster = k;
               }
            }

            partition.setQuick(i, closestCluster, 1);

            if (!changedPartition && prototypeVectors.getQuick(i, closestCluster) == 0) {
               changedPartition = true;
            }
         }
      }
   }

   public DoubleMatrix2D getPartition() {
      return partition;
   }

   public int getMaxIterations() {
      return maxIterations;
   }

   public void setMaxIterations(int maxIterations) {
      this.maxIterations = maxIterations;
   }

   public RandomGenerator getRandomGenerator() {
      return randomGenerator;
   }

   public void setRandomGenerator(RandomGenerator random) {
      this.randomGenerator = random;
   }
}
