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

import cern.colt.list.IntArrayList;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.doublealgo.Statistic;
import cern.colt.matrix.doublealgo.Statistic.VectorVectorFunction;
import cern.colt.matrix.impl.SparseDoubleMatrix2D;
import org.apache.commons.math3.distribution.UniformIntegerDistribution;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;

/**
 * Also known as: PAM (Partitioning around medoids) 
 * It is more robust to noise and outliers
 * See: http://en.wikipedia.org/wiki/K-medoids
 * Also See: Computational Complexity between K-Means and K-Medoids Clustering Algorithms
 *             for Normal and Uniform Distributions of Data Points
 * T. Velmurugan and T. Santhanam
 * Department of Computer Science, DG Vaishnav College, Chennai, India
 * @author tgee
 */
public class KMedoids implements ClusterAlgorithm {
   
   private DoubleMatrix2D partition;
   private int maxIterations = 1000;
   private RandomGenerator randomGenerator = new MersenneTwister();
   private IntArrayList medoids;
   private VectorVectorFunction distanceMeasure = Statistic.EUCLID;

   public KMedoids() {
   }

   @Override
   public void cluster(DoubleMatrix2D data, int clusters) {
      int n = data.rows(); // Number of features
      int p = data.columns(); // Dimensions of features

      partition = new SparseDoubleMatrix2D(n, clusters);
      medoids = new IntArrayList(clusters);

      IntArrayList randomOrdering = new IntArrayList(n);
      for (int i = 0; i < n; ++i) {
         randomOrdering.setQuick(i, i);
      }

      // Choose the medoids by shuffling the data
      for (int i = 0; i < clusters; ++i) {
         // k is the index of the remaining possibilities
         UniformIntegerDistribution uniform = new UniformIntegerDistribution(randomGenerator, i, clusters - 1);
         int k = uniform.sample();

         // Swap x(i) and x(k)
         int medoid = randomOrdering.getQuick(k);
         randomOrdering.setQuick(k, i);                     
         medoids.setQuick(i, medoid);
      }

      boolean changedMedoid = true;

      // Begin the main loop of alternating optimization
      for (int itr = 0; itr < maxIterations && changedMedoid; ++itr) {
         // Get new partition matrix U by
         // assigning each object to the nearest medoid
         for (int i = 0; i < n; i++) {            
            double minDistance = Double.MAX_VALUE;
            int closestCluster = 0;

            for (int k = 0; k < clusters; k++) {
               // U = 1 for the closest medoid
               // U = 0 otherwise
               int medoid = medoids.getQuick(k);
               double distance = distanceMeasure.apply(data.viewRow(medoid), data.viewRow(i));
               if (distance < minDistance) {
                  minDistance = distance;
                  closestCluster = k;
               }
            }

            if (partition.getQuick(i, closestCluster) == 0) {

               for (int k = 0; k < clusters; k++) {
                  partition.setQuick(i, k, (k == closestCluster) ? 1 : 0);
               }
            }
         }

         // Try to find a better set of medoids
         changedMedoid = false;
         for (int k = 0; k < clusters; k++) {

            // For each non-medoid in the cluster
            int medoid = medoids.getQuick(k);
            double lowestCostDelta = 0;
            for (int i = 0; i < n; ++i) {
               int bestMedoid = medoid;
               if (i != medoid && partition.getQuick(i, k) > 0) {
                  // Calculate the change in cost by swapping this configuration
                  int costDelta = 0;
                  for (int j = 0; j < n; ++j) {
                     if (partition.getQuick(j, k) > 0) {                        
                        double oldDistance = distanceMeasure.apply(data.viewRow(medoid), data.viewRow(j));
                        double newDistance = distanceMeasure.apply(data.viewRow(i), data.viewRow(j));
                        costDelta += newDistance - oldDistance;
                     }
                  }

                  if (costDelta < lowestCostDelta) {
                     bestMedoid = i;
                     lowestCostDelta = costDelta;
                  }

                  if (bestMedoid != medoid) {
                     medoids.setQuick(k, bestMedoid);
                     changedMedoid = true;
                  }
               }
            }
         }
      }
   }
   
   public IntArrayList getMedoids() {
      return medoids;
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

   public RandomGenerator getRandomGenerator() {
      return randomGenerator;
   }

   public void setRandomGenerator(RandomGenerator random) {
      this.randomGenerator = random;
   }

   public VectorVectorFunction getDistanceMeasure() {
      return distanceMeasure;
   }

   public void setDistanceMeasure(VectorVectorFunction distanceMeasure) {
      this.distanceMeasure = distanceMeasure;
   }
}
