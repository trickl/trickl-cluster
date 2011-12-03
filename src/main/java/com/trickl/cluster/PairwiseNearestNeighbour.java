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
import cern.colt.list.DoubleArrayList;
import cern.colt.list.IntArrayList;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.doublealgo.Statistic;
import cern.colt.matrix.doublealgo.Statistic.VectorVectorFunction;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.impl.SparseDoubleMatrix1D;
import cern.colt.matrix.impl.SparseDoubleMatrix2D;
import cern.jet.math.Functions;
import java.util.HashMap;
import java.util.Map;

/**
 * A hierachical agglomerative clustering (HAC) approach
 * See Pairwise Nearest Neighbor Method Revisited
 * Olli Virmajoki (dissertation)
 * University of Joensuu
 */
public class PairwiseNearestNeighbour implements ClusterAlgorithm {

   private class BestMergePair {
      
      public int newCluster;
      public int oldCluster;
      public double distance = Double.MAX_VALUE;
   }

   private DoubleMatrix2D codeVectors;
   private DoubleMatrix2D partition;
   private VectorVectorFunction distanceMeasure = Statistic.EUCLID;

   @Override
   public void cluster(DoubleMatrix2D data, int clusters) {

      int n = data.rows(); // Number of features
      int p = data.columns(); // Dimensions of features

      partition = new SparseDoubleMatrix2D(n, clusters);
      codeVectors = new DenseDoubleMatrix2D(n, p);
      codeVectors.assign(data);

      // Set the node weights as the size of the clusters
      final DoubleMatrix1D nodeWeights = new DenseDoubleMatrix1D(n);
      nodeWeights.assign(1.);      

      // Store the history of merges so we can calculate the final partition
      // matrix
      Map<Integer, Integer> merges = new HashMap<Integer, Integer>();

      // Calculate all the distances between nodes O(N^2)
      DoubleMatrix2D distances = new SparseDoubleMatrix2D(n, n);
      for (int i = 0; i < n; i++) {
         for (int j = i + 1; j < n; j++) {
            double n_a = nodeWeights.getQuick(i);
            double n_b = nodeWeights.getQuick(j);
            double distance = ((n_a * n_b) / (n_a + n_b))
                    * distanceMeasure.apply(codeVectors.viewRow(i), codeVectors.viewRow(j));
            distances.setQuick(j, i, distance);
            distances.setQuick(i, j, distance);
         }
      }

      // Repetitively merge clusters until only the required number remain
      for (int itr = 0; itr < n - clusters; itr++) {

         final BestMergePair bestMergePair = new BestMergePair();
         distances.forEachNonZero(new IntIntDoubleFunction() {

            @Override
            public double apply(int i, int j, double distance) {
               if (i < j && nodeWeights.getQuick(i) > 0 && nodeWeights.getQuick(j) > 0) {
                  if (distance < bestMergePair.distance) {
                     bestMergePair.distance = distance;
                     bestMergePair.newCluster = i;
                     bestMergePair.oldCluster = j;
                  }
               }
               return distance;
            }
         });

         double n_a = nodeWeights.getQuick(bestMergePair.newCluster);
         double n_b = nodeWeights.getQuick(bestMergePair.oldCluster);

         // Merge the nearest clusters
         DoubleMatrix1D weightedMeanCodeVector = new SparseDoubleMatrix1D(p);
         weightedMeanCodeVector.assign(codeVectors.viewRow(bestMergePair.newCluster), Functions.plusMult(n_a / (n_a + n_b)));
         weightedMeanCodeVector.assign(codeVectors.viewRow(bestMergePair.oldCluster), Functions.plusMult(n_b / (n_a + n_b)));
         codeVectors.viewRow(bestMergePair.newCluster).assign(weightedMeanCodeVector);
         nodeWeights.setQuick(bestMergePair.newCluster, n_a + n_b);
         nodeWeights.setQuick(bestMergePair.oldCluster, 0);
         merges.put(bestMergePair.oldCluster, bestMergePair.newCluster);

         // Update the distance measure between the merged cluster and the
         // neighbours of the pre-merge clusters
         IntArrayList newClusterNeighbourIndices = new IntArrayList(n);
         IntArrayList oldClusterNeighbourIndices = new IntArrayList(n);
         DoubleArrayList neighbourDistances = new DoubleArrayList(n);
         distances.viewRow(bestMergePair.newCluster).getNonZeros(newClusterNeighbourIndices, neighbourDistances);
         distances.viewRow(bestMergePair.oldCluster).getNonZeros(oldClusterNeighbourIndices, neighbourDistances);

         // Set the oldCluster neighbour distances to zero
         for (int t = 0; t < oldClusterNeighbourIndices.size(); ++t) {
            int i = oldClusterNeighbourIndices.elements()[t];
            distances.setQuick(bestMergePair.oldCluster, i, 0);
            distances.setQuick(i, bestMergePair.oldCluster, 0);
         }

         // Update the new distances
         newClusterNeighbourIndices.addAllOf(oldClusterNeighbourIndices);
         for (int t = 0; t < newClusterNeighbourIndices.size(); ++t) {
            int i = newClusterNeighbourIndices.elements()[t];
            int j = bestMergePair.newCluster;
            if (i != bestMergePair.oldCluster && i != j) {
               n_a = nodeWeights.getQuick(i);
               n_b = nodeWeights.getQuick(j);
               double distance = ((n_a * n_b) / (n_a + n_b))
                       * distanceMeasure.apply(codeVectors.viewRow(i), codeVectors.viewRow(j));
               distances.setQuick(j, i, distance);
               distances.setQuick(i, j, distance);
            }
         }
      }

      // Finally use the merge map to figure out the partition
      Map<Integer, Integer> clusterIndicies = new HashMap<Integer, Integer>();
      int clusterIndex = 0;
      for (int i = 0; i < n; ++i) {
         int j = i;
         while (merges.get(j) != null)
         {
            j = merges.get(j);
         }

         if (!clusterIndicies.containsKey(j)) {
            clusterIndicies.put(j, clusterIndex++);
         }

         partition.setQuick(i, clusterIndicies.get(j), 1.);
      }
   }

   @Override
   public DoubleMatrix2D getPartition() {
      return partition;
   }
}
