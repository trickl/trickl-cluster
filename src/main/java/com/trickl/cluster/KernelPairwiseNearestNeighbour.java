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
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.impl.SparseDoubleMatrix1D;
import cern.colt.matrix.impl.SparseDoubleMatrix2D;
import cern.jet.math.Functions;
import com.trickl.matrix.CompressedSparseColumnMatrix;
import com.trickl.matrix.SparseUtils;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * A hierachical agglomerative clustering (HAC) approach
 * See Pairwise Nearest Neighbor Method Revisited
 * Olli Virmajoki (dissertation)
 * University of Joensuu
 */
public class KernelPairwiseNearestNeighbour implements ClusterAlgorithm {

   private class BestMergePair {

      public int newCluster;
      public int oldCluster;
      public double distance = Double.MAX_VALUE;
   }

   private CompressedSparseColumnMatrix prototypeVectors;
   //private SparseDoubleMatrix2D prototypeVectorsCheck;
   private DoubleMatrix2D partition;
   private final double NON_ZERO_OFFSET = 1;

   @Override
   public void cluster(final DoubleMatrix2D kernel, int clusters) {

      int n = kernel.rows(); // Number of features

      partition = new SparseDoubleMatrix2D(n, clusters);
      prototypeVectors = new CompressedSparseColumnMatrix(n, n);
     // prototypeVectorsCheck = new SparseDoubleMatrix2D(n, n);

      // Initially, every node is a cluster
      for (int i = 0; i < n; i++) {
         prototypeVectors.set(i, i, 1);
         //prototypeVectorsCheck.set(i, i, 1);
      }

      // Set the node weights as the size of the clusters
      final DoubleMatrix1D nodeWeights = new DenseDoubleMatrix1D(n);
      nodeWeights.assign(1.);

      // Store the history of merges so we can calculate the final partition
      // matrix
      Map<Integer, Integer> merges = new HashMap<Integer, Integer>();

      // Calculate the initial cluster variances, taking advantage of the face
      // that each cluster only contains one item
      final DoubleMatrix1D clusterSpans = new DenseDoubleMatrix1D(n);
      for (int i = 0; i < n; i++) {
         clusterSpans.setQuick(i, kernel.getQuick(i, i));
      }

      // Calculate the distances between nodes with a non-zero kernel
      //
      final DoubleMatrix2D distances = new SparseDoubleMatrix2D(n, n);
      kernel.forEachNonZero(new IntIntDoubleFunction() {
         @Override
         public double apply(int i, int j, double value) {            
            double distance = 0.5 * (NON_ZERO_OFFSET + kernel.getQuick(i, i) + kernel.getQuick(j, j)
                                     - 2. * kernel.getQuick(i, j));
            distances.setQuick(i, j, distance);
            distances.setQuick(j, i, distance);
            return value;
         }
      });
      
      // Repetitively merge clusters until only the required number remain
      for (int itr = 0; itr < n - clusters; itr++) {

         final BestMergePair bestMergePair = new BestMergePair();
         final AtomicInteger count = new AtomicInteger(0);
         distances.forEachNonZero(new IntIntDoubleFunction() {

            @Override
            public double apply(int i, int j, double distance) {               
               if (i < j && nodeWeights.getQuick(i) > 0 && nodeWeights.getQuick(j) > 0) {
                  count.incrementAndGet();
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
         DoubleMatrix2D gamma = new SparseDoubleMatrix2D(n, 1);

         //gamma.assign(prototypeVectorsCheck.viewPart(0, bestMergePair.newCluster, n, 1), Functions.plusMult(n_a / (n_a + n_b)));
         //gamma.assign(prototypeVectorsCheck.viewPart(0, bestMergePair.oldCluster, n, 1), Functions.plusMult(n_b / (n_a + n_b)));

         gamma.assign(prototypeVectors.viewColumn(bestMergePair.newCluster), Functions.plusMult(n_a / (n_a + n_b)));
         gamma.assign(prototypeVectors.viewColumn(bestMergePair.oldCluster), Functions.plusMult(n_b / (n_a + n_b)));
         gamma.forEachNonZero(new IntIntDoubleFunction() {

            @Override
            public double apply(int row, int column, double value) {
               prototypeVectors.set(row, bestMergePair.newCluster, value);
               //prototypeVectorsCheck.set(row, bestMergePair.newCluster, value);
               return value;
            }
         });

         nodeWeights.setQuick(bestMergePair.newCluster, n_a + n_b);
         nodeWeights.setQuick(bestMergePair.oldCluster, 0);
         merges.put(bestMergePair.oldCluster, bestMergePair.newCluster);

         // Update the cluster span for the merged cluster         
         DoubleMatrix2D Kgamma = new DenseDoubleMatrix2D(kernel.rows(), 1);
         SparseUtils.zMult(kernel, gamma, Kgamma);

         double clusterSpan = SparseUtils.dot(gamma, Kgamma);
         clusterSpans.setQuick(bestMergePair.newCluster, clusterSpan);

         // Update the distance measure between the merged cluster and the
         // neighbours of the pre-merge clusters
         IntArrayList newClusterNeighbourIndexList = new IntArrayList(n);
         IntArrayList oldClusterNeighbourIndexList = new IntArrayList(n);
         DoubleArrayList neighbourDistances = new DoubleArrayList(n);
         distances.viewRow(bestMergePair.newCluster).getNonZeros(newClusterNeighbourIndexList, neighbourDistances);
         distances.viewRow(bestMergePair.oldCluster).getNonZeros(oldClusterNeighbourIndexList, neighbourDistances);

         // Set the oldCluster neighbour distances to zero
         for (int t = 0; t < oldClusterNeighbourIndexList.size(); ++t) {
            int i = oldClusterNeighbourIndexList.elements()[t];
            distances.setQuick(bestMergePair.oldCluster, i, 0);
            distances.setQuick(i, bestMergePair.oldCluster, 0);
         }

         // Update the new distances
         newClusterNeighbourIndexList.addAllOf(oldClusterNeighbourIndexList);
         for (int t = 0; t < newClusterNeighbourIndexList.size(); ++t) {
            int i = newClusterNeighbourIndexList.elements()[t];
            int j = bestMergePair.newCluster;
            if (i != bestMergePair.oldCluster && i != j) {
               n_a = nodeWeights.getQuick(i);
               n_b = nodeWeights.getQuick(j);

               DoubleMatrix2D neighbourPrototype = prototypeVectors.viewColumn(i);
               //DoubleMatrix2D neighbourPrototype = prototypeVectorsCheck.viewPart(0, i, n, 1);
               
               double clusterDistance = 2. * SparseUtils.dot(Kgamma, neighbourPrototype);
               double distance = ((n_a * n_b) / (n_a + n_b))
                       * (NON_ZERO_OFFSET + clusterSpans.getQuick(i) + clusterSpans.getQuick(j)
                       - clusterDistance);

               distances.setQuick(j, i, distance);
               distances.setQuick(i, j, distance);
            }
         }
      }

      // Finally use the merge map to figure out the partition
      Map<Integer, Integer> clusterIndices = new HashMap<Integer, Integer>();
      int clusterIndex = 0;
      for (int i = 0; i < n; ++i) {
         int j = i;
         while (merges.get(j) != null) {
            j = merges.get(j);
         }

         if (!clusterIndices.containsKey(j)) {
            clusterIndices.put(j, clusterIndex++);
         }

         partition.setQuick(i, clusterIndices.get(j), 1.);
      }
   }

   @Override
   public DoubleMatrix2D getPartition() {
      return partition;
   }
}
