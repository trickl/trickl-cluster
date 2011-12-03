package com.trickl.cluster;

import cern.colt.function.IntIntDoubleFunction;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.SparseDoubleMatrix2D;
import cern.jet.random.engine.MersenneTwister;
import cern.jet.random.engine.RandomEngine;

/**
 * See Graph Nodes Clustering based on the Commute-Time Kernel
 * @author tgee
 */
public class KernelKMeans implements ClusterAlgorithm {

   private DoubleMatrix2D prototypeVectors;
   private DoubleMatrix2D partition;
   private int maxIterations = 1000;
   private RandomEngine randomEngine = new MersenneTwister();
   private PartitionGenerator partitionGenerator = new HardRandomPartitionGenerator();

   public KernelKMeans() {
   }

   /**
    * @param kernel K = X * Xt
    * @param clusters
    */
   @Override
   public void cluster(DoubleMatrix2D kernel, int clusters) {
      int n = kernel.rows(); // Number of features      

      partition = new SparseDoubleMatrix2D(n, clusters);
      partitionGenerator.setRandomEngine(randomEngine);
      partitionGenerator.generate(partition);

      prototypeVectors = new SparseDoubleMatrix2D(n, clusters);

      boolean changedPartition = true;

      final DoubleMatrix1D clusterMembershipSums = new DenseDoubleMatrix1D(clusters);

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

         // Calculate cluster spans O(n*s)
         final DoubleMatrix1D clusterSpans = new DenseDoubleMatrix1D(clusters);
         for (int k = 0; k < clusters; k++) {
            DoubleMatrix1D gamma = prototypeVectors.viewColumn(k);
            DoubleMatrix1D Kgamma = new DenseDoubleMatrix1D(kernel.rows());
            kernel.zMult(gamma, Kgamma);
            double clusterSpan = gamma.zDotProduct(Kgamma);
            clusterSpans.setQuick(k, clusterSpan);
         }

         // Get new partition matrix U:
         changedPartition = false;
         for (int i = 0; i < n; i++) {
            double minDistance = Double.MAX_VALUE;
            int closestCluster = 0;

            for (int k = 0; k < clusters; k++) {
               // U = 1 for the closest prototype
               // U = 0 otherwise
               DoubleMatrix1D gamma = prototypeVectors.viewColumn(k);
               double clusterDistance = 2. * kernel.viewColumn(i).zDotProduct(gamma);
               double distance = Math.sqrt(kernel.getQuick(i, i) + clusterSpans.getQuick(k)
                       - clusterDistance);

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
}
