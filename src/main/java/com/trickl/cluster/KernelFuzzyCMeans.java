package com.trickl.cluster;

import cern.colt.function.IntIntDoubleFunction;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.impl.SparseDoubleMatrix2D;
import cern.jet.random.engine.MersenneTwister;
import cern.jet.random.engine.RandomEngine;

/**
 * See Graph Nodes Clustering based on the Commute-Time Kernel
 * @author tgee
 */
public class KernelFuzzyCMeans implements ClusterAlgorithm {

   private DoubleMatrix2D prototypeVectors;
   private DoubleMatrix2D partition;
   private double fuzzification = 2.0;
   private double epsilon = 1e-7;
   private int maxIterations = 1000;
   private RandomEngine randomEngine = new MersenneTwister();
   private PartitionGenerator partitionGenerator = new HardRandomPartitionGenerator();

   public KernelFuzzyCMeans() {
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

      final DoubleMatrix1D clusterMembershipSums = new DenseDoubleMatrix1D(clusters);

      // Begin the main loop of alternating optimization
      double stepSize = getEpsilon();
      for (int itr = 0; itr < maxIterations && stepSize >= getEpsilon(); ++itr) {
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
               return value;
            }
         });

         // Calculate distance measure d:
         DoubleMatrix2D distancesSquared = new DenseDoubleMatrix2D(n, clusters);
         for (int k = 0; k < clusters; k++) {
            for (int i = 0; i < n; i++) {
               // Euclidean distance calculation
               DoubleMatrix1D gamma = prototypeVectors.viewColumn(k);
               DoubleMatrix1D Kgamma = new DenseDoubleMatrix1D(kernel.rows());
               kernel.zMult(gamma, Kgamma);
               double distanceSquared = kernel.getQuick(i, i)
                       - 2. * kernel.viewColumn(i).zDotProduct(gamma)
                       + gamma.zDotProduct(Kgamma);

               distancesSquared.setQuick(i, k, distanceSquared);
            }
         }

         // Get new partition matrix U:
         stepSize = 0;
         for (int k = 0; k < clusters; k++) {
            for (int i = 0; i < n; i++) {
               double u = 0;

               if (distancesSquared.getQuick(i, k) == 0) {
                  // Handle this awkward case
                  u = 1;
               } else {
                  double sum = 0;
                  for (int j = 0; j < clusters; j++) {
                     // Exact analytic solution given by Lagrange multipliers
                     sum += Math.pow(distancesSquared.getQuick(i, k) / distancesSquared.getQuick(i, j),
                             1.0 / (getFuzzification() - 1.0));
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
}
