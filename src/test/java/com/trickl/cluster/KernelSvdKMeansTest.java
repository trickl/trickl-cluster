package com.trickl.cluster;

import com.trickl.cluster.KernelSvdKMeans;
import com.trickl.cluster.stats.Partition;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import cern.jet.random.engine.MersenneTwister;
import cern.jet.random.engine.RandomEngine;
import com.trickl.dataset.GaussianCircles2D;
import com.trickl.matrix.ColtSvdAlgorithm;
import com.trickl.matrix.SingularValueDecompositionAlgorithm;
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;
import org.junit.Test;

public class KernelSvdKMeansTest {

   public KernelSvdKMeansTest() {
   }

   @Test
   public void ClusterGaussianCircles() throws IOException
   {
      GaussianCircles2D gaussianCircles = new GaussianCircles2D();

      // Set random engine seed for test repeatability
      RandomEngine randomEngine = new MersenneTwister(123456789);
      gaussianCircles.setRandomEngine(randomEngine);
      gaussianCircles.setRadiusStd(0.20);

      DoubleMatrix2D data = gaussianCircles.generate(100);
      DoubleMatrix2D kernel = new DenseDoubleMatrix2D(data.rows(), data.rows());
      data.zMult(data, kernel, 1, 0, false, true);

      SingularValueDecompositionAlgorithm svdAlgorithm = new ColtSvdAlgorithm();
      svdAlgorithm.calculate(kernel);
      double[] singularValues = svdAlgorithm.getSingularValues();
      DoubleMatrix2D U = svdAlgorithm.getU();
      DoubleMatrix2D V = svdAlgorithm.getV();

      KernelSvdKMeans ksvdkm = new KernelSvdKMeans();
      ksvdkm.setRandomEngine(randomEngine);
  
      // Truncate the SVD
      int clusters = 3;
      int maxSingularValues = 2;
      ksvdkm.cluster(U.viewPart(0, 0, kernel.rows(), maxSingularValues),
              Arrays.copyOf(singularValues, maxSingularValues),
              V.viewPart(0, 0, kernel.rows(), maxSingularValues),
              clusters);

      DoubleMatrix2D partition = ksvdkm.getPartition();
      System.out.println("Partition index: " + Partition.partitionCoefficient(partition));
      System.out.println("Partition entropy: " + Partition.partitionEntropy(partition));

      // Output the membership data to separate files
      for (int k = 0; k < partition.columns(); ++k)
      {
         String fileName = "kernel-svd-kmeans-cluster-" + k + ".dat";
         String packagePath = this.getClass().getPackage().getName().replaceAll("\\.", "/");
         File outputFile = new File("src/test/resources/"
              + packagePath
              + "/" + fileName);
         PrintWriter writer = new PrintWriter(outputFile);

         for (int i = 0; i < data.rows(); ++i)
         {
            StringBuffer dataLine = new StringBuffer();
            dataLine.append(data.getQuick(i, 0));

            for (int j = 1; j < data.columns(); ++j)
            {
               dataLine.append(' ');
               dataLine.append(data.getQuick(i, j));
            }

            dataLine.append(' ');
            dataLine.append(partition.getQuick(i, k));
            writer.println(dataLine.toString());
         }

         writer.close();
      }
   }
}
