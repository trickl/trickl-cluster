package com.trickl.cluster;

import com.trickl.cluster.KMeans;
import cern.colt.matrix.DoubleMatrix2D;
import cern.jet.random.engine.MersenneTwister;
import cern.jet.random.engine.RandomEngine;
import com.trickl.dataset.GaussianCircles2D;
import com.trickl.cluster.stats.Partition;
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import org.junit.Test;

public class KMeansTest {

   public KMeansTest() {
   }

   @Test
   public void ClusterGaussianCircles() throws IOException
   {
      GaussianCircles2D gaussianCircles = new GaussianCircles2D();
      // Set seed for repeatable results
      RandomEngine randomEngine = new MersenneTwister(123456789);
      gaussianCircles.setRandomEngine(randomEngine);
      gaussianCircles.setRadiusStd(0.20);
      DoubleMatrix2D data = gaussianCircles.generate(100);

      KMeans km = new KMeans();      
      km.setRandomEngine(randomEngine);
      km.cluster(data, 3);

      DoubleMatrix2D partition = km.getPartition();
      System.out.println("Partition index: " + Partition.partitionCoefficient(partition));
      System.out.println("Partition entropy: " + Partition.partitionEntropy(partition));
      System.out.println("Xie-Beni index: " + Partition.xieBeniIndex(partition, km.getMeans().viewDice(), data, 1));

      // Output the membership data to separate files
      for (int k = 0; k < partition.columns(); ++k)
      {
         String fileName = "kmeans-cluster-" + k + ".dat";
         String packagePath = this.getClass().getPackage().getName().replaceAll("\\.", "/");
         File outputFile = new File("src/test/resources/"
              + packagePath
              + "/" + fileName);
         PrintWriter writer = new PrintWriter(outputFile);

         for (int i = 0; i < data.rows(); ++i)
         {
            StringBuilder dataLine = new StringBuilder();
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
