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
package com.trickl.cluster.stats;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import cern.jet.math.Functions;

public class Partition {

   public static double partitionCoefficient(DoubleMatrix2D partition) {
      double coefficient = 0;

      for (int i = 0; i < partition.rows(); i++) {
         for (int j = 0; j < partition.columns(); j++) {
            coefficient += Math.pow(partition.getQuick(i, j), 2.0);
         }
      }

      coefficient /= partition.rows();

      return coefficient;
   }

   public static double partitionEntropy(DoubleMatrix2D partition) {
      double entropy = 0;

      for (int i = 0; i < partition.rows(); i++) {
         for (int j = 0; j < partition.columns(); j++) {
            double membership = partition.getQuick(i, j);
            entropy -= membership > 0 ? membership * Math.log(membership) : 0;
         }
      }

      entropy /= partition.rows();

      return entropy;
   }

   public static double xieBeniIndex(DoubleMatrix2D U, DoubleMatrix2D V, DoubleMatrix2D X, double m) {
      int n = X.rows(); // Number of features
      int p = X.columns(); // Dimensions of features
      int clusters = U.columns(); // Number of features

      double compactness = 0;
      for (int k = 0; k < clusters; k++) {
         // calculate the distance vector for this cluster
         for (int i = 0; i < n; i++) {
            double Um = Math.pow(U.getQuick(i, k), m);

            DoubleMatrix1D dev_i = new DenseDoubleMatrix1D(p);

            dev_i.assign(V.viewRow(k));
            dev_i.assign(X.viewRow(i), Functions.minus);

            // TODO prototype eigenvectors should be an input
            DoubleMatrix2D A = new DenseDoubleMatrix2D(p, p);
            for (int j = 0; j < p; ++j) {
               A.setQuick(j, j, 1);
            }

            // Euclidean distance using hyperellipsoid clusters
            DoubleMatrix1D Adev_i = new DenseDoubleMatrix1D(p);
            A.zMult(dev_i, Adev_i);

            compactness += dev_i.zDotProduct(Adev_i) * Um;
         }
      }
      compactness /= n;

      double minimal_separation = 0;
      for (int k = 0; k < clusters; k++) {
         for (int k2 = 0; k2 < clusters; k2++) {
            // Euclidean distance
            if (k != k2) {
               DoubleMatrix1D dev_k = new DenseDoubleMatrix1D(p);
               dev_k.assign(V.viewRow(k));
               dev_k.assign(X.viewRow(k2), Functions.minus);

               // Calculate magnitude of dev_k
               double separation = dev_k.aggregate(Functions.plus, Functions.square);

               if (minimal_separation == 0) {
                  minimal_separation = separation;
               } else {
                  minimal_separation = Math.min(separation, minimal_separation);
               }
            }
         }
      }

      return compactness / minimal_separation;
   }

   public static DoubleMatrix2D inclusion(DoubleMatrix2D U) {
      DoubleMatrix2D I = new DenseDoubleMatrix2D(U.columns(), U.columns());

      for (int i = 0; i < I.columns(); ++i) {
         for (int j = 0; j < I.rows(); ++j) {
            if (i == j) {
               I.setQuick(i, j, 1);
            } else if (i < j) {
               DoubleMatrix1D umin = new DenseDoubleMatrix1D(U.columns());
               umin.assign(U.viewColumn(i));
               umin.assign(U.viewColumn(j), Functions.min);

               I.setQuick(i, j, umin.zSum()
                       / Math.min(U.viewColumn(i).zSum(),
                       U.viewColumn(j).zSum()));
            } else {
               I.setQuick(i, j, I.getQuick(j, i));
            }
         }
      }

      return I;
   }
}
