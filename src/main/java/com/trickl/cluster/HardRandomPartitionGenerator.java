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
import org.apache.commons.math3.distribution.UniformIntegerDistribution;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;

public class HardRandomPartitionGenerator implements PartitionGenerator {

   private RandomGenerator randomGenerator;

   public HardRandomPartitionGenerator() {
      randomGenerator = new MersenneTwister();
   }

   @Override
   public void generate(DoubleMatrix2D partition) {
      // Initialise U randomly
      partition.assign(0);

      UniformIntegerDistribution uniform = new UniformIntegerDistribution(randomGenerator, 0, partition.columns() - 1);

      for (int i = 0; i < partition.rows(); ++i)
      {
         // Randomise
         int k = uniform.sample();
         partition.setQuick(i, k, 1);
      }
   }

   public RandomGenerator getRandomGenerator() {
      return randomGenerator;
   }

   @Override
   public void setRandomGenerator(RandomGenerator random) {
      this.randomGenerator = random;
   }
}
