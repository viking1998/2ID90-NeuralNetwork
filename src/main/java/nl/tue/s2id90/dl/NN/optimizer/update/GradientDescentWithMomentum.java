/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nl.tue.s2id90.dl.NN.optimizer.update;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *
 * @author Petar Petrov, 1227549
 */
public class GradientDescentWithMomentum implements UpdateFunction {

    private INDArray vel;
    public static double mu = 0.5;
    
    @Override
    public void update(INDArray array, boolean isBias, double learningRate, int batchSize, INDArray gradient) {
        if (vel == null) {
            vel = gradient.dup('f').assign(0);
        }
        vel = vel.mul(mu).sub(gradient.mul(learningRate/batchSize));
        array.addi(vel);
    }
    
}
