/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nl.tue.s2id90.dl.NN.optimizer.update;

import java.util.function.Supplier;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *
 * @author Emu 1281666
 */
public class L2Decay implements UpdateFunction{
    double decay;
    UpdateFunction f;
    
    public L2Decay(Supplier<UpdateFunction> supplier, double decay){
        this.decay = decay;
        this.f = supplier.get();
    }
    
    @Override
    public void update(INDArray array, boolean isBias, double learningRate, int batchSize, INDArray gradient){
        array = array.add(((array.mul(array)).mul(decay)).div(2));
        f.update(array, isBias, learningRate, batchSize, gradient);
    }
}
