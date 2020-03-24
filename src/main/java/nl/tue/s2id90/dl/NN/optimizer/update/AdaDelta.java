/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nl.tue.s2id90.dl.NN.optimizer.update;

import org.nd4j.linalg.api.ndarray.INDArray;
import static org.nd4j.linalg.ops.transforms.Transforms.sqrt;

/**
 *
 * @author Emu 1281666
 */
public class AdaDelta implements UpdateFunction{
    double ro = 0.9f;//decay
    double epsilon = 0.00001f;// might need tweaking
    INDArray arr;
    INDArray delta;
    
    @Override
    public void update(INDArray array, boolean isBias, double learningRate, int batchSize, INDArray gradient){
        if(arr == null){
            arr = array.mul(0);
            delta = arr.dup();
        }
        arr = (arr.mul(ro)).add(gradient.mul(gradient).mul(1 - ro));
        
        INDArray change = gradient.
                mul(sqrt((delta.add(epsilon)).
                        div(arr.add(epsilon))));
        
        delta = (delta.mul(ro)).add(change.mul(change).mul(1 - ro));
        
        array.subi(change);
    }
}
