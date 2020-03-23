/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nl.tue.s2id90.dl.NN.optimizer.update;

import java.util.function.Supplier;
import org.nd4j.linalg.api.ndarray.INDArray;
import static org.nd4j.linalg.ops.transforms.Transforms.sqrt;

/**
 *
 * @author Emu 1281666
 */
public class AdaDelta implements UpdateFunction{
    double ro = 0.9;//decay
    UpdateFunction f;
    
    double epsilon = 0.00001;// might need tweaking
    INDArray arr;
    INDArray delta;
    
    public AdaDelta(Supplier<UpdateFunction> supplier, double decay){
        this.ro = decay;
        this.f = supplier.get();
    }
    
    
    @Override
    public void update(INDArray array, boolean isBias, double learningRate, int batchSize, INDArray gradient){
        if(arr == null){
            arr = array.dup();
            arr = arr.mul(0);
            delta = arr.dup();
        }
        arr = (arr.mul(ro)).add(gradient.mul(gradient).mul(1-ro));
        
        INDArray change = (gradient.mul(-1)).
                mul(sqrt(delta.mean().add(epsilon)).
                        div(sqrt(gradient.mean().add(epsilon))));
        
        delta = (delta.mul(ro)).add(change.mul(change).mul(1.0 - ro));
        
        array = array.add(change);
    }
}
