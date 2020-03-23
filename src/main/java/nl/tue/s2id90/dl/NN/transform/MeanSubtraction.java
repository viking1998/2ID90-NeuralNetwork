/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nl.tue.s2id90.dl.NN.transform;

import java.util.List;
import nl.tue.s2id90.dl.NN.tensor.TensorPair;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *
 * @author Petar Petrov, 1227549
 */
public class MeanSubtraction implements DataTransform {
    Double mean = 0.0;

    @Override
    public void fit(List<TensorPair> pairs) {
        if (pairs.isEmpty()) {
            throw new IllegalArgumentException("Empty dataset!");
        }

        INDArray perPixelMean = pairs.get(0).model_input.getValues().dup('f').assign(0);
        for (TensorPair pair : pairs) {
            perPixelMean.addi(pair.model_input.getValues());
        }
        
        perPixelMean.divi(pairs.size());
        mean = perPixelMean.meanNumber().doubleValue();
    }

    @Override
    public void transform(List<TensorPair> pairs) {
        for (TensorPair pair : pairs) {
            pair.model_input.getValues().subi(mean);
        }
    }
    
}
