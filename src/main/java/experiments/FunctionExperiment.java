/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package experiments;
import nl.tue.s2id90.dl.experiment.Experiment ;
import java.io.IOException ;
import java.util.concurrent.ThreadLocalRandom;
import nl.tue.s2id90.dl.NN.Model;
import nl.tue.s2id90.dl.NN.activation.RELU;
import nl.tue.s2id90.dl.NN.initializer.Gaussian;
import nl.tue.s2id90.dl.NN.layer.FullyConnected;
import nl.tue.s2id90.dl.NN.layer.InputLayer;
import nl.tue.s2id90.dl.NN.layer.SimpleOutput;
import nl.tue.s2id90.dl.NN.loss.MSE;
import nl.tue.s2id90.dl.NN.optimizer.Optimizer;
import nl.tue.s2id90.dl.NN.optimizer.SGD;
import nl.tue.s2id90.dl.NN.tensor.TensorShape;
import nl.tue.s2id90.dl.NN.validate.Regression;
import nl.tue.s2id90.dl.experiment.GUIExperiment;
import nl.tue.s2id90.dl.input.GenerateFunctionData;
import nl.tue.s2id90.dl.input.InputReader;
        
/**
 *
 * @author Petar Petrov, 1227549
 */
public class FunctionExperiment extends GUIExperiment {
    // ( hyper ) parameters
    double learningRate = 0.08;
    int batchSize = 10;
    int epochs = 100;
    
    public void go() throws IOException {
        // you are going to add code here
        // read input and pr int some informat ion on the data
        InputReader reader = GenerateFunctionData.THREE_VALUED_FUNCTION(batchSize);
        System.out.println(" Reader info :\n" + reader.toString());
        
        int inputs = reader.getInputShape().getNeuronCount();
        int outputs = reader.getOutputShape().getNeuronCount();
        
        Model model = createModel(inputs, outputs);
        model.initialize(new Gaussian());
        
        Optimizer sgd = SGD.builder()
                .model(model)
                .validator(new Regression())
                .learningRate(learningRate)
                .build();
        
        trainModel(sgd, reader, epochs, 0);
    }
    
    Model createModel(int inputs, int outputs) {
        int fcNeurons = 24;
        Model model = new Model(new InputLayer("In", new TensorShape(inputs), true));
        model.addLayer(new FullyConnected("fc1", new TensorShape(inputs), fcNeurons, new RELU()));
        model.addLayer(new SimpleOutput("Out", new TensorShape(fcNeurons), outputs, new MSE(), true));
        return model;
    }

    public static void main(String[] args) throws IOException {
        new FunctionExperiment().go();
    }

}
