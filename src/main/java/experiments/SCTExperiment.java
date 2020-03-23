/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package experiments;

import java.io.IOException;
import nl.tue.s2id90.dl.NN.Model;
import nl.tue.s2id90.dl.NN.activation.RELU;
import nl.tue.s2id90.dl.NN.initializer.Gaussian;
import nl.tue.s2id90.dl.NN.layer.Convolution2D;
import nl.tue.s2id90.dl.NN.layer.Flatten;
import nl.tue.s2id90.dl.NN.layer.FullyConnected;
import nl.tue.s2id90.dl.NN.layer.InputLayer;
import nl.tue.s2id90.dl.NN.layer.OutputSoftmax;
import nl.tue.s2id90.dl.NN.layer.PoolMax2D;
import nl.tue.s2id90.dl.NN.loss.CrossEntropy;
import nl.tue.s2id90.dl.NN.optimizer.Optimizer;
import nl.tue.s2id90.dl.NN.optimizer.SGD;
import nl.tue.s2id90.dl.NN.tensor.TensorShape;
import nl.tue.s2id90.dl.NN.validate.Classification;
import nl.tue.s2id90.dl.experiment.GUIExperiment;
import nl.tue.s2id90.dl.input.InputReader;
import nl.tue.s2id90.dl.input.MNISTReader;
import nl.tue.s2id90.dl.javafx.FXGUI;
import nl.tue.s2id90.dl.javafx.ShowCase;

/**
 *
 * @author Petar Petrov, 1227549
 */
public class SCTExperiment extends GUIExperiment {
        // ( hyper ) parameters
    int batchSize = 32;
    int epochs = 5;
    double learningRate = 0.01;
    String[] labels= {
    "Square","Circle","Triangle"
    };
    ShowCase showCase = new ShowCase(i -> labels[i]);
    
    public void go() throws IOException {
        // you are going to add code here
        // read input and pr int some informat ion on the data
        InputReader reader = MNISTReader.primitives(batchSize);
        System.out.println(" Reader info :\n" + reader.toString());
        
        FXGUI.getSingleton().addTab("show case", showCase.getNode());
        showCase.setItems(reader.getValidationData(100));
        
        int inputs = reader.getInputShape().getNeuronCount();
        int outputs = reader.getOutputShape().getNeuronCount();
        
//        MeanSubtraction meanTransform = new MeanSubtraction();
//        meanTransform.fit(reader.getTrainingData());
//        meanTransform.transform(reader.getTrainingData());
//        meanTransform.transform(reader.getValidationData());

        Model model = createModel(inputs, outputs, reader);
        model.initialize(new Gaussian());
    
        Optimizer sgd = SGD.builder()
                .model(model)
                .validator(new Classification())
//                .updateFunction(GradientDescentWithMomentum::new)
                .learningRate(learningRate)
                .build();
        
        trainModel(sgd, reader, epochs, 0);
    }
    
    Model createModel(int inputs, int outputs, InputReader reader) {
//    int fcNeurons = 100;

//    Model model = new Model(new InputLayer("In", reader.getInputShape(), true));
//    model.addLayer(new Convolution2D("cv1", reader.getInputShape(), 3, 16, new RELU()));
//    model.addLayer(new PoolMax2D("pm2", model.getLayers().get(1).getOutputShape(), 2));
//    model.addLayer(new Flatten("Flatten", model.getLayers().get(2).getOutputShape()));
//    model.addLayer(new FullyConnected("fc1", model.getLayers().get(3).getOutputShape(), fcNeurons, new RELU()));
//    model.addLayer(new OutputSoftmax("Out", new TensorShape(fcNeurons), outputs, new CrossEntropy()));
    Model model = new Model(new InputLayer("In", reader.getInputShape(), true));
    model.addLayer(new Flatten("Flatten", reader.getInputShape()));
    model.addLayer(new OutputSoftmax("Out", new TensorShape(reader.getInputShape().getNeuronCount()), outputs, new CrossEntropy()));

    return model;
    }

    public static void main(String[] args) throws IOException {
        new SCTExperiment().go();
    }
}
