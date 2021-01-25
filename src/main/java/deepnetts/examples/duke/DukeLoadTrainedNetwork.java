package deepnetts.examples.duke;

import deepnetts.core.DeepNetts;
import deepnetts.util.ConvolutionalImageClassifier;
import deepnetts.net.ConvolutionalNetwork;
import deepnetts.util.FileIO;
import deepnetts.util.ImagePreprocessing;
import java.io.File;
import java.io.IOException;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.visrec.ml.ClassificationException;

/**
 * This example shows how to load and create instance of trained network from file.
 * 
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */
public class DukeLoadTrainedNetwork {
    
    public static void main(String[] args) throws ClassificationException {

        try {
            ConvolutionalNetwork neuralNet =  FileIO.createFromFile("deepNetwork1.dnet", ConvolutionalNetwork.class);
            ((ImagePreprocessing)neuralNet.getPreprocessing()).setEnabled(true);	

            ConvolutionalImageClassifier imageClassifier = new ConvolutionalImageClassifier(neuralNet);    // this image recognize shoul dbe used from visrec api
            Map<String, Float> results = imageClassifier.classify(new File("dataset/duke/duke1.jpg"));
            System.out.println("Probability that image belongs into category: " + results.toString());

            DeepNetts.shutdown();
        } catch (IOException | ClassNotFoundException ioe) {
            Logger.getLogger(DukeLoadTrainedNetwork.class.getName()).log(Level.SEVERE, null, ioe);
        }
     
    }    
}
