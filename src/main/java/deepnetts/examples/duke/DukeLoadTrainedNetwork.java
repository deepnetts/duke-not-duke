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
 * This example shows how to load of instance of trained convolutional network from file.
 * 
 * To run this example you need to download and install Deep Netts , which provides free development license.
 * Follow instructions on
 * https://www.deepnetts.com/blog/deep-learning-in-java-getting-started-with-deep-netts.html
 *
 * For more info about licensing see https://www.deepnetts.com/Deep_Netts_End_User_License_Agreement.pdf
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
