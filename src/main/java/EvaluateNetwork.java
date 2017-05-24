import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.collection.ListStringRecordReader;
import org.datavec.api.split.ListStringSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.util.HashMap;
import java.util.List;

public class EvaluateNetwork {

    private static String neuralNetworkFilePath;
    private static final double GREEDY_THRESHOLD = 0.2;


    public EvaluateNetwork(String neuralNetworkFilePath){
        this.neuralNetworkFilePath = neuralNetworkFilePath;
    }

    public static List<String> evaluateNetwork(List<List<String>> dataEval, List<String> log) {
        RecordReader rrTest = new ListStringRecordReader();
        MultiLayerNetwork model = null;
        int batchSize = 1;
        int numOutputs = 2;

        try {
            rrTest.initialize(new ListStringSplit(dataEval));
        } catch (Exception e) {
            System.out.println(e);
        }

        DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest, batchSize, 67, 2);

        try {
            model = ModelSerializer.restoreMultiLayerNetwork(neuralNetworkFilePath);
        } catch (Exception e) {
            System.out.println(e.toString());
        }

        System.out.println("Evaluate model.......");
        Evaluation eval = new Evaluation(numOutputs);

        int positiveDefault = 0;
        int falseDefault = 0;
        int positiveNonDefault = 0;
        int falseNonDefault = 0;
        int amountDefaults = 0;
        int amountNonDefaults = 0;
        int lineCounter = 1;


        while (testIter.hasNext()) {
            DataSet t = testIter.next();
            INDArray features = t.getFeatureMatrix();
            INDArray lables = t.getLabels();
            INDArray predicted = model.output(features, false);
            eval.eval(lables, predicted);
            DataBuffer s = t.getFeatures().data();

            double defaultProbability = predicted.data().getDouble(1L);
            double labelValue = t.getLabels().data().getDouble(1L);

            if(labelValue == 1.0){
                amountDefaults++;
                if(GREEDY_THRESHOLD >= defaultProbability){
                    positiveDefault++;
                }else{
                    falseDefault++;
                }
            }else if(labelValue == 0.0){
                amountNonDefaults++;
                if(GREEDY_THRESHOLD >= defaultProbability){
                    positiveNonDefault++;
                }else{
                    falseNonDefault++;
                }
            }
            System.out.print("Label is: " + labelValue + " Predited was: " + predicted);
            System.out.println(s.getComplexDouble(0L) + "...");
            lineCounter++;
        }
        System.out.println(eval.stats());
        System.out.println(" Richtig erkannte Ausfälle:         " + positiveDefault + "               Von:  " + amountDefaults);
        System.out.println(" Falsch erkannte Ausfälle:          " + falseDefault );
        System.out.println(" Richtig erkannte Nicht-Ausfälle:   " + positiveNonDefault + "            Von:  " + amountNonDefaults);
        System.out.println(" Falsch erkannte Nicht-Ausfälle:    " + falseNonDefault);
        log.add(eval.stats());

        return log;
    }
}
