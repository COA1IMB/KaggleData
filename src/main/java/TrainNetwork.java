import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.collection.ListStringRecordReader;
import org.datavec.api.split.ListStringSplit;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.util.List;

import static org.deeplearning4j.nn.conf.Updater.ADAGRAD;

public class TrainNetwork {

    private static int numHiddenNodes;
    private static int maxEpochs;
    private static String neuralNetworkFilePath;


    public TrainNetwork(int numHiddenNodes, int maxEpochs, String neuralNetworkFilePath){
        this.numHiddenNodes = numHiddenNodes;
        this.maxEpochs = maxEpochs;
        this.neuralNetworkFilePath = neuralNetworkFilePath;
    }

    public static List<String> networkLearn(List<List<String>> data, List<String> log) {
        int seed = 123;
        int batchSize = 50;
        int numInputs = 67;
        int numOutputs = 2;
        RecordReader rr = new ListStringRecordReader();

        try {
            rr.initialize(new ListStringSplit(data));
            data = null;
        } catch (Exception e) {
            System.out.println(e);
        }

        DataSetIterator trainIter = new RecordReaderDataSetIterator(rr, batchSize, 67, 2);

        rr = null;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .iterations(1)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(ADAGRAD)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(numInputs)
                        .nOut(numHiddenNodes)
                        .weightInit(WeightInit.XAVIER)
                        .activation("relu")
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nIn(numHiddenNodes)
                        .nOut(numHiddenNodes)
                        .weightInit(WeightInit.XAVIER)
                        .activation("relu")
                        .build())
                .layer(2, new DenseLayer.Builder()
                        .nIn(numHiddenNodes)
                        .nOut(numHiddenNodes)
                        .weightInit(WeightInit.XAVIER)
                        .activation("relu")
                        .build())
                .layer(3, new DenseLayer.Builder()
                        .nIn(numHiddenNodes)
                        .nOut(numHiddenNodes)
                        .weightInit(WeightInit.XAVIER)
                        .activation("relu")
                        .build())
                .layer(4, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .weightInit(WeightInit.XAVIER)
                        .activation("softmax")
                        .weightInit(WeightInit.XAVIER)
                        .nIn(numHiddenNodes)
                        .nOut(numOutputs)
                        .build()
                )
                .pretrain(false).backprop(true).build();

        // Apply Network and attach Listener to Web-UI
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(10));
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        model.setListeners(new StatsListener(statsStorage));
        uiServer.attach(statsStorage);

        EarlyStoppingConfiguration esConf = new EarlyStoppingConfiguration.Builder()
                .epochTerminationConditions(new MaxEpochsTerminationCondition(maxEpochs))
                .scoreCalculator(new DataSetLossCalculator(trainIter, true))
                .evaluateEveryNEpochs(1)
                .modelSaver(new LocalFileModelSaver("C:\\Users\\fabcot01\\IdeaProjects\\MachineLearning2\\"))
                .build();

        EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf, model, trainIter);
        EarlyStoppingResult result = trainer.fit();

        //Print out the results:
        System.out.println("Termination reason: " + result.getTerminationReason());
        System.out.println("Termination details: " + result.getTerminationDetails());
        System.out.println("Total epochs: " + result.getTotalEpochs());
        System.out.println("Best epoch number: " + result.getBestModelEpoch());
        System.out.println("Score at best epoch: " + result.getBestModelScore());

        log.add("Termination reason: " + result.getTerminationReason());
        log.add("Termination details: " + result.getTerminationDetails());
        log.add("Total epochs: " + result.getTotalEpochs());
        log.add("Best epoch number: " + result.getBestModelEpoch());
        log.add("Score at best epoch: " + result.getBestModelScore());

        File locationToSave = new File(neuralNetworkFilePath);
        boolean saveUpdater = true;

        org.deeplearning4j.nn.api.Model model2 = result.getBestModel();

        try {
            ModelSerializer.writeModel(model2, locationToSave, saveUpdater);
        } catch (Exception e) {
            System.out.println(e.toString());
        }
        return log;
    }
}
