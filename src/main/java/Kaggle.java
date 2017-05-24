import java.io.IOException;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.List;

/**
 * Created by fabcot01 on 24.05.2017.
 */
public class Kaggle {

    private static final String LEARN_FILE_PATH = "C:\\Users\\fabcot01\\IdeaProjects\\KaggleData\\src\\main\\resources\\train_v3.csv";
    private static final String EVAL_FILE_PATH = "C:\\Users\\fabcot01\\IdeaProjects\\KaggleData\\src\\main\\resources\\test_v3.csv";
    private static String NN_FILE_PATH = "C:\\Users\\fabcot01\\IdeaProjects\\KaggleData\\NeuralNetwork.zip";;
    private static int NUM_HIDDEN_NODES = 200;
    private static int MAX_EPOCHS = 10;
    private static List<String> log = new ArrayList<String>();

    public static void main(String [] args){
//        DataGenerator generator = new DataGenerator(LEARN_FILE_PATH);
//        List<List<String>> trainingData = generator.getDataAsList();
//        trainingData = generator.normalize(trainingData);

        DataGenerator generator2n = new DataGenerator(EVAL_FILE_PATH);
        List<List<String>> evalData = generator2n.getDataAsList();
        evalData = generator2n.normalize(evalData);

        String timeLog = new SimpleDateFormat("dd.MM.yyyy HH:mm").format(Calendar.getInstance().getTime());
        log.add("Start time: " + timeLog);
        TrainNetwork trainer = new TrainNetwork(NUM_HIDDEN_NODES, MAX_EPOCHS, NN_FILE_PATH);
        EvaluateNetwork evaluater = new EvaluateNetwork(NN_FILE_PATH);

        //log = trainer.networkLearn(trainingData, log);
        log = evaluater.evaluateNetwork(evalData, log);

        writeLogFile();
    }
    private static void writeLogFile() {

        try {
            PrintWriter writer = new PrintWriter("log.txt", "UTF-8");
            for (String temp : log) {
                writer.println(temp);
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
