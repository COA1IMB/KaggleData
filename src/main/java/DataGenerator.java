import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by fabcot01 on 24.05.2017.
 */
public class DataGenerator {

    String learnFilePath;
    int numberOfDataSets = 70000;
    int numberOfColumns = 68;

    public DataGenerator(String learnFilePath){
        this.learnFilePath = learnFilePath;
    }

    public List<List<String>> getDataAsList() {

        ArrayList<List<String>> data = null;
        try {
            String fileName = learnFilePath;
            BufferedReader br = null;
            String sCurrentLine;
            br = new BufferedReader(new FileReader(fileName));//file name with path
            data = new ArrayList<List<String>>();

            while ((sCurrentLine = br.readLine()) != null) {

                if(sCurrentLine.contains("NA")) {
                    continue;
                }

                String[] parts1 = sCurrentLine.split(",");

                if(Double.parseDouble(parts1[67]) > 0){
                    parts1[67] = "1";
                }

                List<String> data2 = Arrays.asList(parts1);
                data.add(data2);
            }
            return data;
        } catch (IOException e) {
            e.printStackTrace();
        }
        return data;
    }
    public List<List<String>> normalize(List<List<String>> data) {

        double[] values = new double[numberOfDataSets];
        double[] mins = new double[numberOfColumns];
        double[] maxs = new double[numberOfColumns];

        for (int j = 0; j < data.get(0).size(); j++) {
            for (int i = 0; i < data.size(); i++) {
                values[i] = Double.parseDouble(data.get(i).get(j));
            }
            mins[j] = Arrays.stream(values).min().getAsDouble();
            maxs[j] = Arrays.stream(values).max().getAsDouble();
        }
        for (List<String> temp : data) {
            for (int y = 0; y < numberOfColumns; y++) {
                if (mins[y] != maxs[y]) {
                    double tempValue = (Double.parseDouble(temp.get(y)) - mins[y]) / (maxs[y] - mins[y]);

                    if (tempValue == 0.0) {
                        temp.set(y, "0");
                    } else if (tempValue == 1.0) {
                        temp.set(y, "1");
                    } else {
                        temp.set(y, Double.toString(tempValue));
                    }
                }
            }
        }
        return data;
    }
}
