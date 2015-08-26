package utils;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.FileWriter;
import java.io.IOException;
import java.util.List;

/**
 * @author jeffreytang
 */
public class WriteLabelAvgTweetToCsv {
    public static void toCsv(String filePath, List<Integer> targets, List<INDArray> features) throws IOException {

        FileWriter writer = new FileWriter(filePath);
        for (int i=0; i < targets.size(); i++) {
            double[] featureRow = features.get(i).data().asDouble();
            Integer label = targets.get(i);
            writer.append(label.toString());
            writer.append(',');
            for (double num : featureRow) {
                writer.append(String.valueOf(num));
                writer.append(',');
            }
            writer.append('\n');
            writer.flush();
        }
        writer.close();
    }
}
