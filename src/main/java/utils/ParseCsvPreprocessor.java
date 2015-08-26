package utils;

import javafx.util.Pair;

/**
 * @author jeffreytang
 */
public class ParseCsvPreprocessor {
    public Pair<double[], String[]> preProcess(String s) {
        String[] labelAndSentence = s.split(",");
        double label = Double.parseDouble(labelAndSentence[0]);
        double[] labels = {label, 1 - label};
        String[] words = labelAndSentence[1].toLowerCase().split(" ");
        return new Pair<>(labels, words);
    }
}
