package utils;

import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;

/**
 * @author jeffreytang
 */
public class GetSecondColumnCsvPreprocessor implements SentencePreProcessor {

    public String preProcess(String s) {
        return s.split(",")[1].toLowerCase().replaceAll("[^a-zA-Z ]", "");
    }
}
