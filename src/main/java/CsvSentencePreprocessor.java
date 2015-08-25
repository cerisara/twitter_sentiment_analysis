import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;

/**
 * @author jeffreytang
 */
public class CsvSentencePreprocessor implements SentencePreProcessor {

    public String preProcess(String s) {
        return s.split(",")[1];
    }
}
