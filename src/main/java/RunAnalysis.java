/**
 * @author jeffreytang
 */

import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;

public class RunAnalysis {
    private static Logger log = LoggerFactory.getLogger(RunAnalysis.class);


    public static void main(String args[]) throws Exception {

        log.info("Parse CSV file from s3 bucket");
        CsvSentencePreprocessor csvSentencePreprocessor = new CsvSentencePreprocessor();
        S3SentenceIterator it = new S3SentenceIterator(csvSentencePreprocessor,
                "sentiment140twitter", "sentiment140_train.csv");

        InMemoryLookupCache cache = new InMemoryLookupCache();
        WeightLookupTable table = new InMemoryLookupTable.Builder()
                .vectorLength(100)
                .useAdaGrad(false)
                .cache(cache)
                .lr(0.025f).build();

        log.info("Building model....");
        Word2Vec vec = new Word2Vec.Builder()
                .minWordFrequency(5).iterations(1)
                .layerSize(100).lookupTable(table)
                .stopWords(new ArrayList<String>())
                .vocabCache(cache).seed(42)
                .windowSize(5).iterate(it).build();

        log.info("Training model...");
        vec.fit();

        log.info("Writing word vectors to file...");
        WordVectorSerializer.writeWordVectors(vec, "twitter_word_vector.txt");
    }
}
