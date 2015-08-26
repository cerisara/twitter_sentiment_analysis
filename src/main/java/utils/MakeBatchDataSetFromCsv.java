package utils;

import org.apache.commons.io.IOUtils;
import org.apache.commons.io.LineIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.io.FileInputStream;
import java.util.Arrays;

/**
 * @author jeffreytang
 */
public class MakeBatchDataSetFromCsv {
    private LineIterator iter;
    private int batchSize;
    private int ncol;
    private boolean end;

    public MakeBatchDataSetFromCsv(String csvFileName, int batchSize, int ncol) throws Exception {
        this.iter = IOUtils.lineIterator(new FileInputStream(csvFileName), "UTF-8");
        this.ncol = ncol;
        this.batchSize = batchSize;
    }

    public DataSet makeBatch() {
        INDArray target = Nd4j.create(batchSize, 1);
        INDArray feature = Nd4j.create(batchSize, ncol);

        for (int i=0; i < batchSize; i++) {
            if (iter.hasNext()) {
                String[] arr = iter.nextLine().split(",");
                // Deals with label
                target.putRow(i, Nd4j.create(new double[]{Double.parseDouble(arr[0])}));
                // Deals with feature
                double[] doubleFeature = new double[arr.length - 1];
                for (String stringItem : Arrays.copyOfRange(arr, 1, arr.length)) {
                    doubleFeature[i] = Double.parseDouble(stringItem);
                }
                feature.putRow(i, Nd4j.create(doubleFeature));
            } else {
                end = true;
            }
        }
        return new DataSet(feature, target);
    }

    public boolean getEnd() {
        return this.end;
    }
}
