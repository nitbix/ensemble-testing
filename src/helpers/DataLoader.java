package helpers;

import java.io.FileNotFoundException;
import java.util.ArrayList;

import org.encog.NullStatusReportable;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.neural.data.basic.BasicNeuralData;
import org.encog.neural.data.basic.BasicNeuralDataSet;
import org.encog.util.csv.ReadCSV;
import org.encog.util.normalize.DataNormalization;
import org.encog.util.normalize.input.InputField;
import org.encog.util.normalize.input.InputFieldMLDataSet;
import org.encog.util.normalize.output.OutputFieldRangeMapped;
import org.encog.util.normalize.target.NormalizationStorageNeuralDataSet;

public class DataLoader {
	
	private ArrayList<BasicNeuralDataSet> folds;
	private BasicNeuralDataSet _completeSet;
	private boolean _inputsReversed;
	private DataMapper _dataMapper;
	private int _inputs;
	private int _readinputs;
	private int nFolds;

	public DataLoader(DataMapper dataMapper, int readInputs, int inputs, boolean inputsReversed, int nFolds) {
		_dataMapper = dataMapper;
		_readinputs = readInputs;
		_inputs = inputs;
		_inputsReversed = inputsReversed;
		this.nFolds = nFolds;
		folds = new ArrayList<BasicNeuralDataSet>();
		for (int i = 0; i < nFolds; i++)
			folds.add(new BasicNeuralDataSet());
	}
	
	public int readData(String inputFile) throws FileNotFoundException {
		FileLoader fileLoader = new FileLoader();
		int total=0;
		//System.out.println("importing dataset");
		ReadCSV csv = new ReadCSV(fileLoader.openOrFind(inputFile),false,',');
		_completeSet = new BasicNeuralDataSet();
		while(csv.next())
		{
			BasicNeuralData inputData = new BasicNeuralData(getInputs());
			ArrayList<String> readIn = new ArrayList<String>();
			MLData idealData;
			if(_inputsReversed) {
				for(int j = 0; j < getInputs(); j++) {
					inputData.setData(j,csv.getDouble(j));
				}
				for(int k = 0; k < getReadInputs(); k++)
					readIn.add(csv.get(k + getInputs()));
				idealData = getMapper().map(readIn);
			} else {
				for(int k = 0; k < getReadInputs(); k++)
					readIn.add(csv.get(k));
				idealData = getMapper().map(readIn);
				for(int j = 0; j < getInputs(); j++) {
					inputData.setData(j,csv.getDouble(j + getReadInputs()));
				}
			}
			_completeSet.add(inputData,idealData);
			total++;
		}
		BasicNeuralDataSet _normSet = new BasicNeuralDataSet();
		DataNormalization normalizer = new DataNormalization();
		normalizer.setReport(new NullStatusReportable());
		normalizer.setTarget(new NormalizationStorageNeuralDataSet(_normSet));
		InputField[] a = new InputField[getInputs()];
		for(int j = 0; j < getInputs(); j++) {
			normalizer.addInputField(a[j] = new InputFieldMLDataSet(false,_completeSet,j));
			normalizer.addOutputField(new OutputFieldRangeMapped(a[j],0.0,1.0));
		}
		normalizer.process();
		for (int i = 0; i < total; i++)
		{
			folds.get(i % nFolds).add(_normSet.get(i).getInput(),_completeSet.get(i).getIdeal());
		}
		csv.close();
		return total;
		
	}

	public DataMapper getMapper() {
		return _dataMapper;
	}

	private int getInputs() {
		return _inputs;
	}

	public void setInputs(int _inputs) {
		this._inputs = _inputs;
	}

	public BasicNeuralDataSet getTestSet(int fold) {
		return folds.get(fold);
	}

	public int getReadInputs() {
		return _readinputs;
	}

	public void setReadInputs(int _readinputs) {
		this._readinputs = _readinputs;
	}

	public int size() {
		return _completeSet.size();
	}
	public BasicNeuralDataSet getTrainingSet(int fold) {
		BasicNeuralDataSet trainingSet = new BasicNeuralDataSet();
		for (int i = 0; i < nFolds; i++)
			if ((i != fold) && (nFolds > 1)){
				for (MLDataPair k : folds.get(i)) {
					trainingSet.add(k);
				}
			}
		return trainingSet;
	}
}
