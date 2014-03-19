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
	private BasicNeuralDataSet _trainSet;
	private BasicNeuralDataSet _testSet;
	private boolean _inputsReversed;
	private DataMapper _dataMapper;
	private int _inputs;
	private int _readinputs;
	private int nFolds;
	private boolean hasSeparateTestSet;

	public DataLoader(DataMapper dataMapper, int readInputs, int inputs, boolean inputsReversed, int nFolds, boolean hasSeparateTestSet) {
		_dataMapper = dataMapper;
		_readinputs = readInputs;
		_inputs = inputs;
		_inputsReversed = inputsReversed;
		this.hasSeparateTestSet = hasSeparateTestSet;
		this.nFolds = nFolds;
		folds = new ArrayList<BasicNeuralDataSet>();
		for (int i = 0; i < nFolds; i++)
			folds.add(new BasicNeuralDataSet());
	}
	
	private BasicNeuralDataSet readFile(String inputFile) throws FileNotFoundException {
		FileLoader fileLoader = new FileLoader();
		ReadCSV csv = new ReadCSV(fileLoader.openOrFind(inputFile),false,',');
		BasicNeuralDataSet set = new BasicNeuralDataSet();
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
			set.add(inputData,idealData);
		}
		csv.close();
		return set;
		
	}
	
	public int readData(String inputFile) throws FileNotFoundException {
		//System.out.println("importing dataset");
		if(hasSeparateTestSet)
		{
			_trainSet = readFile(inputFile + ".train");
			_testSet = readFile(inputFile + ".test");
			_completeSet = (BasicNeuralDataSet) _trainSet.clone();
			for(MLDataPair p : _testSet)
			{
				_completeSet.add(p);
			}
			return _trainSet.size() + _testSet.size();
		}
		else
		{
			_completeSet = readFile(inputFile);
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
			for (int i = 0; i < _completeSet.size(); i++)
			{
				folds.get(i % nFolds).add(_normSet.get(i).getInput(),_completeSet.get(i).getIdeal());
			}
			return _completeSet.size();
		}
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

	public BasicNeuralDataSet getTestSet() {
		return _testSet;
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
	
	public void setFold(int fold)
	{
		if(!hasSeparateTestSet)
		{
			_trainSet = new BasicNeuralDataSet();
			for (int i = 0; i < nFolds; i++)
			{
				if ((i != fold) && (nFolds > 1)){
					for (MLDataPair k : folds.get(i)) {
						_trainSet.add(k);
					}
				}
			}
			_testSet = folds.get(fold);
		}
	}
	
	public BasicNeuralDataSet getTrainingSet() {
		return _trainSet;
	}
}
