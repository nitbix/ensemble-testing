package helpers;

import java.io.FileNotFoundException;
import java.io.IOException;
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
	private BasicNeuralDataSet _cvSet;
	private BasicNeuralDataSet _testSet;
	private boolean _inputsReversed;
	private DataMapper _dataMapper;
	private int _inputs;
	private int _readinputs;
	private int nFolds;
	private boolean hasSeparateTestSet;
	private boolean gzippedData;

	public DataLoader(DataMapper dataMapper, int readInputs, int inputs, boolean inputsReversed, int nFolds, boolean hasSeparateTestSet, boolean gzippedData) {
		_dataMapper = dataMapper;
		_readinputs = readInputs;
		_inputs = inputs;
		_inputsReversed = inputsReversed;
		this.hasSeparateTestSet = hasSeparateTestSet;
		this.nFolds = nFolds;
		this.gzippedData = gzippedData;
		folds = new ArrayList<BasicNeuralDataSet>();
		for (int i = 0; i < nFolds; i++)
			folds.add(new BasicNeuralDataSet());
	}
	
	private BasicNeuralDataSet readFile(String inputFile) throws FileNotFoundException, IOException {
		FileLoader fileLoader = new FileLoader();
		String inputFileNameFinal = inputFile;
		if(gzippedData)
		{
			inputFileNameFinal += ".gz";
		}
		ReadCSV csv = new ReadCSV(fileLoader.openOrFind(inputFileNameFinal),false,',');
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
	
	public int readData(String inputFile) throws FileNotFoundException, IOException {
		//System.out.println("importing dataset");
		if(hasSeparateTestSet)
		{
			//TODO: normalize
			_trainSet = readFile(inputFile + ".train");
			_testSet = readFile(inputFile + ".test");
			_completeSet = (BasicNeuralDataSet) _trainSet.clone();
			for(MLDataPair p : _testSet)
			{
				_completeSet.add(p);
			}
			for (int i = 0; i < _trainSet.size(); i++)
			{
				folds.get(i % nFolds).add(_trainSet.get(i).getInput(),_trainSet.get(i).getIdeal());
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

	public BasicNeuralDataSet getCVSet() {
		return _cvSet;
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
		_trainSet = new BasicNeuralDataSet();
		if(!hasSeparateTestSet)
		{
			_trainSet = new BasicNeuralDataSet();
			for (int i = 0; i < nFolds; i++)
			{
				if (((i != fold) && (i != (fold + 1) % nFolds)) || nFolds == 1)
				{
					for (MLDataPair k : folds.get(i))
					{
						_trainSet.add(k);
					}
				}
			}
			_testSet = (BasicNeuralDataSet) folds.get(fold).clone();
			_cvSet = (BasicNeuralDataSet) folds.get((fold + 1) % nFolds).clone();
		}
		else
		{
			for (int i = 0; i < nFolds; i++)
			{
				if (i != fold)
				{
					for (MLDataPair k : folds.get(i))
					{
						_trainSet.add(k);
					}
				}
			}
			_cvSet = (BasicNeuralDataSet) folds.get(fold).clone();
		}
	}
	
	public BasicNeuralDataSet getTrainingSet() {
		return _trainSet;
	}
}
