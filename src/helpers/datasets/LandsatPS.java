package helpers.datasets;

import helpers.DataLoader;
import helpers.IntMapper;
import helpers.ProblemDescription;

public class LandsatPS implements ProblemDescription {

	private static int outputs = 7;
	private static int inputs = 36;
	private static int readInputs = 1;
	private static boolean inputsReversed = true;
	private static String inputFile = "data/landsat.data";
	
	@Override
	public DataLoader getDataLoader(double activationThreshold, int nFolds) {
		DataLoader dataLoader = new DataLoader(new IntMapper(outputs,activationThreshold),readInputs,inputs,inputsReversed,nFolds);
		dataLoader.readData(inputFile);
		return dataLoader;
	}

	@Override
	public int getOutputs() {
		return outputs;
	}

	@Override
	public int getInputs() {
		return inputs;
	}

	@Override
	public int getReadInputs() {
		return readInputs;
	}

	@Override
	public boolean areInputsReversed() {
		return inputsReversed;
	}

	@Override
	public String getInputFile() {
		return inputFile;
	}

	@Override
	public String getLabel() {
		// TODO Auto-generated method stub
		return null;
	}

}
