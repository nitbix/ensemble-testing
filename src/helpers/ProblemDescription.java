package helpers;

import java.io.FileNotFoundException;

import helpers.ProblemDescriptionLoader.BadArgument;

public interface ProblemDescription {
	
	public DataLoader getDataLoader(double activationThreshold, int trainingSetSize) throws BadArgument, FileNotFoundException;
	public int getOutputs();
	public int getInputs();
	public int getReadInputs();
	public boolean areInputsReversed();
	public String getInputFile();
	public String getLabel();
	
}
