package helpers;

import java.util.ArrayList;

import org.encog.ml.data.MLData;
import org.encog.neural.data.basic.BasicNeuralData;

public class IntRegressionMapper implements DataMapper {
	
	private static int _outputs;
	private static double _activationThreshold;
	
	public IntRegressionMapper(int outputs, double activationThreshold)
	{
	  _outputs = outputs;
	  _activationThreshold = activationThreshold;
	}
	
	@Override
	public MLData map(ArrayList<String> data) {
		final BasicNeuralData retVal = new BasicNeuralData(_outputs);
		for (int i = 0; i < _outputs; i++)
			retVal.add(i, Integer.parseInt(data.get(i)));
		return retVal;
	}

	@Override
	public ArrayList<String> unmap(MLData dataSet) {
		ArrayList<String> retVal = new ArrayList<String>();
		for(int i=0; i < _outputs; i++)
			retVal.add(Double.toString(dataSet.getData(i)));
		return retVal;
	}

	@Override
	public boolean compare(ArrayList<String> result, ArrayList<String> expected, boolean print) {
		if (print) 
			System.out.println("Exp " + expected.get(0) + " got " + result.get(0));
		return result.get(0).matches(expected.get(0));
	}

	@Override
	public String getClassLabel(int classNumber) {
		return Integer.toString(classNumber);
	}

	
}