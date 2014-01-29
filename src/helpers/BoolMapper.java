package helpers;

import java.util.ArrayList;

import org.encog.ml.data.MLData;
import org.encog.neural.data.basic.BasicNeuralData;

public class BoolMapper implements DataMapper {
	
	private static double _activationThreshold;
	
	public BoolMapper(int outputs, double activationThreshold) throws Exception
	{
		if(outputs != 1)
		{
			throw new Exception("BoolMapper only deals with one output");
		}
		_activationThreshold = activationThreshold;
	}
	
	@Override
	public MLData map(ArrayList<String> data) {
		final BasicNeuralData retVal = new BasicNeuralData(1);
		int value = Integer.parseInt(data.get(0));
		if(value > 1)
			value = 1;
		if(value < 0)
			value = 0;
		retVal.setData(0, value);
		return retVal;
	}

	@Override
	public ArrayList<String> unmap(MLData dataSet) {
		ArrayList<String> retVal = new ArrayList<String>();
		int data = (int) dataSet.getData(0);
		if(data > _activationThreshold)
			retVal.add("1");
		else
			retVal.add("0");
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