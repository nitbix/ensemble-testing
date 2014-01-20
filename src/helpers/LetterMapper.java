package helpers;

import java.util.ArrayList;

import org.encog.ml.data.MLData;
import org.encog.ml.data.basic.BasicMLData;

public class LetterMapper implements DataMapper {
	
	private static int _outputs;
	private static double _activationThreshold;
	private static double _lowBound;
	
	public LetterMapper(int outputs, double activationThreshold, double lowBound)
	{
	  _outputs = outputs;
	  _activationThreshold = activationThreshold;
	  _lowBound = lowBound;
	}
	
	@Override
	public MLData map(ArrayList<String> data) {
		final BasicMLData retVal = new BasicMLData(_outputs);
		for (int i = 0; i < _outputs; i++)
			retVal.add(i, _lowBound);
		int value = data.get(0).charAt(0) - 'A';
		retVal.setData(value, 1.0);
		return retVal;
	}

	@Override
	public ArrayList<String> unmap(MLData dataSet) {
		char max = '_';
		double maxval = _activationThreshold;
		for(int i=0; i < _outputs; i++)
			if (dataSet.getData(i) > maxval)
			{
				max = (char) ('A' + i);
				maxval = dataSet.getData(i);
			}
		ArrayList<String> retVal = new ArrayList<String>();
		retVal.add("" + max);
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
		return "" + (char) ('A' + classNumber);
	}
	
}