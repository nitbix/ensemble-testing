package techniques;

import java.util.List;

import org.encog.ensemble.EnsembleAggregator;
import org.encog.ensemble.EnsembleMLMethodFactory;
import org.encog.ensemble.EnsembleTrainFactory;
import org.encog.ensemble.stacking.Stacking;

import helpers.DataLoader;
import helpers.Labeler;

public class StackingET extends EvaluationTechnique {

	private int dataSetSize;

	public StackingET(List<Integer> sizes, int dataSetSize, Labeler fullLabel, EnsembleMLMethodFactory mlMethod, EnsembleTrainFactory trainFactory, EnsembleAggregator aggregator) {
		this.sizes = sizes;
		this.dataSetSize = dataSetSize;
		this.label = fullLabel;
		this.mlMethod = mlMethod;
		this.trainFactory = trainFactory;
		this.aggregator = aggregator;
	}

	@Override
	public void init(DataLoader dataLoader) {
		ensemble = new Stacking(sizes.get(currentSizeIndex),dataSetSize,mlMethod,trainFactory,aggregator);
		setTrainingSet(dataLoader.getTrainingSet());
		setSelectionSet(dataLoader.getTestSet());
		ensemble.setTrainingData(trainingSet);
	}

	@Override
	public void step(boolean verbose) {
		// TODO Auto-generated method stub
		
	}
	
}
